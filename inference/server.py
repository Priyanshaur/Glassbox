"""Inference server (Phase 5, M6), built on Starlette + pydantic.

Why Starlette directly: the machine this was developed on has FastAPI 0.115.6
installed next to Starlette 1.8.0, which cannot even construct a FastAPI app
(`Router.__init__() got an unexpected keyword argument 'on_startup'`). FastAPI is
only a thin layer over Starlette + pydantic, so this module uses those two
directly (request validation is still pydantic) and works with the versions that
are installed. See docs/server.md.

One model, loaded once at startup, one generation at a time (a single worker
thread), batch size 1 per request plus the same-prompt multi-sample option
(`n`, capped). Variable-length batching across requests is out of scope.

Endpoints
  GET  /health            readiness (503 while loading), model info, limits, queue depth
  POST /generate          JSON in, JSON out (texts, eot_hit, token counts, timing)
  POST /generate/stream   Server-Sent Events: `text` pieces, then `done` (or `error`)

Safety defaults: bind 127.0.0.1; no CORS headers unless explicit origins are
configured (a wildcard is refused); prompt text is not logged unless
`log_prompts` is set; every request has a wall-clock timeout (queue wait
included) that stops the generation cooperatively, so a slow call cannot hold
the queue; a streaming client that disconnects stops its generation; the queue
is bounded (429 when full). There is no authentication: keep it on loopback.

Later (Phase 7) steering would be injected where the engine calls the model;
see docs/server.md. Nothing for it is implemented here.
"""
import argparse
import asyncio
import json
import logging
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from inference.generate import Generator

log = logging.getLogger("inference.server")
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_BODY_BYTES = 64_000


@dataclass
class ServerConfig:
    config_path: str = "configs/base_30m.yaml"
    weights: str = "checkpoints/glassbox-sft-final/sft_final_best.pt"
    precision: str = "fp32"              # fp32 | int8
    device: str = "cpu"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: List[str] = field(default_factory=list)
    max_new_tokens_cap: int = 256
    max_n: int = 8
    max_queue: int = 4                   # admitted requests: running + waiting
    request_timeout_s: float = 60.0      # wall clock from admission, queue wait included
    max_prompt_chars: int = 2000
    log_prompts: bool = False            # prompt text is private by default

    def __post_init__(self):
        if self.precision not in ("fp32", "int8"):
            raise ValueError(f"precision must be fp32 or int8, got {self.precision!r}")
        for o in self.cors_origins:
            if "*" in o or not o.startswith(("http://", "https://")):
                raise ValueError(f"CORS origin {o!r} must be an explicit http(s) origin; "
                                 "a wildcard is never allowed")
        if self.max_queue < 1 or self.max_n < 1 or self.request_timeout_s <= 0:
            raise ValueError("max_queue, max_n must be >= 1 and request_timeout_s > 0")


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")      # unknown fields are a 422, not ignored
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(64, ge=1)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: Optional[int] = Field(50, ge=1)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    n: int = Field(1, ge=1)
    stop_at_eot: bool = True


class StreamRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(64, ge=1)
    temperature: float = Field(0.8, ge=0.0, le=5.0)
    top_k: Optional[int] = Field(50, ge=1)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    seed: int = Field(0, ge=0, le=2**31 - 1)


class Runtime:
    """Everything the handlers share. `pending` (admitted = running + waiting) is
    guarded by a lock: admission happens on the event loop, release in the worker
    thread when the job ends (so it is released even if the request already timed
    out or its event loop is gone)."""

    def __init__(self, cfg, loader, model, tok):
        self.cfg, self.loader = cfg, loader
        self.model = self.tok = self.generator = None
        self.ready, self.load_error = False, None
        self.pending = 0
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "rejected_429": 0, "timeouts": 0, "streams_aborted": 0,
                      "model_loads": 0}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")
        if model is not None:
            self.set_model(model, tok)

    def try_admit(self):
        with self.lock:
            if self.pending >= self.cfg.max_queue:
                return False
            self.pending += 1
            return True

    def release(self):
        with self.lock:
            self.pending -= 1

    def set_model(self, model, tok):
        self.model, self.tok = model, tok
        self.generator = Generator(model, tok)
        self.ready = True
        self.stats["model_loads"] += 1


def default_loader(cfg: ServerConfig):
    """Load tokenizer and weights per the config (called once, in a thread)."""
    import torch

    from common.config import load_config
    from inference.quantize import FORMAT, load_quantized, quantize_model
    from model.gpt import GPT
    from tokenizer.bpe import BPETokenizer

    c = load_config(cfg.config_path)
    tok = BPETokenizer.load(c["data"]["tokenizer_path"])
    blob = torch.load(cfg.weights, map_location="cpu", weights_only=True)
    if isinstance(blob, dict) and blob.get("format") == FORMAT:
        model = load_quantized(c["model"], cfg.weights)           # an int8 file
    else:
        model = GPT(c["model"])
        model.load_state_dict(blob)
        if cfg.precision == "int8":
            model = quantize_model(model, True)
    return model.to(cfg.device).eval(), tok


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def http_exception_handler(request, exc):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code,
                        headers=getattr(exc, "headers", None))


def create_app(cfg: ServerConfig, loader=None, model=None, tok=None) -> Starlette:
    """Build the app. Pass `model`+`tok` for an immediately ready server, or a
    `loader` returning (model, tok) to load in the background (503 until ready)."""
    rt = Runtime(cfg, loader, model, tok)

    @asynccontextmanager
    async def lifespan(app):
        task = None
        if rt.loader is not None and not rt.ready:
            async def load():
                try:
                    m, t = await asyncio.to_thread(rt.loader)
                    rt.set_model(m, t)
                except Exception as e:                       # noqa: BLE001
                    rt.load_error = type(e).__name__
                    log.exception("model load failed")
            task = asyncio.create_task(load())
        yield
        if task is not None and not task.done():
            task.cancel()
        rt.executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ helpers
    def require_ready():
        if not rt.ready:
            raise HTTPException(503, "model is loading" if rt.load_error is None
                                else f"model failed to load ({rt.load_error})")

    async def read_request(request, model_cls):
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            raise HTTPException(413, f"request body exceeds {MAX_BODY_BYTES} bytes")
        try:
            data = json.loads(raw or b"null")
        except ValueError:
            raise HTTPException(422, "body must be valid JSON")
        try:
            return model_cls.model_validate(data)
        except ValidationError as e:
            raise HTTPException(422, [{"loc": list(x["loc"]), "msg": x["msg"], "type": x["type"]}
                                      for x in e.errors()])

    def validate(req, n=1):
        """Context/limit checks that need the server config or the tokenizer (422)."""
        if len(req.prompt) > cfg.max_prompt_chars:
            raise HTTPException(422, f"prompt is {len(req.prompt)} characters; the limit is "
                                     f"{cfg.max_prompt_chars}")
        if req.max_new_tokens > cfg.max_new_tokens_cap:
            raise HTTPException(422, f"max_new_tokens {req.max_new_tokens} exceeds the cap "
                                     f"{cfg.max_new_tokens_cap}")
        if n > cfg.max_n:
            raise HTTPException(422, f"n={n} exceeds the cap {cfg.max_n}")
        ctx = rt.model.cfg["context_len"]
        p = len(rt.tok.encode(req.prompt))
        if p + req.max_new_tokens > ctx:
            raise HTTPException(422, f"prompt ({p} tokens) + max_new_tokens "
                                     f"({req.max_new_tokens}) exceeds the context length ({ctx})")
        return p

    def admit():
        if not rt.try_admit():
            rt.stats["rejected_429"] += 1
            raise HTTPException(429, "server busy: queue is full", headers={"Retry-After": "1"})

    def on_done(f):
        if not f.cancelled():
            f.exception()                                    # mark retrieved

    def log_request(rid, endpoint, status, req, **kw):
        extra = " ".join(f"{k}={v}" for k, v in kw.items())
        log.info("req=%s endpoint=%s status=%s prompt_chars=%d %s", rid, endpoint, status,
                 len(req.prompt), extra)
        if cfg.log_prompts:
            log.info("req=%s prompt=%r", rid, req.prompt[:200])

    # ------------------------------------------------------------------ routes
    async def health(request):
        from inference.quantize import tensor_bytes
        body = {"status": "ok" if rt.ready else ("error" if rt.load_error else "loading"),
                "ready": rt.ready,
                "limits": {"max_new_tokens_cap": cfg.max_new_tokens_cap, "max_n": cfg.max_n,
                           "max_queue": cfg.max_queue, "request_timeout_s": cfg.request_timeout_s,
                           "max_prompt_chars": cfg.max_prompt_chars},
                "queue": {"pending": rt.pending}, "stats": dict(rt.stats)}
        if rt.ready:
            c = rt.model.cfg
            body["model"] = {"precision": cfg.precision, "device": str(cfg.device),
                             "context_len": c["context_len"], "vocab_size": c["vocab_size"],
                             "n_layers": c["n_layers"], "d_model": c["d_model"],
                             "tensor_bytes": tensor_bytes(rt.model)}
        return JSONResponse(body, status_code=200 if rt.ready else 503)

    async def generate(request):
        require_ready()
        req = await read_request(request, GenerateRequest)
        rid = uuid.uuid4().hex[:8]
        validate(req, req.n)
        admit()
        rt.stats["requests"] += 1
        t_admit = time.monotonic()
        deadline = t_admit + cfg.request_timeout_s
        cancel = threading.Event()

        def work():
            try:
                if cancel.is_set():
                    return None
                started = time.monotonic()
                res = rt.generator.generate(
                    req.prompt, n=req.n, max_new=req.max_new_tokens, temperature=req.temperature,
                    top_k=req.top_k, top_p=req.top_p, seed=req.seed, stop_at_eot=req.stop_at_eot,
                    should_stop=lambda: cancel.is_set() or time.monotonic() > deadline)
                return res, started
            finally:
                rt.release()

        fut = asyncio.get_running_loop().run_in_executor(rt.executor, work)
        fut.add_done_callback(on_done)
        try:
            out = await asyncio.wait_for(asyncio.shield(fut), timeout=cfg.request_timeout_s)
        except asyncio.TimeoutError:
            cancel.set()
            rt.stats["timeouts"] += 1
            log_request(rid, "generate", 504, req, reason="timeout_waiting")
            raise HTTPException(504, f"request exceeded the {cfg.request_timeout_s:g}s timeout")
        except asyncio.CancelledError:                       # client went away
            cancel.set()
            raise
        except Exception:                                    # noqa: BLE001
            log.exception("req=%s generation failed", rid)
            raise HTTPException(500, "generation failed")
        res, started = out
        if res.meta.get("stopped"):
            rt.stats["timeouts"] += 1
            log_request(rid, "generate", 504, req, reason="timeout_generating")
            raise HTTPException(504, f"request exceeded the {cfg.request_timeout_s:g}s timeout")
        now = time.monotonic()
        timing = {"queue_ms": round((started - t_admit) * 1000, 1),
                  "ttft_ms": round((res.meta["ttft_s"] or 0.0) * 1000, 1),
                  "decode_ms": round(res.meta["decode_s"] * 1000, 1),
                  "total_ms": round((now - t_admit) * 1000, 1)}
        log_request(rid, "generate", 200, req, n=req.n, new_tokens=res.n_steps,
                    total_ms=timing["total_ms"])
        return JSONResponse({"texts": res.texts, "eot_hit": res.eot_hit,
                             "n_prompt_tokens": res.n_prompt, "n_new_tokens": res.n_steps,
                             "timing": timing})

    async def generate_stream(request):
        require_ready()
        req = await read_request(request, StreamRequest)
        rid = uuid.uuid4().hex[:8]
        validate(req, 1)
        admit()
        rt.stats["requests"] += 1
        loop = asyncio.get_running_loop()
        t_admit = time.monotonic()
        deadline = t_admit + cfg.request_timeout_s
        cancel = threading.Event()
        q: asyncio.Queue = asyncio.Queue()
        info: dict = {}

        def put(item):
            try:
                loop.call_soon_threadsafe(q.put_nowait, item)
            except RuntimeError:                              # loop already closed
                pass

        def work():
            try:
                if cancel.is_set():
                    put(("done", {"n_new": 0, "eot": False, "stopped": True, "ttft_s": None}))
                    return
                for piece in rt.generator.stream(
                        req.prompt, max_new=req.max_new_tokens, temperature=req.temperature,
                        top_k=req.top_k, top_p=req.top_p, seed=req.seed,
                        should_stop=lambda: cancel.is_set() or time.monotonic() > deadline,
                        info=info):
                    put(("text", piece))
                put(("done", dict(info)))
            except Exception:                                 # noqa: BLE001
                log.exception("req=%s streaming generation failed", rid)
                put(("error", "generation failed"))
            finally:
                rt.release()

        fut = loop.run_in_executor(rt.executor, work)
        fut.add_done_callback(on_done)

        async def events():
            finished = False
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    try:
                        kind, payload = await asyncio.wait_for(q.get(), timeout=max(remaining, 0.001))
                    except asyncio.TimeoutError:
                        rt.stats["timeouts"] += 1
                        finished = True
                        yield sse("error", {"detail": f"request exceeded the "
                                                       f"{cfg.request_timeout_s:g}s timeout"})
                        return
                    if kind == "text":
                        yield sse("text", {"text": payload})
                    elif kind == "done":
                        finished = True
                        timed_out = bool(payload.get("stopped")) and time.monotonic() >= deadline
                        reason = ("eot" if payload.get("eot") else "timeout" if timed_out
                                  else "cancelled" if payload.get("stopped") else "length")
                        if timed_out:
                            rt.stats["timeouts"] += 1
                        log_request(rid, "generate/stream", 200, req, new_tokens=payload["n_new"],
                                    finish=reason)
                        yield sse("done", {"finish_reason": reason, "n_new_tokens": payload["n_new"],
                                           "ttft_ms": round((payload["ttft_s"] or 0.0) * 1000, 1),
                                           "total_ms": round((time.monotonic() - t_admit) * 1000, 1)})
                        return
                    else:
                        finished = True
                        yield sse("error", {"detail": payload})
                        return
            finally:
                cancel.set()                                  # disconnect, error or normal end
                if not finished:
                    rt.stats["streams_aborted"] += 1

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    middleware = []
    if cfg.cors_origins:
        middleware.append(Middleware(CORSMiddleware, allow_origins=list(cfg.cors_origins),
                                     allow_methods=["GET", "POST"], allow_headers=["content-type"]))
    app = Starlette(routes=[Route("/health", health, methods=["GET"]),
                            Route("/generate", generate, methods=["POST"]),
                            Route("/generate/stream", generate_stream, methods=["POST"])],
                    lifespan=lifespan, middleware=middleware,
                    exception_handlers={HTTPException: http_exception_handler})
    app.state.runtime = rt
    return app


def parse_args(argv=None) -> ServerConfig:
    d = ServerConfig()
    ap = argparse.ArgumentParser(description="Glassbox inference server")
    ap.add_argument("--config", dest="config_path", default=d.config_path)
    ap.add_argument("--weights", default=d.weights)
    ap.add_argument("--precision", choices=["fp32", "int8"], default=d.precision)
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--host", default=d.host, help="default 127.0.0.1 (loopback)")
    ap.add_argument("--port", type=int, default=d.port)
    ap.add_argument("--cors-origin", dest="cors_origins", action="append", default=[],
                    help="explicit allowed origin, repeatable; '*' is refused")
    ap.add_argument("--max-new-tokens-cap", type=int, default=d.max_new_tokens_cap)
    ap.add_argument("--max-n", type=int, default=d.max_n)
    ap.add_argument("--max-queue", type=int, default=d.max_queue)
    ap.add_argument("--timeout", dest="request_timeout_s", type=float, default=d.request_timeout_s)
    ap.add_argument("--log-prompts", action="store_true", help="log prompt text (off by default)")
    a = ap.parse_args(argv)
    return ServerConfig(**vars(a))


def main(argv=None):
    cfg = parse_args(argv)
    if cfg.host not in LOOPBACK:
        print(f"WARNING: binding to {cfg.host}: the server has no authentication; "
              "exposing it beyond loopback is unsafe.", file=sys.stderr)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app = create_app(cfg, loader=lambda: default_loader(cfg))
    import uvicorn
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
