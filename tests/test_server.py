"""M6 server tests: FastAPI TestClient on the debug model (CPU), plus a raw ASGI
client for the streaming-disconnect case (TestClient buffers streaming responses
and cannot signal a disconnect)."""
import asyncio
import json
import logging
import os
import threading
import time

import pytest
import torch
from starlette.testclient import TestClient

from common.config import load_config
from inference.generate import Generator
from inference.quantize import quantize_model, save_quantized
from inference.server import ServerConfig, create_app, default_loader, main, parse_args
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer

PROMPT = "Write a short story using all of these words: dragon, castle, brave.\n\nStory:"
DEBUG = "configs/debug_cpu.yaml"
SFT = "checkpoints/glassbox-sft-final/sft_final_best.pt"
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        _TOK = BPETokenizer.load("tokenizer/tok8192.json")
    return _TOK


def debug_model():
    torch.manual_seed(4)
    return GPT(load_config(DEBUG)["model"]).eval()


def make_app(**kw):
    return create_app(ServerConfig(**kw), model=debug_model(), tok=tok())


def body(**kw):
    d = {"prompt": PROMPT, "max_new_tokens": 12, "temperature": 1.0, "top_k": None, "seed": 1}
    d.update(kw)
    return d


def parse_sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(l.split(": ", 1) for l in block.splitlines() if ": " in l)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def wait_until(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------- health / loading
def test_health_ready_fields():
    c = TestClient(make_app())
    r = c.get("/health")
    j = r.json()
    assert r.status_code == 200 and j["ready"] and j["status"] == "ok"
    assert j["model"]["precision"] == "fp32" and j["model"]["context_len"] == 64
    assert set(j["limits"]) >= {"max_new_tokens_cap", "max_n", "max_queue", "request_timeout_s"}
    assert j["queue"]["pending"] == 0


def test_503_until_loaded_and_model_loaded_once():
    gate, calls = threading.Event(), []

    def loader():
        calls.append(1)
        gate.wait(10)
        return debug_model(), tok()

    app = create_app(ServerConfig(), loader=loader)
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 503 and r.json()["status"] == "loading"
        assert c.post("/generate", json=body()).status_code == 503
        gate.set()
        assert wait_until(lambda: c.get("/health").status_code == 200)
        for s in range(3):
            assert c.post("/generate", json=body(seed=s)).status_code == 200
    assert len(calls) == 1 and app.state.runtime.stats["model_loads"] == 1


# ---------------------------------------------------------------- /generate
def test_generate_shape_no_token_ids_and_matches_direct_engine():
    app = make_app()
    c = TestClient(app)
    r = c.post("/generate", json=body(n=2, seed=7))
    j = r.json()
    assert r.status_code == 200
    assert set(j) == {"texts", "eot_hit", "n_prompt_tokens", "n_new_tokens", "timing"}   # no raw ids
    assert len(j["texts"]) == 2 and len(j["eot_hit"]) == 2
    assert set(j["timing"]) == {"queue_ms", "ttft_ms", "decode_ms", "total_ms"}
    direct = Generator(app.state.runtime.model, tok()).generate(
        PROMPT, n=2, max_new=12, temperature=1.0, top_k=None, seed=7)
    assert j["texts"] == direct.texts and j["n_new_tokens"] == direct.n_steps


def test_generate_seed_determinism():
    c = TestClient(make_app())
    a = c.post("/generate", json=body(seed=3)).json()["texts"]
    assert a == c.post("/generate", json=body(seed=3)).json()["texts"]
    assert a != c.post("/generate", json=body(seed=4)).json()["texts"]


@pytest.mark.parametrize("payload,cfg_kw,fragment", [
    ({"prompt": ""}, {}, "at least 1"),
    ({"prompt": "x" * 50}, {"max_prompt_chars": 20}, "characters"),
    ({"max_new_tokens": 60}, {}, "context length"),            # debug ctx 64, prompt ~23 tokens
    ({"max_new_tokens": 31}, {"max_new_tokens_cap": 30}, "cap"),
    ({"n": 5}, {"max_n": 4}, "cap"),
    ({"temperature": -1}, {}, "greater than or equal"),
    ({"top_p": 0}, {}, "greater than"),
    ({"top_k": 0}, {}, "greater than or equal"),
    ({"steer": {"layer": 3}}, {}, "Extra inputs"),             # unknown fields are rejected
])
def test_validation_errors_are_422(payload, cfg_kw, fragment):
    c = TestClient(make_app(**cfg_kw))
    r = c.post("/generate", json={**body(), **payload})
    assert r.status_code == 422 and fragment in json.dumps(r.json())


def test_oversized_and_malformed_bodies_are_rejected():
    c = TestClient(make_app())
    assert c.post("/generate", content=b"{not json", headers={"content-type": "application/json"}
                  ).status_code == 422
    assert c.post("/generate", content=b"x" * 70_000).status_code == 413


def test_stream_rejects_n_field():
    c = TestClient(make_app())
    assert c.post("/generate/stream", json={**body(), "n": 2}).status_code == 422


def test_internal_errors_do_not_leak_details(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom-secret-detail")

    monkeypatch.setattr(Generator, "generate", boom)
    r = TestClient(make_app()).post("/generate", json=body())
    assert r.status_code == 500 and "boom-secret-detail" not in r.text


# ---------------------------------------------------------------- streaming
def test_stream_pieces_equal_nonstreaming_text_and_done_event():
    c = TestClient(make_app())
    want = c.post("/generate", json=body(seed=5)).json()["texts"][0]
    r = c.post("/generate/stream", json=body(seed=5))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = parse_sse(r.text)
    assert ev[-1][0] == "done" and ev[-1][1]["finish_reason"] in {"eot", "length"}
    assert "".join(d["text"] for k, d in ev if k == "text") == want
    assert ev[-1][1]["n_new_tokens"] >= 1 and "ttft_ms" in ev[-1][1]


def _asgi_post(app, path, payload, stop_after_chunks=None):
    """Raw ASGI client; after `stop_after_chunks` response chunks it sends http.disconnect."""
    async def run():
        data = json.dumps(payload).encode()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "http_version": "1.1", "method": "POST", "path": path, "raw_path": path.encode(),
                 "query_string": b"", "scheme": "http", "root_path": "",
                 "client": ("127.0.0.1", 5000), "server": ("test", 80),
                 "headers": [(b"host", b"test"), (b"content-type", b"application/json"),
                             (b"content-length", str(len(data)).encode())]}
        sent, disconnect, chunks = False, asyncio.Event(), []

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": data, "more_body": False}
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(msg):
            if msg["type"] == "http.response.body" and msg.get("body"):
                chunks.append(msg["body"])
                if stop_after_chunks and len(chunks) >= stop_after_chunks:
                    disconnect.set()

        await app(scope, receive, send)
        rt = app.state.runtime
        for _ in range(300):                        # let the worker thread finish inside this loop
            if rt.pending == 0:
                break
            await asyncio.sleep(0.02)
        return chunks

    return asyncio.run(run())


def test_streaming_client_disconnect_stops_generation(monkeypatch):
    calls = {"n": 0}
    orig = Generator._sample

    def slow(logits, gen, temperature, top_k, top_p):
        calls["n"] += 1
        time.sleep(0.02)
        return orig(logits, gen, temperature, top_k, top_p)

    monkeypatch.setattr(Generator, "_sample", staticmethod(slow))
    app = make_app(request_timeout_s=30)
    chunks = _asgi_post(app, "/generate/stream", body(max_new_tokens=40), stop_after_chunks=3)
    rt = app.state.runtime
    assert len(chunks) >= 3
    assert rt.pending == 0 and rt.stats["streams_aborted"] == 1
    assert calls["n"] < 40, f"generation kept running after disconnect ({calls['n']} steps)"
    # the worker is free again: a normal request succeeds immediately
    monkeypatch.setattr(Generator, "_sample", staticmethod(orig))
    assert TestClient(app).post("/generate", json=body()).status_code == 200


def test_completed_stream_is_not_counted_as_aborted():
    app = make_app()
    chunks = _asgi_post(app, "/generate/stream", body(max_new_tokens=6))
    assert chunks and app.state.runtime.stats["streams_aborted"] == 0


# ---------------------------------------------------------------- queue and timeouts
def test_queue_full_returns_429_and_recovers(monkeypatch):
    started, release = threading.Event(), threading.Event()
    orig = Generator.generate

    def blocking(self, *a, **k):
        started.set()
        release.wait(10)
        return orig(self, *a, **k)

    monkeypatch.setattr(Generator, "generate", blocking)
    app = make_app(max_queue=1, request_timeout_s=30)
    first = {}

    def first_call():
        first["r"] = TestClient(app).post("/generate", json=body())

    t = threading.Thread(target=first_call)
    t.start()
    assert started.wait(5)
    r = TestClient(app).post("/generate", json=body())
    assert r.status_code == 429 and r.headers.get("retry-after") == "1"
    release.set()
    t.join(10)
    assert first["r"].status_code == 200
    assert wait_until(lambda: app.state.runtime.pending == 0)
    assert TestClient(app).post("/generate", json=body()).status_code == 200


def test_slow_generation_times_out_and_does_not_block_the_queue(monkeypatch):
    orig = Generator._sample
    calls = {"n": 0}

    def slow(logits, gen, temperature, top_k, top_p):
        calls["n"] += 1
        time.sleep(0.05)
        return orig(logits, gen, temperature, top_k, top_p)

    monkeypatch.setattr(Generator, "_sample", staticmethod(slow))
    app = make_app(request_timeout_s=0.4)
    c = TestClient(app)
    t0 = time.time()
    r = c.post("/generate", json=body(max_new_tokens=40))
    assert r.status_code == 504 and time.time() - t0 < 2.0
    assert wait_until(lambda: app.state.runtime.pending == 0, 3.0)
    assert calls["n"] < 40                                       # the generation really stopped
    assert app.state.runtime.stats["timeouts"] >= 1
    monkeypatch.setattr(Generator, "_sample", staticmethod(orig))
    assert c.post("/generate", json=body()).status_code == 200   # queue is free again


def test_request_waiting_in_queue_times_out_without_running(monkeypatch):
    release, started, calls = threading.Event(), threading.Event(), []
    orig = Generator.generate

    def blocking(self, *a, **k):
        calls.append(1)
        started.set()
        release.wait(10)
        return orig(self, *a, **k)

    monkeypatch.setattr(Generator, "generate", blocking)
    app = make_app(max_queue=2, request_timeout_s=0.5)
    out = []
    ta = threading.Thread(target=lambda: out.append(TestClient(app).post("/generate", json=body())))
    ta.start()
    assert started.wait(5)
    rb = TestClient(app).post("/generate", json=body())          # queued behind the blocked worker
    assert rb.status_code == 504
    release.set()
    ta.join(10)
    assert out[0].status_code == 504                              # A also exceeded 0.5 s
    assert wait_until(lambda: app.state.runtime.pending == 0)
    assert len(calls) == 1                                        # B was cancelled before it started


# ---------------------------------------------------------------- privacy, CORS, binding
def test_prompt_text_is_not_logged_by_default_but_can_be_enabled(caplog):
    secret = "ZEBRA-SECRET-PROMPT-98765"
    with caplog.at_level(logging.INFO, logger="inference.server"):
        TestClient(make_app()).post("/generate", json=body(prompt=secret + " story:"))
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "prompt_chars=" in text and secret not in text
        caplog.clear()
        TestClient(make_app(log_prompts=True)).post("/generate", json=body(prompt=secret + " story:"))
        assert secret in "\n".join(r.getMessage() for r in caplog.records)


def test_cors_default_none_explicit_origin_only_and_no_wildcard():
    origin = "http://localhost:3000"
    c = TestClient(make_app())
    r = c.get("/health", headers={"Origin": origin})
    assert "access-control-allow-origin" not in r.headers
    c = TestClient(make_app(cors_origins=[origin]))
    ok = c.options("/generate", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == origin
    bad = c.options("/generate", headers={"Origin": "http://evil.example",
                                          "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    for o in ("*", "localhost:3000", "http://*.example.com"):
        with pytest.raises(ValueError):
            ServerConfig(cors_origins=[o])


def test_default_bind_is_loopback_and_non_loopback_warns(monkeypatch, capsys):
    assert ServerConfig().host == "127.0.0.1" and parse_args([]).host == "127.0.0.1"
    assert parse_args(["--cors-origin", "http://a.example", "--cors-origin", "http://b.example"]
                      ).cors_origins == ["http://a.example", "http://b.example"]
    import uvicorn
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port, log_level: seen.update(host=host))
    main([])
    assert seen["host"] == "127.0.0.1" and "WARNING" not in capsys.readouterr().err
    main(["--host", "0.0.0.0"])
    assert seen["host"] == "0.0.0.0" and "no authentication" in capsys.readouterr().err
    assert not parse_args([]).log_prompts


# ---------------------------------------------------------------- loading paths
def test_default_loader_int8_file_and_fp32_checkpoint_with_int8_flag(tmp_path):
    m = debug_model()
    q = str(tmp_path / "q8.pt")
    save_quantized(quantize_model(m, True), q)
    f = str(tmp_path / "f32.pt")
    torch.save(m.state_dict(), f)
    for weights in (q, f):
        cfg = ServerConfig(config_path=DEBUG, weights=weights, precision="int8")
        app = create_app(cfg, model=None, tok=None, loader=lambda c=cfg: default_loader(c))
        with TestClient(app) as c:
            assert wait_until(lambda: c.get("/health").status_code == 200)
            assert c.post("/generate", json=body()).status_code == 200


@pytest.mark.skipif(not os.path.exists(SFT), reason="SFT checkpoint not present")
def test_real_sft_checkpoint_smoke():
    cfg = ServerConfig(max_new_tokens_cap=64)
    app = create_app(cfg, loader=lambda: default_loader(cfg))
    with TestClient(app) as c:
        assert wait_until(lambda: c.get("/health").status_code == 200, 60)
        r = c.post("/generate", json={"prompt": PROMPT, "max_new_tokens": 24, "seed": 0})
        assert r.status_code == 200 and r.json()["texts"][0].strip()
