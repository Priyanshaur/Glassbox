# Inference server (Phase 5, M6)

`inference/server.py`. One model loaded once, one generation at a time, batch size 1
per request plus a same-prompt multi-sample option (`n`). Variable-length batching
across requests is a documented stretch goal, not implemented.

Run (loopback only by default):

```bash
python -m inference.server --weights checkpoints/glassbox-sft-final/sft_final_best.pt
```

Options: `--precision fp32|int8` (default fp32; `int8` quantizes an fp32 checkpoint at
startup, or loads an int8 file written by `inference.quantize.save_quantized`),
`--device`, `--host` (default `127.0.0.1`), `--port`, `--cors-origin` (repeatable),
`--max-new-tokens-cap` (256), `--max-n` (8), `--max-queue` (4), `--timeout` (60 s),
`--log-prompts` (off).

## Implementation note: Starlette, not FastAPI

The development machine has FastAPI 0.115.6 installed next to Starlette 1.8.0, which
cannot construct a FastAPI app (`Router.__init__() got an unexpected keyword argument
'on_startup'`). FastAPI is a thin layer over Starlette and pydantic, so the server
uses those two directly (request validation is still pydantic). `requirements.txt`
still lists `fastapi`; a consistent `fastapi`/`starlette` pair would also work, but
nothing here needs FastAPI itself.

## Endpoints

### `GET /health`
`200` when ready, `503` while the model loads (status `loading`) or if loading
failed (`error`). Body: `status`, `ready`, `limits`, `queue.pending`, `stats`
(requests, rejected_429, timeouts, streams_aborted, model_loads) and, when ready,
`model` (precision, device, context_len, vocab_size, n_layers, d_model, tensor_bytes).

### `POST /generate`
Request (unknown fields are rejected with 422):

| field | default | limits |
|---|---|---|
| `prompt` | required | 1 to 2000 characters |
| `max_new_tokens` | 64 | 1 to 256; prompt tokens + this must fit the context length |
| `temperature` | 0.8 | 0 to 5 |
| `top_k` | 50 | >= 1, or null |
| `top_p` | null | (0, 1] |
| `seed` | 0 | 0 to 2^31 - 1 (same seed + request = same output) |
| `n` | 1 | 1 to 8 |
| `stop_at_eot` | true | |

Response: `texts` (one per sample), `eot_hit`, `n_prompt_tokens`, `n_new_tokens`
(steps run), and `timing` = `queue_ms`, `ttft_ms` (generation start to first sampled
token, including tokenization, cache allocation and prefill), `decode_ms`,
`total_ms` (admission to completion). **No raw token ids are returned.**

### `POST /generate/stream`
Same fields without `n`. Server-Sent Events: `text` events (`{"text": ...}`), then one
`done` event (`finish_reason` = `eot` | `length` | `timeout` | `cancelled`,
`n_new_tokens`, `ttft_ms`, `total_ms`) or an `error` event.

### Status codes
`422` validation (also unknown fields, over-long prompt, context overflow, caps),
`413` body over 64 KB, `429` queue full (with `Retry-After: 1`), `503` not ready,
`504` request timeout, `500` generation failure (no internals in the body).

## Operational behaviour

- **Serialization.** One worker thread runs generations; `--max-queue` bounds the
  admitted requests (running + waiting) and the rest get 429.
- **Timeouts.** `--timeout` is wall-clock from admission, queue wait included. When it
  expires the request gets 504 and the generation is stopped cooperatively (checked
  after every token), so a slow call cannot hold the queue. A request still waiting is
  cancelled before it starts.
- **Disconnects.** If a streaming client disconnects, the stream is cancelled and the
  generation stops at the next token.
- **Binding and CORS.** Binds to `127.0.0.1`; starting on another host prints a warning
  (there is no authentication). No CORS headers unless explicit origins are given with
  `--cors-origin`; `*` and scheme-less origins are refused at startup.
- **Logging.** One line per request: id, endpoint, status, prompt length in characters,
  token counts and timing. **Prompt text is not logged** unless `--log-prompts`.

## Tests

`tests/test_server.py` (debug model, CPU): readiness and 503-while-loading, model
loaded once, response shape without token ids, HTTP result equal to a direct engine
call, seed determinism, every validation rule, body-size and malformed-JSON handling,
no internals leaked on 500, streamed text equal to the non-streamed text, client
disconnect stops generation (raw ASGI client, because the Starlette test client
buffers streams), queue-full 429 and recovery, slow-generation timeout that frees the
queue, queued request timing out without running, prompt text absent from logs by
default, CORS default/explicit/wildcard-refused, loopback default and warning,
int8-file and fp32-checkpoint-with-int8 loading, and a real-checkpoint smoke test
(skipped when the checkpoint is absent). Engine hooks used by the server
(`should_stop`, timing meta, stream info) are tested in `tests/test_inference.py`.

## Later: where steering would go (Phase 7). Note only, nothing implemented

A steering vector (from the SAE / feature work) adds a vector to the residual stream
after block L. The model already supports this: `GPT.forward(..., inject={L: vec})`
adds `vec` to the output of block L. The only server-side change would be additive:

- **Engine.** `Generator.generate/stream` would accept an optional `steer`
  (`layer`, `vector`, `scale`) and pass `inject={layer: scale * vector}` to every
  `model(...)` call inside `Generator._iter` (prefill and each decode step).
- **API.** `GenerateRequest`/`StreamRequest` gain one optional field, e.g.
  `steer: {"layer": int, "vector_id": str, "scale": float}`. Vectors would be loaded
  server-side from the interp outputs and referenced by id (not sent in requests,
  they are `d_model` floats and need validation). Because request models already
  reject unknown fields, a client cannot use the field before the server supports it,
  and existing clients are unaffected. The response would echo the steering applied.
- **Caveat to carry over from `model/gpt.py`.** An injection only affects tokens
  processed after it is applied; K/V already in the cache keep their old values, so
  steering during prefill and steering only during decode behave differently, and the
  API should say which one it does.
- **Static cache.** Nothing in `StaticKVCache` changes; cached K/V simply reflect the
  steered activations from the step they were written.
