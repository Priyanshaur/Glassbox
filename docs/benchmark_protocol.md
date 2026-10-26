# Inference benchmark protocol (M4 onward)

Written before any timing is run. Results are reported as measured, including
null or negative ones. Deviations from this protocol must be stated next to the
numbers they affect.

## What is compared

| axis | values |
|---|---|
| KV cache | **none** (full recompute every step, as `sample_continuations`), **tuple** (the model's default `torch.cat` cache), **static** (`inference.kv_cache.StaticKVCache`) |
| dtype | fp32 everywhere; fp16 on GPU only; int8 weight-only only for the variant accepted in `docs/quantization_results.md` |
| rows per prompt `n` | 1, 8, 32 (same-prompt multi-sample batch; variable-length batching is out of scope) |
| prompt regime | **short**: the template prompt (about 25 tokens); **long**: the first 256 tokens of a validation gold story |
| hardware | laptop CPU (fixed thread count) and Kaggle T4 |

Out of scope: int4, variable-length batching, server latency (separate, M6).

Expectations, stated up front and **not** assumed in the write-up: a cache beats
no cache clearly; static vs tuple is a modest difference; int8 weight-only is
not faster (there is no fused int8 kernel). If the data disagree, the data win.

## Workload (identical for every config)

- Prompts: the first 8 validation prompts from `train.eval_prompts.val_prompts`
  (short regime) and the same 8 stories' first 256 tokens (long regime). Every
  config sees the same prompts in the same order.
- Generation length is **fixed**: exactly G = 128 new tokens, `stop_at_eot=False`
  (EOT is sampled like any token and does not stop the run), so every run does the
  same amount of work. Long regime: 256 + 128 = 384 <= 512 context.
- Sampling: temperature 0.8, top-k 40, seed = validation story idx (sampling cost
  is included, equally, in every config).

## Correctness gate (before any timing)

For each cache config in fp32, the generated token ids must equal the static-cache
engine's tokens for the same seed (default mode). A config that fails is reported
as failing and is **not** timed against the others; it is not silently dropped.
fp16 and int8 are exempt from token equality (different numerics) and report the
fraction of identical tokens against fp32 instead, plus their quality numbers
(gold-story perplexity). A speed number for a config that fails its quality gate
is flagged as such.

## Timing procedure

- **Warm-up:** 3 full untimed generations per config (CUDA kernels and allocator,
  CPU thread pool, caches).
- **Repeats:** R = 10 timed repetitions per config. One repetition generates for
  all 8 prompts in order; the observation for a repetition is the mean per-prompt
  time.
- **Interleaving:** configs are run round-robin within each repeat (config A rep 1,
  config B rep 1, ..., then rep 2), in a seeded random order, so thermal throttling
  or background load affects all configs alike instead of the last one run.
- **Summary:** median across repeats, with the spread: min, max, interquartile
  range (p25-p75) and std. No outlier is dropped; if the relative IQR exceeds 20%
  on CPU the config is re-run once and the noise is flagged if it persists.
- **Comparisons:** ratio of medians with a paired bootstrap CI over repeats
  (paired by repeat index, B = 10,000). Single runs are never reported.
- **Threads and power:** CPU threads fixed and reported (`torch.get_num_threads()`),
  laptop on AC power, nothing else running; GPU: one process on the T4.
- **GPU timing:** `torch.cuda.synchronize()` immediately before starting each timer
  and before stopping it; wall-clock via `time.perf_counter()`. Peak memory:
  `torch.cuda.reset_peak_memory_stats()` before and `max_memory_allocated()` after
  each config. On CPU report in-memory tensor bytes and KV-cache bytes instead of
  process RSS.

## Definitions

- **Time to first token (TTFT):** from just before the first model forward (the
  prompt is already tokenized and on the device) until the first sampled token
  exists and the device is synchronized. It **includes** cache allocation, the
  prompt prefill forward and sampling the first token. It **excludes**
  tokenization, detokenization and building the result object.
- **Decode time:** from the end of the first token to the end of the G-th token
  (synchronized).
- **Decode throughput:** `(G - 1) * n / decode time`, in tokens per second summed
  over the `n` rows; **per-step latency:** `decode time / (G - 1)`.
- **End-to-end latency:** TTFT + decode time (the whole generation, excluding
  tokenization and detokenization).

## Reporting

- Every config measured appears in the table, with its correctness-gate result.
- Environment is recorded next to the numbers: CPU model, torch version, thread
  count, GPU name, dtype, commit.
- The raw per-repeat timings are saved as JSON next to the markdown table so the
  summary can be recomputed.
- Statements like "faster" require the paired-bootstrap CI of the ratio to exclude
  1; otherwise the result is reported as "no measurable difference".
