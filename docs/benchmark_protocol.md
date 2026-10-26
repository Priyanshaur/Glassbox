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

## CPU run parameters (addendum, fixed before the first timing)

The protocol above is the reference. A first calibration on the laptop CPU (6
threads, fp32, not reported as results) measured these per-step costs:

| case | prefill | decode per step |
|---|---|---|
| cached, n=1, short | 50 ms | 21 ms |
| cached, n=8, long | 1.06 s | 28 ms |
| cached, n=32, long | 4.4 s | 52 ms |
| int8 (accepted variant), n=1, short | 55 ms | 33 ms |
| no cache, n=1, short | - | 32 ms at context 23 (grows with context) |
| no cache, n=8, short | - | 109 ms at context 23 |
| no cache, n=1, long | - | 154 ms at context 256 |

With the reference parameters (8 prompts, G = 128, 13 generations per prompt per
config) the full grid would take about 3.8 hours of CPU, dominated by no-cache and
int8. The CPU run therefore deviates from the reference in exactly these ways;
every deviation is stated again next to the results:

- **K = 4 prompts** (the first 4 of the 8) and **G = 64 new tokens** instead of 8 and
  128. R = 10 repeats, 3 untimed warm-up generations (one generation = one prompt,
  the first), interleaving, medians and spreads are unchanged. "One repetition"
  generates for all K prompts in order.
- **No-cache runs only for** short n=1, short n=8 and long n=1. No-cache n=32
  (both regimes) and long n=8 are not run: their estimated cost is hours each.
  They are listed as "not run" in the table, not omitted.
- **int8** is run with the static cache only.
- **Correctness gate semantics.** tuple vs static are bit-identical by
  construction (M1): their sampled tokens must be equal. The no-cache loop is
  mathematically equivalent but numerically different (about 1e-5 in the logits),
  which can flip a rare sampled token, so its gate is teacher-forced: along the
  static run's tokens, no-cache logits must match cached logits (atol = rtol =
  1e-4). Token agreement with the static run is reported for it, not required.
- Machine state (AC power, background load) is recorded as observed; it was not
  controlled beyond not running anything else during the benchmark.

The full reference grid (K = 8, G = 128, n = 32, all no-cache cells) is intended
for the GPU run.
