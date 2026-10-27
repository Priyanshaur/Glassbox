# Inference benchmark results

Protocol: `docs/benchmark_protocol.md` (including its CPU addendum). Times are medians over 10 interleaved repeats; spread = interquartile range (p25-p75). Raw per-repeat timings: the `_raw.json` next to this file.

## Environment and parameters

- device: cuda
- torch: 2.11.0+cu128
- threads: 2
- platform: Linux-6.18.55+-x86_64-with-glibc2.39
- processor: x86_64
- logical_cpus: 4
- commit: 8f3597d
- started: 2026-10-26 21:31:46
- power_and_background_load: not controlled or verified
- gpu: Tesla T4
- prompts K=4, generated tokens G=128 (fixed, EOT ignored), repeats R=10, warm-up 3, n in [1, 8, 32], grid `reduced`
- sampling: temperature 0.8, top-k 40; short prompts [23, 23, 23, 23] tokens (validation prompts idx [0, 1, 2, 3]), long prompts [256, 256, 256, 256] tokens (first 256 tokens of validation stories idx [1, 189, 190, 191]: the first K stories with enough tokens, not the same stories as the short regime)

## Correctness gate (before timing)

| check | n | regime | result | detail |
|---|---|---|---|---|
| fp16 | 1 | short | PASS | token agreement with fp32 static 1.000 (informational; numerics differ, equality not required) |
| engine | 1 | short | PASS | engine tokens == bench static loop |
| tuple | 1 | short | PASS | tokens identical (agreement 1.000) |
| none | 1 | short | PASS | teacher-forced max logit diff 1.19e-05 (atol=rtol=1e-4); sampled-token agreement with static 1.000 |
| fp16 | 8 | short | PASS | token agreement with fp32 static 0.998 (informational; numerics differ, equality not required) |
| tuple | 8 | short | PASS | tokens identical (agreement 1.000) |
| none | 8 | short | PASS | teacher-forced max logit diff 1.19e-05 (atol=rtol=1e-4); sampled-token agreement with static 1.000 |
| fp16 | 32 | short | PASS | token agreement with fp32 static 0.917 (informational; numerics differ, equality not required) |
| tuple | 32 | short | PASS | tokens identical (agreement 1.000) |
| fp16 | 1 | long | PASS | token agreement with fp32 static 1.000 (informational; numerics differ, equality not required) |
| tuple | 1 | long | PASS | tokens identical (agreement 1.000) |
| none | 1 | long | PASS | teacher-forced max logit diff 2.00e-05 (atol=rtol=1e-4); sampled-token agreement with static 1.000 |
| fp16 | 8 | long | PASS | token agreement with fp32 static 1.000 (informational; numerics differ, equality not required) |
| tuple | 8 | long | PASS | tokens identical (agreement 1.000) |
| fp16 | 32 | long | PASS | token agreement with fp32 static 0.925 (informational; numerics differ, equality not required) |
| tuple | 32 | long | PASS | tokens identical (agreement 1.000) |

## Results, short prompts (median [p25-p75])

| config | TTFT ms | decode ms/step | tokens/s (all rows) | end-to-end s | KV cache MB | rel. IQR (e2e) |
|---|---|---|---|---|---|---|
| fp32/none/n=1/short | 10 [10-11] | 10.2 [10.1-10.2] | 98.3 [97.6-98.8] | 1.30 [1.30-1.31] | 0.0 | 1.2% |
| fp32/tuple/n=1/short | 10 [10-10] | 10.2 [10.1-10.3] | 97.7 [97.2-98.7] | 1.31 [1.30-1.32] | 4.9 | 1.4% |
| fp32/static/n=1/short | 11 [11-11] | 10.3 [10.3-10.3] | 96.9 [96.7-97.4] | 1.32 [1.32-1.32] | 4.9 | 0.6% |
| fp16/static/n=1/short | 12 [11-12] | 10.8 [10.7-10.8] | 92.9 [92.8-93.2] | 1.38 [1.37-1.38] | 2.5 | 0.5% |
| fp32/none/n=8/short | 12 [11-12] | 19.9 [19.5-20.2] | 402.3 [396.3-410.3] | 2.54 [2.49-2.58] | 0.0 | 3.4% |
| fp32/tuple/n=8/short | 11 [11-12] | 10.3 [10.3-10.4] | 773.8 [771.5-777.4] | 1.32 [1.32-1.33] | 39.3 | 0.7% |
| fp32/static/n=8/short | 12 [11-12] | 10.4 [10.3-10.5] | 771.6 [765.5-776.9] | 1.33 [1.32-1.34] | 39.6 | 1.5% |
| fp16/static/n=8/short | 11 [11-11] | 11.0 [11.0-11.0] | 728.2 [725.3-729.0] | 1.41 [1.41-1.41] | 19.8 | 0.5% |
| fp32/tuple/n=32/short | 16 [15-17] | 10.6 [10.5-10.6] | 3031.4 [3016.7-3058.4] | 1.36 [1.35-1.36] | 157.3 | 1.3% |
| fp32/static/n=32/short | 16 [16-17] | 10.6 [10.6-10.7] | 3012.8 [2981.3-3030.0] | 1.37 [1.36-1.38] | 158.3 | 1.5% |
| fp16/static/n=32/short | 12 [12-12] | 11.1 [11.0-11.2] | 2877.6 [2853.5-2897.0] | 1.42 [1.41-1.44] | 79.2 | 1.5% |

## Results, long prompts (median [p25-p75])

| config | TTFT ms | decode ms/step | tokens/s (all rows) | end-to-end s | KV cache MB | rel. IQR (e2e) |
|---|---|---|---|---|---|---|
| fp32/none/n=1/long | 12 [11-12] | 11.7 [11.6-11.7] | 85.8 [85.3-86.4] | 1.49 [1.48-1.50] | 0.0 | 1.3% |
| fp32/tuple/n=1/long | 13 [12-14] | 10.3 [10.2-10.4] | 97.1 [96.4-97.8] | 1.32 [1.31-1.33] | 12.6 | 1.3% |
| fp32/static/n=1/long | 14 [13-15] | 10.3 [10.3-10.3] | 96.9 [96.7-97.3] | 1.32 [1.32-1.33] | 12.6 | 0.6% |
| fp16/static/n=1/long | 11 [11-11] | 10.7 [10.6-10.7] | 93.4 [93.1-93.9] | 1.37 [1.36-1.38] | 6.3 | 0.8% |
| fp32/tuple/n=8/long | 43 [38-44] | 10.3 [10.3-10.4] | 774.1 [765.9-780.1] | 1.35 [1.34-1.37] | 100.4 | 1.8% |
| fp32/static/n=8/long | 41 [39-46] | 10.4 [10.3-10.5] | 770.8 [765.0-773.8] | 1.36 [1.36-1.37] | 100.7 | 1.3% |
| fp16/static/n=8/long | 15 [14-17] | 11.0 [10.9-11.1] | 726.5 [722.0-731.4] | 1.41 [1.40-1.42] | 50.3 | 1.5% |
| fp32/tuple/n=32/long | 202 [199-208] | 12.6 [12.5-12.7] | 2543.2 [2523.0-2555.5] | 1.80 [1.79-1.82] | 401.6 | 1.6% |
| fp32/static/n=32/long | 206 [203-209] | 10.8 [10.7-10.8] | 2967.5 [2959.9-2981.4] | 1.58 [1.57-1.58] | 402.7 | 1.0% |
| fp16/static/n=32/long | 43 [42-47] | 11.0 [10.9-11.1] | 2904.7 [2871.7-2931.1] | 1.45 [1.43-1.46] | 201.3 | 1.9% |

## Not run

- fp32/none/n=32/short: estimated cost too high for the kernel's time cap (tens of GPU minutes per config)
- fp32/none/n=8/long: estimated cost too high for the kernel's time cap (tens of GPU minutes per config)
- fp32/none/n=32/long: estimated cost too high for the kernel's time cap (tens of GPU minutes per config)

## Comparisons (for 'A vs B', speedup = median time of B / median time of A, so >1 means A (the first-named config) is faster)

Paired bootstrap over repeats (B=10,000). 'faster' / 'slower' only when the 95% CI excludes 1.

| comparison | end-to-end | decode | TTFT |
|---|---|---|---|
| static vs tuple, n=1, short | 0.99x [0.98, 1.00] slower | 0.99x [0.98, 1.00] slower | 0.94x [0.92, 0.96] slower |
| tuple vs none, n=1, short | 0.99x [0.99, 1.01] no measurable difference | 0.99x [0.99, 1.01] no measurable difference | 1.00x [0.98, 1.02] no measurable difference |
| static vs none, n=1, short | 0.98x [0.98, 1.00] slower | 0.99x [0.98, 1.00] slower | 0.94x [0.92, 0.97] slower |
| fp16 static vs fp32 static, n=1, short | 0.96x [0.95, 0.97] slower | 0.96x [0.95, 0.97] slower | 0.95x [0.93, 0.97] slower |
| static vs tuple, n=8, short | 1.00x [0.98, 1.01] no measurable difference | 1.00x [0.98, 1.01] no measurable difference | 0.95x [0.88, 1.01] no measurable difference |
| tuple vs none, n=8, short | 1.92x [1.87, 1.96] faster | 1.92x [1.88, 1.96] faster | 1.03x [0.99, 1.10] no measurable difference |
| static vs none, n=8, short | 1.91x [1.86, 1.95] faster | 1.92x [1.87, 1.96] faster | 0.98x [0.94, 1.05] no measurable difference |
| fp16 static vs fp32 static, n=8, short | 0.94x [0.94, 0.96] slower | 0.94x [0.94, 0.96] slower | 1.04x [0.98, 1.08] no measurable difference |
| static vs tuple, n=32, short | 0.99x [0.98, 1.00] no measurable difference | 0.99x [0.98, 1.00] no measurable difference | 1.00x [0.90, 1.09] no measurable difference |
| fp16 static vs fp32 static, n=32, short | 0.96x [0.95, 0.97] slower | 0.96x [0.95, 0.97] slower | 1.38x [1.33, 1.47] faster |
| static vs tuple, n=1, long | 1.00x [0.99, 1.01] no measurable difference | 1.00x [0.99, 1.01] no measurable difference | 0.92x [0.78, 1.09] no measurable difference |
| tuple vs none, n=1, long | 1.13x [1.12, 1.14] faster | 1.13x [1.12, 1.14] faster | 0.91x [0.83, 1.03] no measurable difference |
| static vs none, n=1, long | 1.13x [1.12, 1.13] faster | 1.13x [1.12, 1.14] faster | 0.84x [0.78, 0.93] slower |
| fp16 static vs fp32 static, n=1, long | 0.96x [0.96, 0.97] slower | 0.96x [0.96, 0.97] slower | 1.23x [1.10, 1.35] faster |
| static vs tuple, n=8, long | 1.00x [0.99, 1.01] no measurable difference | 1.00x [0.98, 1.01] no measurable difference | 1.05x [0.90, 1.13] no measurable difference |
| fp16 static vs fp32 static, n=8, long | 0.96x [0.95, 0.98] slower | 0.94x [0.93, 0.95] slower | 2.72x [2.25, 3.17] faster |
| static vs tuple, n=32, long | 1.14x [1.13, 1.16] faster | 1.17x [1.16, 1.18] faster | 0.98x [0.96, 1.02] no measurable difference |
| fp16 static vs fp32 static, n=32, long | 1.09x [1.08, 1.11] faster | 0.98x [0.97, 0.99] slower | 4.75x [4.26, 4.96] faster |
