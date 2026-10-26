# Inference benchmark results

Protocol: `docs/benchmark_protocol.md` (including its CPU addendum). Times are medians over 10 interleaved repeats; spread = interquartile range (p25-p75). Raw per-repeat timings: the `_raw.json` next to this file.

## Environment and parameters

- device: cpu
- torch: 2.9.0+cpu
- threads: 6
- platform: Windows-11-10.0.26300-SP0
- processor: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel
- logical_cpus: 8
- commit: c3d0686
- started: 2026-10-27 02:15:40
- power_and_background_load: not controlled or verified
- prompts K=4, generated tokens G=64 (fixed, EOT ignored), repeats R=10, warm-up 3, n in [1, 8, 32], grid `cpu`
- sampling: temperature 0.8, top-k 40; short prompts [23, 23, 23, 23] tokens (validation prompts idx [0, 1, 2, 3]), long prompts [256, 256, 256, 256] tokens (first 256 tokens of validation stories idx [1, 189, 190, 191]: the first K stories with enough tokens, not the same stories as the short regime)

## Correctness gate (before timing)

| check | n | regime | result | detail |
|---|---|---|---|---|
| engine | 1 | short | PASS | engine tokens == bench static loop |
| tuple | 1 | short | PASS | tokens identical (agreement 1.000) |
| none | 1 | short | PASS | teacher-forced max logit diff 1.19e-05 (atol=rtol=1e-4); sampled-token agreement with static 1.000 |
| tuple | 8 | short | PASS | tokens identical (agreement 1.000) |
| none | 8 | short | PASS | teacher-forced max logit diff 1.19e-05 (atol=rtol=1e-4); sampled-token agreement with static 1.000 |
| tuple | 32 | short | PASS | tokens identical (agreement 1.000) |
| tuple | 1 | long | PASS | tokens identical (agreement 1.000) |
| none | 1 | long | PASS | teacher-forced max logit diff 8.58e-06 (atol=rtol=1e-4); sampled-token agreement with static 1.000 |
| tuple | 8 | long | PASS | tokens identical (agreement 1.000) |
| tuple | 32 | long | PASS | tokens identical (agreement 1.000) |

## Results, short prompts (median [p25-p75])

| config | TTFT ms | decode ms/step | tokens/s (all rows) | end-to-end s | KV cache MB | rel. IQR (e2e) |
|---|---|---|---|---|---|---|
| fp32/none/n=1/short | 33 [32-35] | 50.9 [49.8-52.0] | 19.6 [19.2-20.1] | 3.24 [3.17-3.32] | 0.0 | 4.4% |
| fp32/tuple/n=1/short | 35 [34-42] | 22.9 [22.2-27.2] | 43.6 [36.7-45.1] | 1.48 [1.44-1.75] | 2.8 | 21.1% (re-run: 11.6%) |
| fp32/static/n=1/short | 37 [35-38] | 21.5 [20.7-24.7] | 46.4 [40.6-48.4] | 1.39 [1.34-1.60] | 2.9 | 18.7% |
| int8_all/static/n=1/short | 57 [55-65] | 35.7 [34.0-45.1] | 28.0 [22.4-29.4] | 2.30 [2.21-2.91] | 2.9 | 30.5% (re-run: 3.5%) |
| fp32/none/n=8/short | 128 [124-138] | 255.4 [248.6-274.5] | 31.3 [29.1-32.2] | 16.22 [15.78-17.44] | 0.0 | 10.2% |
| fp32/tuple/n=8/short | 130 [126-136] | 34.3 [34.0-37.4] | 233.5 [214.1-235.4] | 2.29 [2.27-2.49] | 22.5 | 9.3% |
| fp32/static/n=8/short | 142 [136-151] | 32.6 [31.6-36.1] | 245.4 [222.1-253.3] | 2.20 [2.13-2.42] | 22.8 | 13.2% |
| int8_all/static/n=8/short | 163 [159-170] | 46.0 [44.4-57.1] | 174.0 [140.2-180.3] | 3.06 [2.95-3.77] | 22.8 | 26.8% (re-run: 2.8%) |
| fp32/tuple/n=32/short | 405 [397-410] | 66.3 [65.5-69.6] | 482.9 [460.3-488.4] | 4.58 [4.53-4.80] | 90.2 | 5.7% |
| fp32/static/n=32/short | 416 [406-440] | 51.5 [49.3-52.9] | 621.9 [604.7-648.7] | 3.66 [3.51-3.77] | 91.2 | 7.1% |
| int8_all/static/n=32/short | 451 [447-464] | 73.3 [71.0-80.6] | 436.5 [397.0-450.4] | 5.07 [4.92-5.55] | 91.2 | 12.4% |

## Results, long prompts (median [p25-p75])

| config | TTFT ms | decode ms/step | tokens/s (all rows) | end-to-end s | KV cache MB | rel. IQR (e2e) |
|---|---|---|---|---|---|---|
| fp32/none/n=1/long | 172 [169-174] | 193.6 [186.9-198.8] | 5.2 [5.0-5.4] | 12.37 [11.94-12.70] | 0.0 | 6.1% |
| fp32/tuple/n=1/long | 175 [173-181] | 26.1 [24.5-29.3] | 38.4 [34.2-40.9] | 1.82 [1.72-2.02] | 10.5 | 16.7% |
| fp32/static/n=1/long | 179 [170-181] | 22.0 [21.5-26.1] | 45.5 [38.7-46.6] | 1.57 [1.53-1.82] | 10.5 | 18.7% |
| int8_all/static/n=1/long | 204 [196-226] | 35.0 [34.6-43.3] | 28.5 [23.2-28.9] | 2.41 [2.37-2.95] | 10.5 | 23.9% (re-run: 2.7%) |
| fp32/tuple/n=8/long | 1148 [1120-1172] | 56.7 [56.2-57.7] | 141.1 [138.7-142.4] | 4.74 [4.66-4.81] | 83.6 | 3.2% |
| fp32/static/n=8/long | 1266 [1220-1312] | 36.7 [35.4-37.9] | 218.2 [211.0-226.1] | 3.56 [3.47-3.69] | 83.9 | 6.2% |
| int8_all/static/n=8/long | 1315 [1283-1336] | 51.5 [51.0-60.3] | 155.5 [133.7-156.9] | 4.56 [4.54-5.09] | 83.9 | 12.2% |
| fp32/tuple/n=32/long | 4711 [4638-4789] | 136.5 [133.3-141.1] | 234.4 [226.7-240.1] | 13.35 [13.08-13.57] | 334.5 | 3.7% |
| fp32/static/n=32/long | 4786 [4649-4965] | 72.4 [71.1-74.1] | 442.0 [432.1-450.3] | 9.29 [9.20-9.62] | 335.5 | 4.6% |
| int8_all/static/n=32/long | 5004 [4905-5191] | 93.7 [91.0-98.5] | 341.6 [324.8-351.7] | 10.85 [10.79-11.35] | 335.5 | 5.2% |

## Not run

- fp32/none/n=32/short: estimated cost of hours on CPU
- fp32/none/n=8/long: estimated cost of hours on CPU
- fp32/none/n=32/long: estimated cost of hours on CPU

## Comparisons (for 'A vs B', speedup = median time of B / median time of A, so >1 means A (the first-named config) is faster)

Paired bootstrap over repeats (B=10,000). 'faster' / 'slower' only when the 95% CI excludes 1.

| comparison | end-to-end | decode | TTFT |
|---|---|---|---|
| static vs tuple, n=1, short | 1.06x [0.93, 1.27] no measurable difference | 1.06x [0.93, 1.28] no measurable difference | 0.95x [0.89, 1.15] no measurable difference |
| tuple vs none, n=1, short | 2.19x [1.85, 2.27] faster | 2.22x [1.87, 2.30] faster | 0.96x [0.76, 1.07] no measurable difference |
| static vs none, n=1, short | 2.33x [1.98, 2.45] faster | 2.37x [2.01, 2.49] faster | 0.91x [0.82, 1.05] no measurable difference |
| fp32 static vs int8 static, n=1, short | 1.65x [1.43, 2.06] faster | 1.66x [1.43, 2.07] faster | 1.54x [1.49, 1.79] faster |
| static vs tuple, n=8, short | 1.04x [0.98, 1.12] no measurable difference | 1.05x [0.98, 1.13] no measurable difference | 0.92x [0.87, 0.97] slower |
| tuple vs none, n=8, short | 7.10x [6.34, 7.68] faster | 7.46x [6.64, 8.07] faster | 0.98x [0.95, 1.07] no measurable difference |
| static vs none, n=8, short | 7.39x [6.68, 8.20] faster | 7.83x [7.05, 8.68] faster | 0.90x [0.85, 1.00] slower |
| fp32 static vs int8 static, n=8, short | 1.39x [1.30, 1.69] faster | 1.41x [1.31, 1.72] faster | 1.15x [1.08, 1.23] faster |
| static vs tuple, n=32, short | 1.25x [1.22, 1.31] faster | 1.29x [1.25, 1.35] faster | 0.97x [0.91, 1.00] slower |
| fp32 static vs int8 static, n=32, short | 1.39x [1.33, 1.53] faster | 1.42x [1.37, 1.57] faster | 1.08x [1.02, 1.15] faster |
| static vs tuple, n=1, long | 1.16x [1.08, 1.21] faster | 1.19x [1.09, 1.23] faster | 0.98x [0.93, 1.05] no measurable difference |
| tuple vs none, n=1, long | 6.81x [5.79, 7.42] faster | 7.43x [6.22, 8.15] faster | 0.98x [0.93, 1.00] no measurable difference |
| static vs none, n=1, long | 7.90x [6.37, 8.27] faster | 8.80x [6.93, 9.23] faster | 0.96x [0.92, 1.01] no measurable difference |
| fp32 static vs int8 static, n=1, long | 1.54x [1.50, 1.71] faster | 1.59x [1.56, 1.76] faster | 1.14x [1.08, 1.31] faster |
| static vs tuple, n=8, long | 1.33x [1.26, 1.38] faster | 1.55x [1.47, 1.61] faster | 0.91x [0.86, 0.96] slower |
| fp32 static vs int8 static, n=8, long | 1.28x [1.24, 1.43] faster | 1.40x [1.35, 1.66] faster | 1.04x [0.98, 1.10] no measurable difference |
| static vs tuple, n=32, long | 1.44x [1.37, 1.53] faster | 1.89x [1.82, 1.99] faster | 0.98x [0.94, 1.08] no measurable difference |
| fp32 static vs int8 static, n=32, long | 1.17x [1.13, 1.22] faster | 1.29x [1.25, 1.37] faster | 1.05x [1.01, 1.10] faster |
