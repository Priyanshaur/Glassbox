# int8 weight-only quantization check (SFT checkpoint)

Checkpoint: `checkpoints/glassbox-sft-final/sft_final_best.pt` (the **SFT** model, not DPO). CPU, fp32 baseline. Criteria were fixed before running (see `bench/quant_check.py`).

## File size and memory

| variant | file bytes | vs fp32 | in-memory tensor bytes |
|---|---|---|---|
| fp32 | 125,886,485 | 1.000x | 125,863,936 |
| int8_linear | 44,299,739 | 0.352x | 44,255,232 |
| int8_all | 31,749,506 | 0.252x | 31,705,088 |

## Gold-story perplexity (first 100 validation prompts)

Kaggle SFT reference (eval_quality, kernel v3): 2.636. Local fp32: 2.6361.

| variant | gold ppl | relative to fp32 | within 2%? |
|---|---|---|---|
| fp32 | 2.6361 | - | - |
| int8_linear | 2.6369 | +0.032% | PASS |
| int8_all | 2.6372 | +0.042% | PASS |

## Generation check (20 prompts x 6 samples, max_new 256, temp 0.8, top-k 40)

| variant | eot_rate | mean coverage (0-3) |
|---|---|---|
| fp32 | 0.917 | 0.975 |
| int8_linear | 0.883 | 1.025 |
| int8_all | 0.875 | 1.017 |

Difference int8 - fp32 (95% CI over prompts, paired, B=10,000):

| variant | eot_rate diff [CI] | coverage diff [CI] | clear damage? |
|---|---|---|---|
| int8_linear | -0.033 [-0.075, +0.000] | +0.050 [-0.017, +0.117] | no |
| int8_all | -0.042 [-0.083, +0.000] | +0.042 [-0.033, +0.125] | no |

With 20 prompts the check can miss small drops; the perplexity gate is the more sensitive one.

## Verdict

- `int8_linear`: perplexity PASS, generation PASS -> **ACCEPTED**
- `int8_all`: perplexity PASS, generation PASS -> **ACCEPTED**

Accepted int8 variant: **int8_all**.
