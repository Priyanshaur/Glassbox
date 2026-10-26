# Final DPO evaluation (kernel v3, lr 5e-6, two training seeds)

**Result.** DPO raises keyword coverage (about +0.8 of 3, 95% CIs exclude zero, both seeds agree) and lowers `eot_rate` by 12-15 points, to 79-81%, below the 85% guardrail set for this run. This is **not** an overall win: the coverage gain is on the criterion DPO was trained on, while stories got longer, fewer finish within 256 tokens, and gold-story perplexity rose about 5%. No human evaluation has been done yet.

Provenance: Kaggle kernel `priyanshgaur/glassbox-dpo` version 3 (repo commit `f58a309`), 100 validation prompts x 6 samples, `max_new=256`, temperature 0.8, top-k 40, identical prompts and per-prompt sampling seeds for every model. BEST_LR 5e-6 (chosen for stability, see README) was fixed before this evaluation; nothing was tuned afterwards.

## Headline: DPO minus SFT (95% CI bootstrapped over prompts, B=10,000)

| Metric | SFT | DPO seed 0 | diff (seed 0) | DPO seed 1 | diff (seed 1) |
|---|---|---|---|---|---|
| Mean keyword coverage (0-3) | 0.892 | 1.707 | +0.815 [+0.710, +0.917] | 1.768 | +0.877 [+0.773, +0.980] |
| Fraction covering all 3 | 0.028 | 0.170 | +0.142 [+0.108, +0.177] | 0.207 | +0.178 [+0.135, +0.223] |
| `eot_rate` | 0.932 | 0.787 | -0.145 [-0.185, -0.103] | 0.813 | -0.118 [-0.158, -0.078] |
| Validity rate (EOT and 10-500 words) | 0.932 | 0.787 | -0.145 [-0.185, -0.103] | 0.813 | -0.118 [-0.158, -0.078] |
| Mean length (words) | 153.072 | 169.692 | +16.620 [+13.167, +20.107] | 171.302 | +18.230 [+14.927, +21.552] |
| Gold-story perplexity | 2.636 | 2.772 | +0.136 [+0.122, +0.151] | 2.778 | +0.142 [+0.127, +0.159] |

- Seed 0: terminal `epochs`, 444 updates, 198 held-out pairs, mean held-out DPO loss 0.6311. (Seeds also pick different held-out prompt splits.)
- Seed 1: terminal `epochs`, 444 updates, 202 held-out pairs, mean held-out DPO loss 0.6219. (Seeds also pick different held-out prompt splits.)

## Keyword coverage among FINISHED stories only

Finished = the sample ended at its own EOT within `max_new=256`. Rates are pooled over finished samples (total coverage / number finished); CIs resample PROMPTS with replacement (paired across arms for the differences, same resamples for every arm).

| Model | finished / total | mean coverage (finished) | 95% CI | all 3 covered (finished) | 95% CI |
|---|---|---|---|---|---|
| sft | 559 / 600 | 0.889 | [0.788, 0.995] | 0.030 | [0.014, 0.050] |
| dpo_s0 | 472 / 600 | 1.786 | [1.664, 1.907] | 0.197 | [0.149, 0.247] |
| dpo_s1 | 488 / 600 | 1.803 | [1.689, 1.916] | 0.221 | [0.169, 0.277] |

| Difference (finished only) | mean coverage | 95% CI | all 3 covered | 95% CI |
|---|---|---|---|---|
| dpo_s0 minus sft | +0.897 | [+0.787, +1.002] | +0.167 | [+0.127, +0.207] |
| dpo_s1 minus sft | +0.914 | [+0.805, +1.021] | +0.191 | [+0.144, +0.241] |

Caveat: conditioning on finishing selects a different subset of samples for each model (DPO finishes fewer), so this is descriptive, not a causal comparison of the same stories.

## Length (words) of finished vs unfinished stories

Unfinished stories were cut at 256 tokens, so their word counts are truncated, not natural lengths.

| Model | group | n | mean | median | p10 | p90 | max |
|---|---|---|---|---|---|---|---|
| sft | finished | 559 | 149.4 | 149 | 119 | 182 | 214 |
| sft | unfinished | 41 | 202.8 | 204 | 185 | 216 | 227 |
| dpo_s0 | finished | 472 | 161.3 | 161 | 128 | 196 | 221 |
| dpo_s0 | unfinished | 128 | 200.7 | 201 | 185 | 217 | 225 |
| dpo_s1 | finished | 488 | 163.6 | 162 | 133 | 197 | 224 |
| dpo_s1 | unfinished | 112 | 204.7 | 204 | 190 | 219 | 227 |

## Caveats

- DPO was trained on the same keyword-coverage criterion it is scored on, so the coverage gain is circular evidence of quality.
- The `eot_rate` guardrail (85%) was missed: 78.7% (seed 0) and 81.3% (seed 1) against 93.2% for SFT.
- Sampling seeds are fixed per prompt; the bootstrap CIs cover prompt variation only, not sampling or training-seed variation beyond two seeds.
- Only two training seeds; no human evaluation yet (see `docs/human_spotcheck.md`).
