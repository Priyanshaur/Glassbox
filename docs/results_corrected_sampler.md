# DPO generation metrics re-evaluated with the corrected sampler (2026-11-10)

The old sampler (`train.sample.sample_completions`) feeds the last prompt token twice (`docs/decisions.md`, "re-feeds the last
prompt token"). Every generation metric in `docs/results_final.md` used it. This is the same evaluation re-run on the laptop CPU
with `inference.generate.Generator` in its correct mode (`train.eval_quality --sampler engine`): same 100 validation prompts,
6 samples each, `max_new=256`, temperature 0.8, top-k 40, the same per-prompt seed, the same SFT perplexity scorer, the
same checkpoints (SFT `sft_final_best.pt`, DPO seed 0 `weights_step400.pt`, seed 1 `weights_step300.pt`, from the Kaggle DPO kernel's
output). No threshold or rule was changed. The old-sampler column below was recomputed from the published per-prompt files and
matches `docs/results_final.md`.

| Model | metric | old sampler (published) | corrected sampler |
|---|---|---|---|
| SFT | mean keyword coverage (0-3) | 0.892 | 0.903 |
| SFT | `eot_rate` | 0.932 | 0.922 |
| DPO seed 0 | mean coverage | 1.707 | 1.695 |
| DPO seed 0 | `eot_rate` | 0.787 | 0.763 |
| DPO seed 1 | mean coverage | 1.768 | 1.807 |
| DPO seed 1 | `eot_rate` | 0.813 | 0.783 |

DPO minus SFT, paired by prompt (95% CI, bootstrap over prompts, B=10,000):

| | metric | old sampler | corrected sampler |
|---|---|---|---|
| seed 0 | coverage | +0.815 [+0.708, +0.918] | **+0.792 [+0.698, +0.885]** |
| seed 0 | `eot_rate` | -0.145 [-0.187, -0.105] | **-0.158 [-0.203, -0.115]** |
| seed 0 | mean words | +16.6 [+13.2, +20.0] | +17.7 [+14.4, +20.9] |
| seed 1 | coverage | +0.877 [+0.773, +0.982] | **+0.903 [+0.807, +1.000]** |
| seed 1 | `eot_rate` | -0.118 [-0.158, -0.078] | **-0.138 [-0.177, -0.102]** |
| seed 1 | mean words | +18.2 [+14.8, +21.6] | +18.5 [+15.3, +21.5] |

**Conclusion: the DPO result does not change.** Coverage still rises by about +0.8 to +0.9 of 3 (CIs exclude zero), `eot_rate` still falls
by 14-16 points and ends at 76% and 78%, below the 85% guardrail (the miss is slightly larger than published). Gold-story
perplexity is identical (2.636, 2.772, 2.778) because it is teacher-forced. Raw per-prompt files: `checkpoints/glassbox-reeval/`
(git-ignored); the comparison script is `analysis/reeval_compare.py`.

## What was NOT re-done

- **The preference pairs DPO trained on** were built with the old sampler (the sampled candidates came from the slightly distorted
  distribution). Rebuilding them and retraining DPO needs a GPU. The measured cost of the DPO final kernel (2 seeds, training +
  evaluation) was 1,907 s (0.53 GPU h); the earlier pair-building kernel's time is not recorded in this repo, so a full redo is
  roughly 1-2 GPU hours (unmeasured). Not run. The effect on the trained model is unknown; this re-evaluation only shows that
  *scoring* the existing models with the correct sampler gives the same conclusions.
- `bench/win_rate.py` and the human spot-check samples also used the old sampler; no win-rate number is quoted in the README.
- Not affected, so not re-run: all benchmarks and the quantization generation check (they used the new engine), gold-story perplexity,
  DPO losses, and every Phase 6 steering number (the steering experiment samples with the engine in its default correct mode).
