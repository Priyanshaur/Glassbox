# Phase 6 / S4: GPU probe for the layer-4 SAE (pre-registered)

Committed **before the kernel runs**. This is a **probe, not the acceptance run**: nothing in it
counts toward A1-A5 of `docs/phase6_preregistration.md`; the A1-A4 numbers it prints are
labeled "probe, not acceptance". Hard cap: **30 minutes**, private kernel, Tesla T4.

## Inputs

- Base checkpoint `weights_step12158.pt` from the private dataset `glassbox-sft-init-weights`
  (125,893,589 bytes), **SHA-256 gate**: the file's SHA-256 must equal
  `feb065a5bc7d0ca240d756cd7bc48d8e1e284349a35dcd783ece4dc0c899aab0` (the hash of the
  local file); a mismatch stops the run before anything else.
- The `tinystories-shards-8192` dataset (real train and validation shards).
- Only the fast tests run (`test_hooks`, `test_model`, `test_interp_*`); fatal on failure.

## What the probe measures

1. **Normalization scale** recomputed on real training windows: 256 windows x 512 tokens,
   positions >= 1 (Amendment 1), plus layer-4 norm statistics (all positions; position 0 vs
   rest). Reference from the 1,000-story smoke shard: 0.3446.
2. **Extraction speed**: tokens/s of `fill_buffer` (fp32) for chunk sizes 32, 64 and 128 on
   524,288-token buffers (1 warm-up, then the median of 3).
3. **SAE training throughput and peak GPU memory**, from the per-epoch statistics of the
   short run.
4. **A short training run**: layer 4, F = 4,096, k = 32, lr 4e-4, warmup 50 steps, batch
   4,096 tokens, buffer 524,288 tokens (523,264 training rows after dropping position 0),
   planned 24 buffer epochs (about 12.6M tokens), stopped at an epoch boundary once the
   **6-minute training budget** is used. FVU and L0 on 64 held-out validation windows after
   every epoch.
5. **A1-A4 on the short-run SAE** with the real definitions (500,000 evaluation tokens; the
   10M-token dead scan, reduced to 2M only if the time cap requires it, and flagged), plus the
   position-0 extras. Labeled **probe, not acceptance**; A5 is not run (no SFT model attached).

## Stop rule (fixed now): "FVU falling clearly"

Let `f_e` be the validation FVU (all positions) after buffer epoch `e`, for the epochs that
completed (at least 4 are needed to judge). FVU is **falling clearly** iff

- `f_last <= 0.85 * f_1`, **and**
- `f_e < f_{e-1}` for at least **70%** of consecutive epoch pairs.

If it is **not** falling clearly, or fewer than 4 epochs completed, the probe reports
"**FVU not falling clearly: STOP**" and nothing is scaled up; no full run, no layers 2 and 6,
and no labeling until the result has been discussed.

## What gets reported

- Hash gate result; tests result; whether the time cap or the training budget cut anything.
- Throughput: extraction tokens/s by chunk size, SAE training tokens/s, peak GPU memory.
- The FVU and L0 trajectory per epoch (`fvu`, `fvu_excl_pos0`, `l0`, alive fraction) and the
  stop-rule verdict.
- The probe's A1-A4 numbers, FVU by position bucket, and the position-0 variants, labeled as above.
- **Revised GPU estimate for the full 100M-token run**: epochs needed =
  `ceil(100M / 523,264)` = 192; time = 192 x (median fill + train seconds per epoch) +
  the measured full-evaluation time + about 4 minutes of kernel setup. Reported with a range
  from the fastest to the slowest observed epoch.
- GPU quota used by the probe and remaining.

## Artifacts

`probe_report.json`, `probe_summary.md`, `sae_probe/metrics_sae.jsonl` and `sae_probe/sae_latest.pt`
(about 17 MB) in `/kaggle/working`.
