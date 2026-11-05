# Phase 6 S4 - layer-4 full run: parameters fixed before launch

Rules are those of `docs/phase6_preregistration.md` (A1-A4 thresholds and Amendment 1), **unchanged**. This
file only fixes the run parameters. The probe that preceded it is in `docs/decisions.md` (a probe, not acceptance).

| Parameter | Value |
|---|---|
| model / layer | base `weights_step12158.pt` (SHA-256 `feb065a5...` gate), layer 4 |
| SAE | TopK, d_in 512, F 4,096, k 32; AuxK on; unit-norm decoder |
| training data | 192 buffer epochs x 523,264 rows = 100.5M tokens, a fresh buffer each epoch, window position 0 excluded |
| optimiser | Adam, lr 4e-4, warmup 50 steps, batch 4,096 tokens (as in the probe) |
| time limit | `MAX_HOURS` 4.5 h of training; hard-cap watchdog at 5.75 h kills the process |
| saves | atomic checkpoint every 5 epochs (SAE, Adam, epoch, step, scale); resume is exact |
| truncation | limit hit or < 192 epochs: report status TRUNCATED, no evaluation, the kernel raises -> ERROR, never COMPLETE |
| evaluation | A1-A4 only: 500,000 validation tokens, 10M-token dead scan (never reduced), CE recovered vs mean ablation with the full forward |
| not in this run | A5 (SFT transfer), layers 2 and 6, labeling |
| extraction | `record_windows` stops after the recorded layer (bit-identical, tested); the CE-recovered path keeps the full forward |

COMPLETE means "192/192 epochs trained and A1-A4 measured". Whether the layer passes is the printed verdict, reported
as measured. If A1 fails, the pre-registered fallbacks (R1, R2) are separate decisions.
