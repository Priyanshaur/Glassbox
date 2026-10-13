# Design log. Every choice + reason. Update each phase; include failures honestly.

## Phase 0 (2026-10-11)
- Repo root = `Portfolio Project/` (acts as `minigpt-stack/`). Reason: avoid empty nested folder; keep Kaggle clone path simple.
- Added `common/config.py` (deviation from spec tree). Reason: loader is shared by train/inference/interp; duplicating it per folder would diverge.
- Python 3.13 (laptop has 3.13.2, not 3.11). Reason: pin to reality; same `requirements.txt` installs CUDA torch on Kaggle.
- torch 2.x pinned (2.5.1). Reason: need `scaled_dot_product_attention`; T4-safe fp16 path.
- fp16 + GradScaler, NOT bf16 (spec). Reason: T4 handles bf16 poorly.
- wandb OFF by default (`logging.use_wandb: false`). Reason: no paid deps, reproducible local runs; enable explicitly.
- Makefile kept minimal + PowerShell equivalents in README. Reason: `make` often missing on Windows; i3/8GB laptop is Windows.
- Checkpoint contract fixed now (before Phase 3 code): save model, optimizer, scaler, scheduler, RNG state, data position; resume must give continuous loss curve. Reason: user requirement ("save states") + Kaggle sessions die at any time; `save_every` default 1000 steps (~<1h on T4).
- 85M config is a stub; gate: no 85M training until 30M val loss + coherent stories verified.

## Phase 3 (2026-10-14)
- Data position = integer step. Batches redrawn via SeedSequence([seed, step])
  (never hash(): salted per process). Shards sampled weighted by size; every
  window inside one shard (never spans). Tradeoff: sampling with replacement,
  so epochs are statistical. Schedule is a pure function of step (no blob).
- AdamW betas 0.9/0.95; decay on 2-D matrices except embedding; norms/emb exempt.
- fp16+GradScaler on CUDA, fp32 CPU; overflow counted via scale drops (skips).
  NaN/inf parks to ckpt_nan_* and exits 1; last good checkpoint untouched.
- Metrics to stdout + metrics.jsonl; W&B lazy/optional. Time-based ckpts
  (save_every_min) plus step-based; atomic tmp+rename; rotation keeps weights-only copies.
- Sampler (train/sample.py, new file — time-lapse only): own Generator,
  eval/no_grad, restores train mode.
- HF backup lives ONLY in the notebook (huggingface_hub NOT added to
  requirements; Kaggle installs it, skips gracefully without HF_TOKEN secret).
- Notebook runs under Save Version with --max_hours 8; restores newest ckpt
  from attached previous output; full prepare runs once, then shards attach.
- Budget (PROVISIONAL until shakedown measures tok/s): ~490M train tokens
  (user estimate; exact count lands in Kaggle prepare's meta.json — local
  meta.json holds smoke counts only: 231,577 train / 37,743 val).
  At a guessed 60k tok/s, 5.5h ~= 1.19B tokens ~= 2.4 passes ~= 18.1k steps
  @65,536 tok/step -> max_steps=18000 in config. Calibration sets the final
  number; gates: skips/step <1%, LR sweep {3e-4,6e-4,1e-3}x900 steps, micro
  probe 8/16/32, kill-resume drill.
- Bug caught by CPU smoke: logged loss divided by accum TWICE (9.02 showed as
  1.13); gradients were correct (equivalence test passed), only the scalar
  was wrong. Fixed; smoke now shows 9.02 -> decreasing, resume continues.

## Phase 2 (2026-10-11)
- Versions actually used: torch 2.9.0+cpu, Python 3.13.2 (laptop). requirements.txt relaxed to torch>=2.5; Kaggle uses preinstalled torch, never reinstall.
- Vocab reconciled to 8192 everywhere (was 8000 in configs; tokenizer makes 8192). Tokenizer blocks updated to <|endoftext|>/<|pad|>.
- MLP = SwiGLU (3 matrices), hidden via explicit ffn_hidden: base 1536, big 1792, debug 128. Reason: SwiGLU beats GELU at small scale; explicit hidden (not a hidden multiplier) is what makes counts land. big uses 1792 not 2048: 2048 gives ~91.2M (7% over), 1792 gives 84.2M. No rename needed.
- RMSNorm (weight-only) + RoPE (even head_dim asserted) + weight tying (emb counted once).
- Init: GPT-2 N(0,0.02); attn w_o and mlp w_down scaled /sqrt(2L) (residual-stream stability).
- SDPA regimes: no cache is_causal=True; 1-token cached no mask; chunk+cache explicit float mask. Bool-mask convention avoided (ambiguous across builds).
- Hook point: record/inject at output of block L (after both residual adds). Steering caveat: only tokens processed after the change are affected; cached K/V keep old values.
- forward() returns GPTOutput(logits, new_kvs, recorded) always.
- Exact counts (code, not arithmetic): debug 606,528 (was 115,008 at vocab 512); base_30m 31,465,984 (~31.5M); big_85m 84,167,424 (~84.2M).
- Debug vocab fix (2026-10-12): debug_cpu model.vocab_size was 512, so its
  115,008 count proved nothing about real-vocab shapes. Now 8192 everywhere
  (one debug config, no separate synthetic one); overfit test still uses
  synthetic random ids, and Phase 3 adds a CPU smoke run on real shards.
- Phase 2 draft bugs (caught before tests): _scale_residual_init written but never called (fixed: called in __init__); RMSNorm had a double-wrapped Parameter init line (cleaned to torch.ones(d)).
- Phase 1 failures (kept here for honesty):
- First prepare.py print used `≈` (U+2248); Windows cp1252 console crashed with UnicodeEncodeError. Fixed to ASCII `~=`.
- First smoke resume doubled docs (1000->2000) because --max_docs was per-run, not total. Fixed to total-cap semantics; rerun verified no-op.
- First encode-speed one-liner re-created the streaming dataset per doc (`next(iter(ds))` in a loop) and timed out at 120s. Fixed with single-iteration collect + `python -m` script; real speed ~2-5k docs/sec.

## Phase 1 (2026-10-11)
- ID layout (fixed): bytes 0-255, `<|endoftext|>` 256, `<|pad|>` 257, merges from 258, total vocab 8192 including specials. Reason: byte-level covers all unicode; specials up front so streaming prepare.py can append EOT per doc without shifting ids.
- `regex` package (not `re`) for pre-tokenization; added `regex==2024.11.6` to requirements. Reason: `\p{L}`/`\p{N}` unicode classes; `re` cannot do them.
- GPT-2 style pre-tokenizer; merges never cross chunk boundaries. Reason: makes pure-Python counting tractable (short chunks) and avoids junk cross-word merges.
- BPE trained on Counter of UNIQUE chunks weighted by frequency + incremental pair updates (pair->chunks index, only affected chunks revisited per merge). Why fast: TinyStories repeats words heavily (50MB -> 12.65M total chunks but only 21,687 unique); each of the 7934 merges touches a small subset, not the corpus. Measured 35.8s for 50MB on i3-1315U.
- chunk->ids cache in encode(). Measured ~2164 docs/sec cold, ~5246 docs/sec warm (300-doc sample).
- Shards written with buffered tofile() appends (flush every 500 docs or 1M tokens), never a growing memmap. Reason: memmap needs final size up front and rewrites on grow; appends are O(1) RAM on 8GB.
- prepare.py resumable per shard: data/meta.json tracks docs_done per split + shard records; rerun skips finished docs (verified no-op rerun); partial last shard appended to. --max_docs is a TOTAL cap (train) with val cap max(100, max_docs//5); full run omits it.
- Val = official TinyStories `validation` split (overrides earlier doc-holdout wording). Reason: comparable to published numbers, no leakage tuning.
- Append <|endoftext|> (256) after every document in both splits.
- uint16 shards (8192 < 65536).
- Smoke only: --max_docs 1000 train / 200 val. Full run deferred to Kaggle (internet on, save out_dir as dataset). Do NOT run full locally.
- Run modules as `python -m` (e.g. `python -m data.prepare`), not `python path/file.py`. Reason: absolute `from tokenizer...` imports need repo root on sys.path; script-path mode breaks it (hit during smoke test, fixed by convention).
