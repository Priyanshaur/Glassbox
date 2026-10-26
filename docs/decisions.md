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

## Phase 3 calibration (2026-10-19, `priyanshgaur/glassbox-calibrate`)
- 22 tests passed in 12.2s before any GPU was spent.
- micro-batch probe, all three `skip_rate_post30 = 0.0000`:
  `mb=8  tok/s= 9,668 peak_mem=11,997MB`, `mb=16 tok/s=38,709 peak_mem=13,131MB`,
  `mb=32 tok/s=40,245 peak_mem= 8,315MB`. Picked `PROBE_TOKS = 40,245` (best
  passing). Honest flaw: the mb=8 number is a COLD-RAMP artifact — that probe
  run had only reached 15,276 tok/s by step 100 while the other two were already
  at ~42-43k, so the ranking is distorted and peak_mem is non-monotonic (mb=32
  lower than mb=16). It did not hurt: production kept the conservative
  `micro_batch_seqs: 8` (accum = 128 seqs / 8 = 16) and measured
  **~39,600 tok/s steady = 98.4% of the probe**, so the budget was right anyway.
- Budget: `tok/s=40245 target=5.5h -> budget=0.80B tokens`;
  `passes over train data ~= 1.71 -> MAX_STEPS=12158`. Check: 5.5h x 3600 x
  40,245 = 796,851,000 tokens; 796,851,000 // 65,536 = **12,158** exactly.
- Kill-resume drill (`warmup_steps 2000->30`): run 1 stopped early and saved
  `/tmp/drill/ckpt_step135.pt`; run 2 resumed `latest` and reached
  `ckpt_step300.pt`. The resume contract was verified on GPU before the real run.
- LR sweep, 900 steps each, `warmup_steps 2000->60`, `val_every 1000->150`, val
  compared at the SAME step 750 for all three:
  `lr=3e-4 val=2.0146`, `lr=6e-4 val=1.8463`, `lr=1e-3 val=1.7651`.
  Logged `PROPOSED_LR = 1e-3`; set as `CONFIRMED_LR = 1e-3` in the main cell and
  confirmed in every session log (`LR: 0.001`).
- warmup 2000 -> 500 (`configs/base_30m.yaml`, commit `54ea0fa`): at the
  calibrated 12,158 steps, 2000 warmup is 16.4% of the whole run spent ramping;
  500 is 4.1%. Note the sweep's own runs used 60 purely to reach peak inside
  900 steps — that is a test harness value, NOT the production warmup.

## Phase 3 main run — Kaggle sessions 1-3 (2026-10-20/21)
- RESULT: `max_steps 12158` reached, natural exit (line 243 `save(max_steps)` ->
  `return 0`; the `max_hours` branch never fired). **12,157 / 12,158 iterations
  trained**, val **1.294 @ 12000**, **~8.8 GPU h** (`time_used` 31,637.725 s =
  8h47m18s; s1 ~10,932 s + s2 ~9,834 s + s3 ~10,872 s). Val monotonic across
  all three sessions: 1.589@2000, 1.514, 1.461, 1.422, 1.396@6000, 1.369@7000,
  1.347, 1.327, 1.310, 1.299, **1.294@12000**. Never tripped any spike rule;
  skips 0 for the whole run.
- Off-by-one (the entire 12,158 -> 12,157 gap): `pretrain.py:157` runs
  `save(step)` at the TOP of the loop body, before that iteration trains, so
  `ckpt_step6207.pt` holds the state after step 6206; `load_checkpoint` returns
  `step+1`, resume restarts at 6208, and **step 6207 is never trained**. Cause
  is a tagging inconsistency, not the save itself: the periodic path (`:223`)
  saves AFTER training and tags "last step trained", while both exit paths tag
  "last step trained + 1". Line 243's `save(max_steps)` is harmless today —
  `range` stops at `max_steps-1`, so step 12158 does not exist (observed firing
  on the s3 natural exit) — but WOULD silently skip an iteration if `max_steps`
  were ever raised on a resumed run. NOT fixed: sessions 1-3 all ran off one
  commit and retagging breaks resume for checkpoints already on disk. Fix when
  convenient: tag `step-1` on the `max_hours` path.
- Sampler device bug + `check=False` swallow: session 1 died at step 2000 with
  `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`
  (`train/sample.py:39`) — `torch.Generator()` defaults to CPU, then
  `torch.multinomial` was called on a CUDA tensor. Worse, the notebook's
  subprocess call used `check=False`, so the non-zero exit was **swallowed and
  the run still reported COMPLETE**, hiding the crash. Because sampling also ran
  BEFORE the step-2000 save, nothing at 2000 was written — no checkpoint and no
  `samples.txt` block — so session 2 had to rewind to `ckpt_step1662` and
  retrain 1663-1999. Fixed in `7d40bf3`: generator built device-first
  (`torch.Generator(device=device)`, both samplers), save-then-sample in
  `pretrain.py`, cell 18 now raises on non-zero exit.
- Quota `timedelta` misread: the Kaggle API's `to_json()` serializes a
  `timedelta` as a malformed string (`"20765.942.0s"` — seconds and
  microseconds concatenated), so parsing it yields garbage. Read the typed
  `gpu_quota.time_used` / `total_time_allowed` fields as `datetime.timedelta`
  (authoritative) and `quotaRefreshTime` for the reset; never the JSON string.

## Phase 4 (2026-10-16, CPU smoke only — no Kaggle launched)
- Instruct fallback CONFIRMED (amendment 8): no dataset ID verified, so the
  deterministic synthesized-prompt path is the implementation, not a backup:
  3 content words per story (SeedSequence([12345, idx])), plain-text template,
  zero new special tokens. Original stories pass by construction.
- Masking: labels[:P-1] = -100 (prompt predictions out, first response kept);
  right-padding is safe (causal positions never see trailing pads).
- SFT: same Phase 3 checkpoint format (weights init, fresh optim); proposed
  base LR 1e-4, sweep {5e-5, 1e-4, 2e-4} on Kaggle, 1-3 epochs, val on
  response tokens (official validation split). Debug smoke used lr 3e-3.
- Pairs: n=6 completions/prompt via batched KV-cache sampler (identical rows,
  no padding needed), keep min(pass, fail) capped at 2, drop all-pass/all-fail
  with survival stats. Eval: official val prompts, wins/ties/losses + Wilson
  CI on decided pairs. Rule rate is circular (DPO trains to it) -> also val
  ppl, distinct-2, mean length + blind 30-50 human protocol (docs + CSV emit).
- DPO: hand loss, ref frozen (requires_grad_(False), never stepped/saved),
  logs acc/margin/chosen-lp; proposed base lr 1e-6 sweep {5e-7,1e-6,2e-6},
  beta 0.1, held-out early stopping. Debug smoke used lr 1e-4.
- Criterion flaws (honest): rewards keyword-stuffing over story quality
  (Goodhart: expect fluency to sag as win rate climbs — that is why ppl and
  the human check exist), blind to grammar/plot, length-window bias.
- Bugs caught: (1) test fed labels as model input (IndexError on -100);
  (2) masking test overclaimed bit-identical loss (RoPE shifts positions —
  claim corrected to response-only finiteness); (3) win_rate needed --max_new
  + ctx clamping (debug ctx 64 overflows with 64 new tokens).
- SFT final (v5 retry, 2026-10-21): on a flat val curve (64 val examples ->
  <0.01 noise floor), best-val selection is noise-level; the final
  checkpoint is used. Selection code still copies best-logged-val's nearest
  save for auditability, but flat curves make it equivalent to last.
- SFT final run (v7, 2026-10-22): samples show EOT in 8/10 sections (3/5
  openers, 5/5 TEMPLATE), each at a complete story end. TEMPLATE keyword
  adherence is UNDER RECHECK: the first assessment was an artifact —
  sample_continuations decodes the prompt itself (ids start with
  encode(prompt)), so keyword "hits" landed in the echoed prompt prefix;
  strict whole-word matching on pre-EOT story text alone finds 0/5 full
  coverage. A pretrain-weights TEMPLATE baseline (local CPU, same prompts +
  seeds, new file next to stories_256.txt) is part of the recheck; no
  adherence claim is made until it lands.
- Pair-gate criterion change (deliberate, 2026-10-22): pair building no
  longer scores binary pass/fail. Winners/rejecters come from the GRADED
  rule — chosen must cover >= 1 more keyword than rejected, coverage =
  strict whole-word count of the 3 keywords (whole-word matching, not the
  substring check_criterion), and BOTH completions must end at their own
  EOT and fall in the 10-500-word window. Measured on the 200-prompt x 6
  probe (train-stream prompts = upper bound): binary whole-word yield ~18
  pairs per 200 prompts (7.5% prompts kept) vs graded ~311 pairs (90.5%
  kept). Accepted tradeoff: weaker warm-vs-cold contrast (2-vs-1 keyword
  pairs included) in exchange for usable pair quantity. whole_word_ok in
  train/pairs.py is the matched-scoring building block. Final gate: probe
  (template_probe_samples.jsonl) also fixes sampling at max_new=256 (64 new
  tokens cuts nearly every story mid-body and starves the EOT-ended rule).
- Recheck outcome (2026-10-22): strict whole-word / story-prose / pre-first-EOT
  counting gives keyword coverage 0/5 pretrain vs 0/5 SFT (1 stray
  single-word hit baseline vs 4 SFT). Qualitative change confirmed though:
  all 5 SFT outputs are complete stories ending at their EOT, while the
  pretrain model reads the TEMPLATE prompt as story text and wanders into
  invented dialogue; 4/5 SFT stories share a stock opener ("Once upon a
  time, there was a little ...", two with verbatim "little girl named
  Lily"). Next gate before any DPO pair building: n=6-per-prompt keyword
  coverage probe on the first 200 template prompts (exact corpus head,
  reconstructed locally from the fixture shard when its first 5 prompts
  matched PROMPTS_T 5/5), measured by the same strict rule; pair building
  happens only if that probe shows viable coverage.

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

## Corpus mojibake and the "if retraining" fix (2026-10-27) - NOT applied

Finding (evidence: `docs/source_mojibake_check.md`, `docs/results_folded.md`,
`analysis/check_source_mojibake.py`): the Hub's parquet version of
`roneneldan/TinyStories` (what `load_dataset(...)` returns and `data/prepare.py`
tokenizes unchanged) has UTF-8 curly quotes and dashes decoded as cp1252 with
the undefined bytes dropped. A right single quote (U+2019) appears as U+00E2
U+20AC U+2122. About 4-5% of stories are affected (4.9% train, 3.7% validation,
from 3,000 streamed stories per split, so approximate), and no correct U+2019
survives. The original raw `TinyStories-train.txt` / `TinyStories-valid.txt` are
clean UTF-8: over the first 4 MB of the validation file, 282 of 282 non-ASCII
stories equal the raw text re-decoded as cp1252 (errors="ignore"), and the 4,561
ASCII-only stories match the parquet text exactly. So `prepare.py` did not
introduce it.

Impact so far: the tokenizer learned merges for the garbled bytes (26 vocab
entries contain them), the pretrained/SFT/DPO models reproduce the sequence in
1.7-3.3% of samples, and the closing double quote has permanently lost a byte in
the parquet text (not recoverable from it). Folding the garbled apostrophe back
before whole-word scoring changed 0 of 1,800 saved samples, so reported keyword
coverage is unaffected.

Decision: leave the current corpus, tokenizer and models as they are and keep
this as a documented limitation (README Limitations).

If retraining (all of the following would be redone; nothing here is applied):

1. Change only the text source in `data/prepare.py::encode_stream`: read the raw
   files `TinyStories-train.txt` and `TinyStories-valid.txt` from the same Hub
   repo (not the parquet-backed default loader) and split on `<|endoftext|>`,
   stripping the surrounding whitespace. The train file is about 1.9 GB and the
   validation file about 19 MB. Keep appending EOT after every story, the same
   uint16 shards and the same resumability.
2. Verify before trusting it: the first stories of the raw validation file must
   equal the parquet-loader validation stories after whitespace normalisation for
   every ASCII-only story (4,561 of 4,561 matched in the check above), differing
   only on non-ASCII stories. Count U+00E2 U+20AC sequences in the new shards: it
   must be 0, and correct U+2019 bytes (E2 80 99) must now be present.
3. Recommended: retrain the tokenizer on the clean text. The old byte-level
   tokenizer would still work on clean text, but 26 of its vocab entries are
   spent on the garbled bytes. Either way the data must be re-prepared and the
   model pretrained from scratch (about 8.8 GPU hours last time), then SFT, the
   pair builder and DPO. A new tokenizer additionally makes the old checkpoints
   unusable.
4. Keep the validation split comparable: the raw validation file is the same
   official validation set, so val numbers stay comparable to published ones.
5. Caveat: do not repair the existing text with a cp1252 round trip
   (`text.encode("cp1252").decode("utf-8")`). The closing double quote U+201D lost
   its third byte, and the repair fails or corrupts those stories.

## sample_completions re-feeds the last prompt token (found 2026-10-28) - NOT fixed

Found while building the Phase 5 engine (parity test against the old sampler).
`train/sample.py::sample_completions` prefills the prompt, then its first decode
step feeds `ids[:, -1:]` (the LAST PROMPT TOKEN) again at `pos_offset = P-1`. That
token's K/V are therefore in the cache twice (cache length P+1 after the first
step) and stay there for the whole generation. The first next-token distribution
differs slightly from the model's true one: on the SFT model with the template
prompt the total variation distance is 0.047 and the largest logit difference is
1.16; over 60 sampled tokens only 4 of 12 rows were identical between the old
behaviour and the correct one.

What it affects (everything sampled with `sample_completions`): the preference
pairs (`train/pairs.py` via `build_pairs_graded`), `train/eval_quality.py` (the
probe and final-run generation metrics: coverage, `eot_rate`, validity, length),
`bench/win_rate.py`, the sampled stories in the human spot-check, and the
generation numbers in `docs/results_final.md` and `docs/results_folded.md`. (The
Phase 5 quantization generation check in `docs/quantization_results.md` used the
new engine in its correct mode, so it is not affected.)

What it does NOT affect: the DPO training loss and the held-out DPO losses (full
forward passes over winner/loser in `response_logps`), SFT and pretraining
training, gold-story perplexity (teacher-forced), the int8 perplexity gate, and
`sample_continuations` (no cache).

SFT-vs-DPO comparisons used the same sampler, the same prompts and the same
per-prompt seeds for both models, so both carry the same quirk. The comparisons
are consistent with each other, but the absolute `eot_rate` / coverage values come
from a slightly distorted sampling distribution, and the effect on the size of the
DPO-minus-SFT differences was not measured.

Not changed: the old sampler is untouched and no evaluation was re-run. The new
engine (`inference/generate.py`) takes the first token's logits straight from the
prefill (correct); `replay_last_prompt_token=True` reproduces the old behaviour
for parity tests. Proposal, not applied: fix the sampler (or route it through the
engine) and re-run the generation evals as a corrected re-evaluation.
