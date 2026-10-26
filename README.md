# MiniGPT-Stack + Brain Scanner (Phase 0 skeleton)
Repo root is this folder (`Portfolio Project/` acts as `minigpt-stack/` from the spec).

## Setup (Windows, Python 3.13, i3 + 8GB RAM)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If `make` is not installed (common on Windows), use the direct commands:
```powershell
pip install -r requirements.txt
python -m pytest tests/ -v
```

With `make` (Git Bash / WSL):
```
make install
make test
```

## Quickstart
```powershell
python -m pytest tests/ -v
python -c "from common.config import load_config; print(load_config('configs/debug_cpu.yaml')['model'])"
```

## Structure
```
configs/        # one yaml per experiment (30M, 85M stub, debug CPU)
common/         # shared config loader (small deviation from spec, see docs/decisions.md)
tokenizer/ model/ data/ train/ inference/ interp/ demo/ bench/
tests/ docs/ notebooks/
```

## Checkpoints / save states
Every training config already carries a `checkpoint:` block (`dir`, `save_every`, `keep_last_n`).
`train/checkpoint.py` (Phase 3) will save model + optimizer + GradScaler + scheduler + RNG state + data position,
so a dead Kaggle session resumes with a continuous loss curve. See `docs/decisions.md`.

## Limitations

- **Pair prompts overlap SFT training data.** The DPO pair prompts (train-stream
  story idx 200-1499) are the same prompts, with the same gold stories, that SFT
  trained on (SFT used idx 0 to about 20,000 for 2 epochs). DPO therefore starts
  from a policy that has already seen these prompts and their chosen stories. The
  evaluation prompts come from the validation split, which neither SFT nor the
  pairs touch, so this affects how much the DPO signal can teach, not the eval.
- **Eval keyword filter uses a hand-made stopword list.** `train/eval_quality.py`
  drops function words from the eval keyword pool using a hand-written list of
  4+ letter words. It was checked only on a 200-story sample of the local
  validation shard (about 24% of unfiltered keywords and 56% of prompts had at
  least one stopword there), so the list is not validated at scale. SFT and pair
  building still use the unfiltered `synth_prompt`.
- **About 6% of validation gold stories exceed 256 tokens.** On the local
  200-story validation sample, 94% of gold stories (plus EOT) fit in
  `max_new=256`. A model that writes a story that long cannot reach its EOT, so
  `bench/win_rate.py` counts it as invalid (score -1). The sample is small. In
  the final run `eot_rate` fell from 93.2% (SFT) to 78.7% / 81.3% (DPO seeds 0
  / 1); whether those extra unfinished stories are stories that run out of
  tokens or stories that fail to end is not determined, and a `max_new=384`
  re-run is pending.
- **The finished-only comparison is descriptive, not paired.** The coverage
  numbers restricted to finished stories in `docs/results_final.md` condition
  on finishing, which selects a different subset of samples for each model, so
  they do not compare the same stories.
- **DPO was trained on the same criterion it is scored on.** The pair rule is
  whole-word keyword coverage, and the headline metric is keyword coverage, so
  the coverage gain is circular evidence of story quality.
- **The `eot_rate` guardrail (85%) was missed.** After DPO (lr 5e-6),
  `eot_rate` was 78.7% (seed 0) and 81.3% (seed 1) against 93.2% for SFT.
- **Sampling seeds are fixed per prompt.** Every model draws the same
  randomness for a given validation prompt, so the bootstrap CIs (over prompts)
  do not include sampling variation.
- **Two training seeds only.** Variation across training seeds is based on two
  runs, which also use different held-out prompt splits.
- **No human evaluation yet.** The blind spot-check in `docs/human_spotcheck.md`
  has not been rated.
- **The training corpus is partly garbled (mojibake).** In the Hub's parquet
  version of TinyStories (what the default loader and `data/prepare.py` read),
  UTF-8 curly quotes and dashes were decoded as cp1252: a right single quote
  (U+2019) appears as U+00E2 U+20AC U+2122, and the closing double quote loses a
  byte. 5.4% of stories in the local 1,000-story smoke shard contain the garbled
  apostrophe and none contain a correct one; streaming 3,000 stories per split
  from the Hub gives 4.9% (train) and 3.7% (validation). The original raw `.txt`
  files are clean, so the defect is in the parquet files, not in `prepare.py`
  (`docs/source_mojibake_check.md`). The model reproduces it: 1.7% (SFT) and
  3.3% / 2.3% (DPO seeds 0 / 1) of samples contain the garbled apostrophe. The
  whole-word scorer can undercount a keyword next to these characters (U+00E2 is
  a word character), but re-scoring with the apostrophe folded back changed 0
  of 1,800 samples (1 when every garbled sequence is folded), so the reported
  coverage is unaffected here (`docs/results_folded.md`).

## DPO learning rate

The DPO final runs use `lr = 5e-6`. It was chosen for **stability**, not as a proven win over the other arms: in the 3-arm sweep (1e-6, 5e-6, 2e-5; 444 updates, beta 0.1) the 2e-5 arm overfit the training pairs and early-stopped at update 351 (held-out loss rose after step 200), and the 1e-6 arm was still improving at its last eval. Held-out reward accuracy (198 pairs, standard error about 0.035) could not separate the arms, and 5e-6 vs 2e-5 differed by only about 0.012 in best held-out loss, so the choice is a mild preference. It was fixed before the final evaluation and not tuned afterwards.

## DPO result

DPO raises keyword coverage (+0.8 of 3, 95% CIs exclude zero, both training
seeds agree) and lowers `eot_rate` by 12-15 points, below the 85% guardrail.
It is not an overall win. Details, finished-only coverage and length
distributions: `docs/results_final.md`.
