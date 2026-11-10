# analysis/

Scripts that regenerate the written results from the saved final-run outputs.
CPU only, no GPU, no model.

| Script | Produces |
|---|---|
| `finished_only.py` | `docs/results_final.md`: headline DPO-minus-SFT table, finished-only coverage with prompt-bootstrap CIs, finished vs unfinished length tables |
| `make_spotcheck.py` | `docs/human_spotcheck.md` (20 blind, shuffled SFT-vs-DPO pairs, ratings blank) and a key file mapping A/B to the model |
| `rescore_folded.py` | `docs/results_folded.md`: coverage re-scored with the garbled apostrophe (and, as an extra, all garbled sequences) folded back, original numbers alongside, DPO-minus-SFT CIs |
| `check_source_mojibake.py` | `docs/source_mojibake_check.md`: counts the garbled sequence in the Hub's streamed text and compares it with the original raw `.txt` file. **Needs network access** and takes no kernel input |

## Inputs

Both read the outputs of Kaggle kernel `priyanshgaur/glassbox-dpo`, **version 3**
(final run, repo commit `f58a309`): `final_report.json` and
`eval_sft_baseline.jsonl`, `eval_dpo_s0.jsonl`, `eval_dpo_s1.jsonl`. The outputs
also include multi-hundred-MB checkpoints, so fetch only these files:

```bash
kaggle kernels output priyanshgaur/glassbox-dpo/3 -p final_v3 --file-pattern "final_report|eval_.*jsonl"
```

## Run

```bash
python analysis/finished_only.py final_v3 docs/results_final.md
python analysis/make_spotcheck.py final_v3 docs/human_spotcheck.md checkpoints/human_spotcheck_key.json
python analysis/rescore_folded.py final_v3 docs/results_folded.md
python analysis/check_source_mojibake.py 3000 docs/source_mojibake_check.md   # network
```

Notes:

- Everything is seeded (bootstrap rng `20261026`, B=10,000; spot-check selection
  rng `20261027`), so re-running on the same inputs reproduces the committed
  files byte for byte.
- `make_spotcheck.py` refuses to overwrite an existing doc or key without
  `--force`: regenerating wipes the blank-or-filled ratings in the doc.
- The key file lives in `checkpoints/`, which is git-ignored, so it is never
  committed. Do not open it until the spot-check is fully rated.
- The finished-only comparison is descriptive, not paired: each model's
  finished subset differs, so it is not a same-story comparison.

## Phase 6 scripts (also in this folder)

| Script | Purpose |
|---|---|
| `a5_transfer.py` | A5: the base-trained SAE on the SFT (and base) model, plain and template inputs; resumable per cell |
| `scan_features.py`, `label_prep.py`, `blind_check.py`, `steering_candidates.py` | feature scans (train and validation), labeling cards, blind label check, the mechanical eligibility checks and seeded order for steering features |
| `steering.py` | the pre-registered steering experiment (`gen` then `analyze`) |
| `reeval_compare.py` | DPO re-evaluation with the corrected sampler vs the published numbers |
| `gen_steering_nb.py`, `check_*_notebook.py` | generate the steering notebook; run notebooks in a local kernel as a pre-flight check |
| `phase6_smoke.py` | Phase 6 code smoke runs (not evaluations) |

Run them from the repository root (they add the current directory to `sys.path`).
