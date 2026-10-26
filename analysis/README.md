# analysis/

Scripts that regenerate the written results from the saved final-run outputs.
CPU only, no GPU, no model.

| Script | Produces |
|---|---|
| `finished_only.py` | `docs/results_final.md`: headline DPO-minus-SFT table, finished-only coverage with prompt-bootstrap CIs, finished vs unfinished length tables |
| `make_spotcheck.py` | `docs/human_spotcheck.md` (20 blind, shuffled SFT-vs-DPO pairs, ratings blank) and a key file mapping A/B to the model |

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
