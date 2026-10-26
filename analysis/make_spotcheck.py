"""Build the blind human spot-check from the saved kernel-v3 eval samples.

20 random prompts (of the 100 evaluated), one random sample per prompt (the
SAME sample index for both models, so both drew the same randomness), SFT vs
DPO seed 0 side by side in shuffled A/B order. Model labels go ONLY to the key
file. Nothing is rated here.

Usage: python analysis/make_spotcheck.py <input_dir> <out.md> <key.json> [--force]

Refuses to overwrite an existing <out.md> or <key.json> without --force:
re-running with the same seed regenerates BLANK ratings, which would erase a
filled-in spot-check.
"""
import json
import os
import sys

import numpy as np

D, DOC, KEY = sys.argv[1], sys.argv[2], sys.argv[3]
if "--force" not in sys.argv[4:]:
    for _p in (DOC, KEY):
        if os.path.exists(_p):
            sys.exit(f"refusing to overwrite {_p} (it may hold ratings); pass --force")
SEED = 20261027
load = lambda f: [json.loads(l) for l in open(os.path.join(D, f), encoding="utf-8") if l.strip()]
sft, dpo = load("eval_sft_baseline.jsonl"), load("eval_dpo_s0.jsonl")
assert [r["idx"] for r in sft] == [r["idx"] for r in dpo]
assert [r["prompt"] for r in sft] == [r["prompt"] for r in dpo]

rng = np.random.default_rng(SEED)
order = rng.permutation(len(sft))[:20]          # random prompts, random display order
items, key = [], []
for n, p in enumerate(order, 1):
    k = int(rng.integers(0, len(sft[p]["rows"])))
    swap = bool(rng.random() < 0.5)
    a_src, b_src = (("dpo_s0", "sft") if swap else ("sft", "dpo_s0"))
    txt = {"sft": sft[p]["rows"][k]["text"].strip(), "dpo_s0": dpo[p]["rows"][k]["text"].strip()}
    items.append((n, sft[p]["prompt"], txt[a_src], txt[b_src]))
    key.append({"id": n, "val_idx": sft[p]["idx"], "sample_k": k, "A": a_src, "B": b_src})


def quote(t):
    return "\n".join(("> " + l.rstrip()) if l.strip() else ">" for l in t.splitlines()) or ">"


out = []
P = out.append
P("# Human spot-check (blind; rate by hand)\n")
P("The rule-based numbers are circular: DPO was trained on the same keyword-coverage "
  "criterion it is scored on. This blind check asks whether a person prefers the DPO "
  "stories.\n")
P("## How to rate\n")
P("1. Read each item's prompt, then story **A** and story **B**. Which model wrote which "
  "is hidden and the order is shuffled.")
P("2. Judge: (a) does it use all of the prompt's words naturally, (b) does it read like a "
  "coherent story, (c) is it free of repetition and gibberish. Write `A`, `B` or `tie`.")
P("3. Stories are shown exactly as generated, with a 256-token limit, so some end "
  "abruptly. Rate what you read; do not guess which one is which.")
P("4. Do not open the key file until every item is rated. It is deliberately not in git: "
  "`checkpoints/human_spotcheck_key.json` (the `checkpoints/` folder is git-ignored).")
P("5. Afterwards unblind with the key (`A`/`B` map to `sft` or `dpo_s0`) and tally: "
  "how often the DPO story was preferred among non-ties, and ties.\n")
P("Provenance: 20 of the 100 validation prompts, chosen at random "
  f"(rng seed {SEED}); one random sample per prompt, the same sample index for both "
  "models; SFT vs DPO lr 5e-6 seed 0 (kernel `glassbox-dpo` v3). Ratings below are "
  "left blank on purpose.\n")
for n, prompt, a, b in items:
    P(f"## Item {n}\n")
    P("**Prompt:** " + prompt.split("\n")[0] + "\n")
    P("**Story A**\n")
    P(quote(a) + "\n")
    P("**Story B**\n")
    P(quote(b) + "\n")
    P(f"**Rating (item {n}):** preferred = ____ (A / B / tie)  |  notes: \n")
P("## Fill in\n")
P("- Rater:")
P("- Date:")
P("- Items rated: /20")
P("- Prefers A: , B: , tie:")
P("- After unblinding, DPO preferred: , SFT preferred: , ties:")
P("- Notes (failure examples welcome):")
open(DOC, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
json.dump({"seed": SEED, "dpo_model": "dpo_s0 (kernel v3, lr 5e-6, seed 0)",
           "note": "A/B give the model for each displayed story; do not open before rating",
           "items": key}, open(KEY, "w", encoding="utf-8"), indent=1)
print("items:", len(items), "| A=sft count:", sum(k["A"] == "sft" for k in key))
