"""Mechanical part of the steering-feature eligibility rules (docs/steering_features.md, rules 2 and 3) for ALL 4096
features, followed by the fixed random order (seed 20261102). Writes candidates.json. Nothing here labels a feature or
looks at any steering result: label validity (rule 1) and the keyword set (rule 4) are checked afterwards, in this order,
until 5 features pass.

  checks per feature (train scan, position 0 skipped): top token share <= 0.5, >= 6 distinct token types,
  non-alphabetic share <= 0.5, >= 5 distinct documents among the 24 top contexts, and >= 5 distinct documents among the
  10 true-active contexts of the disjoint validation scan."""
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from common.config import load_config
from interp import feature_scan as S
from tokenizer.bpe import BPETokenizer
from train.data_loader import ShardLoader

SEED = 20_261_102
OUT = "checkpoints/glassbox-sae-l4/labeling/candidates.json"


def checks(res, maps, tok, f, n):
    ctxs = S.top_contexts(res, maps, f, n=n)
    at = Counter(tok.decode([c["tokens"][c["at"]]]).strip().lower() for c in ctxs)
    k = max(len(ctxs), 1)
    nonalpha = sum(v for t, v in at.items() if not any(ch.isalpha() for ch in t))
    return {"n": len(ctxs), "docs": len({c["doc"] for c in ctxs}), "types": len(at),
            "top_token": at.most_common(1)[0][0] if at else "", "top_share": round(at.most_common(1)[0][1] / k, 3) if at else 1.0,
            "nonalpha_share": round(nonalpha / k, 3)}


def main():
    cfg = load_config("configs/base_30m.yaml")
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    tr, va = torch.load("checkpoints/glassbox-sae-l4/scan_train.pt", weights_only=False), \
        torch.load("checkpoints/glassbox-sae-l4/scan_val.pt", weights_only=False)
    mt, mv = ShardLoader("data/full", "train").maps, ShardLoader("data/full", "validation").maps
    rows, ok = {}, []
    for f in range(tr["counts"].shape[0]):
        a, b = checks(tr, mt, tok, f, 24), checks(va, mv, tok, f, 10)
        passes = (a["top_share"] <= 0.5 and a["types"] >= 6 and a["nonalpha_share"] <= 0.5 and a["docs"] >= 5
                  and b["docs"] >= 5 and b["n"] >= 10)
        rows[f] = {"train24": a, "val10": b, "mechanical_pass": bool(passes)}
        if passes:
            ok.append(f)
    order = [int(x) for x in np.random.default_rng(SEED).permutation(ok)]
    json.dump({"seed": SEED, "n_features": len(rows), "n_mechanical_pass": len(ok), "order": order, "rows": rows}, open(OUT, "w"))
    print(f"mechanical pass: {len(ok)}/{len(rows)}; first in seeded order: {order[:15]}")
    for name, cond in (("top_share<=0.5", lambda r: r["train24"]["top_share"] <= 0.5), ("types>=6", lambda r: r["train24"]["types"] >= 6),
                       ("nonalpha<=0.5", lambda r: r["train24"]["nonalpha_share"] <= 0.5)):
        print(f"  alone {name}: {sum(cond(r) for r in rows.values())}")


if __name__ == "__main__":
    main()
