"""Prepare the labeling material for a seeded random sample of alive features (docs/phase6_preregistration.md, B).

    python analysis/label_prep.py --scan checkpoints/glassbox-sae-l4/scan_train.pt --out_dir checkpoints/glassbox-sae-l4/labeling

Writes sample.json (feature ids, seed) and, per feature, the 24 strongest train contexts (one per document, +-12 tokens,
the activating token between << >>), its firing count, a_f (90th percentile of non-zero activations), the tokens its
decoder direction promotes/suppresses through the tied unembedding, and the token-level checks of
docs/steering_features.md. Labels are written by the assistant from this file; nothing here labels anything."""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from common.config import load_config
from interp import feature_scan as S
from interp.sae import load_sae
from tokenizer.bpe import BPETokenizer
from train.data_loader import ShardLoader
from analysis.a5_transfer import load_model

SEED = 20_261_101
N_SAMPLE = 60


def tokstr(tok, i):
    return tok.decode([int(i)]).replace("\n", "\n")


def feature_card(f, res, maps, tok, wte, sae):
    ctxs = S.top_contexts(res, maps, f)
    lens = wte @ sae.W_dec[f].detach()
    lens[256:] = float("nan")                                       # specials are not words
    lens = torch.nan_to_num(lens, nan=0.0)
    up, down = lens.topk(12).indices, (-lens).topk(12).indices
    at = Counter(tokstr(tok, c["tokens"][c["at"]]).strip().lower() for c in ctxs)
    top_tok, top_n = at.most_common(1)[0]
    punct = sum(1 for t, n in at.items() if not any(ch.isalpha() for ch in t) for _ in range(n))
    checks = {"n_contexts": len(ctxs), "distinct_docs": len({c["doc"] for c in ctxs}),
              "top_token": top_tok, "top_token_share": round(top_n / max(len(ctxs), 1), 3),
              "distinct_token_types": len(at), "non_alphabetic_share": round(punct / max(len(ctxs), 1), 3)}
    lines = []
    for k, c in enumerate(ctxs):
        t = c["tokens"]
        pre, mid, post = tok.decode(t[:c["at"]]), tok.decode([t[c["at"]]]), tok.decode(t[c["at"] + 1:])
        lines.append(f"{k + 1:>2}. [{c['act']:.2f}] " + (pre + "<<" + mid + ">>" + post).replace("\n", " "))
    return checks, lines, [tokstr(tok, i) for i in up], [tokstr(tok, i) for i in down]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/base_30m.yaml")
    ap.add_argument("--sae", default="checkpoints/glassbox-sae-l4/sae_latest.pt")
    ap.add_argument("--weights", default="checkpoints/glassbox-s3-final/weights_step12158.pt")
    ap.add_argument("--data_dir", default="data/full")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    res = torch.load(a.scan, weights_only=False)
    alive = (res["counts"] >= S.MIN_FIRINGS).nonzero().squeeze(1).tolist()
    sample = sorted(int(x) for x in np.random.default_rng(SEED).choice(alive, N_SAMPLE, replace=False))
    json.dump({"seed": SEED, "n_alive": len(alive), "features": sample, "scan": a.scan}, open(f"{a.out_dir}/sample.json", "w"))
    cfg = load_config(a.config)
    sae, _ = load_sae(a.sae)
    wte = load_model(cfg, a.weights).wte.weight.detach()
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    maps = ShardLoader(a.data_dir, "train").maps
    with open(f"{a.out_dir}/cards.md", "w", encoding="utf-8") as out:
        for f in sample:
            checks, lines, up, down = feature_card(f, res, maps, tok, wte, sae)
            p90 = S.percentile_from_hist(res["hist"][f], 0.9)
            out.write(f"## feature {f}  (fires {int(res['counts'][f])}x in {res['tokens']:,} tokens; a_f~{p90:.2f})\n")
            out.write(f"checks: {json.dumps(checks)}\npromotes: {' | '.join(up)}\nsuppresses: {' | '.join(down)}\n")
            out.write("\n".join(lines) + "\n\n")
    print(f"alive {len(alive)}/{sae.n_features}; sample of {N_SAMPLE} written to {a.out_dir}")


if __name__ == "__main__":
    main()
