"""Blind check of feature labels (docs/phase6_preregistration.md, B).

  build:  python analysis/blind_check.py build --labels L.json --scan scan_val.pt --out_dir D [--seed S]
          For each labeled feature: 10 TRUE contexts (the strongest activations of the disjoint validation scan, one per
          document) and 10 CONTROL contexts (random validation positions, window position >= 1, where the feature is
          inactive), shuffled together. D/sheet.md holds label + the 20 numbered contexts WITHOUT any activation or
          marker of which are true; D/key.json (the answer) must not be opened before the predictions are written.
  score:  python analysis/blind_check.py score --key D/key.json --pred P.json
          P.json = {feature: [numbers of the 10 contexts predicted active]}. Accuracy = correct classifications / 20;
          a label is valid at >= 16/20."""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from analysis.a5_transfer import load_model
from common.config import load_config
from interp import activations as A
from interp import feature_scan as S
from interp.sae import load_sae
from tokenizer.bpe import BPETokenizer
from train.data_loader import ShardLoader

CONTROL_SEED = 20_261_104
N_CONTROL_WINDOWS = 400
VALID_AT = 16


def _context(maps, shard, ap, tok, half=S.CTX_HALF):
    lo, hi = max(0, ap - half), min(len(maps[shard]), ap + half + 1)
    t = [int(x) for x in maps[shard][lo:hi]]
    k = ap - lo
    return (tok.decode(t[:k]) + "<<" + tok.decode([t[k]]) + ">>" + tok.decode(t[k + 1:])).replace("\n", " ")


def build(a):
    labels = {int(k): v for k, v in json.load(open(a.labels)).items()}
    res = torch.load(a.scan, weights_only=False)
    cfg = load_config(a.config)
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    loader = ShardLoader(a.data_dir, "validation")
    sae, sm = load_sae(a.sae)
    model = load_model(cfg, a.weights)
    feats = sorted(labels)
    # control pool: activations of the labeled features at random validation windows (positions >= 1)
    rng = np.random.default_rng(np.random.SeedSequence([CONTROL_SEED, 1]))
    starts = S.window_starts(loader, N_CONTROL_WINDOWS, 512, seed=CONTROL_SEED)
    ids = torch.from_numpy(np.stack([np.asarray(loader.maps[s][b:b + 512], dtype=np.int64) for s, b in starts]))
    with torch.no_grad():
        x = A.record_windows(model, ids, 4, 16)[:, 1:] * sm["act_scale"]
        acts = torch.cat([sae.encode(x[i:i + 8].reshape(-1, x.shape[-1]))[0][:, feats].reshape(-1, 511, len(feats))
                          for i in range(0, len(x), 8)])                              # (W, 511, nf)
    sheet, key = [], {}
    for j, f in enumerate(feats):
        true = S.top_contexts(res, loader.maps, f, n=a.n_true)
        items = [(True, _context(loader.maps, *_loc(res, f, c), tok), c["doc"]) for c in true]
        inactive = (acts[:, :, j] == 0).nonzero()
        pick = rng.choice(len(inactive), size=a.n_control, replace=False)
        for p in pick:
            w, pos = (int(v) for v in inactive[p])
            s, b = starts[w]
            items.append((False, _context(loader.maps, s, b + pos + 1, tok), ("ctrl", s, b)))
        order = rng.permutation(len(items))
        items = [items[i] for i in order]
        key[str(f)] = {"is_true": [bool(t) for t, _, _ in items], "n_true": sum(t for t, _, _ in items),
                       "true_docs": len({tuple(d) for t, _, d in items if t})}
        sheet.append(f"## feature {f}\nLabel: {labels[f]}\n" + "\n".join(f"{i + 1:>2}. {c}" for i, (_, c, _) in enumerate(items)) + "\n")
    os.makedirs(a.out_dir, exist_ok=True)
    open(f"{a.out_dir}/sheet.md", "w", encoding="utf-8").write("\n".join(sheet))
    json.dump(key, open(f"{a.out_dir}/key.json", "w"))
    print(f"wrote {len(feats)} features to {a.out_dir}/sheet.md; key in key.json (do not open before predicting)")


def _loc(res, f, c):
    """(shard, absolute position) of a top_contexts entry: found again from the pool by activation value."""
    j = int((res["pool_vals"][f] - c["act"]).abs().argmin())
    shard, start, pos = (int(t) for t in res["pool_loc"][f, j])
    return shard, start + pos


def score(a):
    key, pred = json.load(open(a.key)), json.load(open(a.pred))
    out, valid = {}, 0
    for f, k in key.items():
        guess = set(int(i) for i in pred.get(f, []))
        correct = sum(1 for i, t in enumerate(k["is_true"], 1) if (i in guess) == t)
        n = len(k["is_true"])
        out[f] = {"correct": correct, "of": n, "valid": correct >= VALID_AT * n // 20, "n_predicted": len(guess)}
        valid += out[f]["valid"]
    print(json.dumps({"valid": valid, "of": len(out), "fraction_valid": valid / len(out)}))
    json.dump(out, open(a.pred + ".scored.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    ap.add_argument("--labels"), ap.add_argument("--scan"), ap.add_argument("--out_dir")
    ap.add_argument("--key"), ap.add_argument("--pred")
    ap.add_argument("--config", default="configs/base_30m.yaml")
    ap.add_argument("--sae", default="checkpoints/glassbox-sae-l4/sae_latest.pt")
    ap.add_argument("--weights", default="checkpoints/glassbox-s3-final/weights_step12158.pt")
    ap.add_argument("--data_dir", default="data/full")
    ap.add_argument("--n_true", type=int, default=10)
    ap.add_argument("--n_control", type=int, default=10)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)
