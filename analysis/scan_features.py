"""Scan a split with the accepted layer-4 SAE on the BASE model (the model the SAE was trained on); resumable.

    python analysis/scan_features.py --split train --tokens 10000000 --out checkpoints/glassbox-sae-l4/scan_train.pt
    python analysis/scan_features.py --split validation --tokens 4000000 --sequential --out .../scan_val.pt

Logs the SHA-256 of the SAE and weights (stops on mismatch with --expect_*), the code commit, seeds and speed. The
state is saved every --block windows (atomic); a rerun with the same arguments continues where it stopped."""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from analysis.a5_transfer import load_model, sha256
from common.config import load_config
from interp import feature_scan as S
from interp.sae import load_sae
from train.data_loader import ShardLoader


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_30m.yaml")
    ap.add_argument("--sae", default="checkpoints/glassbox-sae-l4/sae_latest.pt")
    ap.add_argument("--weights", default="checkpoints/glassbox-s3-final/weights_step12158.pt")
    ap.add_argument("--data_dir", default="data/full")
    ap.add_argument("--split", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--block", type=int, default=400, help="windows per saved block")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect_sae_sha", default=None)
    ap.add_argument("--expect_weights_sha", default=None)
    a = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)
    shas = {"sae": sha256(a.sae), "weights": sha256(a.weights)}
    for k, e in (("sae", a.expect_sae_sha), ("weights", a.expect_weights_sha)):
        log(f"[scan] sha256 {k} {shas[k]}" + (f" expected -> {'MATCH' if shas[k] == e else 'MISMATCH'}" if e else ""))
        assert not e or shas[k] == e, f"{k} hash mismatch; stopping"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "interp", "analysis", "model"],
                           capture_output=True, text=True).stdout.strip()
    meta = {"split": a.split, "tokens_requested": a.tokens, "sequential": a.sequential, "seed": S.SCAN_SEED, "ctx": a.ctx,
            "layer": a.layer, "sha256": shas, "commit": commit + ("+uncommitted" if dirty else ""),
            "skip_first_position": True, "pool": S.POOL, "command": " ".join(sys.argv)}
    log(f"[scan] commit {meta['commit']}")
    cfg = load_config(a.config)
    sae, sm = load_sae(a.sae)
    sae.eval()
    model = load_model(cfg, a.weights)
    loader = ShardLoader(a.data_dir, a.split)
    n_win = a.tokens // a.ctx
    starts = S.window_starts(loader, n_win, a.ctx, sequential=a.sequential)
    log(f"[scan] {len(starts)} windows x {a.ctx} = {len(starts) * a.ctx:,} tokens ({a.split}); act_scale {sm['act_scale']:.5f}")
    state, done = None, 0
    if os.path.exists(a.out):
        old = torch.load(a.out, weights_only=False)
        if old["meta"]["sha256"] == shas and old["meta"]["split"] == a.split and old["meta"]["seed"] == S.SCAN_SEED \
                and old["meta"]["tokens_requested"] == a.tokens and old["meta"]["sequential"] == a.sequential:
            state, done = old, old["windows_done"]
            log(f"[scan] RESUME at window {done}/{len(starts)} (block saved earlier; same hashes/seed/args)")
        else:
            log("[scan] existing file does not match: starting fresh")
    t0 = time.time()
    first = done
    while done < len(starts):
        blk = starts[done:done + a.block]
        state = S.scan(model, sae, sm["act_scale"], loader, a.layer, blk, a.ctx, a.chunk, state=state)
        done += len(blk)
        state.update({"meta": meta, "windows_done": done})
        torch.save(state, a.out + ".tmp")
        os.replace(a.out + ".tmp", a.out)
        rate = (done - first) * (a.ctx - 1) / (time.time() - t0)
        log(f"[scan] {done}/{len(starts)} windows | {rate:,.0f} tok/s | eta {(len(starts) - done) * (a.ctx - 1) / rate / 60:.0f} min")
    alive = int((state["counts"] >= S.MIN_FIRINGS).sum())
    log(f"[scan] DONE tokens {state['tokens']:,} | features firing >= {S.MIN_FIRINGS}x: {alive}/{sae.n_features} "
        f"| never fired: {int((state['counts'] == 0).sum())}")
    with open(a.out + ".json", "w") as f:
        json.dump({**meta, "windows_done": done, "tokens_scanned": state["tokens"], "alive": alive}, f, indent=1)


if __name__ == "__main__":
    main()
