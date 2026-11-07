"""A5 (SFT transfer), pre-registered in docs/phase6_preregistration.md, evaluated locally.

The layer-4 SAE trained on the BASE model is applied, unchanged, to the SFT model's layer-4 residual:
FVU and CE recovered (with the SFT model's own CE_clean and its own mean-ablation baseline) on
  (1) plain validation windows (the same fixed draw as A1-A4: EVAL_SEED, >= 500,000 tokens), and
  (2) template-formatted prompt+story sequences (validation stories, response tokens counted for CE;
      all non-pad positions for FVU).
Pre-registered rule, thresholds unchanged (interp.eval_sae.THRESHOLDS): FVU <= 0.25 AND CE recovered >= 0.80
on BOTH input types; A5 fails if either fails. The BASE model on the same windows is reported for
comparison only (it does not gate). Nothing is trained or tuned here.

    python analysis/a5_transfer.py --sae checkpoints/glassbox-sae-l4/sae_latest.pt \
        --base_weights checkpoints/glassbox-s3-final/weights_step12158.pt \
        --sft_weights checkpoints/glassbox-sft-final/sft_final_best.pt \
        --data_dir data/full --out checkpoints/glassbox-sae-l4/a5_report.json \
        --expect_sae_sha bd9ec5e3192861b16b9976eb3050077e959debb611f051dc382c494e64a502e9
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
from common.config import load_config
from interp import activations as A
from interp import eval_sae as E
from interp.sae import load_sae
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer
from train.data_loader import ShardLoader


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(cfg, path):
    m = GPT(cfg["model"])
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def measure(model, sae, layer, scale, plain, template, chunk, log, out=None, on_cell=lambda: None):
    """FVU/L0 and CE recovered of the SAE on one model, for both input types. `out` may already hold finished cells
    (a resumed run): those are skipped; `on_cell` is called after each new cell so the report is saved per cell."""
    out = {} if out is None else out
    inputs, targets = plain
    t = time.time()
    if "plain" in out:
        log("    plain    SKIPPED (already saved)")
    else:
        _plain(model, sae, layer, scale, inputs, targets, chunk, log, out, t)
        on_cell()
    if "template" in out:
        log("    template SKIPPED (already saved)")
    else:
        _template(model, sae, layer, scale, template, chunk, log, out)
        on_cell()
    return out


def _plain(model, sae, layer, scale, inputs, targets, chunk, log, out, t):
    acts = (A.record_windows(model, inputs, layer, chunk) * scale).reshape(-1, sae.d_in)
    m = E.reconstruction_metrics(sae, acts)
    c = E.ce_recovered(model, sae, layer, scale, inputs, targets, None, chunk)
    out["plain"] = {"fvu": m["fvu"], "l0": m["l0"], "tokens": int(inputs.numel()), **{k: c[k] for k in (
        "ce_clean", "ce_sae", "ce_meanablate", "ce_recovered")}}
    log(f"    plain    fvu {m['fvu']:.4f}  ce_recovered {c['ce_recovered']:.4f}  ({time.time() - t:.0f}s)")


def _template(model, sae, layer, scale, template, chunk, log, out):
    ti, tt, tm = template
    t = time.time()
    f = E.template_fvu(model, sae, layer, scale, ti, tm, chunk)
    c = E.ce_recovered(model, sae, layer, scale, ti, tt, tm, chunk)
    out["template"] = {"fvu": f["fvu"], "l0": f["l0"], "sequences": int(ti.shape[0]), "response_tokens": int(c["tokens"]),
                       **{k: c[k] for k in ("ce_clean", "ce_sae", "ce_meanablate", "ce_recovered")}}
    log(f"    template fvu {f['fvu']:.4f}  ce_recovered {c['ce_recovered']:.4f}  ({time.time() - t:.0f}s)")


def judge(res):
    """Per-input-type flags. A NaN CE recovered (mean-ablation no worse than clean: the metric is undefined) is
    never read as a pass or a silent fail: the flag is None and A5_transfer is None (undetermined)."""
    T = E.THRESHOLDS
    flags = {}
    for kind in ("plain", "template"):
        r = res[kind]
        ce_def = r["ce_recovered"] == r["ce_recovered"]                      # not NaN
        flags[kind] = {"fvu_ok": bool(r["fvu"] <= T["sft_fvu_max"]),
                       "ce_ok": bool(r["ce_recovered"] >= T["sft_ce_recovered_min"]) if ce_def else None}
        if not flags[kind]["fvu_ok"] or flags[kind]["ce_ok"] is False:
            flags[kind]["passes"] = False                                     # a definite failure is a failure
        else:
            flags[kind]["passes"] = None if not ce_def else True
    ps = [flags[k]["passes"] for k in ("plain", "template")]
    if any(p is False for p in ps):
        flags["A5_transfer"] = False                                          # fails if either fails
    elif any(p is None for p in ps):
        flags["A5_transfer"] = None
    else:
        flags["A5_transfer"] = True
    return flags


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_30m.yaml")
    ap.add_argument("--sae", required=True)
    ap.add_argument("--base_weights", required=True)
    ap.add_argument("--sft_weights", required=True)
    ap.add_argument("--data_dir", required=True, help="dir with validation shards + meta.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--eval_tokens", type=int, default=E.MIN_EVAL_TOKENS)
    ap.add_argument("--n_template", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--expect_sae_sha", default=None)
    ap.add_argument("--expect_base_sha", default=None)
    ap.add_argument("--expect_sft_sha", default=None)
    a = ap.parse_args(argv)
    log = lambda s: print(s, flush=True)
    t0 = time.time()

    shas = {"sae_latest.pt": sha256(a.sae), "base_weights": sha256(a.base_weights), "sft_weights": sha256(a.sft_weights)}
    for k, v in shas.items():
        log(f"[a5] sha256 {k:<14} {v}")
    for key, exp in (("sae_latest.pt", a.expect_sae_sha), ("base_weights", a.expect_base_sha), ("sft_weights", a.expect_sft_sha)):
        if exp:
            ok = shas[key] == exp
            log(f"[a5] {key} expected {exp[:16]}... -> {'MATCH' if ok else 'MISMATCH'}")
            assert ok, f"{key} hash mismatch; stopping"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "analysis/a5_transfer.py"]).returncode:
            commit += "+uncommitted-edits-to-a5_transfer.py"       # honest: the code run differs from the commit
    except OSError:
        commit = "unknown"
    log(f"[a5] code commit {commit}")
    partial = {"label": "A5 partial (run in progress)", "sha256": shas, "commit": commit}
    saved = {}
    if os.path.exists(a.out):                                   # resume: keep finished cells of a killed run
        old = json.load(open(a.out))
        if old.get("sha256") == shas and old.get("label", "").startswith("A5 partial"):
            saved = {m: old[m] for m in ("sft", "base") if m in old}
            partial["resumed_from_commit"] = old.get("resumed_from_commit") or old.get("commit")
            log(f"[a5] RESUME: report has cells {{ {', '.join(f'{m}: {sorted(c)}' for m, c in saved.items())} }} "
                f"from commit {partial['resumed_from_commit']}; same hashes")
        else:
            log("[a5] existing report does not match this run (hashes/label): ignoring it, starting fresh")
    for m, c in saved.items():
        partial[m] = c
    def save():
        tmp = a.out + ".tmp"
        json.dump(partial, open(tmp, "w"), indent=1)
        os.replace(tmp, a.out)                                  # atomic: a kill cannot leave a half-written report
    save()

    cfg = load_config(a.config)
    sae, meta = load_sae(a.sae)
    sae.eval()
    scale = meta["act_scale"]
    assert meta.get("layer", a.layer) == a.layer
    log(f"[a5] SAE d_in {sae.d_in} F {sae.n_features} k {sae.k}, layer {a.layer}, act_scale {scale:.5f}, "
        f"epochs_done {meta.get('epochs_done')}")
    base, sft = load_model(cfg, a.base_weights), load_model(cfg, a.sft_weights)

    va = ShardLoader(a.data_dir, "validation")
    n_win = max(a.eval_tokens // a.ctx, 1)
    ids = va.batch_ids(E.EVAL_SEED, 0, n_win, a.ctx)
    plain = (ids[:, :a.ctx], ids[:, 1:a.ctx + 1])
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    template = E.template_sequences(tok, a.data_dir, a.n_template, a.ctx)
    log(f"[a5] plain windows {n_win} x {a.ctx} = {plain[0].numel():,} tokens | template sequences {template[0].shape[0]}"
        f" (len {template[0].shape[1]}, response tokens {int(template[2].sum()):,})")

    # a resumed cell must have been measured on these exact inputs
    for m, c in saved.items():
        if "plain" in c: assert c["plain"]["tokens"] == plain[0].numel(), "saved plain cell: different windows"
        if "template" in c: assert c["template"]["sequences"] == template[0].shape[0], "saved template cell: different sequences"
    res = {}
    for name, model in (("sft", sft), ("base", base)):
        log(f"[a5] {name} model:")
        partial.setdefault(name, {})
        def on_cell(name=name):
            partial["label"] = f"A5 partial: cells saved {[(m, sorted(c)) for m, c in partial.items() if m in ('sft','base')]}"
            save()
        res[name] = measure(model, sae, a.layer, scale, plain, template, a.chunk, log, out=partial[name], on_cell=on_cell)
    flags = judge(res["sft"])
    base_flags = judge(res["base"])
    T = E.THRESHOLDS
    report = {"label": "A5 SFT transfer (pre-registered rule; thresholds unchanged)",
              "thresholds": {"fvu_max": T["sft_fvu_max"], "ce_recovered_min": T["sft_ce_recovered_min"]},
              "sha256": shas, "commit": commit, "layer": a.layer, "act_scale": scale, "eval_seed": E.EVAL_SEED,
              "sft": res["sft"], "base_same_windows_reference": res["base"],
              "verdict_sft": flags, "A5_transfer": flags["A5_transfer"],
              "base_reference_flags_not_gating": base_flags,
              "steering_target": {True: "sft", False: "base", None: "undetermined"}[flags["A5_transfer"]],
              "seconds": time.time() - t0}
    json.dump(report, open(a.out, "w"), indent=1)
    T = report["thresholds"]
    log("")
    log(f"[a5] thresholds: FVU <= {T['fvu_max']}, CE recovered >= {T['ce_recovered_min']} on BOTH input types")
    log(f"[a5] {'input':<9}{'model':<6}{'FVU':>8}{'CE rec':>9}   FVU ok  CE ok")
    for kind in ("plain", "template"):
        for name in ("sft", "base"):
            r, fl = res[name][kind], (flags if name == "sft" else base_flags)[kind]
            log(f"[a5] {kind:<9}{name:<6}{r['fvu']:>8.4f}{r['ce_recovered']:>9.4f}   {str(fl['fvu_ok']):<7} {fl['ce_ok']}"
                + ("" if name == "sft" else "   (reference, not gating)"))
    log(f"[a5] VERDICT A5_transfer (SFT, fails if either input type fails): {report['A5_transfer']}"
        f" -> steering target: {report['steering_target']} | {report['seconds']:.0f}s")
    return report


if __name__ == "__main__":
    main()
