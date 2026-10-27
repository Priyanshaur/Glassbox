"""Phase 6 smoke runs (code checks, NOT evaluations against the pre-registered thresholds).

1. The debug model (random init, 606k parameters) through the real CLIs: train_sae -> eval_sae
   with the same model as the "SFT" model for the transfer path.
2. Activation-norm statistics of the real base model at every layer on the local smoke
   shard, with no SAE training: informs the normalization scale and the position-0 norm.

Usage: python analysis/phase6_smoke.py docs/phase6_smoke.md
"""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import load_config  # noqa: E402
from interp import activations as A  # noqa: E402
from interp import eval_sae, train_sae  # noqa: E402
from model.gpt import GPT  # noqa: E402
from train.data_loader import ShardLoader  # noqa: E402

BASE_WEIGHTS = "checkpoints/glassbox-s3-final/weights_step12158.pt"


def debug_smoke(L):
    cfg = load_config("configs/debug_cpu.yaml")
    torch.manual_seed(2)
    with tempfile.TemporaryDirectory() as d:
        w = os.path.join(d, "debug.pt")
        torch.save(GPT(cfg["model"]).state_dict(), w)
        out = os.path.join(d, "sae")
        train_sae.main(["--config", "configs/debug_cpu.yaml", "--weights", w, "--out_dir", out,
                        "--layer", "1", "--n_features", "128", "--k", "4", "--lr", "0.002",
                        "--warmup_steps", "4", "--batch_tokens", "256", "--buffer_tokens", "1024",
                        "--epochs", "6", "--ctx", "64", "--dead_steps", "6", "--aux_k", "8",
                        "--norm_windows", "8", "--chunk", "8", "--log_every", "2", "--eval_every", "1",
                        "--ckpt_every", "2", "--eval_windows", "8"])
        rows = [json.loads(x) for x in open(os.path.join(out, "metrics_sae.jsonl"), encoding="utf-8")]
        evals = [(r["epoch"] + 1, r["eval"]) for r in rows if "eval" in r]
        rep_path = os.path.join(d, "report.json")
        eval_sae.main(["--config", "configs/debug_cpu.yaml", "--weights", w,
                       "--sae", os.path.join(out, "sae_latest.pt"), "--sft_weights", w, "--out", rep_path,
                       "--eval_tokens", "512", "--dead_tokens", "4096"])
        rep = json.load(open(rep_path))
    L.append("## 1. Debug-model smoke (random-init 606k-parameter model, layer 1, F=128, k=4)\n")
    L.append("Goes through the real CLIs (`interp.train_sae`, `interp.eval_sae`). The model is untrained, "
             "so the numbers say nothing about interpretability; the point is that the pipeline runs end to end.\n")
    L.append("| buffer epoch | val FVU | L0 | latents alive |")
    L.append("|---|---|---|---|")
    for e, ev in evals:
        L.append(f"| {e} | {ev['fvu']:.4f} | {ev['l0']:.2f} | {ev['alive_frac']:.3f} |")
    L.append("")
    tr = rep["transfer"]
    L.append(f"Evaluation report: FVU {rep['fvu']:.4f}, L0 {rep['l0']:.2f}, dead fraction "
             f"{rep['dead_fraction']:.3f}, CE recovered {rep['ce_recovered']:.3f} "
             f"(CE clean {rep['ce_clean']:.4f}, SAE {rep['ce_sae']:.4f}, mean-ablate "
             f"{rep['ce_meanablate']:.4f}). Transfer path (same model as SFT): plain FVU "
             f"{tr['plain']['fvu']:.4f}, template FVU {tr['template']['fvu']:.4f}, template CE recovered "
             f"{tr['template']['ce_recovered']:.3f}.\n")
    L.append("Verdict flags from the same run (an untrained 128-latent SAE on a tiny model is not "
             f"expected to pass): `{rep['verdict']}`.\n")


def base_norms(L):
    if not os.path.exists(BASE_WEIGHTS):
        L.append("## 2. Base-model activation norms\n\nBase weights not present; skipped.\n")
        return
    cfg = load_config("configs/base_30m.yaml")
    m = GPT(cfg["model"])
    m.load_state_dict(torch.load(BASE_WEIGHTS, map_location="cpu", weights_only=True))
    m.eval()
    tr = ShardLoader(cfg["data"]["out_dir"], "train")
    L.append("## 2. Base-model residual norms (no SAE; local smoke shard, 64 windows x 512 tokens)\n")
    L.append("`weights_step12158.pt`, output of each block. `max/mean` is the largest token norm over the "
             "mean; `pos0` and `rest` are the mean norm at position 0 and at positions 1+.\n")
    L.append("| layer | mean norm | std | p99 | max/mean | pos0 mean | rest mean | scale = sqrt(512)/mean |")
    L.append("|---|---|---|---|---|---|---|---|")
    for layer in range(8):
        s = A.norm_stats(m, tr, layer, 0, 64, 512)
        L.append(f"| {layer} | {s['mean_norm']:.2f} | {s['std_norm']:.2f} | {s['p99']:.2f} | "
                 f"{s['max_over_mean']:.1f} | {s['pos0_mean_norm']:.2f} | {s['rest_mean_norm']:.2f} | "
                 f"{(512 ** 0.5) / s['mean_norm']:.4f} |")
    L.append("")
    L.append("Windows start at random offsets inside documents, so position 0 here is a mid-document "
             "token, not a start-of-text token.\n")


def main(out_md):
    L = ["# Phase 6 smoke runs (code checks, not threshold evaluations)\n",
         "Generated by `analysis/phase6_smoke.py` on CPU. Nothing here is compared against "
         "`docs/phase6_preregistration.md`; no SAE has been trained on the base model.\n"]
    debug_smoke(L)
    base_norms(L)
    open(out_md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
