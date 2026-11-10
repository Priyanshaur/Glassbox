"""Corrected re-evaluation (engine sampler, correct first token) vs the published numbers (old sampler), paired by prompt.
Reads the per-prompt jsonl files written by train.eval_quality; bootstrap over PROMPTS (B=10,000), as in docs/results_final.md.
    python analysis/reeval_compare.py <new_dir> <old_dir>      # new_dir: sft_engine/dpo_s0_engine/dpo_s1_engine .jsonl; old_dir: eval_*.jsonl"""
import json
import sys

import numpy as np


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["idx"]: {"cov": np.mean([x["cov"] for x in r["rows"]]), "eot": np.mean([x["eot"] for x in r["rows"]]),
                       "words": np.mean([x["words"] for x in r["rows"]])} for r in rows}


def diff_ci(a, b, key, B=10_000, seed=0):
    idx = sorted(a)
    d = np.array([a[i][key] - b[i][key] for i in idx])
    m = d[np.random.default_rng(seed).integers(0, len(d), size=(B, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(new_dir, old_dir):
    new = {"sft": load(f"{new_dir}/sft_engine.jsonl"), "dpo_s0": load(f"{new_dir}/dpo_s0_engine.jsonl"), "dpo_s1": load(f"{new_dir}/dpo_s1_engine.jsonl")}
    old = {"sft": load(f"{old_dir}/eval_sft_baseline.jsonl"), "dpo_s0": load(f"{old_dir}/eval_dpo_s0.jsonl"), "dpo_s1": load(f"{old_dir}/eval_dpo_s1.jsonl")}
    out = {}
    for name, label in (("old sampler (published)", old), ("corrected sampler", new)):
        for s in ("dpo_s0", "dpo_s1"):
            for key in ("cov", "eot", "words"):
                out[f"{label is new}|{s}|{key}"] = diff_ci(label[s], label["sft"], key)
    print(f"{'DPO minus SFT':<26}{'metric':<8}{'old sampler':>30}{'corrected sampler':>34}")
    for s in ("dpo_s0", "dpo_s1"):
        for key in ("cov", "eot", "words"):
            o, n = out[f"False|{s}|{key}"], out[f"True|{s}|{key}"]
            print(f"{s:<26}{key:<8}{o[0]:+9.3f} [{o[1]:+.3f},{o[2]:+.3f}]{n[0]:+14.3f} [{n[1]:+.3f},{n[2]:+.3f}]")
    for name in ("sft", "dpo_s0", "dpo_s1"):
        print(name, "mean cov %.3f -> %.3f | eot %.3f -> %.3f" % (np.mean([v["cov"] for v in old[name].values()]),
              np.mean([v["cov"] for v in new[name].values()]), np.mean([v["eot"] for v in old[name].values()]),
              np.mean([v["eot"] for v in new[name].values()])))
    json.dump({k: list(v) for k, v in out.items()}, open(f"{new_dir}/compare.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
