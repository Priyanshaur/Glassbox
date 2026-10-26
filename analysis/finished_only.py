"""Finished-only keyword coverage + length distributions from the saved eval
samples of the final run (kernel v3), and the generated docs/results_final.md.

Usage: python analysis/finished_only.py <dir with eval_*.jsonl + final_report.json> <out.md>
Everything is computed from existing samples: no model, no GPU.
"""
import json
import os
import sys

import numpy as np

D, OUT = sys.argv[1], sys.argv[2]
ARMS = [("sft", "eval_sft_baseline.jsonl"), ("dpo_s0", "eval_dpo_s0.jsonl"),
        ("dpo_s1", "eval_dpo_s1.jsonl")]
rep = json.load(open(os.path.join(D, "final_report.json")))
data = {}
for name, f in ARMS:
    rows = [json.loads(l) for l in open(os.path.join(D, f), encoding="utf-8") if l.strip()]
    data[name] = rows
n = len(data["sft"])
assert all([r["idx"] for r in data[k]] == [r["idx"] for r in data["sft"]] for k in data)

# ---- per-prompt sufficient statistics for finished-only ratios ----------------
def stats(rows):
    f = np.array([sum(bool(x["eot"]) for x in r["rows"]) for r in rows], float)
    c = np.array([sum(x["cov"] for x in r["rows"] if x["eot"]) for r in rows], float)
    a3 = np.array([sum(x["cov"] == 3 for x in r["rows"] if x["eot"]) for r in rows], float)
    return f, c, a3


S = {k: stats(v) for k, v in data.items()}
BI = np.random.default_rng(20261026).integers(0, n, size=(10_000, n))


def ratio(num, den, idx=None):
    if idx is None:
        return num.sum() / den.sum()
    return num[idx].sum(axis=1) / den[idx].sum(axis=1)


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


lines = []
P = lines.append
P("# Final DPO evaluation (kernel v3, lr 5e-6, two training seeds)\n")
P("**Result.** DPO raises keyword coverage (about +0.8 of 3, 95% CIs exclude zero, "
  "both seeds agree) and lowers `eot_rate` by 12-15 points, to 79-81%, below the 85% "
  "guardrail set for this run. This is **not** an overall win: the coverage gain is on the "
  "criterion DPO was trained on, while stories got longer, fewer finish within 256 tokens, "
  "and gold-story perplexity rose about 5%. No human evaluation has been done yet.\n")
P("Provenance: Kaggle kernel `priyanshgaur/glassbox-dpo` version 3 (repo commit `f58a309`), "
  "100 validation prompts x 6 samples, `max_new=256`, temperature 0.8, top-k 40, identical "
  "prompts and per-prompt sampling seeds for every model. BEST_LR 5e-6 (chosen for "
  "stability, see README) was fixed before this evaluation; nothing was tuned afterwards.\n")

P("## Headline: DPO minus SFT (95% CI bootstrapped over prompts, B=10,000)\n")
P("| Metric | SFT | DPO seed 0 | diff (seed 0) | DPO seed 1 | diff (seed 1) |")
P("|---|---|---|---|---|---|")
names = [("cov", "Mean keyword coverage (0-3)"), ("full3", "Fraction covering all 3"),
         ("eot", "`eot_rate`"), ("valid", "Validity rate (EOT and 10-500 words)"),
         ("words", "Mean length (words)"), ("gold_ppl", "Gold-story perplexity")]
for k, label in names:
    a, b = rep["seeds"]["0"][k], rep["seeds"]["1"][k]
    f = (lambda v: f"{v:.3f}")
    d = lambda x: f"{x['diff']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]" + \
        ("" if x["excludes_zero"] else " (includes 0)")
    P(f"| {label} | {f(a['sft'])} | {f(a['dpo'])} | {d(a)} | {f(b['dpo'])} | {d(b)} |")
P("")
for sd in ("0", "1"):
    h = rep["seeds"][sd]["heldout"]
    P(f"- Seed {sd}: terminal `{h['terminal']}`, {h['updates_done']} updates, "
      f"{h['n_pairs']} held-out pairs, mean held-out DPO loss {h['mean_loss']:.4f}. "
      f"(Seeds also pick different held-out prompt splits.)")
P("")

P("## Keyword coverage among FINISHED stories only\n")
P("Finished = the sample ended at its own EOT within `max_new=256`. Rates are pooled "
  "over finished samples (total coverage / number finished); CIs resample PROMPTS with "
  "replacement (paired across arms for the differences, same resamples for every arm).\n")
P("| Model | finished / total | mean coverage (finished) | 95% CI | all 3 covered (finished) | 95% CI |")
P("|---|---|---|---|---|---|")
for k in ("sft", "dpo_s0", "dpo_s1"):
    f, c, a3 = S[k]
    cv, cl = ratio(c, f), ci(ratio(c, f, BI))
    a3v, a3l = ratio(a3, f), ci(ratio(a3, f, BI))
    P(f"| {k} | {int(f.sum())} / {n * 6} | {cv:.3f} | [{cl[0]:.3f}, {cl[1]:.3f}] | "
      f"{a3v:.3f} | [{a3l[0]:.3f}, {a3l[1]:.3f}] |")
P("")
P("| Difference (finished only) | mean coverage | 95% CI | all 3 covered | 95% CI |")
P("|---|---|---|---|---|")
for k in ("dpo_s0", "dpo_s1"):
    f, c, a3 = S[k]
    f0, c0, a30 = S["sft"]
    d = ratio(c, f) - ratio(c0, f0)
    dd = ci(ratio(c, f, BI) - ratio(c0, f0, BI))
    e = ratio(a3, f) - ratio(a30, f0)
    ee = ci(ratio(a3, f, BI) - ratio(a30, f0, BI))
    P(f"| {k} minus sft | {d:+.3f} | [{dd[0]:+.3f}, {dd[1]:+.3f}]"
      f"{'' if (dd[0] > 0 or dd[1] < 0) else ' (includes 0)'} | {e:+.3f} | "
      f"[{ee[0]:+.3f}, {ee[1]:+.3f}]{'' if (ee[0] > 0 or ee[1] < 0) else ' (includes 0)'} |")
P("")
P("Caveat: conditioning on finishing selects a different subset of samples for each model "
  "(DPO finishes fewer), so this is descriptive, not a causal comparison of the same stories.\n")

P("## Length (words) of finished vs unfinished stories\n")
P("Unfinished stories were cut at 256 tokens, so their word counts are truncated, not "
  "natural lengths.\n")
P("| Model | group | n | mean | median | p10 | p90 | max |")
P("|---|---|---|---|---|---|---|---|")
for k in ("sft", "dpo_s0", "dpo_s1"):
    allr = [x for r in data[k] for x in r["rows"]]
    for grp, sel in (("finished", [x for x in allr if x["eot"]]),
                     ("unfinished", [x for x in allr if not x["eot"]])):
        w = np.array([x["words"] for x in sel], float)
        if len(w) == 0:
            P(f"| {k} | {grp} | 0 | - | - | - | - | - |")
            continue
        P(f"| {k} | {grp} | {len(w)} | {w.mean():.1f} | {np.median(w):.0f} | "
          f"{np.percentile(w, 10):.0f} | {np.percentile(w, 90):.0f} | {w.max():.0f} |")
P("")

P("## Caveats\n")
P("- DPO was trained on the same keyword-coverage criterion it is scored on, so the "
  "coverage gain is circular evidence of quality.")
P("- The `eot_rate` guardrail (85%) was missed: 78.7% (seed 0) and 81.3% (seed 1) against 93.2% for SFT.")
P("- Sampling seeds are fixed per prompt; the bootstrap CIs cover prompt variation only, "
  "not sampling or training-seed variation beyond two seeds.")
P("- Only two training seeds; no human evaluation yet (see `docs/human_spotcheck.md`).")
open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("\n".join(lines))
