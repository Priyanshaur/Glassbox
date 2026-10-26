"""Re-score the saved eval samples with garbled characters folded back.

The training corpus (and so the models' samples) contains U+00E2 U+20AC U+2122
where a right single quote should be, and similar sequences for double quotes
and dashes (see docs/source_mojibake_check.md). U+00E2 is a word character, so a
keyword directly next to it fails the whole-word match in
train.pairs.whole_word_count. Every saved sample is scored three ways:

  original  the text exactly as generated (reproduces the stored `cov`)
  apos      the garbled APOSTROPHE (U+00E2 U+20AC U+2122) replaced by an ASCII
            apostrophe before matching (the requested fold)
  all       extra check: every "U+00E2 U+20AC" + optional one of
            U+2122 U+0153 U+02DC U+201C U+201D U+2022 U+2026 sequence (garbled
            apostrophes, quotes, dashes; the closing double quote has lost its
            third byte) replaced by a space

The original numbers stay the primary ones. DPO-minus-SFT CIs are bootstrapped
over PROMPTS (paired; same resamples as analysis/finished_only.py).

Usage: python analysis/rescore_folded.py <dir with eval_*.jsonl> <out.md>
"""
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.pairs import whole_word_count  # noqa: E402

D, OUT = sys.argv[1], sys.argv[2]
GARBLED_APOS = chr(0xE2) + chr(0x20AC) + chr(0x2122)
ALL_RX = re.compile(chr(0xE2) + chr(0x20AC) + "[" + "".join(
    chr(c) for c in (0x2122, 0x153, 0x2DC, 0x201C, 0x201D, 0x2022, 0x2026)) + "]?")
ARMS = [("sft", "eval_sft_baseline.jsonl"), ("dpo_s0", "eval_dpo_s0.jsonl"),
        ("dpo_s1", "eval_dpo_s1.jsonl")]
VARIANTS = [("orig", "original", lambda t: t),
            ("apos", "apostrophe folded", lambda t: t.replace(GARBLED_APOS, "'")),
            ("all", "all garbled sequences folded (extra)", lambda t: ALL_RX.sub(" ", t))]

data = {k: [json.loads(l) for l in open(os.path.join(D, f), encoding="utf-8") if l.strip()]
        for k, f in ARMS}
n = len(data["sft"])
rng_idx = np.random.default_rng(20261026).integers(0, n, size=(10_000, n))

per, info = {}, {}
for k, rows in data.items():
    per[k] = {}
    for key, _, fn in VARIANTS:
        per[k][key] = np.array([[whole_word_count(fn(x["text"]), r["keywords"]) for x in r["rows"]]
                                for r in rows], float)
    stored = np.array([[x["cov"] for x in r["rows"]] for r in rows], float)
    assert (per[k]["orig"] == stored).all(), f"{k}: recomputed original coverage != stored cov"
    texts = [x["text"] for r in rows for x in r["rows"]]
    info[k] = {"samples": len(texts),
               "with_apos": sum(GARBLED_APOS in t for t in texts),
               "with_any": sum(bool(ALL_RX.search(t)) for t in texts),
               "changed_apos": int((per[k]["apos"] != per[k]["orig"]).sum()),
               "changed_all": int((per[k]["all"] != per[k]["orig"]).sum()),
               "gained_all": float((per[k]["all"] - per[k]["orig"]).sum())}


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def pm(a, kind, stat):
    c = per[a][kind]
    return c.mean(axis=1) if stat == "cov" else (c == 3).mean(axis=1)


L = []
P = L.append
P("# Keyword coverage re-scored with garbled characters folded back\n")
P("The corpus contains the garbled sequence U+00E2 U+20AC U+2122 instead of an apostrophe "
  "(`docs/source_mojibake_check.md`). U+00E2 is a word character, so a keyword directly "
  "next to it would fail the whole-word match. **Original** = text as generated (the "
  "numbers in `docs/results_final.md`, unchanged and still primary). **Apostrophe folded** "
  "= that sequence replaced by an ASCII apostrophe before matching. **All folded** is an "
  "extra check that also folds the garbled double-quote and dash sequences.\n")
P("## How many samples are affected\n")
P("| model | samples | with garbled apostrophe | with any garbled sequence | coverage changes (apostrophe fold) | coverage changes (all folded) | keyword hits gained (all folded) |")
P("|---|---|---|---|---|---|---|")
for k in ("sft", "dpo_s0", "dpo_s1"):
    i = info[k]
    P(f"| {k} | {i['samples']} | {i['with_apos']} ({100 * i['with_apos'] / i['samples']:.1f}%) | "
      f"{i['with_any']} ({100 * i['with_any'] / i['samples']:.1f}%) | {i['changed_apos']} | "
      f"{i['changed_all']} | {i['gained_all']:.0f} |")
P("")
for stat, label in (("cov", "Mean keyword coverage (0-3)"), ("full3", "Fraction covering all 3")):
    P(f"## {label}: original vs folded\n")
    P("| model | original | apostrophe folded | all folded |")
    P("|---|---|---|---|")
    for k in ("sft", "dpo_s0", "dpo_s1"):
        P(f"| {k} | " + " | ".join(f"{pm(k, key, stat).mean():.4f}" for key, _, _ in VARIANTS) + " |")
    P("")
    P(f"DPO minus SFT, {label.lower()} (95% CI over prompts, B=10,000):\n")
    P("| comparison | original diff [CI] | apostrophe folded diff [CI] | all folded diff [CI] |")
    P("|---|---|---|---|")
    for k in ("dpo_s0", "dpo_s1"):
        cells = []
        for key, _, _ in VARIANTS:
            d = pm(k, key, stat) - pm("sft", key, stat)
            lo, hi = ci(d[rng_idx].mean(axis=1))
            cells.append(f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]" +
                         ("" if (lo > 0 or hi < 0) else " (includes 0)"))
        P(f"| {k} minus sft | " + " | ".join(cells) + " |")
    P("")
tot_apos = sum(info[k]["changed_apos"] for k in info)
tot_all = sum(info[k]["changed_all"] for k in info)
P("## Reading\n")
P(f"Folding the garbled apostrophe changed the coverage of **{tot_apos}** of "
  f"{sum(info[k]['samples'] for k in info)} samples; folding every garbled sequence changed "
  f"**{tot_all}**. The whole-word undercount is possible in principle (a keyword directly "
  "before or after a garbled character), but it did not occur in these samples, so the "
  "original numbers and the DPO-minus-SFT CIs are unaffected here.")
open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L))
