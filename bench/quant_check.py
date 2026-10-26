"""M3 quantization gate: file size, gold-story perplexity, small generation check.

Checkpoint under test: the SFT checkpoint (checkpoints/glassbox-sft-final/
sft_final_best.pt), NOT a DPO model. CPU, fp32 baseline.

Variants: fp32 | int8_linear (all block Linears) | int8_all (Linears + tied
embedding/head). "int8" is accepted as the most aggressive variant that passes.

PRE-REGISTERED CRITERIA (written before any result was produced)
-----------------------------------------------------------------
1. Perplexity gate: gold-story perplexity of the int8 variant must be within 2%
   of fp32, i.e. |ppl_int8 / ppl_fp32 - 1| <= 0.02. Perplexity is computed as in
   train/eval_quality.py (response tokens of the gold story given the eval
   prompt, exp of total NLL / total tokens) on the first 100 validation prompts.
   Sanity: the local fp32 value should be close to the Kaggle SFT value 2.636.
2. Generation check: the first 20 of those prompts x 6 samples, max_new 256,
   temperature 0.8, top-k 40, seed = validation story idx, same prompts and seeds
   for every variant, the inference engine in its default (correct) mode.
   Metrics: eot_rate and mean keyword coverage (whole-word, 0-3).
   "Clear damage" for a metric means EITHER the point estimate of
   (int8 - fp32) is worse than -0.10 (eot_rate) / -0.20 (coverage), OR its 95%
   CI (bootstrap over prompts, paired) lies entirely below zero. No clear damage
   on both metrics = pass. Only 20 prompts, so the check is low-powered: it can
   miss a small drop and says so.
3. int8 is accepted only if (1) and (2) both pass.

Usage: python -m bench.quant_check [--out docs/quantization_results.md]
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import torch

from common.config import load_config
from inference.generate import Generator
from inference.quantize import quantize_model, save_quantized, tensor_bytes
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer
from train.eval_prompts import val_prompts
from train.eval_quality import mean_nll
from train.pairs import whole_word_count

KAGGLE_SFT_GOLD_PPL = 2.636     # eval_quality on the 100 val prompts, kernel v3
PPL_TOL = 0.02
EOT_DROP, COV_DROP = -0.10, -0.20


def gold_ppl(model, tok, vp, ctx):
    nll = ntok = 0.0
    with torch.no_grad():
        for _, prompt, _, gold in vp:
            a, b = mean_nll(model, tok, prompt, gold, ctx, "cpu")
            nll, ntok = nll + a * b, ntok + b
    return float(np.exp(nll / ntok))


def gen_check(model, tok, vp20, log):
    g = Generator(model, tok)
    eot, cov = [], []
    for k, (idx, prompt, kw, _) in enumerate(vp20, 1):
        r = g.generate(prompt, n=6, max_new=256, temperature=0.8, top_k=40, seed=idx)
        eot.append(np.mean(r.eot_hit))
        cov.append(np.mean([whole_word_count(t, kw) for t in r.texts]))
        if k % 5 == 0:
            log(f"    generation {k}/{len(vp20)} prompts")
    return np.array(eot), np.array(cov)


def boot_ci(d, idx):
    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="checkpoints/glassbox-sft-final/sft_final_best.pt")
    ap.add_argument("--config", default="configs/base_30m.yaml")
    ap.add_argument("--out", default="docs/quantization_results.md")
    ap.add_argument("--n_ppl", type=int, default=100)
    ap.add_argument("--n_gen", type=int, default=20)
    a = ap.parse_args()
    log = lambda s: print(s, flush=True)

    cfg = load_config(a.config)
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    ctx = cfg["train"]["context_len"]
    vp = val_prompts(tok, cfg["data"]["out_dir"], a.n_ppl)
    fp = GPT(cfg["model"])
    fp.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    fp.eval()
    variants = {"fp32": fp,
                "int8_linear": quantize_model(fp, quantize_embedding=False),
                "int8_all": quantize_model(fp, quantize_embedding=True)}

    sizes, mem, ppl, gen, secs = {}, {}, {}, {}, {}
    with tempfile.TemporaryDirectory() as d:
        for name, m in variants.items():
            p = os.path.join(d, name + ".pt")
            if name == "fp32":
                torch.save(m.state_dict(), p)
            else:
                save_quantized(m, p)
            sizes[name], mem[name] = os.path.getsize(p), tensor_bytes(m)
    log(f"file sizes: {sizes}")
    for name, m in variants.items():
        t0 = time.time()
        ppl[name] = gold_ppl(m, tok, vp, ctx)
        log(f"{name}: gold ppl {ppl[name]:.4f} ({time.time() - t0:.0f}s)")
    for name, m in variants.items():
        t0 = time.time()
        log(f"{name}: generation check")
        gen[name] = gen_check(m, tok, vp[:a.n_gen], log)
        secs[name] = time.time() - t0
        log(f"{name}: eot {gen[name][0].mean():.3f} cov {gen[name][1].mean():.3f} ({secs[name]:.0f}s)")

    n = a.n_gen
    idx = np.random.default_rng(20261027).integers(0, n, size=(10_000, n))
    res = {}
    for name in ("int8_linear", "int8_all"):
        rel = ppl[name] / ppl["fp32"] - 1
        d_eot, d_cov = gen[name][0] - gen["fp32"][0], gen[name][1] - gen["fp32"][1]
        ci_e, ci_c = boot_ci(d_eot, idx), boot_ci(d_cov, idx)
        dmg_e = d_eot.mean() <= EOT_DROP or ci_e[1] < 0
        dmg_c = d_cov.mean() <= COV_DROP or ci_c[1] < 0
        res[name] = {"ppl": ppl[name], "ppl_rel": rel, "ppl_pass": abs(rel) <= PPL_TOL,
                     "d_eot": float(d_eot.mean()), "ci_eot": ci_e, "damage_eot": bool(dmg_e),
                     "d_cov": float(d_cov.mean()), "ci_cov": ci_c, "damage_cov": bool(dmg_c),
                     "gen_pass": not (dmg_e or dmg_c)}
        res[name]["accepted"] = res[name]["ppl_pass"] and res[name]["gen_pass"]
    accepted = next((k for k in ("int8_all", "int8_linear") if res[k]["accepted"]), None)

    L = []
    P = L.append
    P("# int8 weight-only quantization check (SFT checkpoint)\n")
    P("Checkpoint: `checkpoints/glassbox-sft-final/sft_final_best.pt` (the **SFT** model, "
      "not DPO). CPU, fp32 baseline. Criteria were fixed before running (see "
      "`bench/quant_check.py`).\n")
    P("## File size and memory\n")
    P("| variant | file bytes | vs fp32 | in-memory tensor bytes |")
    P("|---|---|---|---|")
    for k in variants:
        P(f"| {k} | {sizes[k]:,} | {sizes[k] / sizes['fp32']:.3f}x | {mem[k]:,} |")
    P("")
    P("## Gold-story perplexity (first 100 validation prompts)\n")
    P(f"Kaggle SFT reference (eval_quality, kernel v3): {KAGGLE_SFT_GOLD_PPL}. "
      f"Local fp32: {ppl['fp32']:.4f}.\n")
    P("| variant | gold ppl | relative to fp32 | within 2%? |")
    P("|---|---|---|---|")
    P(f"| fp32 | {ppl['fp32']:.4f} | - | - |")
    for k in ("int8_linear", "int8_all"):
        P(f"| {k} | {ppl[k]:.4f} | {res[k]['ppl_rel']:+.3%} | {'PASS' if res[k]['ppl_pass'] else 'FAIL'} |")
    P("")
    P(f"## Generation check ({n} prompts x 6 samples, max_new 256, temp 0.8, top-k 40)\n")
    P("| variant | eot_rate | mean coverage (0-3) |")
    P("|---|---|---|")
    for k in variants:
        P(f"| {k} | {gen[k][0].mean():.3f} | {gen[k][1].mean():.3f} |")
    P("")
    P("Difference int8 - fp32 (95% CI over prompts, paired, B=10,000):\n")
    P("| variant | eot_rate diff [CI] | coverage diff [CI] | clear damage? |")
    P("|---|---|---|---|")
    for k in ("int8_linear", "int8_all"):
        r = res[k]
        P(f"| {k} | {r['d_eot']:+.3f} [{r['ci_eot'][0]:+.3f}, {r['ci_eot'][1]:+.3f}] | "
          f"{r['d_cov']:+.3f} [{r['ci_cov'][0]:+.3f}, {r['ci_cov'][1]:+.3f}] | "
          f"{'YES' if not r['gen_pass'] else 'no'} |")
    P("")
    P(f"With {n} prompts the check can miss small drops; the perplexity gate is the more sensitive one.\n")
    P("## Verdict\n")
    for k in ("int8_linear", "int8_all"):
        P(f"- `{k}`: perplexity {'PASS' if res[k]['ppl_pass'] else 'FAIL'}, generation "
          f"{'PASS' if res[k]['gen_pass'] else 'FAIL'} -> "
          f"**{'ACCEPTED' if res[k]['accepted'] else 'REJECTED'}**")
    P(f"\nAccepted int8 variant: **{accepted or 'none'}**.")
    text = "\n".join(L) + "\n"
    open(a.out, "w", encoding="utf-8", newline="\n").write(text)
    json.dump({"sizes": sizes, "ppl": ppl, "results": res, "accepted": accepted},
              open(a.out.replace(".md", ".json"), "w"), indent=1)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
