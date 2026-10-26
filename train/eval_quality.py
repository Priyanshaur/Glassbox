"""DPO/SFT generation-quality eval on val-split template prompts.

For one weights file (an SFT baseline or a DPO arm) it reports, over N
val-split template prompts x n samples each (same sampler settings as the
probe: max_new=256, temp=0.8, top_k=40):

  cov3          mean distinct-keywords covered (whole-word, prose-only, pre-EOT)
  full3_rate    fraction of samples covering all 3 keywords
  eot_rate      fraction of samples that hit their own EOT
  mean_words    mean response length in words (EOT-cut)
  distinct2     unique bigrams / total bigrams over concatenated prose
  ppl_sft       perplexity of the samples under the SFT model (mean token
                NLL, exp) -- always scored by --ppl_weights (the frozen SFT
                ref), so arms are compared under a fixed scorer.

Prompts come from the VALIDATION split (never trained on), synth_prompt on
val story idx 0..N-1. Seeds key on prompt idx so every arm draws the same
randomness (comparable generations). One JSON row per prompt + a summary.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from common.config import load_config
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer, EOT_ID
from train.instruct import synth_prompt, iter_shard_texts
from train.pairs import whole_word_count
from train.sample import sample_completions


@torch.no_grad()
def mean_nll(model, tok, prompt, response, context_len, device):
    """Mean per-token NLL of response+EOT given prompt (prompt masked)."""
    p = tok.encode(prompt)
    r = tok.encode(response)
    full = (p + r + [EOT_ID])[:context_len]
    P = min(len(p), len(full) - 1)
    ids = torch.tensor([full], dtype=torch.long, device=device)
    logp = torch.log_softmax(model(ids).logits[0], dim=-1)
    idx = torch.tensor(full[1:], dtype=torch.long, device=logp.device)
    tok_lp = logp[:-1].gather(1, idx.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(tok_lp)
    mask[:P - 1] = 0
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0
    return float(-(tok_lp * mask).sum()) / n, n


def distinct2(texts):
    grams = []
    for t in texts:
        w = t.split()
        grams.extend(zip(w, w[1:]))
    return (len(set(grams)) / len(grams)) if grams else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True, help="model under test (SFT or DPO arm)")
    ap.add_argument("--ppl_weights", required=True, help="frozen SFT scorer for perplexity")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    ctx = cfg["train"]["context_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])

    model = GPT(mcfg).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device,
                                     weights_only=True))
    model.eval()
    scorer = GPT(mcfg).to(device)
    scorer.load_state_dict(torch.load(args.ppl_weights, map_location=device,
                                      weights_only=True))
    scorer.eval()

    # val-split template prompts (validation split never trained on)
    prompts, idx = [], 0
    for text in iter_shard_texts(tok, cfg["data"]["out_dir"], "validation"):
        sp = synth_prompt(text, idx)
        if sp is not None:
            prompts.append((idx, sp[0], sp[1]))
        idx += 1
        if len(prompts) >= args.n_prompts:
            break
    print(f"[eval] {len(prompts)} val prompts | model={args.label or args.weights}",
          flush=True)

    t0 = time.time()
    all_prose, sum_cov, n_full3, n_eot, n_words, nll_sum, tok_sum = [], 0, 0, 0, 0, 0.0, 0
    with open(args.out, "w", encoding="utf-8") as f:
        for k, (i, prompt, kw) in enumerate(prompts, 1):
            texts, flags = sample_completions(
                model, tok, prompt, n=args.n_samples, max_new=args.max_new,
                temperature=args.temperature, top_k=args.top_k,
                seed=i, device=device, return_flags=True)
            rows = []
            for t, fl in zip(texts, flags):
                cov = whole_word_count(t, kw)
                nw = len(t.split())
                nl, nt = mean_nll(scorer, tok, prompt, t, ctx, device)
                rows.append({"cov": cov, "eot": fl["eot_hit"], "words": nw,
                             "nll": nl, "ntok": nt, "text": t})
                sum_cov += cov
                n_full3 += (cov == 3)
                n_eot += fl["eot_hit"]
                n_words += nw
                nll_sum += nl * nt
                tok_sum += nt
                all_prose.append(t)
            f.write(json.dumps({"idx": i, "prompt": prompt, "keywords": kw,
                                "rows": rows}, ensure_ascii=False) + "\n")
            if k % 20 == 0:
                print(f"[eval] {k}/{len(prompts)} prompts, {time.time()-t0:.0f}s",
                      flush=True)

    n = len(prompts) * args.n_samples
    summary = {
        "label": args.label or os.path.basename(args.weights),
        "weights": args.weights,
        "n_prompts": len(prompts), "n_samples_per_prompt": args.n_samples,
        "cov3": sum_cov / n, "full3_rate": n_full3 / n, "eot_rate": n_eot / n,
        "mean_words": n_words / n, "distinct2": distinct2(all_prose),
        "ppl_sft": math.exp(nll_sum / tok_sum) if tok_sum else float("inf"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(args.out + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("[eval] SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
