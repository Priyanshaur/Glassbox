"""DPO vs SFT win rate + unoptimized quality metrics + blind spot-check CSV.

Eval prompts: official validation split (pretraining saw all of train).
Per prompt: one completion per model, identical decoding. Rule verdict per
completion -> wins (dpo pass, sft fail) / ties / losses, Wilson 95% CI on
DECIDED pairs only. The rule win rate is CIRCULAR (DPO optimizes it), so we
also report what DPO did not optimize: response-token val perplexity,
distinct-2, mean length. Plus a blind shuffled CSV for the human spot-check
(docs/human_spotcheck.md).
"""
import argparse
import csv
import json
import math
import os
import sys

import torch

from common.config import load_config
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer
from train.instruct import build_sft_example, check_criterion, docs_from_shard, synth_prompt
from train.sample import prompt_ids, sample_completions
from train.sft import response_loss


def wilson(wins, n, z=1.96):
    """Wilson 95% interval for wins/n. Returns (lo, hi)."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(c - m, 0) / d, min(c + m, 1) / d


def distinct_2(texts):
    bis, n = set(), 0
    for t in texts:
        w = t.split()
        for a, b in zip(w, w[1:]):
            bis.add((a, b))
            n += 1
    return len(bis) / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--sft_weights", required=True)
    ap.add_argument("--dpo_weights", required=True)
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--out", default="bench/win_rate.json")
    ap.add_argument("--emit_spotcheck", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    val_texts = docs_from_shard(tok, cfg["data"]["out_dir"], "validation")
    prompts = []
    for i, t in enumerate(val_texts):
        sp = synth_prompt(t, 10_000 + i)
        if sp is not None:
            prompts.append((sp[0], sp[1], t))  # keep source story for val ppl
        if len(prompts) >= args.n_prompts:
            break

    sft = GPT(mcfg).to(device)
    sft.load_state_dict(torch.load(args.sft_weights, map_location=device,
                                   weights_only=True))
    dpo = GPT(mcfg).to(device)
    dpo.load_state_dict(torch.load(args.dpo_weights, map_location=device,
                                   weights_only=True))

    wins = ties = losses = skipped = 0
    sft_texts, dpo_texts, rows = [], [], []
    for i, (prompt, kw, _src) in enumerate(prompts):
        plen = len(prompt_ids(tok, prompt))
        if plen + 8 > cfg["model"]["context_len"]:
            skipped += 1  # prompt alone nearly fills ctx; nothing to generate
            continue
        mn = min(args.max_new, cfg["model"]["context_len"] - plen - 1)
        a = sample_completions(sft, tok, prompt, n=1, max_new=mn, seed=args.seed + i,
                               device=device)[0]
        b = sample_completions(dpo, tok, prompt, n=1, max_new=mn, seed=args.seed + i,
                               device=device)[0]
        sft_texts.append(a)
        dpo_texts.append(b)
        pa, pb = check_criterion(a, kw)["ok"], check_criterion(b, kw)["ok"]
        outcome = "tie" if pa == pb else ("win" if pb else "loss")
        wins += outcome == "win"
        ties += outcome == "tie"
        losses += outcome == "loss"
        rows.append({"prompt": prompt, "sft": a, "dpo": b, "outcome": outcome})

    decided = wins + losses
    lo, hi = wilson(wins, decided)
    val_ex = [{"input": x, "labels": y} for x, y in
              (build_sft_example(tok, p, t, cfg["train"]["context_len"])
               for p, t in ((p, s) for p, _, s in prompts))]
    rep = {"n": len(prompts), "skipped_ctx": skipped,
           "wins": wins, "ties": ties, "losses": losses,
           "win_rate_decided": wins / max(decided, 1),
           "ci95": [lo, hi],
           "note": "rule rate is circular (DPO optimizes it); see unoptimized metrics",
           "sft_val_ppl": float(torch.exp(torch.tensor(response_loss(
               sft, val_ex, device, mcfg["vocab_size"])))),
           "dpo_val_ppl": float(torch.exp(torch.tensor(response_loss(
               dpo, val_ex, device, mcfg["vocab_size"])))),
           "sft_distinct2": distinct_2(sft_texts),
           "dpo_distinct2": distinct_2(dpo_texts),
           "sft_mean_len": sum(len(t.split()) for t in sft_texts) / max(len(sft_texts), 1),
           "dpo_mean_len": sum(len(t.split()) for t in dpo_texts) / max(len(dpo_texts), 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    print(json.dumps(rep, indent=2))

    if args.emit_spotcheck:
        import random
        rng = random.Random(args.seed)
        picks = rng.sample(rows, min(args.emit_spotcheck, len(rows)))
        with open(args.out + ".spotcheck.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "prompt", "A", "B", "human_choice"])
            mapping = {}
            for j, r in enumerate(picks):
                flip = rng.random() < 0.5
                w.writerow([j, r["prompt"],
                            r["dpo"] if flip else r["sft"],
                            r["sft"] if flip else r["dpo"], ""])
                mapping[j] = "dpo_is_A" if flip else "dpo_is_B"
        json.dump(mapping, open(args.out + ".spotcheck_map.json", "w"))
        print(f"spot-check CSV + hidden mapping written ({len(picks)} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
