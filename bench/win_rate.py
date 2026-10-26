"""DPO vs SFT win rate + unoptimized quality metrics + blind spot-check CSV.

Eval prompts: official validation split (pretraining saw all of train).
Per prompt: one completion per model, identical decoding (max_new=256 by
default so stories can reach their EOT). Each completion is scored with the
SAME graded rule the DPO pairs were built with (train.pairs): it is VALID iff
it ended at its own EOT and has 10-500 words; a valid completion scores its
whole-word keyword coverage (0..3), an invalid one scores -1. DPO wins when
its score is higher, loses when lower, ties when equal; Wilson 95% CI on
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
from train.eval_prompts import val_prompts
from train.instruct import build_sft_example
from train.pairs import whole_word_count
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


def graded_score(text, keywords, eot_hit):
    """Same eligibility as train.pairs.make_graded_pairs: ended at its own EOT
    and 10-500 words. Valid -> whole-word keyword coverage; invalid -> -1."""
    if not eot_hit or not 10 <= len(text.split()) <= 500:
        return -1
    return whole_word_count(text, keywords)


TIE_RULE = ("tie = equal graded score: both completions invalid (-1) or both "
            "valid with equal keyword coverage. Ties are excluded from "
            "win_rate_decided and its CI, and counted as non-wins in "
            "win_rate_all_pairs. Validity (EOT-ended, 10-500 words) is "
            "reported per arm in *_valid_rate and is NOT part of the win rate "
            "alone: wins/losses split into 'validity' (one side invalid) and "
            "'coverage' (both valid, different coverage).")


def pair_outcome(sft_score, dpo_score):
    """(outcome, kind) for one pair of graded scores (-1 = invalid).

    kind says what decided it: 'validity' (exactly one side valid),
    'coverage' (both valid, different coverage), or for ties
    'both_invalid' / 'equal_valid'."""
    if sft_score == dpo_score:
        return "tie", ("both_invalid" if sft_score < 0 else "equal_valid")
    kind = "validity" if min(sft_score, dpo_score) < 0 else "coverage"
    return ("win" if dpo_score > sft_score else "loss"), kind


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
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--out", default="bench/win_rate.json")
    ap.add_argument("--emit_spotcheck", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    # same prompts, same idx keying as train/eval_quality.py (shared function);
    # keep the source story for val ppl
    prompts = [(p, kw, text) for _, p, kw, text in
               val_prompts(tok, cfg["data"]["out_dir"], args.n_prompts)]

    sft = GPT(mcfg).to(device)
    sft.load_state_dict(torch.load(args.sft_weights, map_location=device,
                                   weights_only=True))
    dpo = GPT(mcfg).to(device)
    dpo.load_state_dict(torch.load(args.dpo_weights, map_location=device,
                                   weights_only=True))

    wins = ties = losses = skipped = 0
    sft_texts, dpo_texts, rows = [], [], []
    sft_sc, dpo_sc, sft_eot, dpo_eot = [], [], 0, 0
    kinds = {}
    for i, (prompt, kw, _src) in enumerate(prompts):
        plen = len(prompt_ids(tok, prompt))
        if plen + 8 > cfg["model"]["context_len"]:
            skipped += 1  # prompt alone nearly fills ctx; nothing to generate
            continue
        mn = min(args.max_new, cfg["model"]["context_len"] - plen - 1)
        ta, fa = sample_completions(sft, tok, prompt, n=1, max_new=mn,
                                    seed=args.seed + i, device=device,
                                    return_flags=True)
        tb, fb = sample_completions(dpo, tok, prompt, n=1, max_new=mn,
                                    seed=args.seed + i, device=device,
                                    return_flags=True)
        a, b = ta[0], tb[0]
        sft_texts.append(a)
        dpo_texts.append(b)
        sa = graded_score(a, kw, fa[0]["eot_hit"])
        sb = graded_score(b, kw, fb[0]["eot_hit"])
        sft_sc.append(sa)
        dpo_sc.append(sb)
        sft_eot += fa[0]["eot_hit"]
        dpo_eot += fb[0]["eot_hit"]
        outcome, kind = pair_outcome(sa, sb)
        wins += outcome == "win"
        ties += outcome == "tie"
        losses += outcome == "loss"
        kinds[(outcome, kind)] = kinds.get((outcome, kind), 0) + 1
        rows.append({"prompt": prompt, "sft": a, "dpo": b, "outcome": outcome,
                     "kind": kind, "sft_score": sa, "dpo_score": sb})

    decided = wins + losses
    lo, hi = wilson(wins, decided)
    val_ex = [{"input": x, "labels": y} for x, y in
              (build_sft_example(tok, p, t, cfg["train"]["context_len"])
               for p, t in ((p, s) for p, _, s in prompts))]
    rep = {"n": len(prompts), "skipped_ctx": skipped,
           "n_evaluated": len(rows),
           "wins": wins, "ties": ties, "losses": losses,
           "win_rate_decided": wins / max(decided, 1),
           "ci95_decided": [lo, hi],
           "win_rate_all_pairs": wins / max(len(rows), 1),
           "tie_rule": TIE_RULE,
           "wins_validity": kinds.get(("win", "validity"), 0),
           "wins_coverage": kinds.get(("win", "coverage"), 0),
           "losses_validity": kinds.get(("loss", "validity"), 0),
           "losses_coverage": kinds.get(("loss", "coverage"), 0),
           "ties_both_invalid": kinds.get(("tie", "both_invalid"), 0),
           "ties_equal_valid": kinds.get(("tie", "equal_valid"), 0),
           "note": "rule rate is circular (DPO optimizes it); see unoptimized metrics",
           "max_new": args.max_new,
           "sft_eot_rate": sft_eot / max(len(sft_texts), 1),
           "dpo_eot_rate": dpo_eot / max(len(dpo_texts), 1),
           "sft_valid_rate": sum(x >= 0 for x in sft_sc) / max(len(sft_sc), 1),
           "dpo_valid_rate": sum(x >= 0 for x in dpo_sc) / max(len(dpo_sc), 1),
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
