"""Preference-pair building: 4-8 completions per prompt, keep pass-vs-fail.

Per prompt (deterministic completion seeds from SeedSequence([salt, idx])):
sample n completions from the SFT model, score each with the rule criterion.
Keep min(#pass, #fail) pairs (zipped, winners/losers shuffled deterministically).
Drop all-pass and all-fail prompts (ties carry no DPO signal) and REPORT the
survival counts -- a collapsed survival rate means the criterion or the
model, not DPO, needs fixing.
"""
import numpy as np

from train.instruct import check_criterion
from train.sample import sample_completions

SALT = 777


def make_pairs(prompt, scored, max_pairs=2):
    """Pure pairing helper (tested with crafted scores, no model needed).

    scored: list of (text, passed_bool). Returns (pairs, kept_prompt_bool).
    """
    wins = [t for t, p in scored if p]
    losses = [t for t, p in scored if not p]
    if not wins or not losses:
        return [], False
    k = min(len(wins), len(losses), max_pairs)
    return ([{"prompt": prompt, "winner": w, "loser": l}
             for w, l in zip(wins[:k], losses[:k])], True)


def build_pairs(model, tok, prompts, n=6, max_new=64, temperature=0.8,
                top_k=50, seed=0, device=None, max_pairs=2):
    """prompts: list of (prompt_text, keywords). Returns (pairs, stats)."""
    pairs, kept, drop_pass, drop_fail = [], 0, 0, 0
    for i, (prompt, kw) in enumerate(prompts):
        texts = sample_completions(model, tok, prompt, n=n, max_new=max_new,
                                   temperature=temperature, top_k=top_k,
                                   seed=int(np.random.SeedSequence(
                                       [seed, SALT, i]).generate_state(1)[0]),
                                   device=device)
        scored = [(t, check_criterion(t, kw)["ok"]) for t in texts]
        made, keep = make_pairs(prompt, scored, max_pairs)
        pairs.extend(made)
        kept += keep
        drop_pass += (not keep and all(p for _, p in scored))
        drop_fail += (not keep and not any(p for _, p in scored))
    stats = {"prompts": len(prompts), "kept_prompts": kept, "pairs": len(pairs),
             "dropped_all_pass": drop_pass, "dropped_all_fail": drop_fail}
    print(f"[pairs] {stats}", flush=True)
    return pairs, stats
