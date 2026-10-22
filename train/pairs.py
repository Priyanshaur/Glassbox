"""Preference-pair building: 4-8 completions per prompt, keep pass-vs-fail.

Per prompt (deterministic completion seeds from SeedSequence([salt, idx])):
sample n completions from the SFT model, score each with the rule criterion.
Keep min(#pass, #fail) pairs (zipped in completion order -- the sampler's
own order, no re-shuffle; the pair list order is deterministic here).
Drop all-pass and all-fail prompts (ties carry no DPO signal) and REPORT the
survival counts -- a collapsed survival rate means the criterion or the
model, not DPO, needs fixing.
"""
import json
import os

import numpy as np
import re

from train.sample import sample_completions

SALT = 777


def whole_word_ok(response, keywords):
    """Whole-word constraint check (gate 2026-10-22, see decisions.md).

    Every keyword must appear as a standalone word (case-insensitive), plus
    the same 10-500-word window as check_criterion. Substring matching counts
    keywords inside other words ('when' in 'whenever', 'tree' in 'street'),
    which the probe showed inflates pass verdicts; the probe switched the
    pair gate to whole-word matching. (check_criterion stays untouched: it
    remains the documented substring reference for criterion-flaw notes.)

    NOTE: the graded rule (decision 2026-10-22) pairs by keyword-count
    margins, not by this boolean; this helper stays the whole-word building
    block for count scoring in build_pairs.
    """
    low = response.lower()
    for k in keywords:
        if not re.search(r"(?<!\w)" + re.escape(k.lower()) + r"(?!\w)", low):
            return False
    return 10 <= len(response.split()) <= 500


def whole_word_count(response, keywords):
    """Number of DISTINCT keywords covered, whole-word matching."""
    low = response.lower()
    return sum(1 for k in keywords
               if re.search(r"(?<!\w)" + re.escape(k.lower()) + r"(?!\w)",
                            low))


def make_graded_pairs(prompt, rows, max_pairs=2):
    """Deliberate criterion change (decisions.md 2026-10-22).

    A pair (chosen, rejected) is valid iff chosen covers >= 1 MORE keyword
    than rejected -- coverage = whole-word count of distinct keywords
    (0..3) -- and BOTH rows end at their own EOT and fall in the
    10-500-word window. rows: {'text','cov','eot','nwords'}.

    Pair selection: keep min(#winners, #losers, max_pairs) pairs like
    make_pairs, by EXACT max-cardinality/max-margin matching (n <= 8 rows:
    brute force). Output edges are ordered margin-desc; deterministic
    tie-break via stable row order.
    """
    import itertools
    el = [r for r in rows if r["eot"] and 10 <= r["nwords"] <= 500]
    if len(el) < 2:
        return [], False
    wins = [r for r in el if any(r["cov"] - o["cov"] >= 1 for o in el)]
    loses = [r for r in el if any(o["cov"] - r["cov"] >= 1 for o in el)]
    k = min(len(wins), len(loses), max_pairs)
    edges = None
    while k > 0 and edges is None:
        best_score = -1
        for wcombo in itertools.combinations(range(len(wins)), k):
            for lperm in itertools.permutations(range(len(loses)), k):
                cand = [(wins[i], loses[j]) for i, j in zip(wcombo, lperm)]
                if not all(w["cov"] - l["cov"] >= 1 for w, l in cand):
                    continue
                score = sum(w["cov"] - l["cov"] for w, l in cand)
                if score > best_score:
                    edges, best_score = cand, score
        if edges is None:
            k -= 1
    if not edges:
        return [], False
    edges.sort(key=lambda e: -(e[0]["cov"] - e[1]["cov"]))
    out = [{"prompt": prompt, "winner": w["text"], "loser": l["text"],
            "margin": w["cov"] - l["cov"]} for w, l in edges]
    return out, True


def build_pairs_graded(model, tok, prompts, out_path, n=6, max_new=256,
                       temperature=0.8, top_k=40, seed=0, device=None,
                       max_pairs=2, survival_check=100, min_survival=0.75):
    """Graded pair building with per-prompt resume, one JSON line per prompt.

    prompts: [(global_idx, prompt_text, keywords)] -- seeds key on the
    global story index, SeedSequence([seed, SALT, global_idx]), so the run
    is resumable and independent of how the slice was batched.

    Per prompt the record {"idx", "prompt", "keywords", "pairs", "counts"}
    is appended and flushed, so an interrupted run loses at most one
    prompt. Records already in out_path are skipped WITHOUT model calls.

    Survival gate: after each survival_check prompts, kept/past prompts
    below min_survival raise (checked against the probe projection of 90.5%
    kept at n=6 per prompt).
    """
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec["idx"])
    total = len(prompts)
    todo = [(g, p, kw) for g, p, kw in prompts if g not in done]
    print(f"[pairs-graded] {total} prompts, {len(done)} done (resume), "
          f"{len(todo)} to go", flush=True)
    kept = past = 0
    eot_hits = rows_seen = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for k, (g, prompt, kw) in enumerate(todo, 1):
            texts, flags = sample_completions(
                model, tok, prompt, n=n, max_new=max_new,
                temperature=temperature, top_k=top_k,
                seed=int(np.random.SeedSequence(
                    [seed, SALT, g]).generate_state(1)[0]),
                device=device, return_flags=True)
            rows = [{"text": t, "cov": whole_word_count(t, kw),
                     "eot": fl["eot_hit"], "nwords": len(t.split())}
                    for t, fl in zip(texts, flags)]
            rows_seen += len(rows)
            eot_hits += sum(r["eot"] for r in rows)
            pairs, keep = make_graded_pairs(prompt, rows, max_pairs)
            past += 1
            kept += keep
            rec = {"idx": g, "prompt": prompt, "keywords": list(kw),
                   "pairs": pairs,
                   "counts": {"rows": len(rows),
                              "eot_hit": sum(r["eot"] for r in rows)}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if survival_check and past % survival_check == 0:
                rate = kept / past
                print(f"[pairs-graded] {past}/{len(todo)} prompts this run, "
                      f"kept={kept} ({rate:.1%}), eot={eot_hits}/{rows_seen} "
                      f"({eot_hits/max(rows_seen,1):.1%})", flush=True)
                if rate < min_survival:
                    raise RuntimeError(
                        f"[pairs-graded] survival rate {rate:.1%} fell below "
                        f"floor {min_survival:.0%} after {past} prompts; "
                        f"stopping before the long tail (criterion/model "
                        f"check, not DPO)")
    return {"prompts": total, "done_before": len(done), "run": past,
            "kept": kept, "eot_hits": eot_hits, "rows": rows_seen}


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
        scored = [(t, whole_word_ok(t, kw)) for t in texts]
        made, keep = make_pairs(prompt, scored, max_pairs)
        pairs.extend(made)
        kept += keep
        drop_pass += (not keep and all(p for _, p in scored))
        drop_fail += (not keep and not any(p for _, p in scored))
    stats = {"prompts": len(prompts), "kept_prompts": kept, "pairs": len(pairs),
             "dropped_all_pass": drop_pass, "dropped_all_fail": drop_fail}
    print(f"[pairs] {stats}", flush=True)
    return pairs, stats
