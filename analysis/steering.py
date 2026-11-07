"""Phase 6 / C: SAE-feature steering experiment, exactly as pre-registered (docs/phase6_preregistration.md, section C).

  gen:      generate every (feature, condition, prompt) cell, resumable, one JSON line per cell
  analyze:  per-feature success verdict (rules a-d) and the claim (>= 3 of 5 succeed)

What a "steer" is: at the output of block 4 the vector  s * a_f * d_f / act_scale  is ADDED to the residual at every
position, in the prefill and in every decode step (GPT.forward(inject=...), via Generator.generate(inject=...)).
d_f is the unit decoder direction of the feature, a_f its 90th-percentile activation (SAE-normalised units), act_scale
the SAE's input scale, so the vector is in raw residual units. Controls: a random direction of the same norm (3 seeds),
s = -2, and ablation (subtract the feature's own SAE contribution a_f(x) * d_f / act_scale at every position).
Prompts are the first n validation prompts of train.eval_prompts.val_prompts; the sampling seed is the prompt index and is
the same in every condition. Fluency uses the UNSTEERED target model to score the steered texts.

    python analysis/steering.py gen --out_dir D [--smoke]      # D/cells.jsonl (+ meta.json)
    python analysis/steering.py analyze --out_dir D            # D/verdict.json
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
SCALES = (1, 2, 4, 8)
NEG_SCALE = -2
RANDOM_SEEDS = (101, 102, 103)
LAYER = 4
BOOT = 10_000
# fixed in docs/steering_features.md before any steering run: feature -> (a_f on validation scan, keyword set)
FEATURES = {
    1319: (2.97, ["apples", "bananas", "carrots", "oranges", "cheese", "cows", "pigs", "chickens", "horses"]),
    3651: (1.72, ["vase", "bookcase", "bookshelf", "couch", "ceiling", "shelf"]),
    1672: (6.36, ["shirt", "coat", "jacket", "sweater", "suit", "pants"]),
    1131: (2.26, ["stars", "moon", "twinkling", "dots"]),
    3990: (3.31, ["crocodile", "alligator", "kangaroo", "tiger", "turtle", "hippo", "ostrich", "bear", "wolf", "frog",
                  "spider"]),
}


# ----------------------------------------------------------------------------------- injection vectors
def feature_vector(sae, act_scale, f, a_f, s):
    """s * a_f * d_f / act_scale, shape (d,), raw residual units."""
    return s * a_f * sae.W_dec[f].detach() / act_scale


def random_vector(norm, seed, d):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return v / v.norm() * norm


def ablation_hook(sae, act_scale, f):
    """Callable inject: delta = -a_f(x) * d_f / act_scale at every position (removes the feature's SAE contribution)."""
    d_f = sae.W_dec[f].detach()

    def hook(x):
        acts, _ = sae.encode(x.reshape(-1, x.shape[-1]) * act_scale)
        a = acts[:, f].reshape(x.shape[:-1]).unsqueeze(-1)
        return -a * d_f / act_scale
    return hook


def conditions(sae, act_scale, f, a_f):
    """[(name, kind, scale, inject)] for one feature (baseline is shared and handled by the caller)."""
    d = sae.W_dec.shape[1]
    out = [(f"s{s}", "feature", s, {LAYER: feature_vector(sae, act_scale, f, a_f, s)}) for s in SCALES]
    for s in SCALES:
        for r in RANDOM_SEEDS:
            out.append((f"rand_s{s}_{r}", "random", s, {LAYER: random_vector(abs(s) * a_f / act_scale, r, d)}))
    out.append((f"s{NEG_SCALE}", "negative", NEG_SCALE, {LAYER: feature_vector(sae, act_scale, f, a_f, NEG_SCALE)}))
    out.append(("ablate", "ablation", 0, {LAYER: ablation_hook(sae, act_scale, f)}))
    return out


# ----------------------------------------------------------------------------------- generation stage
def cell_key(feature, cond, idx):
    return f"{feature}|{cond}|{idx}"


def load_done(path):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[cell_key(r["feature"], r["cond"], r["idx"])] = r
    return done


def run_gen(a):
    from analysis.a5_transfer import load_model, sha256
    from common.config import load_config
    from inference.generate import Generator
    from interp.sae import load_sae
    from tokenizer.bpe import BPETokenizer
    from train.eval_prompts import val_prompts
    from train.eval_quality import distinct2, mean_nll  # noqa: F401
    cfg = load_config(a.config)
    dev = a.device
    sae, sm = load_sae(a.sae)
    sae = sae.to(dev).eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    scale = sm["act_scale"]
    model = load_model(cfg, a.weights).to(dev)
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    shas = {"sae": sha256(a.sae), "weights": sha256(a.weights)}
    for k, e in (("sae", a.expect_sae_sha), ("weights", a.expect_weights_sha)):
        print(f"[steer] sha256 {k} {shas[k]}" + (f" expected -> {'MATCH' if shas[k] == e else 'MISMATCH'}" if e else ""), flush=True)
        assert not e or shas[k] == e, f"{k} hash mismatch; stopping"
    prompts = val_prompts(tok, a.data_dir, a.n_prompts)
    feats = {f: FEATURES[f] for f in (a.features or FEATURES)}
    os.makedirs(a.out_dir, exist_ok=True)
    path = f"{a.out_dir}/cells.jsonl"
    done = load_done(path)
    meta = {"shas": shas, "layer": LAYER, "act_scale": scale, "n_prompts": len(prompts), "n": a.n, "max_new": a.max_new,
            "temperature": a.temperature, "top_k": a.top_k, "features": {str(f): v for f, v in feats.items()},
            "scales": SCALES, "random_seeds": RANDOM_SEEDS, "commit": a.commit, "smoke": a.smoke}
    json.dump(meta, open(f"{a.out_dir}/meta.json", "w"), indent=1)
    gen = Generator(model, tok, device=dev)
    todo = [(None, "baseline", "baseline", 0, None)]
    for f, (a_f, _) in feats.items():
        todo += [(f, name, kind, s, inj) for name, kind, s, inj in conditions(sae, scale, f, a_f)]
    total, t0, ncell = len(todo) * len(prompts), time.time(), 0
    out = open(path, "a", encoding="utf-8")
    for f, name, kind, s, inj in todo:
        for idx, prompt, kws, _gold in prompts:
            key = cell_key(f, name, idx)
            if key in done:
                continue
            res = gen.generate(prompt, n=a.n, max_new=a.max_new, temperature=a.temperature, top_k=a.top_k, seed=idx,
                               inject=inj)
            nll = [mean_nll(model, tok, prompt, t, cfg["model"]["context_len"], dev)[0] for t in res.texts]   # UNSTEERED
            row = {"feature": f, "cond": name, "kind": kind, "scale": s, "idx": idx, "keywords": kws, "texts": res.texts,
                   "eot": res.eot_hit, "steps": res.n_steps, "nll": nll}
            out.write(json.dumps(row) + "\n")
            out.flush()
            ncell += 1
            if ncell % 25 == 0:
                el = time.time() - t0
                print(f"[steer] {ncell} new cells | {el:.0f}s | {el / ncell:.2f} s/cell | remaining "
                      f"{(total - len(done) - ncell) * el / ncell / 60:.0f} min (extrapolated)", flush=True)
    print(f"[steer] DONE: {ncell} new cells in {time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------------- analysis stage
def has_concept(text, concept, banned):
    words = set(re.findall(r"[a-z']+", text.lower()))
    return any(w in words for w in concept if w not in banned)


def cond_table(rows, concept):
    """{idx: dict(concept_rate, eot_rate, nll, texts)} for the rows of one condition."""
    t = {}
    for r in rows:
        banned = set(w.lower() for w in r["keywords"])
        hits = [has_concept(x, concept, banned) for x in r["texts"]]
        t[r["idx"]] = {"concept": float(np.mean(hits)), "eot": float(np.mean(r["eot"])), "nll": float(np.mean(r["nll"])),
                       "texts": r["texts"]}
    return t


def distinct2(texts):
    grams = []
    for x in texts:
        w = x.split()
        grams.extend(zip(w, w[1:]))
    return len(set(grams)) / len(grams) if grams else 0.0


def boot_ci(diff, B=BOOT, seed=0):
    """95% percentile CI of the mean of per-prompt differences (resampling prompts)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, dtype=float)
    m = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def judge_feature(cells, f, concept):
    """cells: {(feature or None, cond): rows}. Returns the full per-scale table and the verdict of rules a-d."""
    base = cond_table(cells[(None, "baseline")], concept)
    idxs = sorted(base)
    b_conc = np.array([base[i]["concept"] for i in idxs])
    b_eot = float(np.mean([base[i]["eot"] for i in idxs]))
    b_d2 = distinct2([x for i in idxs for x in base[i]["texts"]])
    b_ppl = math.exp(float(np.mean([base[i]["nll"] for i in idxs])))
    table = {}
    for s in SCALES:
        feat = cond_table(cells[(f, f"s{s}")], concept)
        rnd = [cond_table(cells[(f, f"rand_s{s}_{r}")], concept) for r in RANDOM_SEEDS]
        fc = np.array([feat[i]["concept"] for i in idxs])
        rc = np.mean([[t[i]["concept"] for i in idxs] for t in rnd], axis=0)
        diff = fc - rc
        lo, hi = boot_ci(diff)
        table[s] = {"concept_feature": float(fc.mean()), "concept_random": float(rc.mean()), "diff": float(diff.mean()),
                    "ci": [lo, hi], "eot": float(np.mean([feat[i]["eot"] for i in idxs])),
                    "distinct2": distinct2([x for i in idxs for x in feat[i]["texts"]]),
                    "ppl_ratio": math.exp(float(np.mean([feat[i]["nll"] for i in idxs]))) / b_ppl}
    ctrl = {}
    for name in (f"s{NEG_SCALE}", "ablate"):
        t = cond_table(cells[(f, name)], concept)
        c = np.array([t[i]["concept"] for i in idxs])
        lo, hi = boot_ci(c - b_conc)
        ctrl[name] = {"concept": float(c.mean()), "diff_vs_baseline": float((c - b_conc).mean()), "ci": [lo, hi],
                      "not_above_baseline": bool(lo <= 0)}              # CI lower bound <= 0: no significant rise
    c_ok = all(v["not_above_baseline"] for v in ctrl.values())
    works = {}
    for k, s in enumerate(SCALES):
        r = table[s]
        a_ok = r["diff"] >= 0.10 and r["ci"][0] > 0
        diffs = [table[x]["diff"] for x in SCALES[:k + 1]]
        b_ok = all(diffs[i + 1] >= diffs[i] - 0.02 for i in range(len(diffs) - 1))
        d_ok = r["eot"] >= b_eot - 0.10 and r["distinct2"] >= b_d2 - 0.05 and r["ppl_ratio"] <= 1.25
        works[s] = {"a_gain_ci": bool(a_ok), "b_dose_response": bool(b_ok), "c_controls": bool(c_ok), "d_fluency": bool(d_ok),
                    "all": bool(a_ok and b_ok and c_ok and d_ok)}
    ok = [s for s, v in works.items() if v["all"]]
    return {"feature": f, "baseline": {"concept": float(b_conc.mean()), "eot": b_eot, "distinct2": b_d2, "ppl": b_ppl},
            "scales": table, "controls": ctrl, "rules_by_scale": works, "working_scales": ok, "success": bool(ok)}


def run_analyze(a):
    meta = json.load(open(f"{a.out_dir}/meta.json"))
    cells = {}
    for line in open(f"{a.out_dir}/cells.jsonl", encoding="utf-8"):
        r = json.loads(line)
        cells.setdefault((r["feature"], r["cond"]), []).append(r)
    verdicts = [judge_feature(cells, int(f), FEATURES[int(f)][1]) for f in meta["features"]]
    n_ok = sum(v["success"] for v in verdicts)
    out = {"features": verdicts, "n_success": n_ok, "n_features": len(verdicts),
           "claim_holds": bool(n_ok >= 3 and len(verdicts) >= 5), "meta": meta}
    json.dump(out, open(f"{a.out_dir}/verdict.json", "w"), indent=1)
    for v in verdicts:
        print(f"feature {v['feature']}: success={v['success']} working scales={v['working_scales']} "
              + " | ".join(f"s{s}: diff {t['diff']:+.2f} CI[{t['ci'][0]:+.2f},{t['ci'][1]:+.2f}]" for s, t in v["scales"].items()))
    print(f"CLAIM {'HOLDS' if out['claim_holds'] else 'NOT ESTABLISHED'}: {n_ok}/{len(verdicts)} features succeed (need >= 3 of >= 5)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "analyze"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="configs/base_30m.yaml")
    ap.add_argument("--sae", default="checkpoints/glassbox-sae-l4/sae_latest.pt")
    ap.add_argument("--weights", default="checkpoints/glassbox-sft-final/sft_final_best.pt")
    ap.add_argument("--data_dir", default="data/full")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_prompts", type=int, default=50)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--features", type=int, nargs="*")
    ap.add_argument("--expect_sae_sha"), ap.add_argument("--expect_weights_sha")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--smoke", action="store_true", help="label only: this is not a pre-registered run")
    a = ap.parse_args(argv)
    return run_gen(a) if a.cmd == "gen" else run_analyze(a)


if __name__ == "__main__":
    main()
