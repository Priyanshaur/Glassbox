"""Inference speed benchmark, following docs/benchmark_protocol.md.

Compares KV-cache implementations (none / tuple / static), rows per prompt n, two
prompt regimes (short template prompt, long 256-token story prefix) and dtypes
(fp32, int8 weight-only with the static cache, fp16 on GPU). Fixed generation length,
identical prompts per config, warm-up, interleaved repeats, median + spread,
paired-bootstrap ratio CIs, cuda synchronize for GPU timing.

Usage:  python -m bench.speed [--device cpu] [--prompts 4] [--gen 64] [--repeats 10]
                              [--warmup 3] [--ns 1,8,32] [--grid cpu|full]
                              [--out_prefix docs/benchmark_cpu]
`--grid cpu` applies the reduced no-cache grid from the protocol's CPU addendum;
`--grid full` runs every cell.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

from common.config import load_config
from inference.generate import Generator
from inference.kv_cache import StaticKVCache
from inference.quantize import quantize_model, tensor_bytes, to_half
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer
from train.eval_prompts import val_prompts

TEMPERATURE, TOP_K = 0.8, 40


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------- one generation
@torch.no_grad()
def run_once(model, prompt_ids, n, G, cache, seed, dev):
    """One fixed-length generation of G tokens for n identical rows.
    Returns (tokens (n, G), ttft_s, decode_s, cache_bytes). TTFT covers cache
    allocation, the prompt forward and sampling the first token (synchronized)."""
    P = prompt_ids.shape[1]
    gen = torch.Generator(device=dev).manual_seed(seed)
    base = prompt_ids.repeat(n, 1)
    sync(dev)
    t0 = time.perf_counter()
    kv = past = None
    if cache == "static":
        kv = StaticKVCache.from_model(model, n, P + G)
        out = model(base, past_kvs=kv.layers, pos_offset=0)
    else:
        out = model(base)
        past = out.new_kvs if cache == "tuple" else None
    # .float() is a no-op for fp32; fp16 logits are sampled in fp32 (same cost in every fp16 config)
    nxt = Generator._sample(out.logits[:, -1, :].float(), gen, TEMPERATURE, TOP_K, None)
    sync(dev)
    t1 = time.perf_counter()
    cols, ids = [nxt], torch.cat([base, nxt], dim=1)
    for _ in range(1, G):
        if cache == "static":
            logits = model(ids[:, -1:], past_kvs=kv.layers, pos_offset=ids.shape[1] - 1).logits
        elif cache == "tuple":
            s = model(ids[:, -1:], past_kvs=past, pos_offset=ids.shape[1] - 1)
            past, logits = s.new_kvs, s.logits
        else:
            logits = model(ids).logits
        nxt = Generator._sample(logits[:, -1, :].float(), gen, TEMPERATURE, TOP_K, None)
        cols.append(nxt)
        ids = torch.cat([ids, nxt], dim=1)
    sync(dev)
    t2 = time.perf_counter()
    if cache == "static":
        cbytes = kv.nbytes
    elif cache == "tuple":
        cbytes = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in past)
    else:
        cbytes = 0
    return torch.cat(cols, dim=1), t1 - t0, t2 - t1, cbytes


@torch.no_grad()
def teacher_forced_maxdiff(model, prompt_ids, toks):
    """(max |no-cache logits - static-cache logits|, allclose(atol=rtol=1e-4)) along a fixed
    token sequence: the gate for the numerically different no-cache loop."""
    P, G = prompt_ids.shape[1], toks.shape[1]
    seq = torch.cat([prompt_ids, toks[:1, :-1]], dim=1)
    full = model(seq).logits[:, P - 1:]
    kv = StaticKVCache.from_model(model, 1, seq.shape[1])
    steps = [model(seq[:, :P], past_kvs=kv.layers, pos_offset=0).logits[:, -1:]]
    for t in range(P, seq.shape[1]):
        steps.append(model(seq[:, t:t + 1], past_kvs=kv.layers, pos_offset=t).logits)
    cached = torch.cat(steps, dim=1)
    return float((full - cached).abs().max()), bool(torch.allclose(full, cached, atol=1e-4, rtol=1e-4))


# ---------------------------------------------------------------- statistics
def summarize(x):
    x = np.asarray(x, float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"median": float(med), "min": float(x.min()), "max": float(x.max()),
            "p25": float(q1), "p75": float(q3), "std": float(x.std()),
            "rel_iqr": float((q3 - q1) / med) if med else float("nan")}


def paired_ratio_ci(slow, fast, B=10_000, seed=0):
    """Speedup = median(slow)/median(fast) with a paired bootstrap CI over repeats.
    >1 means `fast` is faster. Returns (point, lo, hi, verdict)."""
    slow, fast = np.asarray(slow, float), np.asarray(fast, float)
    idx = np.random.default_rng(seed).integers(0, len(slow), size=(B, len(slow)))
    r = np.median(slow[idx], axis=1) / np.median(fast[idx], axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    point = float(np.median(slow) / np.median(fast))
    verdict = ("faster" if lo > 1 else "slower" if hi < 1 else "no measurable difference")
    return point, float(lo), float(hi), verdict


def environment(dev):
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
    except Exception:
        sha, dirty = "unknown", False
    env = {"device": dev, "torch": torch.__version__, "threads": torch.get_num_threads(),
           "platform": platform.platform(), "processor": platform.processor(),
           "logical_cpus": os.cpu_count(), "commit": sha + ("+dirty" if dirty else ""),
           "started": time.strftime("%Y-%m-%d %H:%M:%S"),
           "power_and_background_load": "not controlled or verified"}
    if dev == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


# ---------------------------------------------------------------- main
def build_configs(models, ns, grid, dev):
    cfgs, skipped = [], []
    for regime in ("short", "long"):
        for n in ns:
            for cache in ("none", "tuple", "static"):
                if cache == "none" and grid in ("cpu", "reduced") and not (
                        (regime == "short" and n in (1, 8)) or (regime == "long" and n == 1)):
                    why = ("estimated cost of hours on CPU" if dev == "cpu" else
                           "estimated cost too high for the kernel's time cap (tens of GPU minutes per config)")
                    skipped.append((f"fp32/none/n={n}/{regime}", why))
                    continue
                cfgs.append(("fp32", cache, n, regime))
            if "int8_all" in models:
                cfgs.append(("int8_all", "static", n, regime))
            if "fp16" in models:
                cfgs.append(("fp16", "static", n, regime))
    return cfgs, skipped


def cname(c):
    return f"{c[0]}/{c[1]}/n={c[2]}/{c[3]}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_30m.yaml")
    ap.add_argument("--weights", default="checkpoints/glassbox-sft-final/sft_final_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="0 = torch default")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ns", default="1,8,32")
    ap.add_argument("--grid", choices=["cpu", "reduced", "full"], default="cpu")
    ap.add_argument("--long_len", type=int, default=256)
    ap.add_argument("--no_int8", action="store_true")
    ap.add_argument("--allow_cpu_fp16", action="store_true",
                    help="time the fp16 model on CPU too (only to exercise the code path in tests)")
    ap.add_argument("--models", default="fp32,int8_all,fp16",
                    help="which of fp32,int8_all,fp16 to time (fp32 is always loaded as the reference)")
    ap.add_argument("--budget_s", type=float, default=0.0,
                    help="stop starting new repeats after this many seconds (needs >= 2 repeats); 0 = off")
    ap.add_argument("--out_prefix", default="docs/benchmark_cpu")
    a = ap.parse_args(argv)
    dev = a.device
    if a.threads:
        torch.set_num_threads(a.threads)
    log = lambda s: print(s, flush=True)

    cfg = load_config(a.config)
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    fp32 = GPT(cfg["model"])
    if a.weights != "none":
        fp32.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    fp32 = fp32.to(dev).eval()
    wanted = set(a.models.split(","))
    models = {"fp32": fp32}
    if not a.no_int8 and "int8_all" in wanted:
        models["int8_all"] = quantize_model(fp32.cpu(), True).to(dev)
        fp32.to(dev)
    if (dev == "cuda" or a.allow_cpu_fp16) and "fp16" in wanted:
        models["fp16"] = to_half(fp32)
    t_start = time.time()
    ns = [int(x) for x in a.ns.split(",")]
    K, G, R, W = a.prompts, a.gen, a.repeats, a.warmup

    # prompts: identical for every config
    vp = val_prompts(tok, cfg["data"]["out_dir"], 400)
    short = [(i, torch.tensor([tok.encode(p)], device=dev), p) for i, p, _, _ in vp[:K]]
    long_ = []
    for i, p, _, gold in vp:
        t = tok.encode(gold)
        if len(t) >= a.long_len:
            long_.append((i, torch.tensor([t[:a.long_len]], device=dev), None))
        if len(long_) == K:
            break
    assert len(long_) == K, "not enough validation stories of the long length"
    prompts = {"short": short, "long": long_}
    for reg in prompts:
        assert prompts[reg][0][1].shape[1] + G <= cfg["model"]["context_len"], reg
    cfgs, skipped = build_configs(models, ns, a.grid, dev)
    log(f"{len(cfgs)} configs, {len(skipped)} not run; K={K} G={G} R={R} W={W} device={dev}")

    # -------- correctness gate (before any timing)
    gate, dropped, tf = {}, set(), {}
    for regime in ("short", "long"):
        pid, seed = prompts[regime][0][1], prompts[regime][0][0]
        if any(c[1] == "none" and c[3] == regime for c in cfgs):
            ref1 = run_once(fp32, pid, 1, G, "static", seed, dev)[0]
            tf[regime] = teacher_forced_maxdiff(fp32, pid, ref1)       # (maxdiff, ok)
        for n in ns:
            toks = {}
            for cache in ("static", "tuple", "none"):
                if cache == "static" or ("fp32", cache, n, regime) in cfgs:
                    toks[cache] = run_once(fp32, pid, n, G, cache, seed, dev)[0]
            ref = toks["static"]
            if ("fp16", "static", n, regime) in cfgs:   # exempt from equality: report agreement
                h = run_once(models["fp16"], pid, n, G, "static", seed, dev)[0]
                gate[("fp16", n, regime)] = {"pass": True, "detail": (
                    f"token agreement with fp32 static {float((h == ref).float().mean()):.3f} "
                    "(informational; numerics differ, equality not required)")}
            if regime == "short" and n == 1:      # the real engine must match the bench loop
                eng = Generator(fp32, tok).generate(prompts["short"][0][2], n=1, max_new=G,
                                                    temperature=TEMPERATURE, top_k=TOP_K,
                                                    seed=seed, stop_at_eot=False).new_ids
                gate[("engine", n, regime)] = {"pass": eng == ref.tolist(),
                                               "detail": "engine tokens == bench static loop"}
            for cache in ("tuple", "none"):
                if cache not in toks:
                    continue
                agree = float((toks[cache] == ref).float().mean())
                if cache == "tuple":
                    ok, detail = bool(torch.equal(toks[cache], ref)), f"tokens identical (agreement {agree:.3f})"
                else:
                    md, ok = tf[regime]
                    detail = (f"teacher-forced max logit diff {md:.2e} (atol=rtol=1e-4); "
                              f"sampled-token agreement with static {agree:.3f}")
                gate[(cache, n, regime)] = {"pass": ok, "detail": detail}
                if not ok:
                    dropped.add(("fp32", cache, n, regime))
    for k, v in gate.items():
        if k[0] in ("tuple", "none", "engine", "fp16"):
            log(f"gate {k}: {'PASS' if v['pass'] else 'FAIL'} - {v['detail']}")
    cfgs = [c for c in cfgs if c not in dropped]

    # -------- warm-up, then interleaved timed repeats
    for c in cfgs:
        m, cache, n, regime = c
        pid, seed = prompts[regime][0][1], prompts[regime][0][0]
        for _ in range(W):
            run_once(models[m], pid, n, G, cache, seed, dev)
    log("warm-up done")
    raw = {cname(c): [] for c in cfgs}          # per repeat: list over prompts of [ttft, decode]
    cbytes = {}
    peak = {}

    def one_repeat(c):
        m, cache, n, regime = c
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        per = []
        for pi, pid, _ in prompts[regime]:
            _, ttft, dec, cb = run_once(models[m], pid, n, G, cache, pi, dev)
            per.append([ttft, dec])
            cbytes[cname(c)] = cb
        if dev == "cuda":
            peak[cname(c)] = max(peak.get(cname(c), 0), torch.cuda.max_memory_allocated())
        return per

    completed = 0
    for rep in range(R):
        if a.budget_s and completed >= 2 and time.time() - t_start > a.budget_s:
            log(f"time budget {a.budget_s:.0f}s reached: stopping after {completed} of {R} repeats")
            break
        order = list(cfgs)
        random.Random(1000 + rep).shuffle(order)
        for c in order:
            raw[cname(c)].append(one_repeat(c))
        completed = rep + 1
        log(f"repeat {completed}/{R} done ({time.time() - t_start:.0f}s elapsed)")
        json.dump({"partial": True, "repeats_completed": completed, "raw": raw},
                  open(a.out_prefix + "_raw.json", "w"))
    rerun = {}
    R = completed
    for c in cfgs:                               # protocol: re-run once if relative IQR > 20% (CPU)
        e2e = [np.mean([t + d for t, d in rp]) for rp in raw[cname(c)]]
        if dev == "cpu" and summarize(e2e)["rel_iqr"] > 0.20:
            log(f"re-running noisy config {cname(c)}")
            rerun[cname(c)] = [one_repeat(c) for _ in range(R)]

    # -------- summaries
    def series(name, r=None):
        rp = (r or raw)[name]
        ttft = [np.mean([t for t, _ in x]) for x in rp]
        dec = [np.mean([d for _, d in x]) for x in rp]
        return ttft, dec, [t + d for t, d in zip(ttft, dec)]

    rows = {}
    for c in cfgs:
        name = cname(c)
        ttft, dec, e2e = series(name)
        n = c[2]
        thr = [(G - 1) * n / d for d in dec]
        rows[name] = {"ttft": summarize(ttft), "decode": summarize(dec), "e2e": summarize(e2e),
                      "throughput_tok_s": summarize(thr),
                      "step_latency_ms": summarize([d / (G - 1) * 1000 for d in dec]),
                      "kv_cache_bytes": cbytes.get(name),
                      "model_tensor_bytes": tensor_bytes(models[c[0]]),
                      "peak_gpu_bytes": peak.get(name)}
        if name in rerun:
            t2, d2, e22 = series(name, rerun)
            rows[name]["rerun_e2e"] = summarize(e22)
    comps = []

    def add(label, slow_c, fast_c):
        if cname(slow_c) in raw and cname(fast_c) in raw:
            s, f = series(cname(slow_c)), series(cname(fast_c))
            comps.append({"comparison": label, "e2e": paired_ratio_ci(s[2], f[2]),
                          "decode": paired_ratio_ci(s[1], f[1]), "ttft": paired_ratio_ci(s[0], f[0])})
    for regime in ("short", "long"):
        for n in ns:
            add(f"static vs tuple, n={n}, {regime}", ("fp32", "tuple", n, regime), ("fp32", "static", n, regime))
            add(f"tuple vs none, n={n}, {regime}", ("fp32", "none", n, regime), ("fp32", "tuple", n, regime))
            add(f"static vs none, n={n}, {regime}", ("fp32", "none", n, regime), ("fp32", "static", n, regime))
            add(f"fp32 static vs int8 static, n={n}, {regime}", ("int8_all", "static", n, regime), ("fp32", "static", n, regime))
            add(f"fp16 static vs fp32 static, n={n}, {regime}", ("fp32", "static", n, regime), ("fp16", "static", n, regime))

    env = environment(dev)
    params = {"prompts": K, "gen": G, "repeats": R, "repeats_requested": a.repeats, "warmup": W, "ns": ns, "grid": a.grid,
              "temperature": TEMPERATURE, "top_k": TOP_K, "long_len": a.long_len,
              "short_prompt_idx": [i for i, _, _ in short], "long_prompt_idx": [i for i, _, _ in long_],
              "short_prompt_tokens": [int(p.shape[1]) for _, p, _ in short],
              "long_prompt_tokens": [int(p.shape[1]) for _, p, _ in long_]}
    json.dump({"environment": env, "params": params, "raw_per_repeat_per_prompt_ttft_decode_s": raw,
               "rerun": rerun, "summary": rows, "comparisons": comps,
               "gate": {f"{k[0]}|n={k[1]}|{k[2]}": v for k, v in gate.items()},
               "skipped": skipped, "dropped_by_gate": sorted(cname(c) for c in dropped)},
              open(a.out_prefix + "_raw.json", "w"), indent=1)
    write_markdown(a.out_prefix + ".md", env, params, cfgs, rows, comps, gate, skipped, dropped, G)
    return 0


def write_markdown(path, env, params, cfgs, rows, comps, gate, skipped, dropped, G):
    L, P = [], None
    P = L.append
    P("# Inference benchmark results\n")
    P("Protocol: `docs/benchmark_protocol.md` (including its CPU addendum). Times are medians over "
      f"{params['repeats']} interleaved repeats"
      f"{' (requested ' + str(params['repeats_requested']) + ', stopped by the time budget)' if params['repeats'] != params['repeats_requested'] else ''}; spread = interquartile range (p25-p75). Raw "
      "per-repeat timings: the `_raw.json` next to this file.\n")
    P("## Environment and parameters\n")
    for k, v in env.items():
        P(f"- {k}: {v}")
    P(f"- prompts K={params['prompts']}, generated tokens G={G} (fixed, EOT ignored), repeats "
      f"R={params['repeats']}, warm-up {params['warmup']}, n in {params['ns']}, grid `{params['grid']}`")
    P(f"- sampling: temperature {params['temperature']}, top-k {params['top_k']}; short prompts "
      f"{params['short_prompt_tokens']} tokens (validation prompts idx {params['short_prompt_idx']}), long "
      f"prompts {params['long_prompt_tokens']} tokens (first {params['long_len']} tokens of validation "
      f"stories idx {params['long_prompt_idx']}: the first K stories with enough tokens, not the same "
      "stories as the short regime)\n")
    P("## Correctness gate (before timing)\n")
    P("| check | n | regime | result | detail |")
    P("|---|---|---|---|---|")
    for k, v in gate.items():
        if k[0] in ("tuple", "none", "engine", "fp16"):
            P(f"| {k[0]} | {k[1]} | {k[2]} | {'PASS' if v['pass'] else 'FAIL'} | {v['detail']} |")
    if dropped:
        P(f"\nDropped by the gate (not timed): {', '.join(sorted(cname(c) for c in dropped))}")
    P("")
    for regime in ("short", "long"):
        P(f"## Results, {regime} prompts (median [p25-p75])\n")
        P("| config | TTFT ms | decode ms/step | tokens/s (all rows) | end-to-end s | KV cache MB | rel. IQR (e2e) |")
        P("|---|---|---|---|---|---|---|")
        for c in cfgs:
            if c[3] != regime:
                continue
            r = rows[cname(c)]
            f = lambda s, k, sc=1.0, d=1: f"{s[k]['median'] * sc:.{d}f} [{s[k]['p25'] * sc:.{d}f}-{s[k]['p75'] * sc:.{d}f}]"
            kv = r["kv_cache_bytes"]
            P(f"| {cname(c)} | {f(r, 'ttft', 1000, 0)} | {r['step_latency_ms']['median']:.1f} "
              f"[{r['step_latency_ms']['p25']:.1f}-{r['step_latency_ms']['p75']:.1f}] | "
              f"{f(r, 'throughput_tok_s', 1, 1)} | {f(r, 'e2e', 1, 2)} | "
              f"{(kv or 0) / 1e6:.1f} | {r['e2e']['rel_iqr']:.1%}"
              f"{' (re-run: ' + format(r['rerun_e2e']['rel_iqr'], '.1%') + ')' if 'rerun_e2e' in r else ''} |")
        P("")
    if skipped:
        P("## Not run\n")
        for n_, why in skipped:
            P(f"- {n_}: {why}")
        P("")
    P("## Comparisons (for 'A vs B', speedup = median time of B / median time of A, so >1 means A (the first-named config) is faster)\n")
    P("Paired bootstrap over repeats (B=10,000). 'faster' / 'slower' only when the 95% CI excludes 1.\n")
    P("| comparison | end-to-end | decode | TTFT |")
    P("|---|---|---|---|")
    fmt = lambda t: f"{t[0]:.2f}x [{t[1]:.2f}, {t[2]:.2f}] {t[3]}"
    for c in comps:
        P(f"| {c['comparison']} | {fmt(c['e2e'])} | {fmt(c['decode'])} | {fmt(c['ttft'])} |")
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
