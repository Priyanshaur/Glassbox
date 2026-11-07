"""Phase 6 / B: scan a corpus with the accepted SAE and record, per feature, how often it fires, a histogram of its
non-zero activations, and its strongest activations (the raw material for labeling and for choosing steering features).

Rules fixed by docs/phase6_preregistration.md and Amendment 1:
  * window position 0 is skipped everywhere (the SAE never trained on it);
  * an "alive" feature fires at least MIN_FIRINGS (20) times in a 10M-token scan;
  * contexts used for labeling are the 24 highest activations, at most ONE per document (doc = text between two
    <|endoftext|> tokens), +-12 tokens around the activating token.

What is stored (nothing but numbers and file positions; token text is re-read from the shard when needed):
  counts (F,)            number of (non-position-0) tokens where the feature fires
  hist (F, NBINS)        histogram of the non-zero activations, log-spaced bins (for the 90th percentile a_f)
  pool_vals (F, P)       the P strongest per-window maxima (one candidate per window, so a candidate is ~one document)
  pool_loc (F, P, 3)     (shard index, absolute start of the window in the shard, position inside the window)
Every window's maximum is a candidate; a window contributes at most one candidate per feature.
"""
import json
import os

import numpy as np
import torch

from interp import activations as A

MIN_FIRINGS = 20
POOL = 128
NBINS = 256
EDGES = torch.logspace(-3, 3, NBINS - 1)          # activation bin edges (in SAE-normalised units)
EOT_ID = 256
CTX_HALF = 12
N_CONTEXTS = 24
SCAN_SEED = 20_261_103                             # fresh windows: not a training or dead-scan seed


def window_starts(loader, n_windows, ctx, seed=SCAN_SEED, sequential=False):
    """[(shard, start)] of n_windows windows of `ctx` tokens. Random (with replacement over shards, weighted by
    size) for the train scan; sequential non-overlapping from the start for small splits (validation)."""
    out = []
    if sequential:
        for si, m in enumerate(loader.maps):
            for s in range(0, len(m) - ctx + 1, ctx):
                out.append((si, s))
                if len(out) == n_windows:
                    return out
        return out
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    for _ in range(n_windows):
        si = int(rng.choice(len(loader.maps), p=loader.probs))
        out.append((si, int(rng.integers(0, int(loader.sizes[si]) - ctx + 1))))
    return out


@torch.no_grad()
def scan(model, sae, scale, loader, layer, starts, ctx, chunk=16, device=None, log=None, state=None):
    """One pass over the given windows. Returns the dict described in the module docstring. `state` (a previous
    result) is continued, so a long scan can be run in blocks and resumed."""
    device = device or next(model.parameters()).device
    Fn = sae.n_features
    if state is None:
        counts = torch.zeros(Fn, dtype=torch.long)
        hist = torch.zeros(Fn * NBINS, dtype=torch.long)
        pool_vals = torch.full((Fn, POOL), -1.0)
        pool_loc = torch.zeros(Fn, POOL, 3, dtype=torch.long)
        tokens = 0
    else:
        counts, hist = state["counts"].clone(), state["hist"].reshape(-1).clone()
        pool_vals, pool_loc, tokens = state["pool_vals"].clone(), state["pool_loc"].clone(), state["tokens"]
    for i in range(0, len(starts), chunk):
        sub = starts[i:i + chunk]
        ids = torch.from_numpy(np.stack([np.asarray(loader.maps[s][a:a + ctx], dtype=np.int64) for s, a in sub]))
        x = A.record_windows(model, ids, layer, chunk, device)[:, 1:] * scale          # skip window position 0
        B, T, d = x.shape
        acts = sae.encode(x.reshape(-1, d))[0].reshape(B, T, Fn).cpu()
        counts += (acts > 0).sum((0, 1))
        nz = acts.nonzero()                                                       # (n, 3): b, t, f
        v = acts[nz[:, 0], nz[:, 1], nz[:, 2]]
        hist += torch.bincount(nz[:, 2] * NBINS + torch.bucketize(v, EDGES), minlength=Fn * NBINS)
        wmax, wpos = acts.max(dim=1)                                              # (B, F): per-window strongest
        loc = torch.stack([torch.tensor([s for s, _ in sub]).unsqueeze(1).expand(B, Fn),
                           torch.tensor([a for _, a in sub]).unsqueeze(1).expand(B, Fn),
                           wpos + 1], dim=-1)                                     # +1: undo the position-0 skip
        all_vals = torch.cat([pool_vals, wmax.t()], dim=1)
        all_loc = torch.cat([pool_loc, loc.transpose(0, 1)], dim=1)
        top = all_vals.topk(POOL, dim=1).indices
        pool_vals = all_vals.gather(1, top)
        pool_loc = all_loc.gather(1, top.unsqueeze(-1).expand(-1, -1, 3))
        tokens += B * T
        if log and (i // chunk) % 100 == 0:
            log(f"    scanned {tokens:,} tokens ({i + len(sub)}/{len(starts)} windows)")
    return {"counts": counts, "hist": hist.reshape(Fn, NBINS), "pool_vals": pool_vals, "pool_loc": pool_loc,
            "tokens": tokens}


def percentile_from_hist(hist_row, q):
    """Approximate q-quantile (0..1) of the non-zero activations of one feature from its log-bin histogram
    (returns the upper edge of the bin containing it)."""
    c = hist_row.cumsum(0)
    if c[-1] == 0:
        return float("nan")
    b = int(torch.searchsorted(c, torch.tensor(q * float(c[-1]))))
    edges = torch.cat([EDGES, EDGES[-1:] * 1.0])
    return float(edges[min(b, len(edges) - 1)])


def doc_id(maps, shard, abs_pos, window_start, lookback=4096):
    """Identifier of the document containing absolute token `abs_pos`: the position of the last EOT before it.
    If no EOT is found within the lookback, the window start (stable, and distinct windows differ)."""
    lo = max(0, abs_pos - lookback)
    seg = np.asarray(maps[shard][lo:abs_pos])
    hit = np.nonzero(seg == EOT_ID)[0]
    return (shard, int(lo + hit[-1])) if len(hit) else (shard, -1 - window_start)


def top_contexts(res, maps, f, n=N_CONTEXTS, half=CTX_HALF):
    """The up-to-n strongest contexts of feature f, one per document: list of dicts with the token ids around the
    activating token, the index of that token inside them, the activation and the document id."""
    out, seen = [], set()
    order = res["pool_vals"][f].argsort(descending=True)
    for j in order.tolist():
        v = float(res["pool_vals"][f, j])
        if v <= 0:
            break
        shard, start, pos = (int(t) for t in res["pool_loc"][f, j])
        ap = start + pos
        did = doc_id(maps, shard, ap, start)
        if did in seen:
            continue
        seen.add(did)
        lo, hi = max(0, ap - half), min(len(maps[shard]), ap + half + 1)
        out.append({"act": v, "doc": did, "tokens": [int(t) for t in maps[shard][lo:hi]], "at": ap - lo})
        if len(out) == n:
            break
    return out


def save_scan(res, path, meta):
    torch.save({**res, "meta": meta}, path)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
