"""SAE evaluation against the pre-registered rules in docs/phase6_preregistration.md.

  A1 fvu        sum ||x - x_hat||^2 / sum ||x - mean(x)||^2           <= 0.15
  A2 L0         mean non-zero latents per token                        8 .. 32
  A3 dead       fraction of latents that never fire over 10M tokens    <= 0.10
  A4 CE rec.    1 - (CE_sae - CE_clean) / (CE_meanablate - CE_clean)   >= 0.85
  A5 transfer   base-trained SAE on the SFT model: FVU <= 0.25 and CE recovered >= 0.80,
                on plain validation windows AND template-formatted prompt+story sequences

CE_sae / CE_meanablate patch the layer's residual through `GPT.forward(inject=...)` with an
exact delta (replacement - original), so the model code is untouched. The constants in
THRESHOLDS are checked against the pre-registration document by a test.
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from interp import activations as A
from interp.sae import load_sae

THRESHOLDS = {"fvu_max": 0.15, "l0_max": 32, "l0_min": 8, "dead_max": 0.10,
              "ce_recovered_min": 0.85, "sft_fvu_max": 0.25, "sft_ce_recovered_min": 0.80}
MIN_EVAL_TOKENS = 500_000
DEAD_SCAN_TOKENS = 10_000_000
DEAD_SCAN_SEED = 314_159                    # fresh windows: not a training seed
EVAL_SEED = 999_983                         # same fixed validation draw as A.eval_activations


# ---------------------------------------------------------------- reconstruction metrics
@torch.no_grad()
def reconstruction_metrics(sae, acts, batch=8192):
    """FVU, L0 (mean and histogram), mean norms, and which latents fired, over (N, d) acts."""
    mean = acts.mean(0)
    err = var = 0.0
    hist = torch.zeros(sae.k + 1, dtype=torch.long)
    fired = torch.zeros(sae.n_features, dtype=torch.bool)
    for i in range(0, acts.shape[0], batch):
        x = acts[i:i + batch]
        x_hat, a, _ = sae(x)
        err += float(((x - x_hat) ** 2).sum())
        var += float(((x - mean) ** 2).sum())
        nz = (a > 0).sum(-1)
        hist += torch.bincount(nz, minlength=sae.k + 1).cpu()
        fired |= (a > 0).any(0).cpu()
    l0 = float((hist * torch.arange(sae.k + 1)).sum() / hist.sum())
    return {"fvu": err / var, "l0": l0, "l0_hist": hist.tolist(), "tokens": int(acts.shape[0]),
            "mean_act_norm": float(acts.norm(dim=-1).mean()),
            "mean_decoder_norm": float(sae.W_dec.norm(dim=1).mean()),
            "fired_in_eval": int(fired.sum())}


@torch.no_grad()
def dead_fraction(model, sae, loader, layer, scale, tokens, ctx, seed=DEAD_SCAN_SEED, chunk=32,
                  windows_per_round=64, device=None):
    """Fraction of latents that never fire over `tokens` fresh train-distribution tokens."""
    device = device or next(model.parameters()).device
    fired = torch.zeros(sae.n_features, dtype=torch.bool, device=device)
    seen, rnd = 0, 0
    while seen < tokens:
        ids = A.draw_windows(loader, seed, rnd, windows_per_round, ctx)
        acts = A.record_windows(model, ids, layer, chunk, device).reshape(-1, sae.d_in) * scale
        for i in range(0, acts.shape[0], 8192):
            fired |= (sae.encode(acts[i:i + 8192])[0] > 0).any(0)
        seen += acts.shape[0]
        rnd += 1
    return {"dead_fraction": float(1 - fired.float().mean()), "scan_tokens": seen}


@torch.no_grad()
def fvu_by_position(sae, acts3, split=16, batch=8192):
    """FVU for positions < split vs the rest, both against the global mean: (N, T, d) input."""
    d = acts3.shape[-1]
    mean = acts3.reshape(-1, d).mean(0)
    out = {}
    for name, sl in (("first_%d" % split, slice(0, split)), ("rest", slice(split, None))):
        x = acts3[:, sl].reshape(-1, d)
        err = var = 0.0
        for i in range(0, x.shape[0], batch):
            xb = x[i:i + batch]
            err += float(((xb - sae(xb)[0]) ** 2).sum())
            var += float(((xb - mean) ** 2).sum())
        out[name] = err / var if var > 0 else float("nan")
    return out


# ---------------------------------------------------------------- CE with patched residual
def _ce_sum(logits, targets, mask=None):
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    if mask is not None:
        m = mask.reshape(-1)
        return float(ce[m].sum()), int(m.sum())
    return float(ce.sum()), int(ce.numel())


@torch.no_grad()
def ce_recovered(model, sae, layer, scale, inputs, targets, masks=None, chunk=16, device=None):
    """CE with the clean model, with the SAE reconstruction patched in, and with the
    dataset-mean residual patched in; plus the recovered fraction.
    inputs/targets: (N, T) ids; masks: optional (N, T) bool of positions that count."""
    device = device or next(model.parameters()).device
    # mean residual over the (counted) positions
    tot, n = torch.zeros(sae.d_in, device=device), 0
    for i in range(0, inputs.shape[0], chunk):
        r = model(inputs[i:i + chunk].to(device), record_layers=[layer]).recorded[layer]
        if masks is None:
            tot += r.reshape(-1, r.shape[-1]).sum(0)
            n += r.shape[0] * r.shape[1]
        else:
            mk = masks[i:i + chunk].to(device)
            tot += r[mk].sum(0)
            n += int(mk.sum())
    mean_res = tot / n
    sums = {"clean": [0.0, 0], "sae": [0.0, 0], "ablate": [0.0, 0]}
    for i in range(0, inputs.shape[0], chunk):
        x = inputs[i:i + chunk].to(device)
        y = targets[i:i + chunk].to(device)
        mk = None if masks is None else masks[i:i + chunk].to(device)
        out = model(x, record_layers=[layer])
        r = out.recorded[layer]
        rec = sae(r * scale)[0] / scale
        for name, logits in (("clean", out.logits),
                             ("sae", model(x, inject={layer: rec - r}).logits),
                             ("ablate", model(x, inject={layer: mean_res.view(1, 1, -1) - r}).logits)):
            s, c = _ce_sum(logits, y, mk)
            sums[name][0] += s
            sums[name][1] += c
    ce = {k: v[0] / v[1] for k, v in sums.items()}
    denom = ce["ablate"] - ce["clean"]
    rec_frac = 1 - (ce["sae"] - ce["clean"]) / denom if denom > 1e-9 else float("nan")
    return {"ce_clean": ce["clean"], "ce_sae": ce["sae"], "ce_meanablate": ce["ablate"],
            "ce_recovered": rec_frac, "tokens": sums["clean"][1]}


# ---------------------------------------------------------------- template (SFT-format) data
def template_sequences(tok, out_dir, n, ctx):
    """First n validation stories as SFT-formatted prompt+story+EOT sequences:
    (inputs (N, L) padded, targets (N, L), mask (N, L) True on response-token positions)."""
    from tokenizer.bpe import PAD_ID
    from train.instruct import build_sft_example, iter_shard_texts, synth_prompt
    rows = []
    for idx, text in enumerate(iter_shard_texts(tok, out_dir, "validation")):
        sp = synth_prompt(text, idx)
        if sp is None:
            continue
        inp, lab = build_sft_example(tok, sp[0], text, ctx)
        rows.append((inp, lab))
        if len(rows) >= n:
            break
    L = max(len(i) for i, _ in rows)
    inputs = torch.full((len(rows), L), PAD_ID, dtype=torch.long)
    targets = torch.zeros((len(rows), L), dtype=torch.long)
    mask = torch.zeros((len(rows), L), dtype=torch.bool)
    for r, (i, l) in enumerate(rows):
        inputs[r, :len(i)] = i
        keep = l != -100
        targets[r, :len(l)] = torch.where(keep, l, torch.zeros_like(l))
        mask[r, :len(l)] = keep
    return inputs, targets, mask


@torch.no_grad()
def template_fvu(model, sae, layer, scale, inputs, mask, chunk=16, device=None):
    """Reconstruction metrics over all non-pad positions of the template sequences
    (mask is the response mask; non-pad = inputs != PAD)."""
    from tokenizer.bpe import PAD_ID
    device = device or next(model.parameters()).device
    xs = []
    for i in range(0, inputs.shape[0], chunk):
        x = inputs[i:i + chunk].to(device)
        r = model(x, record_layers=[layer]).recorded[layer] * scale
        xs.append(r[(x != PAD_ID)])
    return reconstruction_metrics(sae, torch.cat(xs))


# ---------------------------------------------------------------- rules
def verdict(report):
    """Pass/fail per pre-registered rule from a report dict (see evaluate())."""
    T = THRESHOLDS
    v = {"A1_fvu": report["fvu"] <= T["fvu_max"],
         "A2_l0": T["l0_min"] <= report["l0"] <= T["l0_max"],
         "A3_dead": report["dead_fraction"] <= T["dead_max"],
         "A4_ce": report["ce_recovered"] >= T["ce_recovered_min"]}
    v["layer_passes_A1_A4"] = all(v.values())
    t = report.get("transfer")
    if t is not None:
        v["A5_transfer"] = all(t[k]["fvu"] <= T["sft_fvu_max"] and t[k]["ce_recovered"] >= T["sft_ce_recovered_min"]
                               for k in ("plain", "template"))
    return v


def evaluate(model, sae, scale, layer, val_loader, train_loader, ctx, tok=None, out_dir=None,
             eval_tokens=MIN_EVAL_TOKENS, dead_tokens=DEAD_SCAN_TOKENS, sft_model=None, chunk=16,
             n_template=200, device=None):
    """Full report for one SAE. `sft_model` (optional) enables A5."""
    device = device or next(model.parameters()).device
    n_win = max(eval_tokens // ctx, 1)
    ids = val_loader.batch_ids(EVAL_SEED, 0, n_win, ctx)
    inputs, targets = ids[:, :ctx], ids[:, 1:ctx + 1]
    acts = (A.record_windows(model, inputs, layer, chunk, device) * scale).reshape(-1, sae.d_in)
    rep = reconstruction_metrics(sae, acts)
    rep["fvu_by_position"] = fvu_by_position(sae, acts.reshape(inputs.shape[0], inputs.shape[1], -1))
    rep.update(dead_fraction(model, sae, train_loader, layer, scale, dead_tokens, ctx, device=device))
    rep.update(ce_recovered(model, sae, layer, scale, inputs, targets, None, chunk, device))
    rep["eval_tokens"] = int(inputs.numel())
    if sft_model is not None:
        plain = {**reconstruction_metrics(sae, (A.record_windows(sft_model, inputs, layer, chunk, device)
                                                * scale).reshape(-1, sae.d_in)),
                 **ce_recovered(sft_model, sae, layer, scale, inputs, targets, None, chunk, device)}
        ti, tt, tm = template_sequences(tok, out_dir, n_template, ctx)
        temp = {**template_fvu(sft_model, sae, layer, scale, ti, tm, chunk, device),
                **ce_recovered(sft_model, sae, layer, scale, ti, tt, tm, chunk, device)}
        rep["transfer"] = {"plain": plain, "template": temp}
    rep["verdict"] = verdict(rep)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True, help="model the SAE was trained on (base)")
    ap.add_argument("--sae", required=True)
    ap.add_argument("--sft_weights", default=None)
    ap.add_argument("--out", required=True, help="json report path")
    ap.add_argument("--eval_tokens", type=int, default=MIN_EVAL_TOKENS)
    ap.add_argument("--dead_tokens", type=int, default=DEAD_SCAN_TOKENS)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    from common.config import load_config
    from model.gpt import GPT
    from tokenizer.bpe import BPETokenizer
    from train.data_loader import ShardLoader
    c = load_config(a.config)
    load = lambda p: _load_model(c, p, a.device)
    model = load(a.weights)
    sft = load(a.sft_weights) if a.sft_weights else None
    sae, meta = load_sae(a.sae)
    sae = sae.to(a.device).eval()
    tok = BPETokenizer.load(c["data"]["tokenizer_path"])
    out_dir = c["data"]["out_dir"]
    rep = evaluate(model, sae, meta["act_scale"], meta["layer"], ShardLoader(out_dir, "validation"),
                   ShardLoader(out_dir, "train"), c["train"]["context_len"], tok, out_dir,
                   a.eval_tokens, a.dead_tokens, sft, device=a.device)
    rep["sae_meta"] = {k: v for k, v in meta.items() if k != "train_cfg"}
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("fvu", "l0", "dead_fraction", "ce_recovered", "verdict")}, indent=1))
    return 0


def _load_model(c, path, device):
    from model.gpt import GPT
    m = GPT(c["model"])
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(device).eval()


if __name__ == "__main__":
    sys.exit(main())
