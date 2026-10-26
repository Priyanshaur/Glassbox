"""DPO by hand: frozen SFT reference, policy learns pass-over-fail margins.

loss = -log sigmoid(beta * ((lp_w - lp_l) - (lr_w - lr_l))), response-token
logprob SUMS (prompt masked like SFT). Logs: loss, reward accuracy
(fraction with margin > 0), mean margin, mean chosen logprob. Low LR
(proposed 1e-6 for 30M, sweep {5e-7, 1e-6, 2e-6} on Kaggle), few epochs,
early stopping on held-out pairs (best held-out loss wins; patience stops).
Same Phase 3 checkpoint format (policy only; reference rebuilt from the SFT
weights file and never saved or stepped).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.config import load_config
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer, EOT_ID, PAD_ID
from train import checkpoint as ckpt_mod
from train.optim import build_param_groups


def load_pairs(path):
    """Accept BOTH formats: flat {prompt,winner,loser} rows and the graded
    per-prompt records written by train.pairs.build_pairs_graded
    ({idx, prompt, pairs: [{prompt,winner,loser,margin}]}). Returns a flat
    list of {prompt, winner, loser} dicts."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "pairs" in rec and isinstance(rec["pairs"], list):
                out.extend({"prompt": p["prompt"], "winner": p["winner"],
                            "loser": p["loser"]} for p in rec["pairs"])
            else:
                out.append({"prompt": rec["prompt"],
                            "winner": rec["winner"], "loser": rec["loser"]})
    return out


def response_logps(model, tok, prompt, completion, context_len, device):
    """Sum of logprobs over completion+EOT tokens given the prompt."""
    p = tok.encode(prompt)
    r = tok.encode(completion)
    full = (p + r + [EOT_ID])[:context_len]
    P = min(len(p), len(full) - 1)
    ids = torch.tensor([full], dtype=torch.long, device=device)
    logp = torch.log_softmax(model(ids).logits[0], dim=-1)  # caller sets mode
    idx = torch.tensor(full[1:], dtype=torch.long, device=logp.device)
    tok_lp = logp[:-1].gather(1, idx.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(tok_lp)
    mask[:P - 1] = 0  # prompt predictions excluded, first response kept
    return (tok_lp * mask).sum()


def dpo_loss(lp_w, lp_l, lr_w, lr_l, beta=0.1):
    margin = (lp_w - lp_l) - (lr_w - lr_l)
    return -F.logsigmoid(beta * margin), margin


def split_holdout(pairs, heldout_frac, seed=0):
    """Hold out BY PROMPT, not by pair (fix 2026-10-22).

    The old split permuted the pair list, so one prompt's pair 1 could sit
    in train while its pair 2 was held out -- prompt leakage across the
    split. Deterministic: prompt permutation via SeedSequence([seed, 1]).
    Returns (train_pairs, hold_pairs) with disjoint prompt sets.
    """
    prompts = sorted({p["prompt"] for p in pairs})
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    order = rng.permutation(len(prompts))
    n_hold = max(int(len(prompts) * heldout_frac), 1)
    hold_prompts = {prompts[i] for i in order[:n_hold]}
    hold = [p for p in pairs if p["prompt"] in hold_prompts]
    train = [p for p in pairs if p["prompt"] not in hold_prompts]
    return train, hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init_policy", required=True)
    ap.add_argument("--pairs", required=True, help="jsonl {prompt,winner,loser}")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="none",
                    help="'latest' (resume from ckpt_dir), 'none', or a ckpt path")
    ap.add_argument("--max_steps", type=int, default=0,
                    help="stop after this many total updates (0 = run all)")
    ap.add_argument("--max_hours", type=float, default=0.0,
                    help="exit cleanly at a step boundary after this many "
                         "wall-clock hours (0 = no cap)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    ctx = cfg["train"]["context_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])

    all_pairs = load_pairs(args.pairs)
    train, hold = split_holdout(all_pairs, args.heldout_frac, args.seed)
    hold_prompts = sorted({p["prompt"] for p in hold})
    print(f"[dpo] pairs={len(all_pairs)} train={len(train)} "
          f"heldout={len(hold)} (holdout by prompt: {len(hold_prompts)} "
          f"of {len({p['prompt'] for p in all_pairs})})",
          flush=True)

    policy = GPT(mcfg).to(device)
    policy.load_state_dict(torch.load(args.init_policy, map_location=device,
                                      weights_only=True))
    ref = GPT(mcfg).to(device)
    ref.load_state_dict(torch.load(args.init_policy, map_location=device,
                                   weights_only=True))
    ref.eval()
    ref.requires_grad_(False)  # frozen: never stepped, never saved
    opt = torch.optim.AdamW(build_param_groups(policy, 0.0),
                            lr=args.lr, betas=(0.9, 0.95))

    def batch_stats(pairs):
        tot_loss, tot_acc, tot_marg, tot_ch = 0.0, 0.0, 0.0, 0.0
        policy.train()
        for i in range(0, len(pairs), args.batch):
            chunk = pairs[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                refs = [(response_logps(ref, tok, p["prompt"], p["winner"], ctx, device),
                         response_logps(ref, tok, p["prompt"], p["loser"], ctx, device))
                        for p in chunk]
            lps = [(response_logps(policy, tok, p["prompt"], p["winner"], ctx, device),
                    response_logps(policy, tok, p["prompt"], p["loser"], ctx, device))
                   for p in chunk]
            losses = [dpo_loss(a, b, c, d, args.beta)
                      for (a, b), (c, d) in zip(lps, refs)]
            loss = torch.stack([l for l, _ in losses]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            margs = torch.stack([m for _, m in losses]).detach()
            tot_loss += loss.item() * len(chunk)
            tot_acc += float((margs > 0).float().mean()) * len(chunk)
            tot_marg += float(margs.mean()) * len(chunk)
            tot_ch += float(sum(float(a.detach()) for a, _ in lps))
        n = len(pairs)
        return tot_loss / n, tot_acc / n, tot_marg / n, tot_ch / n

    @torch.no_grad()
    def heldout():
        policy.eval()
        tot_loss, tot_acc, n = 0.0, 0.0, 0
        for p in hold:
            lw = response_logps(policy, tok, p["prompt"], p["winner"], ctx, device)
            ll = response_logps(policy, tok, p["prompt"], p["loser"], ctx, device)
            rw = response_logps(ref, tok, p["prompt"], p["winner"], ctx, device)
            rl = response_logps(ref, tok, p["prompt"], p["loser"], ctx, device)
            l, m = dpo_loss(lw, ll, rw, rl, args.beta)
            tot_loss += l.item()
            tot_acc += float(m > 0)
            n += 1
        policy.train()
        return tot_loss / n, tot_acc / n

    os.makedirs(args.ckpt_dir, exist_ok=True)
    metrics_path = os.path.join(args.ckpt_dir, "metrics_dpo.jsonl")
    prior_hl = []                      # F1: held-out losses already on disk
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "h_loss" in r:
                        prior_hl.append(r["h_loss"])
    metrics_f = open(metrics_path, "a", encoding="utf-8")
    tok_hash = ckpt_mod.file_hash(cfg["data"]["tokenizer_path"])
    full_cfg = {"model": mcfg, "seed": args.seed}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start_step = 0
    step = 0
    if args.resume == "latest":
        latest = ckpt_mod.latest_checkpoint(args.ckpt_dir)
        if latest:
            start_step = ckpt_mod.load_checkpoint(latest, policy, opt, scaler,
                                                  cfg["data"]["tokenizer_path"],
                                                  mcfg)
            print(f"[dpo] resumed {latest} -> next step {start_step}", flush=True)
        else:
            print("[dpo] --resume latest: no checkpoint yet; starting fresh",
                  flush=True)
    elif args.resume != "none":
        start_step = ckpt_mod.load_checkpoint(args.resume, policy, opt, scaler,
                                              cfg["data"]["tokenizer_path"],
                                              mcfg)
        print(f"[dpo] resumed {args.resume} -> next step {start_step}",
              flush=True)
    best, stale = float("inf"), 0
    if start_step and prior_hl:
        # Resume: the best held-out loss and the patience counter come from
        # the metrics already on disk, so a resumed run neither forgets its
        # best checkpoint nor restarts early stopping from zero.
        best = min(prior_hl)
        stale = len(prior_hl) - 1 - prior_hl.index(best)
    loss = acc = marg = ch = t_step = None  # last fresh update; None until one runs

    def log_final_step():
        """Record the last completed update even when it falls off the
        every-10 logging boundary (a 20-step run ends at step 19)."""
        if loss is None or step <= 0 or (step - 1) % 10 == 0:
            return
        rec = {"step": step - 1, "loss": loss, "acc": acc,
               "margin": marg, "chosen_lp": ch, "t_step": round(t_step, 3)}
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        print(f"[dpo] final step={step - 1} loss={loss:.3f} acc={acc:.2f} "
              f"marg={marg:.2f} ch_lp={ch:.1f} t={t_step:.2f}s", flush=True)

    if args.max_steps and start_step >= args.max_steps:
        print(f"[dpo] already at/past max_steps ({start_step}>={args.max_steps})",
              flush=True)
        metrics_f.close()
        return 0
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        rng_e = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        idx = rng_e.permutation(len(train))
        for s in range(0, len(train), args.batch):
            if start_step and step < start_step:
                step += 1
                continue  # replayed on resume; skip the already-done updates
            if args.max_steps and step >= args.max_steps:
                print(f"[dpo] reached max_steps={args.max_steps}; stopping",
                      flush=True)
                log_final_step()
                metrics_f.close()
                return 0
            if args.max_hours and (time.perf_counter() - t_start
                                   ) >= args.max_hours * 3600:
                print(f"[dpo] reached max_hours={args.max_hours:g}; exiting "
                      "cleanly (best ckpt on disk; --resume latest continues)",
                      flush=True)
                log_final_step()
                metrics_f.close()
                return 0
            t0 = time.perf_counter()
            chunk = [train[int(j)] for j in idx[s:s + args.batch]]
            loss, acc, marg, ch = batch_stats(chunk)
            t_step = time.perf_counter() - t0
            if step % 10 == 0:
                rec = {"step": step, "loss": loss, "acc": acc,
                       "margin": marg, "chosen_lp": ch, "t_step": round(t_step, 3)}
                if step % args.eval_every == 0:
                    te0 = time.perf_counter()
                    hl, ha = heldout()
                    rec["t_eval"] = round(time.perf_counter() - te0, 3)
                    rec["h_loss"], rec["h_acc"] = hl, ha
                    if hl < best:
                        best, stale = hl, 0
                        ckpt_mod.save_checkpoint(
                            args.ckpt_dir, policy, opt, scaler, step, full_cfg,
                            tok_hash, cfg["checkpoint"]["keep_last_n"])
                    else:
                        stale += 1
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
                print(f"[dpo] step={step} loss={loss:.3f} acc={acc:.2f} "
                      f"marg={marg:.2f} ch_lp={ch:.1f} t={t_step:.2f}s " +
                      (f"h_loss={rec.get('h_loss', float('nan')):.3f} "
                       f"h_acc={rec.get('h_acc', float('nan')):.2f} "
                       f"t_eval={rec.get('t_eval', float('nan')):.1f}s" if "h_loss" in rec else ""),
                      flush=True)
            step += 1
        if stale >= args.patience:
            print(f"[dpo] early stop at epoch {epoch} (best h_loss={best:.3f})",
                  flush=True)
            break
    log_final_step()
    metrics_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
