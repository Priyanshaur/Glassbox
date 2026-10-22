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

import numpy as np
import torch
import torch.nn.functional as F

from common.config import load_config
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer, EOT_ID, PAD_ID
from train import checkpoint as ckpt_mod
from train.optim import build_param_groups


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
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    ctx = cfg["train"]["context_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])

    with open(args.pairs, encoding="utf-8") as f:
        all_pairs = [json.loads(l) for l in f]
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
            margs = torch.stack([m for _, m in losses])
            tot_loss += loss.item() * len(chunk)
            tot_acc += float((margs > 0).float().mean()) * len(chunk)
            tot_marg += float(margs.mean()) * len(chunk)
            tot_ch += float(sum(float(a) for a, _ in lps))
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
    metrics_f = open(os.path.join(args.ckpt_dir, "metrics_dpo.jsonl"), "a",
                     encoding="utf-8")
    tok_hash = ckpt_mod.file_hash(cfg["data"]["tokenizer_path"])
    full_cfg = {"model": mcfg, "seed": args.seed}
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    best, stale, step = float("inf"), 0, 0
    for epoch in range(args.epochs):
        rng_e = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        idx = rng_e.permutation(len(train))
        for s in range(0, len(train), args.batch):
            chunk = [train[int(j)] for j in idx[s:s + args.batch]]
            loss, acc, marg, ch = batch_stats(chunk)
            if step % 10 == 0:
                rec = {"step": step, "loss": loss, "acc": acc,
                       "margin": marg, "chosen_lp": ch}
                if step % args.eval_every == 0:
                    hl, ha = heldout()
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
                      f"marg={marg:.2f} ch_lp={ch:.1f} " +
                      (f"h_loss={rec.get('h_loss', float('nan')):.3f}" if "h_loss" in rec else ""),
                      flush=True)
            step += 1
        if stale >= args.patience:
            print(f"[dpo] early stop at epoch {epoch} (best h_loss={best:.3f})",
                  flush=True)
            break
    metrics_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
