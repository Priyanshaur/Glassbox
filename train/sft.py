"""SFT: response-only loss on instruct examples, Phase 3 checkpoint format.

Init: --init_weights (a Phase 3 weights_step{N}.pt). Fresh optimizer/step,
same atomic format, so SFT -> DPO chains with the same resume machinery and
arch/tokenizer-hash checks. Peak LR lower than pretraining (proposed 1e-4 for
30M; debug smoke overrides via CLI). 1-3 epochs; val = response-token loss on
the official validation split.
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
from model.gpt import GPT, count_params
from tokenizer.bpe import BPETokenizer, PAD_ID
from train import checkpoint as ckpt_mod
from train.instruct import build_sft_example, collate, docs_from_shard, synth_prompt
from train.optim import build_param_groups
from train.schedule import lr_at


def build_examples(tok, texts, context_len, offset=0):
    """(prompt, keywords, input_ids, labels) list; skips thin stories."""
    ex, skipped = [], 0
    for i, t in enumerate(texts):
        sp = synth_prompt(t, offset + i)
        if sp is None:
            skipped += 1
            continue
        prompt, kw = sp
        inp, labels = build_sft_example(tok, prompt, t, context_len)
        ex.append({"prompt": prompt, "keywords": kw,
                   "input": inp, "labels": labels})
    return ex, skipped


@torch.no_grad()
def response_loss(model, examples, device, vocab):
    """Mean CE over non-masked (response) tokens only."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), 8):
        chunk = examples[i:i + 8]
        inp, lab = collate([(e["input"], e["labels"]) for e in chunk], PAD_ID)
        inp = inp.to(device)
        logits = model(inp).logits
        loss = F.cross_entropy(logits.reshape(-1, vocab),
                               lab.reshape(-1).to(device), ignore_index=-100,
                               reduction="sum")
        tot += loss.item()
        n += int((lab != -100).sum())
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init_weights", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_seqs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    ctx = cfg["train"]["context_len"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = BPETokenizer.load(cfg["data"]["tokenizer_path"])
    train_texts = docs_from_shard(tok, cfg["data"]["out_dir"], "train")
    val_texts = docs_from_shard(tok, cfg["data"]["out_dir"], "validation")
    train_ex, skip_tr = build_examples(tok, train_texts, ctx)
    val_ex, skip_va = build_examples(tok, val_texts, ctx, offset=len(train_texts))
    print(f"[sft] train={len(train_ex)} (skip {skip_tr}) val={len(val_ex)} "
          f"(skip {skip_va})", flush=True)

    model = GPT(mcfg).to(device)
    model.load_state_dict(torch.load(args.init_weights, map_location=device,
                                     weights_only=True))
    opt = torch.optim.AdamW(build_param_groups(model, cfg["train"]["weight_decay"]),
                            lr=args.lr, betas=(0.9, 0.95))

    steps_per_epoch = max(len(train_ex) // args.batch_seqs, 1)
    max_steps = args.epochs * steps_per_epoch
    warmup = max(max_steps // 20, 1)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    metrics_f = open(os.path.join(args.ckpt_dir, "metrics_sft.jsonl"), "a",
                     encoding="utf-8")
    tok_hash = ckpt_mod.file_hash(cfg["data"]["tokenizer_path"])
    full_cfg = {"model": mcfg, "seed": args.seed}
    step = 0
    for epoch in range(args.epochs):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        order = rng.permutation(len(train_ex))
        for b in range(steps_per_epoch):
            sel = [(b * args.batch_seqs + k) % len(train_ex)
                   for k in range(args.batch_seqs)]
            batch = [train_ex[int(order[i])] for i in sel]
            inp, lab = collate([(e["input"], e["labels"]) for e in batch], PAD_ID)
            inp, lab = inp.to(device), lab.to(device)
            lr = lr_at(step, args.lr, warmup, max_steps, 0.1)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(inp).logits.reshape(-1, mcfg["vocab_size"]),
                                   lab.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 20 == 0:
                v = response_loss(model, val_ex[:64], device, mcfg["vocab_size"])
                rec = {"step": step, "loss": loss.item(), "val": v, "lr": lr}
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
                print(f"[sft] step={step} loss={loss.item():.3f} val={v:.3f}",
                      flush=True)
            step += 1
    ckpt_mod.save_checkpoint(args.ckpt_dir, model, opt,
                             torch.amp.GradScaler("cuda", enabled=False),
                             step, full_cfg, tok_hash,
                             cfg["checkpoint"]["keep_last_n"])
    metrics_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
