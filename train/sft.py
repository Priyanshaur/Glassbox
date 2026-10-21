"""SFT: response-only loss on instruct examples, Phase 3 checkpoint format.

Init: --init_weights (a Phase 3 weights_step{N}.pt). Fresh optimizer/step,
same atomic format, so SFT -> DPO chains with the same resume machinery and
arch/tokenizer-hash checks. Peak LR lower than pretraining (proposed 1e-4 for
30M; debug smoke overrides via CLI). 1-3 epochs; val = response-token loss on
the official validation split.

Data is streamed straight out of the shard .bins and capped by --max_examples
(0 = the whole split): the real train split is ~2M stories, and materialising
it as tensors is ~8 GB, so an uncapped run would OOM both Kaggle (~13 GB) and a
laptop. Only the cap's worth of examples is ever held.

Checkpointing mirrors pretrain: every --save_every steps, every
--save_every_min minutes, and once at the end. --resume latest restores
model+optimizer+scaler+RNG and derives the epoch/batch position from the step
alone (batch order is a pure function of (seed, epoch)), so a killed run
resumes bit-identically. The saved step is the index of the last COMPLETED
update -- load_checkpoint returns step + 1, i.e. the next update to run.
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
from train.instruct import (build_sft_example, collate, iter_shard_texts,
                            synth_prompt)
from train.optim import build_param_groups
from train.schedule import lr_at

VAL_CAP = 64  # response_loss only ever reads val_ex[:64]; never load more


def _example(tok, text, context_len, idx):
    """(example, usable) for one story; usable=False if too thin to prompt."""
    sp = synth_prompt(text, idx)
    if sp is None:
        return None, False
    prompt, kw = sp
    inp, labels = build_sft_example(tok, prompt, text, context_len)
    return {"prompt": prompt, "keywords": kw,
            "input": inp, "labels": labels}, True


def build_examples(tok, texts, context_len, offset=0):
    """(examples, skipped) from an in-memory list of texts. Unit-test helper.

    Training goes through collect_examples instead, which streams. Both call
    _example, so they cannot drift apart.
    """
    ex, skipped = [], 0
    for i, t in enumerate(texts):
        e, usable = _example(tok, t, context_len, offset + i)
        if usable:
            ex.append(e)
        else:
            skipped += 1
    return ex, skipped


def collect_examples(tok, out_dir, split, context_len, max_examples=0,
                     offset=0):
    """(examples, skipped, stories_seen) from a split, streamed not materialised.

    Stops the instant max_examples are built (0 = no cap), so cost tracks the
    cap rather than the size of the split. The story index keeps counting up to
    the stop so prompt-word selection stays deterministic under the cap.
    """
    ex, skipped, seen = [], 0, 0
    for text in iter_shard_texts(tok, out_dir, split):
        if max_examples and len(ex) >= max_examples:
            break
        e, usable = _example(tok, text, context_len, offset + seen)
        seen += 1
        if usable:
            ex.append(e)
        else:
            skipped += 1
    return ex, skipped, seen


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
    ap.add_argument("--max_examples", type=int, default=50000,
                    help="cap on examples per split; 0 = no cap")
    ap.add_argument("--save_every", type=int, default=0,
                    help="also save every N steps; 0 = time-based only")
    ap.add_argument("--save_every_min", type=float, default=10.0,
                    help="also save if this many minutes since the last save")
    ap.add_argument("--max_hours", type=float, default=0.0,
                    help="clean save+exit after this long; 0 = no limit")
    ap.add_argument("--resume", default="latest",
                    help="'latest', 'none', or a ckpt path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t_start = time.time()  # --max_hours bounds data loading + training together

    cfg = load_config(args.config)
    mcfg = cfg["model"]
    ctx = cfg["train"]["context_len"]
    tok_path = cfg["data"]["tokenizer_path"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = BPETokenizer.load(tok_path)
    out_dir = cfg["data"]["out_dir"]
    train_ex, skip_tr, n_stories = collect_examples(
        tok, out_dir, "train", ctx, args.max_examples)
    # val is only ever consumed as val_ex[:64], so cap at VAL_CAP: validation
    # cost stays fixed even if --max_examples is raised or disabled.
    val_ex, skip_va, _ = collect_examples(
        tok, out_dir, "validation", ctx,
        min(args.max_examples, VAL_CAP) if args.max_examples else VAL_CAP,
        offset=n_stories)
    print(f"[sft] train={len(train_ex)} (skip {skip_tr}) val={len(val_ex)} "
          f"(skip {skip_va}) max_examples={args.max_examples or 'none'}",
          flush=True)
    if not train_ex:
        raise SystemExit("[sft] no usable train examples; nothing to train on")

    model = GPT(mcfg).to(device)
    model.load_state_dict(torch.load(args.init_weights, map_location=device,
                                     weights_only=True))
    opt = torch.optim.AdamW(build_param_groups(model, cfg["train"]["weight_decay"]),
                            lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(device, enabled=False)

    steps_per_epoch = max(len(train_ex) // args.batch_seqs, 1)
    max_steps = args.epochs * steps_per_epoch
    warmup = max(max_steps // 20, 1)
    keep = cfg["checkpoint"]["keep_last_n"]

    os.makedirs(args.ckpt_dir, exist_ok=True)
    tok_hash = ckpt_mod.file_hash(tok_path)
    full_cfg = {"model": mcfg, "seed": args.seed}

    def save(done_step, why):
        p = ckpt_mod.save_checkpoint(args.ckpt_dir, model, opt, scaler,
                                     done_step, full_cfg, tok_hash, keep)
        print(f"[sft] {why} -> {os.path.basename(p)} "
              f"(updates {done_step + 1}/{max_steps})", flush=True)

    start_step = 0
    if args.resume != "none":
        path = (ckpt_mod.latest_checkpoint(args.ckpt_dir)
                if args.resume == "latest" else args.resume)
        if path:
            # returns last_completed + 1 == the next update to run
            start_step = ckpt_mod.load_checkpoint(path, model, opt, scaler,
                                                  tok_path, mcfg)
            print(f"[sft] resumed {path} -> next step {start_step}", flush=True)
        elif args.resume == "latest":
            print("[sft] --resume latest: no checkpoint yet; starting fresh",
                  flush=True)
        else:
            raise SystemExit(f"[sft] --resume {path!r}: no such checkpoint")

    if start_step >= max_steps:
        print(f"[sft] already complete ({start_step}/{max_steps} updates); "
              "nothing to do", flush=True)
        return 0

    metrics_f = open(os.path.join(args.ckpt_dir, "metrics_sft.jsonl"), "a",
                     encoding="utf-8")
    last_save, last_saved_step = time.time(), None
    try:
        cur_epoch, order = -1, None
        for step in range(start_step, max_steps):
            epoch, b = divmod(step, steps_per_epoch)
            if epoch != cur_epoch:
                cur_epoch = epoch
                order = np.random.default_rng(
                    np.random.SeedSequence([args.seed, epoch])
                ).permutation(len(train_ex))
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
            timed_out = (time.time() - last_save) / 60.0 >= args.save_every_min
            periodic = args.save_every and (step + 1) % args.save_every == 0
            if (timed_out or periodic) and step != last_saved_step:
                save(step, f"periodic step={step}")
                last_save, last_saved_step = time.time(), step
            # Checked at the END of the body, where `step` is the update just
            # completed -- so the saved tag resumes at step + 1 and nothing is
            # skipped (checking at the top would save the not-yet-run index).
            if args.max_hours and (time.time() - t_start) > args.max_hours * 3600:
                if step != last_saved_step:
                    save(step, f"max_hours at step={step}")
                print(f"[sft] max_hours {args.max_hours:g}h reached; clean exit "
                      f"at step {step} ({step + 1}/{max_steps} updates)",
                      flush=True)
                return 0
        if step != last_saved_step:
            save(step, "final")
        else:
            print(f"[sft] final state already saved at step {step}", flush=True)
    finally:
        metrics_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
