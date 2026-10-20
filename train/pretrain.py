"""Pretraining loop: AdamW, cosine+warmup, grad accumulation, fp16+scaler.

Usage:
  python -m train.pretrain --config configs/base_30m.yaml --resume latest
  python -m train.pretrain --config configs/debug_cpu.yaml --resume none --max_steps 20

Data position = integer step only (see train/data_loader.py): batches are a
pure function of (seed, step), so resume restores the data stream exactly.

Checkpointing: every save_every steps AND every save_every_min minutes,
atomic tmp+rename, keep_last_n rotation. --max_hours saves and exits 0
before Kaggle kills the session. NaN/inf loss parks under ckpt_nan_* and
exits 1 WITHOUT touching the last good checkpoint.

Metrics go to stdout AND metrics.jsonl (one JSON object per line); W&B only
if logging.use_wandb is true (imported lazily, warns and continues if missing).
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.config import load_config
from model.gpt import GPT, count_params
from tokenizer.bpe import BPETokenizer
from train import checkpoint as ckpt_mod
from train.data_loader import FixedVal, ShardLoader, split_inputs_targets
from train.optim import build_param_groups
from train.sample import sample_continuations
from train.schedule import lr_at

PROMPTS = ["Once upon a time,", "One day, Lily"]


def set_nested(cfg, dotted, value):
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="latest",
                    help="'latest', 'none', or a ckpt path")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--micro_batch_seqs", type=int, default=None)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--max_hours", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        set_nested(cfg, "train.max_steps", args.max_steps)
    if args.micro_batch_seqs is not None:
        set_nested(cfg, "train.micro_batch_seqs", args.micro_batch_seqs)
    if args.max_hours is not None:
        set_nested(cfg, "checkpoint.max_hours", args.max_hours)
    if args.lr is not None:
        set_nested(cfg, "train.lr", args.lr)
    ckpt_dir = args.ckpt_dir or cfg["checkpoint"]["dir"]

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda" and cfg["train"]["use_fp16"]
    if device == "cuda" and cfg["train"]["use_fp16"] is False:
        print("[pretrain] WARN: fp16 off on CUDA; slow but exact.", flush=True)

    tok_path = cfg["data"]["tokenizer_path"]
    tok = BPETokenizer.load(tok_path)
    tok_hash = ckpt_mod.file_hash(tok_path)

    out_dir = cfg["data"]["out_dir"]
    train_loader = ShardLoader(out_dir, "train")
    val_loader = ShardLoader(out_dir, "validation")
    mcfg = cfg["model"]
    assert mcfg["vocab_size"] == train_loader.meta.get("vocab_size", mcfg["vocab_size"]), \
        "model vocab != shard vocab"
    assert mcfg["vocab_size"] == tok.vocab_size, "model vocab != tokenizer vocab"

    ctx = cfg["train"]["context_len"]
    assert ctx <= mcfg["context_len"], "train ctx exceeds model ctx"
    total_seqs = cfg["train"]["batch_tokens"] // ctx
    micro = cfg["train"]["micro_batch_seqs"]
    assert cfg["train"]["batch_tokens"] % ctx == 0, "batch_tokens not divisible by ctx"
    assert total_seqs % micro == 0, "seqs/step not divisible by micro_batch_seqs"
    accum = total_seqs // micro

    model = GPT(mcfg).to(device)
    print(f"[pretrain] params={count_params(model)} device={device} "
          f"amp={use_amp} accum={accum}", flush=True)
    opt = torch.optim.AdamW(build_param_groups(model, cfg["train"]["weight_decay"]),
                            lr=cfg["train"]["lr"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and
                                                cfg["train"]["use_scaler"]))

    step = 0
    if args.resume != "none":
        path = ckpt_mod.latest_checkpoint(ckpt_dir) if args.resume == "latest" \
            else args.resume
        if path is None:
            print("[pretrain] no checkpoint found; training from scratch", flush=True)
        else:
            step = ckpt_mod.load_checkpoint(path, model, opt, scaler,
                                            tok_path, mcfg)
            print(f"[pretrain] resumed {path} -> step {step}", flush=True)

    max_steps = cfg["train"]["max_steps"]
    peak, warmup = cfg["train"]["lr"], cfg["train"]["warmup_steps"]
    min_ratio = cfg["train"]["min_lr_ratio"]
    fixed_val = FixedVal(val_loader, seed, cfg["train"]["val_windows"], ctx)

    os.makedirs(ckpt_dir, exist_ok=True)
    metrics_f = open(os.path.join(ckpt_dir, "metrics.jsonl"), "a", encoding="utf-8")
    samples_path = os.path.join(ckpt_dir, "samples.txt")

    use_wandb = cfg["logging"].get("use_wandb", False)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=cfg["logging"].get("project", "minigpt-stack"),
                       name=cfg["run_name"], config=cfg)
        except Exception as e:  # optional dep: never kill a run over logging
            print(f"[pretrain] wandb disabled ({e})", flush=True)
            use_wandb = False

    max_hours = cfg["checkpoint"].get("max_hours")
    save_every = cfg["checkpoint"]["save_every"]
    save_min = cfg["checkpoint"].get("save_every_min", 0)
    t_start, last_ckpt_t, skip_count = time.time(), time.time(), 0

    def save(tag):
        p = ckpt_mod.save_checkpoint(ckpt_dir, model, opt, scaler, tag, cfg,
                                     tok_hash, cfg["checkpoint"]["keep_last_n"])
        print(f"[pretrain] saved {p}", flush=True)
        return p

    for step in range(step, max_steps):
        if max_hours and (time.time() - t_start) > max_hours * 3600:
            save(step)
            print("[pretrain] max_hours reached; clean exit", flush=True)
            metrics_f.close()
            return 0
        lr = lr_at(step, peak, warmup, max_steps, min_ratio)
        for g in opt.param_groups:
            g["lr"] = lr

        t0 = time.time()
        batch = train_loader.batch_ids(seed, step, total_seqs, ctx).to(device)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for mb in batch.chunk(accum):
            inp, tgt = split_inputs_targets(mb)
            with torch.autocast("cuda", torch.float16, enabled=use_amp):
                logits = model(inp).logits
                # Divide by accum so the summed micro-gradients equal the
                # full-batch mean gradient (verified in test_accum_equivalence).
                loss = F.cross_entropy(logits.reshape(-1, mcfg["vocab_size"]),
                                       tgt.reshape(-1)) / accum
            loss_sum += loss.detach()  # already scaled: sum == step mean
            scaler.scale(loss).backward()
        step_loss = float(loss_sum)
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        prev_scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if scaler.get_scale() < prev_scale:
            skip_count += 1  # fp16 overflow: scaler skipped this step
        dt = time.time() - t0

        if not math.isfinite(step_loss):
            bad = ckpt_mod.save_nan(ckpt_dir, model, opt, step)
            print(f"[pretrain] NON-FINITE loss at step {step}; parked at {bad}; "
                  "last good checkpoint untouched. aborting.", flush=True)
            metrics_f.close()
            return 1

        rec = {"step": step, "loss": step_loss, "lr": lr,
               "tok_s": cfg["train"]["batch_tokens"] / dt,
               "grad_norm": grad_norm, "scale": float(scaler.get_scale()),
               "skips": skip_count}
        if step % cfg["train"]["log_every"] == 0:
            if step % cfg["train"]["val_every"] == 0:
                model.eval()
                with torch.no_grad(), torch.autocast(
                        "cuda", torch.float16, enabled=use_amp):
                    vb = fixed_val.batch.to(device)
                    vi, vt = split_inputs_targets(vb)
                    vloss = F.cross_entropy(
                        model(vi).logits.reshape(-1, mcfg["vocab_size"]),
                        vt.reshape(-1)).item()
                model.train()
                rec["val"] = vloss
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            print(f"[pretrain] step={step} loss={step_loss:.3f} "
                  f"val={rec.get('val', float('nan')):.3f} lr={lr:.2e} "
                  f"tok/s={rec['tok_s']:.0f} gnorm={grad_norm:.2f} "
                  f"skips={skip_count}", flush=True)
            if use_wandb:
                wandb.log(rec, step=step)
        # Checkpoint BEFORE sampling. Sampling is optional garnish; a crash
        # there must never cost us the step's save (at step 2000 it cost us
        # both the save and the rest of the session).
        if step % save_every == 0 and step > 0:
            save(step)
            last_ckpt_t = time.time()
        elif save_min and (time.time() - last_ckpt_t) > save_min * 60:
            save(step)
            last_ckpt_t = time.time()
        if step % cfg["train"]["sample_every"] == 0 and step > 0:
            try:
                outs = sample_continuations(model, tok, PROMPTS, max_new=64,
                                            seed=seed + step, device=device)
            except Exception as e:
                print(f"[pretrain] WARNING: sampling failed at step {step}: "
                      f"{type(e).__name__}: {e}. training continues.",
                      flush=True)
            else:
                with open(samples_path, "a", encoding="utf-8") as f:
                    f.write(f"--- step {step} ---\n")
                    for o in outs:
                        f.write(o + "\n\n")

    save(max_steps)
    metrics_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
