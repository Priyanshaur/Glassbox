"""SAE training loop, resumable exactly at buffer-epoch boundaries.

One "buffer epoch" = fill a buffer of `buffer_tokens` activations from train-split
windows (a pure function of (seed, epoch)), then take one pass over a fixed permutation
of it in minibatches of `batch_tokens`. Checkpoints are written at epoch boundaries
and hold the SAE (including dead-latent counters), the Adam state, the epoch, the step
and the activation scale, so a resumed run is identical to an uninterrupted one.
The LR is a pure function of the step (warmup + cosine), like pretraining.

Usage (GPU run, later): python -m interp.train_sae --config configs/base_30m.yaml \
    --weights <base weights> --layer 4 --features 4096 --k 32 --epochs 190 --out_dir ...
"""
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import torch

from common.config import load_config
from interp import activations as A
from interp.sae import TopKSAE, save_sae
from model.gpt import GPT
from train.schedule import lr_at


@dataclass
class TrainConfig:
    layer: int = 4
    n_features: int = 4096
    k: int = 32
    seed: int = 0
    lr: float = 4e-4
    warmup_steps: int = 200
    batch_tokens: int = 4096
    buffer_tokens: int = 524_288
    epochs: int = 190                   # buffer epochs; tokens seen = epochs * buffer_tokens
    ctx: int = 512
    aux_coef: float = 1 / 32
    aux_k: int = 64
    dead_steps: int = 1000
    norm_windows: int = 64
    chunk: int = 32
    log_every: int = 50                 # steps
    eval_every: int = 5                 # buffer epochs
    ckpt_every: int = 5                 # buffer epochs
    eval_windows: int = 64
    max_hours: float = 0.0              # 0 = off; checked at epoch boundaries
    stop_after_epoch: int = 0           # 0 = off; stop cleanly after this many epochs (probes, tests)

    @property
    def steps_per_epoch(self):
        return self.buffer_tokens // self.batch_tokens


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


@torch.no_grad()
def quick_eval(sae, val_acts):
    """FVU, mean L0 and the fraction of latents that fired at least once on `val_acts`."""
    x_hat, acts, _ = sae(val_acts)
    err = ((val_acts - x_hat) ** 2).sum()
    var = ((val_acts - val_acts.mean(0)) ** 2).sum()
    return {"fvu": float(err / var), "l0": float((acts > 0).sum(-1).float().mean()),
            "alive_frac": float((acts > 0).any(0).float().mean())}


def train(model, train_loader, val_loader, cfg: TrainConfig, out_dir, device="cpu", log=print,
          meta_extra=None):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    metrics_f = open(os.path.join(out_dir, "metrics_sae.jsonl"), "a", encoding="utf-8")
    total_steps = cfg.epochs * cfg.steps_per_epoch
    assert cfg.warmup_steps < total_steps, "warmup must be shorter than the run"
    model = model.to(device).eval()
    d = model.cfg["d_model"]

    sae = TopKSAE(d, cfg.n_features, cfg.k, seed=cfg.seed).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    epoch0, step, scale, first_buffer = 0, 0, None, None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sae.load_state_dict(ck["sae"])
        opt.load_state_dict(ck["opt"])
        epoch0, step, scale = ck["epoch"], ck["step"], ck["scale"]
        log(f"[sae] resumed epoch {epoch0} step {step}")
    else:
        scale = A.estimate_norm_scale(model, train_loader, cfg.layer, cfg.seed, cfg.norm_windows,
                                      cfg.ctx, cfg.chunk, device)
        first_buffer = A.fill_buffer(model, train_loader, cfg.layer, cfg.seed, 0, cfg.buffer_tokens,
                                     cfg.ctx, scale, cfg.chunk, device)
        sae.init_b_dec(first_buffer.mean(0))
        log(f"[sae] fresh run: act_scale {scale:.5f}, total steps {total_steps}")
    val_acts, _ = A.eval_activations(model, val_loader, cfg.layer, cfg.eval_windows, cfg.ctx, scale,
                                     chunk=cfg.chunk, device=device)

    def save_all(epoch_done):
        _atomic_save({"sae": sae.state_dict(), "opt": opt.state_dict(), "epoch": epoch_done,
                      "step": step, "scale": scale, "cfg": asdict(cfg)}, ckpt_path)
        save_sae(os.path.join(out_dir, "sae_latest.pt"), sae,
                 {"layer": cfg.layer, "act_scale": scale, "epochs_done": epoch_done, "steps": step,
                  "train_cfg": asdict(cfg), **(meta_extra or {})})

    t_start, reason = time.time(), "epochs"
    for epoch in range(epoch0, cfg.epochs):
        buf = first_buffer if (epoch == 0 and first_buffer is not None) else A.fill_buffer(
            model, train_loader, cfg.layer, cfg.seed, epoch, cfg.buffer_tokens, cfg.ctx, scale,
            cfg.chunk, device)
        last = {}
        for batch in A.epoch_batches(buf, cfg.batch_tokens, cfg.seed, epoch):
            lr = lr_at(step, cfg.lr, cfg.warmup_steps, total_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss, parts = sae.loss(batch, cfg.aux_coef, cfg.aux_k, cfg.dead_steps)
            loss.backward()
            sae.project_decoder_grad()
            opt.step()
            sae.normalize_decoder()
            step += 1
            last = parts
            if step % cfg.log_every == 0:
                rec = {"step": step, "epoch": epoch, "lr": lr, "loss": float(loss.detach()), **parts}
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
        first_buffer = None
        done = epoch + 1
        if done % cfg.eval_every == 0 or done == cfg.epochs:
            ev = quick_eval(sae, val_acts)
            metrics_f.write(json.dumps({"step": step, "epoch": epoch, "eval": ev,
                                        "n_dead_now": last.get("n_dead")}) + "\n")
            metrics_f.flush()
            log(f"[sae] epoch {done}/{cfg.epochs} step {step} val fvu {ev['fvu']:.4f} l0 {ev['l0']:.1f} "
                f"alive {ev['alive_frac']:.3f}")
        stop_time = cfg.max_hours and (time.time() - t_start) / 3600 >= cfg.max_hours
        stop_early = cfg.stop_after_epoch and done >= cfg.stop_after_epoch and done < cfg.epochs
        if done % cfg.ckpt_every == 0 or done == cfg.epochs or stop_time or stop_early:
            save_all(done)
        if stop_time or stop_early:
            reason = "max_hours" if stop_time else "stop_after_epoch"
            break
    metrics_f.write(json.dumps({"terminal": reason, "epochs_done": epoch0 if epoch0 >= cfg.epochs else done,
                                "steps": step}) + "\n")
    metrics_f.close()
    return {"sae": sae, "scale": scale, "steps": step, "terminal": reason}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True, help="state dict of the model the SAE reads")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cpu")
    d = TrainConfig()
    for name, default in asdict(d).items():
        ap.add_argument("--" + name, type=type(default), default=default)
    a = ap.parse_args(argv)
    cfg = TrainConfig(**{k: getattr(a, k) for k in asdict(d)})
    c = load_config(a.config)
    model = GPT(c["model"])
    model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    from train.data_loader import ShardLoader
    tr = ShardLoader(c["data"]["out_dir"], "train")
    va = ShardLoader(c["data"]["out_dir"], "validation")
    res = train(model, tr, va, cfg, a.out_dir, a.device,
                meta_extra={"weights_sha256": file_sha256(a.weights),
                            "tokenizer_sha256": file_sha256(c["data"]["tokenizer_path"])})
    print("[sae] done:", res["terminal"], "steps", res["steps"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
