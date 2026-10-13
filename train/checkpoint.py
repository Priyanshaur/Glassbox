"""Atomic checkpoints with full resume state.

Saved per checkpoint: model, optimizer, GradScaler, integer step (the LR
schedule is a pure function of step, so no scheduler blob), torch/CUDA/
numpy/python RNG states, config snapshot, tokenizer-file hash, arch record.

- Atomic: write to tmp in the same dir, then os.replace (never half-files).
- Rotation: keep newest keep_last_n ckpt_step{N}.pt (+ matching weights).
- Weights-only copy weights_step{N}.pt alongside, for later phases.
- NaN states go to ckpt_nan_step{N}.pt via save_nan(): never touches the
  last good checkpoint, excluded from rotation and from latest_checkpoint.
- Refuses to load on tokenizer-hash or arch mismatch (silent shape bugs
  are worse than a loud error).
"""
import glob
import hashlib
import os
import random

import numpy as np
import torch

ARCH_KEYS = ("vocab_size", "n_layers", "d_model", "n_heads", "ffn_hidden",
             "context_len")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arch_of(model_cfg):
    return {k: model_cfg[k] for k in ARCH_KEYS}


def ckpt_path(ckpt_dir, step):
    return os.path.join(ckpt_dir, f"ckpt_step{step}.pt")


def weights_path(ckpt_dir, step):
    return os.path.join(ckpt_dir, f"weights_step{step}.pt")


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate()}


def _rng_restore(s):
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def _rotate(ckpt_dir, keep_last_n):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt")),
                   key=_step_of)
    for p in paths[:-keep_last_n]:
        try:
            os.remove(p)
        except OSError:
            pass
        w = p.replace("ckpt_step", "weights_step")
        try:
            os.remove(w)
        except OSError:
            pass


def _step_of(path):
    base = os.path.basename(path)
    return int(base[len("ckpt_step"):-len(".pt")])


def save_checkpoint(ckpt_dir, model, optim, scaler, step, config, tok_hash,
                    keep_last_n):
    """Save full state + weights-only copy. Returns full-checkpoint path."""
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {"step": step,
             "model": model.state_dict(),
             "optim": optim.state_dict(),
             "scaler": scaler.state_dict(),
             "rng": _rng_state(),
             "config": config,
             "tok_hash": tok_hash,
             "arch": arch_of(config["model"])}
    path = ckpt_path(ckpt_dir, step)
    _atomic_save(state, path)
    _atomic_save(model.state_dict(), weights_path(ckpt_dir, step))
    _rotate(ckpt_dir, keep_last_n)
    return path


def save_nan(ckpt_dir, model, optim, step):
    """Park a NaN/inf state under a separate name. Never rotates good ckpts."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_nan_step{step}.pt")
    _atomic_save({"step": step, "model": model.state_dict(),
                  "optim": optim.state_dict()}, path)
    return path


def latest_checkpoint(ckpt_dir):
    paths = [p for p in glob.glob(os.path.join(ckpt_dir, "ckpt_step*.pt"))
             if "nan" not in os.path.basename(p)]
    if not paths:
        return None
    return max(paths, key=_step_of)


def load_checkpoint(path, model, optim, scaler, tok_path, model_cfg):
    """Load + verify + restore RNG. Returns next step (ckpt step + 1)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["tok_hash"] != file_hash(tok_path):
        raise RuntimeError(
            f"tokenizer mismatch: {path} was trained with a different "
            "tokenizer file. Refusing to resume.")
    if ckpt["arch"] != arch_of(model_cfg):
        raise RuntimeError(
            f"arch mismatch: checkpoint {ckpt['arch']} vs config "
            f"{arch_of(model_cfg)}. Refusing to resume.")
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    scaler.load_state_dict(ckpt["scaler"])
    _rng_restore(ckpt["rng"])
    return ckpt["step"] + 1
