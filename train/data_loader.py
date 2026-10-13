"""Shard data loader. Batches are a deterministic function of (seed, step).

Why this design: Kaggle sessions die at any time, and any data cursor
(shuffled index list, file offset) is extra state that can desync from the
checkpoint. Here the whole data position collapses to the integer `step`:
re-seed a Generator with SeedSequence([seed, step]) and you redraw the exact
same batch. Resume needs nothing but step. (No Python hash(): hash() is
salted per process, so it would differ across runs. SeedSequence is stable.)

Tradeoff, stated plainly: sampling WITH replacement, weighted by shard
size, so windows can repeat/overlap across steps. "Epochs" are statistical.
Fine for LM training; determinism is what we need.

Each window sits INSIDE one shard (never spans a boundary): draw shard,
then a uniform offset in [0, tokens - (T+1)]. Windows may cross document
EOT boundaries inside the shard -- that is normal LM data.
"""
import json
import os

import numpy as np
import torch

VAL_TAG = 999999  # fixed SeedSequence tag for validation windows


class ShardLoader:
    def __init__(self, out_dir, split="train"):
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        self.meta = meta
        recs = [s for s in meta["shards"] if s["split"] == split]
        assert recs, f"no shards for split {split} in {out_dir}"
        self.maps = [np.memmap(os.path.join(out_dir, r["file"]),
                               dtype=np.uint16, mode="r") for r in recs]
        self.sizes = np.array([len(m) for m in self.maps])
        assert (self.sizes > 0).all()
        self.total = int(self.sizes.sum())
        self.probs = self.sizes / self.total

    def batch_ids(self, seed, step, batch_seqs, context_len):
        """(batch_seqs, context_len+1) long tensor. Deterministic in (seed, step)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
        out = np.empty((batch_seqs, context_len + 1), dtype=np.int64)
        span = context_len + 1
        for i in range(batch_seqs):
            si = int(rng.choice(len(self.maps), p=self.probs))
            hi = int(self.sizes[si]) - span
            assert hi >= 0, "shard smaller than one window"
            off = int(rng.integers(0, hi + 1))
            out[i] = np.asarray(self.maps[si][off:off + span], dtype=np.int64)
        return torch.from_numpy(out).long()


def split_inputs_targets(batch):
    """(B, T+1) -> inputs (B, T), targets shifted by one (B, T)."""
    return batch[:, :-1].contiguous(), batch[:, 1:].contiguous()


class FixedVal:
    """Materialized once: same windows every eval, every run, every resume.

    Drawn with SeedSequence([seed, VAL_TAG]), independent of training steps,
    so val loss is comparable across checkpoints.
    """

    def __init__(self, loader, seed, n_windows, context_len):
        rng = np.random.default_rng(np.random.SeedSequence([seed, VAL_TAG]))
        wins = []
        span = context_len + 1
        for _ in range(n_windows):
            si = int(rng.choice(len(loader.maps), p=loader.probs))
            hi = int(loader.sizes[si]) - span
            assert hi >= 0, "val shard smaller than one window"
            off = int(rng.integers(0, hi + 1))
            wins.append(np.asarray(loader.maps[si][off:off + span],
                                   dtype=np.int64))
        self.batch = torch.from_numpy(np.stack(wins)).long()
