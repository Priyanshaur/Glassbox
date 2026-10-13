"""Phase 3 tests: schedule, param groups, loader, accumulation, checkpoints.

All CPU, all fast: tiny 1-layer model, small synthetic uint16 shards in tmp.
Real-shard smoke (debug config on data/*.bin) is a manual step, not a test.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from model.gpt import GPT
from train import checkpoint as ckpt_mod
from train.data_loader import FixedVal, ShardLoader, split_inputs_targets
from train.optim import build_param_groups
from train.schedule import lr_at

TINY = {"vocab_size": 128, "n_layers": 1, "d_model": 32, "n_heads": 4,
        "ffn_hidden": 64, "context_len": 32, "weight_tying": True,
        "dropout": 0.0}


def make_shards(path, seed=0, vocab=128):
    """Two train shards + one val shard of random uint16 tokens + meta.json."""
    rng = np.random.default_rng(seed)
    sizes = {"train_00000": 5000, "train_00001": 3000, "validation_00000": 1000}
    shards = []
    for name, n in sizes.items():
        split = "train" if name.startswith("train") else "validation"
        arr = rng.integers(0, vocab, size=n).astype(np.uint16)
        arr.tofile(os.path.join(path, name + ".bin"))
        shards.append({"split": split, "index": len(
            [s for s in shards if s["split"] == split]),
            "file": name + ".bin", "tokens": n})
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"shards": shards, "vocab_size": vocab}, f)
    return path


def write_tok(path, content=b"tok-a"):
    p = os.path.join(path, "tok.json")
    with open(p, "wb") as f:
        f.write(content)
    return p


# -- schedule --
def test_warmup_cosine_values():
    peak, warm, mx, ratio = 0.1, 10, 100, 0.1
    assert lr_at(0, peak, warm, mx, ratio) == peak / warm  # ramps from ~0
    assert lr_at(9, peak, warm, mx, ratio) == peak        # continuous at joint
    assert lr_at(10, peak, warm, mx, ratio) == peak
    assert lr_at(50, peak, warm, mx, ratio) < peak        # decaying after
    assert lr_at(60, peak, warm, mx, ratio) < lr_at(50, peak, warm, mx, ratio)
    assert abs(lr_at(99, peak, warm, mx, ratio) - peak * ratio) < 1e-3
    assert max(lr_at(s, peak, warm, mx, ratio) for s in range(mx)) <= peak


# -- param groups --
def test_weight_decay_groups():
    m = GPT(TINY)
    d, nd = build_param_groups(m, 0.1)
    dn = {n for n, p in m.named_parameters() if p in set(d["params"])}
    nn = {n for n, p in m.named_parameters() if p in set(nd["params"])}
    assert any("w_q" in n for n in dn) and any("w_down" in n for n in dn)
    assert any("wte" in n for n in nn) and "ln_f.w" in nn
    assert any("attn_norm" in n for n in nn)
    assert d["weight_decay"] == 0.1 and nd["weight_decay"] == 0.0
    assert len(dn | nn) == len(list(m.parameters()))  # every param grouped


# -- loader --
def test_loader_shift_bounds_determinism(tmp_path):
    d = make_shards(str(tmp_path))
    L = ShardLoader(d, "train")
    b = L.batch_ids(seed=7, step=3, batch_seqs=4, context_len=16)
    assert b.shape == (4, 17) and b.dtype == torch.long
    assert int(b.max()) < 128 and int(b.min()) >= 0  # in-bounds ids
    inp, tgt = split_inputs_targets(b)
    assert torch.equal(tgt[:, :-1], inp[:, 1:])  # targets = inputs shifted by one
    again = L.batch_ids(seed=7, step=3, batch_seqs=4, context_len=16)
    assert torch.equal(b, again)  # deterministic in (seed, step)
    other = L.batch_ids(seed=7, step=4, batch_seqs=4, context_len=16)
    assert not torch.equal(b, other)  # different steps differ


def test_fixed_val_stable(tmp_path):
    d = make_shards(str(tmp_path))
    L = ShardLoader(d, "validation")
    v1 = FixedVal(L, seed=7, n_windows=8, context_len=16).batch
    v2 = FixedVal(L, seed=7, n_windows=8, context_len=16).batch
    assert torch.equal(v1, v2) and v1.shape == (8, 17)


# -- accumulation equivalence --
def test_grad_accumulation_equivalence():
    torch.manual_seed(0)
    ids = torch.randint(0, 128, (4, 17))
    inp, tgt = split_inputs_targets(ids)
    ma = GPT(TINY)
    mb = GPT(TINY)
    mb.load_state_dict(ma.state_dict())
    oa = torch.optim.AdamW(ma.parameters(), lr=1e-3)
    ob = torch.optim.AdamW(mb.parameters(), lr=1e-3)
    oa.zero_grad()
    F.cross_entropy(ma(inp).logits.reshape(-1, 128), tgt.reshape(-1)).backward()
    torch.nn.utils.clip_grad_norm_(ma.parameters(), 1.0)
    oa.step()
    ob.zero_grad()
    for half in (slice(0, 2), slice(2, 4)):  # 2 micro-batches, loss halved
        (F.cross_entropy(mb(inp[half]).logits.reshape(-1, 128),
                         tgt[half].reshape(-1)) / 2).backward()
    torch.nn.utils.clip_grad_norm_(mb.parameters(), 1.0)
    ob.step()
    for pa, pb in zip(ma.parameters(), mb.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


# -- checkpoints --
def ckpt_setup(tmp_path):
    m = GPT(TINY)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sc = torch.amp.GradScaler("cuda", enabled=False)
    cfg = {"model": dict(TINY), "seed": 0}
    tok = write_tok(str(tmp_path))
    return m, opt, sc, cfg, tok, ckpt_mod.file_hash(tok)


def test_atomic_write_and_rotation(tmp_path):
    m, opt, sc, cfg, tok, th = ckpt_setup(tmp_path)
    cd = str(tmp_path / "ck")
    for s in (1, 2, 3):
        ckpt_mod.save_checkpoint(cd, m, opt, sc, s, cfg, th, keep_last_n=2)
    left = sorted(os.listdir(cd))
    assert "ckpt_step3.pt" in left and "ckpt_step2.pt" in left
    assert "ckpt_step1.pt" not in left and "weights_step1.pt" not in left
    assert "weights_step3.pt" in left  # weights-only copies rotate too
    assert not [f for f in left if f.endswith(".tmp")]  # no tmp leftovers


def test_resume_refuses_mismatch(tmp_path):
    m, opt, sc, cfg, tok, th = ckpt_setup(tmp_path)
    cd = str(tmp_path / "ck")
    ckpt_mod.save_checkpoint(cd, m, opt, sc, 5, cfg, th, keep_last_n=2)
    path = ckpt_mod.latest_checkpoint(cd)
    other_tok = write_tok(str(tmp_path), b"tok-B")
    try:
        ckpt_mod.load_checkpoint(path, m, opt, sc, other_tok, TINY)
    except RuntimeError:
        pass
    else:
        raise AssertionError("tokenizer mismatch not refused")
    bad_arch = dict(TINY, n_layers=2)
    try:
        ckpt_mod.load_checkpoint(path, m, opt, sc, tok, bad_arch)
    except RuntimeError:
        pass
    else:
        raise AssertionError("arch mismatch not refused")


def tiny_step(m, opt, loader, seed, step, lr=1e-3):
    for g in opt.param_groups:
        g["lr"] = lr
    b = loader.batch_ids(seed, step, 4, 16)
    inp, tgt = split_inputs_targets(b)
    opt.zero_grad()
    loss = F.cross_entropy(m(inp).logits.reshape(-1, 128), tgt.reshape(-1))
    loss.backward()
    opt.step()
    return loss.item()


def test_resume_matches_straight_run(tmp_path):
    """Acceptance: 2N straight == N, save, resume, N more (tight tol, CPU)."""
    d = make_shards(str(tmp_path))
    L = ShardLoader(d, "train")
    torch.manual_seed(0)
    ma = GPT(TINY)
    oa = torch.optim.AdamW(ma.parameters(), lr=1e-3)
    straight = [tiny_step(ma, oa, L, 0, s) for s in range(8)]
    torch.manual_seed(0)
    mb = GPT(TINY)
    ob = torch.optim.AdamW(mb.parameters(), lr=1e-3)
    sc = torch.amp.GradScaler("cuda", enabled=False)
    cfg = {"model": dict(TINY), "seed": 0}
    tok = write_tok(str(tmp_path))
    cd = str(tmp_path / "ck")
    resumed = [tiny_step(mb, ob, L, 0, s) for s in range(4)]
    ckpt_mod.save_checkpoint(cd, mb, ob, sc, 3, cfg,
                             ckpt_mod.file_hash(tok), keep_last_n=2)
    mc = GPT(TINY)
    oc = torch.optim.AdamW(mc.parameters(), lr=1e-3)
    sc2 = torch.amp.GradScaler("cuda", enabled=False)
    nxt = ckpt_mod.load_checkpoint(ckpt_mod.latest_checkpoint(cd), mc, oc,
                                   sc2, tok, TINY)
    assert nxt == 4  # data position = step, restored exactly
    resumed += [tiny_step(mc, oc, L, 0, s) for s in range(4, 8)]
    assert len(straight) == len(resumed) == 8
    for a, b in zip(straight, resumed):
        assert abs(a - b) < 1e-6, f"{a} vs {b}"
