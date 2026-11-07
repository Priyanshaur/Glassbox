"""Phase 6 / B: the feature scan agrees with a brute-force recomputation."""
import json

import numpy as np
import torch

from common.config import load_config
from interp import activations as A
from interp import feature_scan as S
from interp.sae import TopKSAE
from model.gpt import GPT
from train.data_loader import ShardLoader

CTX = 32


def _setup(tmp_path):
    cfg = load_config("configs/debug_cpu.yaml")
    torch.manual_seed(0)
    model = GPT(cfg["model"]).eval()
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 250, 4000).astype(np.uint16)
    toks[::50] = S.EOT_ID
    (tmp_path / "train_00000.bin").write_bytes(toks.tobytes())
    json.dump({"shards": [{"split": "train", "index": 0, "file": "train_00000.bin", "tokens": 4000}]},
              open(tmp_path / "meta.json", "w"))
    loader = ShardLoader(str(tmp_path), "train")
    d = cfg["model"]["d_model"]
    sae = TopKSAE(d, 96, 4, seed=1)
    return model, sae, loader, 1


def test_scan_matches_brute_force(tmp_path):
    model, sae, loader, layer = _setup(tmp_path)
    starts = S.window_starts(loader, 20, CTX, sequential=True)
    res = S.scan(model, sae, 1.7, loader, layer, starts, CTX, chunk=7)
    ids = torch.from_numpy(np.stack([np.asarray(loader.maps[s][a:a + CTX], dtype=np.int64) for s, a in starts]))
    x = A.record_windows(model, ids, layer, 7)[:, 1:] * 1.7
    acts = sae.encode(x.reshape(-1, x.shape[-1]))[0].reshape(20, CTX - 1, 96)
    assert torch.equal(res["counts"], (acts > 0).sum((0, 1)))
    assert res["tokens"] == 20 * (CTX - 1)
    assert int(res["hist"].sum()) == int((acts > 0).sum())
    wmax = acts.max(1).values                                              # (20, F): per-window maxima
    for f in (0, 5, 40):
        want = wmax[:, f].sort(descending=True).values[:S.POOL]
        got = res["pool_vals"][f][:20]
        assert torch.allclose(got.sort(descending=True).values, want, rtol=1e-5, atol=1e-7)
    # the recorded location really is where that activation occurs (position 0 skip undone)
    f = 3
    j = int(res["pool_vals"][f].argmax())
    shard, start, pos = (int(t) for t in res["pool_loc"][f, j])
    w = starts.index((shard, start))
    assert pos >= 1 and abs(float(acts[w, pos - 1, f]) - float(res["pool_vals"][f, j])) < 1e-6


def test_top_contexts_one_per_document_and_percentile(tmp_path):
    model, sae, loader, layer = _setup(tmp_path)
    starts = S.window_starts(loader, 100, CTX, sequential=True)
    res = S.scan(model, sae, 1.0, loader, layer, starts, CTX)
    f = int(res["counts"].argmax())
    ctxs = S.top_contexts(res, loader.maps, f, n=24)
    docs = [c["doc"] for c in ctxs]
    assert len(docs) == len(set(docs)) and len(ctxs) >= 2
    assert all(c["act"] > 0 and 0 <= c["at"] < len(c["tokens"]) for c in ctxs)
    assert [c["act"] for c in ctxs] == sorted((c["act"] for c in ctxs), reverse=True)
    p50, p90 = (S.percentile_from_hist(res["hist"][f], q) for q in (0.5, 0.9))
    assert 0 < p50 <= p90


def test_scan_in_blocks_equals_one_pass(tmp_path):
    model, sae, loader, layer = _setup(tmp_path)
    starts = S.window_starts(loader, 30, CTX, sequential=True)
    whole = S.scan(model, sae, 1.0, loader, layer, starts, CTX, chunk=8)
    st = S.scan(model, sae, 1.0, loader, layer, starts[:13], CTX, chunk=8)
    st = S.scan(model, sae, 1.0, loader, layer, starts[13:], CTX, chunk=8, state=st)
    for k in ("counts", "hist", "tokens"):
        assert torch.equal(torch.as_tensor(whole[k]), torch.as_tensor(st[k]))
    assert torch.allclose(whole["pool_vals"], st["pool_vals"], rtol=1e-4, atol=1e-6)   # batch sizes differ: ~1e-5 rounding
