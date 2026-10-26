"""Phase 5 inference tests: static KV cache (M1), generation engine (M2), quantization (M3).

Tolerance for logits comparisons: atol=1e-5, rtol=1e-5, the same as the existing
tests/test_model.py::test_kv_cache_equivalence. Tests that need the real SFT
checkpoint skip when it is not present (it is not in git).
"""
import os

import pytest
import torch

from common.config import load_config
from inference.kv_cache import StaticKVCache
from model.gpt import GPT

ATOL = RTOL = 1e-5
SFT = "checkpoints/glassbox-sft-final/sft_final_best.pt"
needs_sft = pytest.mark.skipif(not os.path.exists(SFT), reason="SFT checkpoint not present")


def debug_model(seed=4):
    torch.manual_seed(seed)
    return GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()


def sft_model():
    m = GPT(load_config("configs/base_30m.yaml")["model"])
    m.load_state_dict(torch.load(SFT, map_location="cpu", weights_only=True))
    return m.eval()


# ---------------------------------------------------------------- M1: static cache
def _incremental(m, ids, cache=None):
    """Token-by-token logits, with a StaticKVCache or the default tuple cache."""
    steps, past = [], None
    for t in range(ids.shape[1]):
        if cache is not None:
            o = m(ids[:, t:t + 1], past_kvs=cache.layers, pos_offset=cache.length)
        else:
            o = m(ids[:, t:t + 1], past_kvs=past, pos_offset=t)
            past = o.new_kvs
        steps.append(o.logits)
    return torch.cat(steps, dim=1)


def test_static_cache_matches_full_forward_and_tuple_cache():
    m = debug_model()
    ids = torch.randint(0, 8192, (2, 12))
    with torch.no_grad():
        full = m(ids).logits
        cat = _incremental(m, ids)
        static = _incremental(m, ids, StaticKVCache.from_model(m, 2, 12))
    assert torch.allclose(full, static, atol=ATOL, rtol=RTOL)
    assert torch.allclose(cat, static, atol=ATOL, rtol=RTOL)


def test_static_cache_chunked_prefill_matches_full_forward():
    """Prefill 5 then a 7-token chunk exercises the explicit-mask regime."""
    m = debug_model()
    ids = torch.randint(0, 8192, (1, 12))
    with torch.no_grad():
        full = m(ids).logits
        cache = StaticKVCache.from_model(m, 1, 12)
        o1 = m(ids[:, :5], past_kvs=cache.layers, pos_offset=0)
        o2 = m(ids[:, 5:], past_kvs=cache.layers, pos_offset=5)
    assert torch.allclose(full[:, :5], o1.logits, atol=ATOL, rtol=RTOL)
    assert torch.allclose(full[:, 5:], o2.logits, atol=ATOL, rtol=RTOL)
    assert cache.length == 12


def test_static_cache_does_not_reallocate():
    m = debug_model()
    cache = StaticKVCache.from_model(m, 1, 10)
    ptrs = [(l.k.data_ptr(), l.v.data_ptr()) for l in cache.layers]
    ids = torch.randint(0, 8192, (1, 10))
    with torch.no_grad():
        _incremental(m, ids, cache)
    assert [(l.k.data_ptr(), l.v.data_ptr()) for l in cache.layers] == ptrs
    assert cache.length == 10


def test_static_cache_overflow_reset_truncate_repeat():
    m = debug_model()
    cache = StaticKVCache.from_model(m, 1, 6)
    ids = torch.randint(0, 8192, (1, 8))
    with torch.no_grad():
        m(ids[:, :6], past_kvs=cache.layers, pos_offset=0)
        with pytest.raises(ValueError, match="overflow"):
            m(ids[:, 6:7], past_kvs=cache.layers, pos_offset=6)
        cache.truncate(4)
        assert cache.length == 4
        with pytest.raises(ValueError):
            cache.truncate(5)
        # after truncating, decoding the same token again reproduces the logits
        ref = m(ids[:, :5]).logits[:, 4]
        again = m(ids[:, 4:5], past_kvs=cache.layers, pos_offset=4).logits[:, 0]
        assert torch.allclose(ref, again, atol=ATOL, rtol=RTOL)
        cache.reset()
        assert cache.length == 0
        # repeat_batch: prefill once, fan out to 3 rows, rows stay identical to a fresh run
        m(ids[:, :5], past_kvs=cache.layers, pos_offset=0)
        wide = cache.repeat_batch(3)
        assert wide.batch == 3 and wide.length == 5
        nxt = ids[:, 5:6].repeat(3, 1)
        out = m(nxt, past_kvs=wide.layers, pos_offset=5).logits
        want = m(ids[:, :6]).logits[:, 5]
    for r in range(3):
        assert torch.allclose(out[r, 0], want[0], atol=ATOL, rtol=RTOL)
    assert wide.nbytes == 3 * cache.nbytes


@needs_sft
def test_static_cache_real_sft_model_matches_full_forward_and_tuple_cache():
    """Gate M1 on the real 31.5M SFT checkpoint (CPU, fp32)."""
    m = sft_model()
    ids = torch.randint(0, 8192, (1, 40))
    with torch.no_grad():
        full = m(ids).logits
        cat = _incremental(m, ids)
        static = _incremental(m, ids, StaticKVCache.from_model(m, 1, 40))
    assert torch.allclose(full, static, atol=ATOL, rtol=RTOL), float((full - static).abs().max())
    assert torch.allclose(cat, static, atol=ATOL, rtol=RTOL), float((cat - static).abs().max())
