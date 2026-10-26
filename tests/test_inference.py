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


# ---------------------------------------------------------------- M2: generation engine
from inference.generate import Generator  # noqa: E402
from tokenizer.bpe import BPETokenizer, EOT_ID  # noqa: E402
from train.sample import sample_completions  # noqa: E402

PROMPT = "Write a short story using all of these words: dragon, castle, brave.\n\nStory:"
_TOK = None


def tok():
    global _TOK
    if _TOK is None:
        _TOK = BPETokenizer.load("tokenizer/tok8192.json")
    return _TOK


def eot_prone_model():
    """Debug model whose tied head strongly prefers EOT, so EOT paths are exercised."""
    m = debug_model()
    hs = []
    h = m.ln_f.register_forward_hook(lambda mod, i, o: hs.append(o.detach()))
    with torch.no_grad():
        m(torch.tensor([tok().encode(PROMPT)]))
    h.remove()
    mean = hs[0][0].mean(dim=0)
    with torch.no_grad():
        m.wte.weight[EOT_ID] = 6.0 * mean / mean.norm()
    return m


@pytest.mark.parametrize("n", [1, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_engine_matches_sample_completions_same_seed(n, seed):
    """Gate M2 (debug model): identical completions and EOT flags for the same seed."""
    m = debug_model()
    want, wflags = sample_completions(m, tok(), PROMPT, n=n, max_new=24, temperature=0.8,
                                      top_k=50, seed=seed, device="cpu", return_flags=True)
    got = Generator(m, tok()).generate(PROMPT, n=n, max_new=24, temperature=0.8, top_k=50,
                                       seed=seed, replay_last_prompt_token=True)
    assert got.texts == want
    assert got.eot_hit == [f["eot_hit"] for f in wflags]


def test_engine_parity_without_topk_and_with_eot_stops():
    """Same parity with top_k=None / temperature 1.0, and on a model that actually samples EOT."""
    m = eot_prone_model()
    for seed in range(4):
        want, wflags = sample_completions(m, tok(), PROMPT, n=3, max_new=24, temperature=1.0,
                                          top_k=None, seed=seed, device="cpu", return_flags=True)
        got = Generator(m, tok()).generate(PROMPT, n=3, max_new=24, temperature=1.0, top_k=None,
                                           seed=seed, replay_last_prompt_token=True)
        assert got.texts == want
        assert got.eot_hit == [f["eot_hit"] for f in wflags]
    # the EOT path really ran: at least one row finished early with a cut text
    assert any(got.eot_hit) or any(f["eot_hit"] for f in wflags)


@needs_sft
@pytest.mark.parametrize("seed", [0, 1])
def test_engine_matches_sample_completions_real_sft_model(seed):
    """Gate M2 on the real SFT checkpoint (CPU, fp32): identical text and EOT flags."""
    m = sft_model()
    want, wflags = sample_completions(m, tok(), PROMPT, n=3, max_new=40, temperature=0.8,
                                      top_k=40, seed=seed, device="cpu", return_flags=True)
    got = Generator(m, tok()).generate(PROMPT, n=3, max_new=40, temperature=0.8, top_k=40,
                                       seed=seed, replay_last_prompt_token=True)
    assert got.texts == want
    assert got.eot_hit == [f["eot_hit"] for f in wflags]


def test_default_mode_uses_true_next_token_distribution_old_sampler_does_not():
    """The old sampler re-feeds the last prompt token, which changes the first
    next-token logits; the engine default does not."""
    m = debug_model()
    ids = torch.tensor([tok().encode(PROMPT)])
    P = ids.shape[1]
    with torch.no_grad():
        true = m(ids).logits[:, -1]
        cache = StaticKVCache.from_model(m, 1, P + 2)
        pre = m(ids, past_kvs=cache.layers, pos_offset=0)
        assert torch.allclose(pre.logits[:, -1], true, atol=ATOL, rtol=RTOL)   # engine default
        replay = m(ids[:, -1:], past_kvs=cache.layers, pos_offset=P - 1).logits[:, -1]
    assert cache.length == P + 1                                   # token cached twice
    assert not torch.allclose(replay, true, atol=1e-4, rtol=1e-4)  # old sampler's first step differs


def test_default_greedy_matches_naive_full_forward_rollout():
    m = debug_model()
    ids = torch.tensor([tok().encode(PROMPT)])
    with torch.no_grad():
        cur = ids
        for _ in range(10):
            cur = torch.cat([cur, m(cur).logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    got = Generator(m, tok()).generate(PROMPT, n=1, max_new=10, temperature=1e-6, top_k=None,
                                       seed=0, stop_at_eot=False)
    assert got.new_ids[0] == cur[0, ids.shape[1]:].tolist()


def test_seed_determinism_and_sensitivity():
    g = Generator(debug_model(), tok())
    a = g.generate(PROMPT, n=2, max_new=20, seed=3).new_ids
    assert a == g.generate(PROMPT, n=2, max_new=20, seed=3).new_ids
    assert a != g.generate(PROMPT, n=2, max_new=20, seed=4).new_ids


def test_top_p_one_is_noop_and_tiny_top_p_is_greedy():
    g = Generator(debug_model(), tok())
    base = g.generate(PROMPT, n=2, max_new=16, top_k=50, seed=5).new_ids
    assert base == g.generate(PROMPT, n=2, max_new=16, top_k=50, top_p=1.0, seed=5).new_ids
    greedy = g.generate(PROMPT, n=1, max_new=16, temperature=1e-6, top_k=None, seed=0).new_ids
    for sd in (0, 1, 2):
        assert g.generate(PROMPT, n=1, max_new=16, top_k=None, top_p=1e-9, seed=sd).new_ids == greedy


def test_stop_at_eot_and_fixed_length(monkeypatch):
    g = Generator(debug_model(), tok())
    assert g.generate(PROMPT, n=1, max_new=12, stop_at_eot=False).n_steps == 12
    calls = {"i": 0}

    def fake(logits, gen, temperature, top_k, top_p):
        calls["i"] += 1
        return torch.full((logits.shape[0], 1), EOT_ID if calls["i"] == 3 else 5)

    monkeypatch.setattr(Generator, "_sample", staticmethod(fake))
    r = g.generate(PROMPT, n=1, max_new=12)
    assert r.n_steps == 3 and r.eot_hit == [True]
    calls["i"] = 0
    assert g.generate(PROMPT, n=1, max_new=12, stop_at_eot=False).n_steps == 12


def test_stream_concatenates_to_generate_text():
    """Streaming pieces join to the generate() text, with and without an early EOT."""
    for model in (debug_model(), eot_prone_model()):
        g = Generator(model, tok())
        for seed in range(3):
            full = g.generate(PROMPT, n=1, max_new=24, temperature=1.0, top_k=None, seed=seed)
            pieces = list(g.stream(PROMPT, max_new=24, temperature=1.0, top_k=None, seed=seed))
            assert "".join(pieces) == full.texts[0]
    plain = Generator(debug_model(), tok())
    assert len(list(plain.stream(PROMPT, max_new=24, temperature=1.0, top_k=None, seed=0))) >= 1
