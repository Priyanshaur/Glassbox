"""Phase 6 / S1: the model's record and inject hooks, tested against independent captures.

`record_layers=[L]` returns the residual stream after block L (both residual adds);
`inject={L: v}` adds v after block L (before it is recorded and before block L+1).
`v` may be a (d,) vector, a (B,T,d) tensor, or a callable of the residual.
"""
import pytest
import torch

from common.config import load_config
from inference.kv_cache import StaticKVCache
from model.gpt import GPT


def debug_model(seed=11):
    torch.manual_seed(seed)
    return GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()


def ids(b=2, t=10, v=8192, seed=0):
    return torch.randint(0, v, (b, t), generator=torch.Generator().manual_seed(seed))


def test_recorded_activations_equal_an_independent_forward_hook_capture():
    """Inject test 0: recorded[L] is exactly the output of block L, captured by a
    plain PyTorch forward hook on that block, for every layer."""
    m = debug_model()
    x = ids()
    caught = {}
    handles = [blk.register_forward_hook(lambda mod, i, o, L=L: caught.__setitem__(L, o[0].detach()))
               for L, blk in enumerate(m.blocks)]
    with torch.no_grad():
        out = m(x, record_layers=[0, 1])
    for h in handles:
        h.remove()
    assert set(out.recorded) == {0, 1}
    for L in (0, 1):
        assert torch.equal(out.recorded[L], caught[L])


def test_recorded_are_clones_not_views_of_later_state():
    m = debug_model()
    with torch.no_grad():
        a = m(ids(), record_layers=[0]).recorded[0]
        b = m(ids(), record_layers=[0]).recorded[0]
    assert torch.equal(a, b) and a.data_ptr() != b.data_ptr()


def test_inject_vector_shifts_the_recorded_residual_by_exactly_v_and_only_downstream():
    """Inject test 1: adding v after block 0 shifts recorded[0] by exactly v (broadcast
    over batch and time) and changes everything downstream; it changes nothing upstream."""
    m = debug_model()
    x = ids()
    v = torch.randn(64) * 3.0
    with torch.no_grad():
        base = m(x, record_layers=[0, 1])
        shifted = m(x, record_layers=[0, 1], inject={0: v})
    d0 = shifted.recorded[0] - base.recorded[0]
    assert torch.allclose(d0, v.view(1, 1, -1).expand_as(d0), atol=1e-6)
    assert not torch.allclose(shifted.recorded[1], base.recorded[1], atol=1e-4)
    assert not torch.allclose(shifted.logits, base.logits, atol=1e-4)
    # injecting after the LAST block leaves every earlier record untouched
    with torch.no_grad():
        last = m(x, record_layers=[0, 1], inject={1: v})
    assert torch.equal(last.recorded[0], base.recorded[0])
    assert torch.allclose(last.recorded[1] - base.recorded[1], v.view(1, 1, -1).expand_as(d0), atol=1e-6)


def test_inject_toward_a_token_embedding_raises_that_tokens_logit():
    """Inject test 2: pushing the last block's residual along a token's (tied) embedding
    direction raises that token's logit, monotonically in the scale, at every position."""
    m = debug_model()
    x = ids()
    tok_id = 123
    u = m.wte.weight[tok_id].detach()
    u = u / u.norm()
    with torch.no_grad():
        resid = m(x, record_layers=[1]).recorded[1]
        unit = resid.norm(dim=-1).mean()
        logit = []
        for s in (0.0, 1.0, 4.0, 16.0, 64.0):
            logit.append(m(x, inject={1: s * unit * u}).logits[..., tok_id].mean().item())
    assert all(b > a for a, b in zip(logit, logit[1:])), logit
    with torch.no_grad():
        big = m(x, inject={1: 64 * unit * u}).logits
    assert (big.argmax(-1) == tok_id).all()          # at a large scale it wins everywhere


def test_inject_accepts_btd_tensor_and_callable_and_they_agree():
    m = debug_model()
    x = ids()
    v = torch.randn(64)
    with torch.no_grad():
        a = m(x, inject={0: v}).logits
        b = m(x, inject={0: v.view(1, 1, -1).expand(2, 10, 64).clone()}).logits
        c = m(x, inject={0: lambda r: v.view(1, 1, -1).expand_as(r)}).logits
    assert torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6)


def test_callable_inject_can_replace_the_residual_exactly():
    """A callable returning (target - x) replaces the residual with target: the basis for
    SAE reconstruction patching and mean ablation."""
    m = debug_model()
    x = ids()
    with torch.no_grad():
        rec = m(x, record_layers=[0]).recorded[0]
        same = m(x, inject={0: lambda r: rec - r}).logits
        assert torch.allclose(same, m(x).logits, atol=1e-5)          # replace by itself: identity
        zero = m(x, record_layers=[0], inject={0: lambda r: -r})
        assert torch.allclose(zero.recorded[0], torch.zeros_like(rec), atol=1e-6)
        assert not torch.allclose(zero.logits, m(x).logits, atol=1e-3)


def test_static_cache_decode_with_inject_at_every_step_matches_full_forward():
    """Injecting at every position (prefill and each decode step) is consistent with a
    full forward pass; injecting only during decode is not (cached K/V keep old values)."""
    m = debug_model()
    x = ids(1, 12)
    v = torch.randn(64) * 2.0
    with torch.no_grad():
        full = m(x, inject={0: v}).logits
        cache = StaticKVCache.from_model(m, 1, 12)
        steps = [m(x[:, :6], past_kvs=cache.layers, pos_offset=0, inject={0: v}).logits]
        for t in range(6, 12):
            steps.append(m(x[:, t:t + 1], past_kvs=cache.layers, pos_offset=t, inject={0: v}).logits)
        every = torch.cat(steps, dim=1)
        cache2 = StaticKVCache.from_model(m, 1, 12)
        steps2 = [m(x[:, :6], past_kvs=cache2.layers, pos_offset=0).logits]      # prefill NOT injected
        for t in range(6, 12):
            steps2.append(m(x[:, t:t + 1], past_kvs=cache2.layers, pos_offset=t, inject={0: v}).logits)
        decode_only = torch.cat(steps2, dim=1)
    assert torch.allclose(full, every, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(full[:, 6:], decode_only[:, 6:], atol=1e-3)


def test_recording_under_the_static_cache_matches_full_forward_rows():
    m = debug_model()
    x = ids(1, 12)
    with torch.no_grad():
        full = m(x, record_layers=[1]).recorded[1]
        cache = StaticKVCache.from_model(m, 1, 12)
        rows = [m(x[:, :5], past_kvs=cache.layers, pos_offset=0, record_layers=[1]).recorded[1]]
        for t in range(5, 12):
            rows.append(m(x[:, t:t + 1], past_kvs=cache.layers, pos_offset=t, record_layers=[1]).recorded[1])
    assert torch.allclose(full, torch.cat(rows, dim=1), atol=1e-5, rtol=1e-5)
