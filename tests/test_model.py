"""Phase 2 tests: shapes, causality, SDPA==manual, KV-cache, counts, overfit."""
import math

import torch
import torch.nn.functional as F

from common.config import load_config
from model.attention import Attention, manual_attn, sdpa_path
from model.gpt import GPT, count_params


def debug_cfg():
    return load_config("configs/debug_cpu.yaml")["model"]


def test_output_shapes():
    torch.manual_seed(0)
    c = debug_cfg()
    m = GPT(c).eval()
    ids = torch.randint(0, c["vocab_size"], (2, 16))
    out = m(ids)
    assert out.logits.shape == (2, 16, c["vocab_size"])
    assert len(out.new_kvs) == 2
    k, v = out.new_kvs[0]
    assert k.shape == (2, 4, 16, 16) and v.shape == (2, 4, 16, 16)
    assert out.recorded == {}


def test_head_dim_even():
    for p in ("configs/debug_cpu.yaml", "configs/base_30m.yaml",
              "configs/big_85m.yaml"):
        c = load_config(p)["model"]
        assert c["d_model"] % c["n_heads"] == 0
        assert (c["d_model"] // c["n_heads"]) % 2 == 0


def test_causal_mask_tight_allclose():
    """Changing future tokens never changes earlier logits (tight tol)."""
    torch.manual_seed(1)
    c = debug_cfg()
    m = GPT(c).eval()
    ids = torch.randint(0, c["vocab_size"], (1, 16))
    ids2 = ids.clone()
    ids2[0, 12:] = torch.randint(0, c["vocab_size"], (4,))
    with torch.no_grad():
        a = m(ids).logits
        b = m(ids2).logits
    assert torch.allclose(a[:, :12], b[:, :12], atol=1e-5, rtol=1e-5)


def test_sdpa_vs_manual_all_regimes():
    """SDPA fast path == manual reference: no-cache, 1-token cached, chunk."""
    torch.manual_seed(2)
    attn = Attention(64, 4).eval()
    B, H, D = 2, 4, 16
    with torch.no_grad():
        # Regime 1: no cache, causal over T=8.
        q = torch.randn(B, H, 8, D)
        k = torch.randn(B, H, 8, D)
        v = torch.randn(B, H, 8, D)
        assert torch.allclose(sdpa_path(q, k, v, 0),
                              manual_attn(q, k, v, 0), atol=1e-5, rtol=1e-5)
        # Regime 2: cache 7 + single token, no mask.
        kc = torch.randn(B, H, 7, D)
        vc = torch.randn(B, H, 7, D)
        q1 = torch.randn(B, H, 1, D)
        k1 = torch.cat([kc, torch.randn(B, H, 1, D)], dim=2)
        v1 = torch.cat([vc, torch.randn(B, H, 1, D)], dim=2)
        assert torch.allclose(sdpa_path(q1, k1, v1, 7),
                              manual_attn(q1, k1, v1, 7), atol=1e-5, rtol=1e-5)
        # Regime 3: cache 7 + chunk of 5, explicit mask.
        q5 = torch.randn(B, H, 5, D)
        k5 = torch.cat([kc, torch.randn(B, H, 5, D)], dim=2)
        v5 = torch.cat([vc, torch.randn(B, H, 5, D)], dim=2)
        assert torch.allclose(sdpa_path(q5, k5, v5, 7),
                              manual_attn(q5, k5, v5, 7), atol=1e-5, rtol=1e-5)


def test_model_sdpa_vs_manual_impl():
    torch.manual_seed(3)
    c = debug_cfg()
    m = GPT(c).eval()
    ids = torch.randint(0, c["vocab_size"], (1, 12))
    with torch.no_grad():
        a = m(ids, impl="sdpa").logits
        b = m(ids, impl="manual").logits
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_kv_cache_equivalence():
    """Full-sequence logits == incremental decode with past_kvs (amendment 4)."""
    torch.manual_seed(4)
    c = debug_cfg()
    m = GPT(c).eval()
    ids = torch.randint(0, c["vocab_size"], (1, 12))
    with torch.no_grad():
        full = m(ids).logits
        # Token-by-token.
        past, steps = None, []
        for t in range(12):
            o = m(ids[:, t:t + 1], past_kvs=past, pos_offset=t)
            past, steps = o.new_kvs, steps + [o.logits]
        inc = torch.cat(steps, dim=1)
        assert torch.allclose(full, inc, atol=1e-5, rtol=1e-5)
        # Chunked: 5 then 7 (exercises explicit-mask path).
        o1 = m(ids[:, :5])
        o2 = m(ids[:, 5:], past_kvs=o1.new_kvs, pos_offset=5)
        assert torch.allclose(full[:, 5:], o2.logits, atol=1e-5, rtol=1e-5)


def hand_count(c):
    V, L, D, Ff = (c["vocab_size"], c["n_layers"], c["d_model"],
                   c["ffn_hidden"])
    per_layer = 2 * D + 4 * D * D + 3 * D * Ff  # norms + attn + swiglu
    return V * D + D + L * per_layer  # tied emb (once) + final norm


def test_param_count_matches_hand_calc():
    for p in ("configs/debug_cpu.yaml", "configs/base_30m.yaml",
              "configs/big_85m.yaml"):
        c = load_config(p)["model"]
        assert count_params(GPT(c)) == hand_count(c)


def test_initial_loss_about_ln_vocab():
    """Random-init mean logits ~0, so CE ~ ln(V)."""
    torch.manual_seed(5)
    c = debug_cfg()
    m = GPT(c).eval()
    ids = torch.randint(0, c["vocab_size"], (4, 32))
    with torch.no_grad():
        loss = F.cross_entropy(m(ids).logits[:, :-1].reshape(-1, c["vocab_size"]),
                               ids[:, 1:].reshape(-1))
    assert abs(loss.item() - math.log(c["vocab_size"])) < 0.3


def test_zero_inject_changes_nothing():
    torch.manual_seed(6)
    c = debug_cfg()
    m = GPT(c).eval()
    ids = torch.randint(0, c["vocab_size"], (1, 10))
    with torch.no_grad():
        a = m(ids).logits
        b = m(ids, inject={0: torch.zeros(64)}).logits
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6)


def test_record_layers_shapes():
    torch.manual_seed(7)
    c = debug_cfg()
    m = GPT(c).eval()
    ids = torch.randint(0, c["vocab_size"], (2, 10))
    with torch.no_grad():
        out = m(ids, record_layers=[0, 1])
    assert set(out.recorded) == {0, 1}
    assert out.recorded[0].shape == (2, 10, 64)


def test_debug_overfits_tiny_batch():
    """Acceptance: debug model drives a single batch to near-zero loss (CPU)."""
    torch.manual_seed(0)
    c = debug_cfg()
    m = GPT(c).train()
    ids = torch.randint(0, c["vocab_size"], (2, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first, last = None, None
    for _ in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(m(ids).logits[:, :-1].reshape(-1, c["vocab_size"]),
                               ids[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    assert first > 3.0, f"unexpected first loss {first}"
    assert last < 0.1, f"failed to overfit: {first:.2f} -> {last:.3f}"
