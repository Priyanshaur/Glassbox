"""Multi-head self-attention with RoPE. SDPA fast path + manual reference.

Masking rules (see decisions.md):
- No cache (training / full prompt): is_causal=True triangle over T.
- Cache + single new token: no mask (is_causal=False, attn_mask=None).
  The single query legally sees all cached keys; a causal mask over L=1
  would be all-allowed anyway, and SDPA forbids is_causal=True when L!=S
  on some builds, so we pass no mask.
- Cache + multi-token chunk: explicit float mask (T queries x S keys),
  row i allows keys j <= cache_len + i. Float additive (0 / -inf) so the
  convention is unambiguous; the test compares this path to manual_attn.

RoPE + cache: rotary is applied to new K at ABSOLUTE positions
(pos_offset..pos_offset+T-1) BEFORE appending to the cache, so
incremental decode matches a full forward. Cached K already carry their
rotation and are never re-rotated.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cos_sin(positions, head_dim, base=10000.0, dtype=torch.float32,
                 device=None):
    """Return cos, sin tables of shape (T, head_dim//2) for given positions."""
    assert head_dim % 2 == 0, "RoPE needs even head_dim"
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    inv = inv.to(dtype=dtype, device=device)
    pos = positions.to(dtype=dtype, device=device)
    ang = torch.outer(pos, inv)  # (T, D/2)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    """Rotate last-dim-paired queries/keys. x: (B,H,T,D); cos/sin: (T,D/2)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # Broadcast over batch/heads: (1,1,T,D/2).
    c = cos.unsqueeze(0).unsqueeze(0).to(x.dtype)
    s = sin.unsqueeze(0).unsqueeze(0).to(x.dtype)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


def causal_float_mask(t_q, cache_len):
    """Additive mask (T, S=cache_len+T): 0 allowed, -inf blocked."""
    t = t_q
    s = cache_len + t_q
    row = torch.arange(cache_len, cache_len + t).unsqueeze(1)  # (T,1)
    col = torch.arange(s).unsqueeze(0)                          # (1,S)
    allowed = col <= row
    return torch.where(allowed, torch.zeros(()),
                       torch.tensor(float("-inf")))


def sdpa_path(q, k_full, v_full, cache_len):
    """q/k/v: (B,H,T or S,D). Chooses mask regime by shapes (amendment 3)."""
    t = q.shape[2]
    if cache_len == 0:
        return F.scaled_dot_product_attention(q, k_full, v_full,
                                              is_causal=True)
    if t == 1:
        return F.scaled_dot_product_attention(q, k_full, v_full,
                                              is_causal=False)
    mask = causal_float_mask(t, cache_len).to(dtype=q.dtype,
                                              device=q.device)
    return F.scaled_dot_product_attention(q, k_full, v_full,
                                          attn_mask=mask, is_causal=False)


def manual_attn(q, k_full, v_full, cache_len):
    """Slow explicit attention. Test reference only (never in training)."""
    d = q.shape[-1]
    scores = q @ k_full.transpose(-2, -1) / math.sqrt(d)
    t = q.shape[2]
    if cache_len == 0:
        # Lower-triangular allowed (query i sees keys <= i).
        row = torch.arange(t).unsqueeze(1)
        col = torch.arange(t).unsqueeze(0)
        scores = scores.masked_fill(col > row, float("-inf"))
    elif t > 1:
        mask = causal_float_mask(t, cache_len).to(dtype=scores.dtype,
                                                  device=scores.device)
        scores = scores + mask
    # t == 1 with cache: no mask needed.
    return torch.softmax(scores, dim=-1) @ v_full


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must split across heads"
        assert (d_model // n_heads) % 2 == 0, "head_dim must be even (RoPE)"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, past_kv=None, pos_offset=0, impl="sdpa"):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        q = self.w_q(x).view(B, T, H, D).transpose(1, 2)
        k_new = self.w_k(x).view(B, T, H, D).transpose(1, 2)
        v_new = self.w_v(x).view(B, T, H, D).transpose(1, 2)
        # Absolute-position RoPE on the NEW keys/queries only.
        pos = torch.arange(pos_offset, pos_offset + T)
        cos, sin = rope_cos_sin(pos, D, device=x.device)
        q, k_new = apply_rope(q, cos, sin), apply_rope(k_new, cos, sin)
        if past_kv is not None:
            pk, pv = past_kv
            k_full = torch.cat([pk, k_new], dim=2)
            v_full = torch.cat([pv, v_new], dim=2)
            cache_len = pk.shape[2]
        else:
            k_full, v_full = k_new, v_new
            cache_len = 0
        fn = sdpa_path if impl == "sdpa" else manual_attn
        a = fn(q, k_full, v_full, cache_len)
        a = a.transpose(1, 2).contiguous().view(B, T, -1)
        return self.drop(self.w_o(a)), (k_full.detach(), v_full.detach())
