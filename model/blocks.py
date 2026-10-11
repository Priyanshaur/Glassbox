"""Transformer blocks: RMSNorm, SwiGLU MLP, pre-norm residual block."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import Attention


class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2) + eps) * w. No centering/bias (LLaMA-style)."""

    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps) * self.w


class SwiGLU(nn.Module):
    """3-matrix MLP: silu(xWg) * (xWu) then Wd. Hidden size from config."""

    def __init__(self, d_model, hidden, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    """Pre-norm block: x += attn(norm(x)); x += mlp(norm(x))."""

    def __init__(self, d_model, n_heads, ffn_hidden, dropout=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.mlp_norm = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.mlp = SwiGLU(d_model, ffn_hidden, dropout)

    def forward(self, x, past_kv=None, pos_offset=0, impl="sdpa"):
        a, kv = self.attn(self.attn_norm(x), past_kv, pos_offset, impl)
        x = x + a
        x = x + self.mlp(self.mlp_norm(x))
        return x, kv
