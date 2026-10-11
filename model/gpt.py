"""Decoder-only GPT: embedding, blocks, final norm, tied LM head.

Residual hook point (amendment 5): record/inject happen at the OUTPUT of
block L, i.e. after BOTH residual adds (attn + mlp) of that block.
`record_layers=[L]` returns that tensor; `inject={L: vec}` adds vec to it.

Steering caveat: injected vectors only affect tokens processed AFTER the
change. Cached K/V already in past_kvs keep old values, so in-flight
context is not rewritten (see decisions.md).

forward() always returns GPTOutput(logits, new_kvs, recorded) — a fixed
named structure (amendment 6), never shape-shifting tuples.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.blocks import Block, RMSNorm


@dataclass
class GPTOutput:
    logits: torch.Tensor
    new_kvs: list = field(default_factory=list)
    recorded: dict = field(default_factory=dict)


class GPT(nn.Module):
    def __init__(self, cfg):
        """cfg: model dict from YAML (vocab_size, n_layers, d_model, n_heads,
        context_len, ffn_hidden, norm/pos/mlp tags, weight_tying, dropout)."""
        super().__init__()
        self.cfg = dict(cfg)
        V, L, D, H = cfg["vocab_size"], cfg["n_layers"], cfg["d_model"], cfg["n_heads"]
        assert D % H == 0 and (D // H) % 2 == 0, "need even head_dim for RoPE"
        self.context_len = cfg["context_len"]
        self.tying = cfg.get("weight_tying", True)
        self.wte = nn.Embedding(V, D)
        self.blocks = nn.ModuleList(
            Block(D, H, cfg["ffn_hidden"], cfg.get("dropout", 0.0)) for _ in range(L))
        self.ln_f = RMSNorm(D)
        self.drop = nn.Dropout(cfg.get("dropout", 0.0))
        if not self.tying:
            self.lm_head = nn.Linear(D, V, bias=False)
        self.apply(self._init_fn)
        self._scale_residual_init()

    def _init_fn(self, m):
        """GPT-2 init: N(0, 0.02); residual-output projs scaled /sqrt(2L).

        Output projections (attn w_o, mlp w_down) feed the residual stream,
        so their variance is divided by 2*n_layers to keep stream variance
        stable at init; otherwise deep stacks blow up pre-training.
        """
        if isinstance(m, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, RMSNorm):
            torch.nn.init.ones_(m.w)

    def _scale_residual_init(self):
        L = len(self.blocks)
        s = (2 * L) ** -0.5
        with torch.no_grad():
            for b in self.blocks:
                b.attn.w_o.weight.mul_(s)
                b.mlp.w_down.weight.mul_(s)

    def forward(self, ids, past_kvs=None, pos_offset=0, record_layers=None,
                inject=None, impl="sdpa"):
        B, T = ids.shape
        assert T > 0, "empty sequence"
        assert pos_offset + T <= self.context_len, "exceeds context_len"
        if past_kvs is not None:
            assert len(past_kvs) == len(self.blocks), "past_kvs per layer"
        rec = set(record_layers or [])
        x = self.drop(self.wte(ids))
        new_kvs, recorded = [], {}
        for i, blk in enumerate(self.blocks):
            pk = past_kvs[i] if past_kvs is not None else None
            x, kv = blk(x, pk, pos_offset, impl)
            if inject and i in inject:
                v = inject[i]
                # (d,) broadcast over batch+time, or exact (B,T,d).
                x = x + (v.view(1, 1, -1) if v.dim() == 1 else v)
            if i in rec:
                recorded[i] = x.clone()
            new_kvs.append(kv)
        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight) if self.tying else self.lm_head(x)
        return GPTOutput(logits=logits, new_kvs=new_kvs, recorded=recorded)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
