"""fp16 and int8 weight-only quantization for the GPT in model/gpt.py.

int8 here is weight-only and symmetric with one scale per OUTPUT row:
    q = round(w / scale), scale = absmax(row) / 127, q in [-127, 127]
and a layer computes  (x @ q^T) * scale  (the scale commutes with the matmul, so
this equals F.linear(x, q * scale)). Activations stay in the model's own dtype.
There is no fused int8 kernel: q is cast to the activation dtype on every call,
so this saves DISK and MEMORY, and is not expected to be faster (measured in
bench/, not assumed).

What is quantized: every nn.Linear in the blocks (attention q/k/v/o, SwiGLU
gate/up/down) and, optionally, the tied embedding / output head (one scale per
vocabulary row serves both the lookup and the head). RMSNorm weights stay fp32.
The tied head reads `wte.weight`, which QuantEmbedding dequantizes on access.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "int8-rowwise-v1"


def quantize_rows(w):
    """(R, C) float -> (int8 (R, C), float32 scale (R,)); symmetric per row."""
    w = w.detach().float()
    absmax = w.abs().amax(dim=1)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    q = torch.round(w / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


class QuantLinear(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        self.register_buffer("q", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features))

    @classmethod
    def from_linear(cls, lin):
        assert lin.bias is None, "bias-free layers only (the GPT has none)"
        m = cls(*lin.weight.shape)
        m.q, m.scale = quantize_rows(lin.weight)
        return m

    def forward(self, x):
        return F.linear(x, self.q.to(x.dtype)) * self.scale.to(x.dtype)


class QuantEmbedding(nn.Module):
    def __init__(self, num, dim):
        super().__init__()
        self.register_buffer("q", torch.zeros(num, dim, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(num))

    @classmethod
    def from_embedding(cls, emb):
        m = cls(*emb.weight.shape)
        m.q, m.scale = quantize_rows(emb.weight)
        return m

    def forward(self, ids):
        return self.q[ids].float() * self.scale[ids].unsqueeze(-1)

    @property
    def weight(self):
        """Dequantized (V, D) matrix, rebuilt on every access (used by the tied head)."""
        return self.q.float() * self.scale[:, None]


_BLOCK_LINEARS = (("attn", ("w_q", "w_k", "w_v", "w_o")),
                  ("mlp", ("w_gate", "w_up", "w_down")))


def quantize_model(model, quantize_embedding=True):
    """Return an int8 weight-only copy of `model` (the original is untouched)."""
    m = copy.deepcopy(model).eval()
    for blk in m.blocks:
        for part, names in _BLOCK_LINEARS:
            mod = getattr(blk, part)
            for name in names:
                setattr(mod, name, QuantLinear.from_linear(getattr(mod, name)))
    if quantize_embedding:
        m.wte = QuantEmbedding.from_embedding(m.wte)
    m.quant = {"format": FORMAT, "embedding": bool(quantize_embedding)}
    return m


def to_half(model):
    """fp16 copy (for the GPU; CPU half support is slow)."""
    return copy.deepcopy(model).eval().half()


def save_quantized(model, path):
    torch.save({"format": FORMAT, "embedding": model.quant["embedding"],
                "state": model.state_dict()}, path)


def load_quantized(cfg_model, path):
    """Rebuild a GPT from its config and load an int8 file written by save_quantized."""
    from model.gpt import GPT
    blob = torch.load(path, map_location="cpu", weights_only=True)
    assert blob["format"] == FORMAT, blob["format"]
    m = quantize_model(GPT(cfg_model), quantize_embedding=blob["embedding"])
    m.load_state_dict(blob["state"])
    return m.eval()


def tensor_bytes(model):
    """In-memory bytes of all parameters and buffers."""
    return sum(t.numel() * t.element_size()
               for t in list(model.parameters()) + list(model.buffers()))
