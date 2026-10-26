"""Static (preallocated) KV cache.

model/attention.py's default cache is a tuple of (K, V) tensors that is
re-created with torch.cat on every decoded token: each step allocates and copies
the whole history. StaticKVCache allocates K/V buffers once, (batch, heads,
max_len, head_dim) per layer, and writes new keys/values in place; attention then
reads a view of the filled prefix. Same maths, no per-step allocation.

Usage with the model (no model change needed beyond the `update` hook that
Attention already understands):

    cache = StaticKVCache.from_model(model, batch=1, max_len=prompt_len + max_new)
    out = model(prompt_ids, past_kvs=cache.layers, pos_offset=0)          # prefill
    out = model(next_tok, past_kvs=cache.layers, pos_offset=cache.length) # decode

The caller still owns `pos_offset` (RoPE position of the first new token); it
must equal `cache.length`. Cached K already carry their rotation.
"""
import torch


class LayerCache:
    """One layer's buffers. Duck-typed by Attention.forward via `update`."""

    def __init__(self, k, v):
        self.k, self.v, self.length = k, v, 0

    @property
    def capacity(self):
        return self.k.shape[2]

    def update(self, k_new, v_new):
        """Write (B,H,T,D) keys/values at [length, length+T); return views of the
        filled prefix plus the cache length BEFORE the write (what Attention
        calls cache_len, selecting the mask regime)."""
        t = k_new.shape[2]
        start = self.length
        if start + t > self.capacity:
            raise ValueError(
                f"KV cache overflow: {start} cached + {t} new > capacity {self.capacity}")
        self.k[:, :, start:start + t] = k_new
        self.v[:, :, start:start + t] = v_new
        self.length = start + t
        return self.k[:, :, :self.length], self.v[:, :, :self.length], start


class StaticKVCache:
    def __init__(self, n_layers, batch, n_heads, head_dim, max_len,
                 dtype=torch.float32, device=None):
        shape = (batch, n_heads, max_len, head_dim)
        self.layers = [LayerCache(torch.zeros(shape, dtype=dtype, device=device),
                                  torch.zeros(shape, dtype=dtype, device=device))
                       for _ in range(n_layers)]

    @classmethod
    def from_model(cls, model, batch, max_len, dtype=None, device=None):
        """Size the cache from the model config; dtype/device default to the
        model's own parameters."""
        c = model.cfg
        p = next(model.parameters())
        return cls(c["n_layers"], batch, c["n_heads"], c["d_model"] // c["n_heads"],
                   max_len, dtype or p.dtype, device or p.device)

    @property
    def length(self):
        return self.layers[0].length

    @property
    def capacity(self):
        return self.layers[0].capacity

    @property
    def batch(self):
        return self.layers[0].k.shape[0]

    @property
    def nbytes(self):
        return sum(l.k.numel() * l.k.element_size() * 2 for l in self.layers)

    def reset(self):
        """Forget the contents (buffers are kept and overwritten on next use)."""
        for l in self.layers:
            l.length = 0

    def truncate(self, n):
        """Roll back to the first n positions (e.g. to reuse a shared prompt)."""
        if not 0 <= n <= self.length:
            raise ValueError(f"cannot truncate a cache of length {self.length} to {n}")
        for l in self.layers:
            l.length = n

    def repeat_batch(self, n):
        """New cache with every batch row repeated n times (same-prompt
        multi-sample: prefill once with batch 1, then fan out)."""
        c = self.layers[0]
        out = StaticKVCache(len(self.layers), c.k.shape[0] * n, c.k.shape[1],
                            c.k.shape[3], c.capacity, c.k.dtype, c.k.device)
        for src, dst in zip(self.layers, out.layers):
            dst.k.copy_(src.k.repeat_interleave(n, dim=0))
            dst.v.copy_(src.v.repeat_interleave(n, dim=0))
            dst.length = src.length
        return out
