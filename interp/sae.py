"""TopK sparse autoencoder for residual-stream activations.

    pre   = (x - b_dec) @ W_enc + b_enc
    acts  = relu(top-k of pre), everything else 0          (exactly <= k non-zeros / token)
    x_hat = acts @ W_dec + b_dec                           (W_dec rows have unit norm)

Dead latents (no activation for `dead_steps` steps) get an auxiliary loss (AuxK, as in
the OpenAI TopK SAE recipe): the top `aux_k` dead pre-activations must reconstruct the
residual error of the main reconstruction. Decoder rows are kept at unit norm by
projecting their gradient and renormalizing after each step.

`steps_since_active` is a (persistent) buffer, so dead-latent bookkeeping is part of the
state dict and training resumes exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in, n_features, k, seed=0):
        super().__init__()
        assert 1 <= k <= n_features
        self.d_in, self.n_features, self.k = d_in, n_features, k
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(n_features, d_in, generator=g)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w.clone())            # (F, d), unit-norm rows
        self.W_enc = nn.Parameter(w.t().contiguous())   # (d, F), tied at init
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.register_buffer("steps_since_active", torch.zeros(n_features, dtype=torch.long))

    # ------------------------------------------------------------------ forward pieces
    def pre_acts(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        """(acts (B,F) with <= k non-zeros, pre (B,F))."""
        pre = self.pre_acts(x)
        vals, idx = pre.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
        return acts, pre

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, x):
        acts, pre = self.encode(x)
        return self.decode(acts), acts, pre

    # ------------------------------------------------------------------ training helpers
    def loss(self, x, aux_coef=1 / 32, aux_k=64, dead_steps=1000):
        """(total, parts). main = per-element MSE; aux = AuxK on the dead latents."""
        x_hat, acts, pre = self(x)
        err = x - x_hat
        main = (err ** 2).mean()
        dead = self.steps_since_active >= dead_steps
        n_dead = int(dead.sum())
        aux = x.new_zeros(())
        if n_dead > 0 and aux_coef > 0:
            kk = min(aux_k, n_dead)
            pre_dead = pre.masked_fill(~dead, float("-inf"))
            vals, idx = pre_dead.topk(kk, dim=-1)
            a = torch.zeros_like(pre).scatter(-1, idx, F.relu(vals))
            e = err.detach()
            aux = ((e - a @ self.W_dec) ** 2).sum(-1).mean() / (e ** 2).sum(-1).mean().clamp_min(1e-8)
        with torch.no_grad():
            fired = (acts > 0).any(dim=0)
            self.steps_since_active += 1
            self.steps_since_active[fired] = 0
            fvu = (err ** 2).sum() / ((x - x.mean(0)) ** 2).sum().clamp_min(1e-8)
            l0 = (acts > 0).sum(-1).float().mean()
        return main + aux_coef * aux, {"main": float(main.detach()), "aux": float(aux.detach()),
                                       "n_dead": n_dead, "fvu": float(fvu), "l0": float(l0)}

    @torch.no_grad()
    def project_decoder_grad(self):
        """Remove the gradient component along each decoder row (keeps rows on the sphere)."""
        if self.W_dec.grad is not None:
            g = self.W_dec.grad
            g -= (g * self.W_dec).sum(-1, keepdim=True) * self.W_dec

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def init_b_dec(self, mean):
        self.b_dec.copy_(mean)

    def config(self):
        return {"d_in": self.d_in, "n_features": self.n_features, "k": self.k}


def save_sae(path, sae, meta=None):
    """Atomic save of the weights plus config and free-form provenance `meta`
    (layer, act_scale, model hash, ...)."""
    import os
    tmp = path + ".tmp"
    torch.save({"config": sae.config(), "state": sae.state_dict(), "meta": meta or {}}, tmp)
    os.replace(tmp, path)


def load_sae(path, map_location="cpu"):
    blob = torch.load(path, map_location=map_location, weights_only=True)
    c = blob["config"]
    sae = TopKSAE(c["d_in"], c["n_features"], c["k"])
    sae.load_state_dict(blob["state"])
    return sae, blob["meta"]
