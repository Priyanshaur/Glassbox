"""Phase 6 / S2: the TopK SAE module."""
import numpy as np
import pytest
import torch

from interp.sae import TopKSAE, load_sae, save_sae


def test_encode_has_at_most_k_nonzeros_nonnegative_and_decoder_rows_unit_norm():
    sae = TopKSAE(16, 64, 5, seed=1)
    x = torch.randn(32, 16)
    acts, pre = sae.encode(x)
    assert acts.shape == (32, 64) and (acts >= 0).all()
    assert ((acts > 0).sum(-1) <= 5).all()
    kept = pre.topk(5, dim=-1).values.clamp_min(0)
    assert torch.allclose(acts.sum(-1), kept.sum(-1), atol=1e-5)           # exactly the top-k values
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-6)


def test_decode_is_linear_in_acts_and_forward_is_consistent():
    sae = TopKSAE(8, 20, 3, seed=2)
    x = torch.randn(10, 8)
    x_hat, acts, pre = sae(x)
    assert torch.allclose(x_hat, acts @ sae.W_dec + sae.b_dec, atol=1e-6)
    sae.init_b_dec(torch.ones(8))
    assert torch.allclose(sae.b_dec, torch.ones(8))


def test_gradient_projection_and_renormalization_keep_unit_norm_rows():
    sae = TopKSAE(8, 20, 3, seed=3)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    x = torch.randn(64, 8)
    for _ in range(20):
        opt.zero_grad()
        loss, _ = sae.loss(x)
        loss.backward()
        sae.project_decoder_grad()
        g = sae.W_dec.grad
        assert torch.allclose((g * sae.W_dec).sum(-1), torch.zeros(20), atol=1e-6)   # no radial component
        opt.step()
        sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(20), atol=1e-5)


def test_dead_latents_are_tracked_and_the_aux_loss_only_exists_when_some_are_dead():
    sae = TopKSAE(8, 40, 2, seed=4)
    x = torch.randn(1, 8).repeat(128, 1)                          # identical rows: only 2 latents can ever fire
    _, parts = sae.loss(x, dead_steps=3)
    assert parts["n_dead"] == 0 and parts["aux"] == 0.0
    for _ in range(4):
        sae.loss(x, dead_steps=3)
    never = (sae.steps_since_active >= 3)
    assert int(never.sum()) == 38                                 # only the 2 firing latents are alive
    total, parts = sae.loss(x, dead_steps=3, aux_k=4)
    assert parts["n_dead"] == int(never.sum()) and parts["aux"] > 0
    total.backward()
    assert sae.W_enc.grad is not None
    # the aux term gives the dead latents' encoder columns a gradient; the main term cannot
    dead_cols = sae.W_enc.grad[:, never]
    assert dead_cols.abs().sum() > 0


def test_sae_learns_a_synthetic_sparse_dictionary():
    """Data = 3 of 32 random unit atoms with positive coefficients (d=16): a TopK SAE
    with k=3 recovers it well (fraction of variance unexplained drops far below 1)."""
    torch.manual_seed(0)
    d, n_atoms = 16, 32
    atoms = torch.randn(n_atoms, d)
    atoms = atoms / atoms.norm(dim=1, keepdim=True)

    def batch(n):
        idx = torch.stack([torch.randperm(n_atoms)[:3] for _ in range(n)])
        coef = torch.rand(n, 3) + 0.5
        return (coef.unsqueeze(-1) * atoms[idx]).sum(1)

    sae = TopKSAE(d, 64, 3, seed=0)
    opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
    for _ in range(1500):
        opt.zero_grad()
        loss, _ = sae.loss(batch(512), aux_coef=0.0)
        loss.backward()
        sae.project_decoder_grad()
        opt.step()
        sae.normalize_decoder()
    x = batch(4096)
    with torch.no_grad():
        x_hat, _, _ = sae(x)
    fvu = float(((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
    assert fvu < 0.15, fvu


def test_save_load_roundtrip(tmp_path):
    sae = TopKSAE(8, 24, 4, seed=5)
    sae.loss(torch.randn(16, 8))                                  # make steps_since_active non-trivial
    p = str(tmp_path / "s.pt")
    save_sae(p, sae, {"layer": 4, "act_scale": 0.5})
    back, meta = load_sae(p)
    assert meta == {"layer": 4, "act_scale": 0.5} and back.config() == sae.config()
    x = torch.randn(5, 8)
    assert torch.equal(sae(x)[0], back(x)[0])
    assert torch.equal(sae.steps_since_active, back.steps_since_active)
