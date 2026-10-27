"""Phase 6 / S3: SAE evaluation metrics and their agreement with the pre-registration."""
import math
import os

import pytest
import torch

from common.config import load_config
from interp import eval_sae as E
from interp.sae import TopKSAE
from model.gpt import GPT
from tokenizer.bpe import BPETokenizer
from train.data_loader import ShardLoader

CTX, LAYER = 64, 1


def structured_model():
    """Debug model with weights scaled up so the residual stream matters to the loss."""
    torch.manual_seed(3)
    m = GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(6.0)
    return m


def identity_sae(d=64):
    s = TopKSAE(d, 2 * d, 2 * d)                       # k = F: keeps every latent
    with torch.no_grad():
        s.W_dec.zero_()
        s.W_enc.zero_()
        s.W_dec[:d] = torch.eye(d)
        s.W_dec[d:] = -torch.eye(d)                    # positive and negative parts
        s.W_enc[:, :d] = torch.eye(d)
        s.W_enc[:, d:] = -torch.eye(d)
    return s


def mean_sae(mean, d=64):
    s = TopKSAE(d, 8, 2)
    with torch.no_grad():
        s.W_dec.zero_()
        s.b_dec.copy_(mean)
    return s


def test_preregistration_document_contains_exactly_the_coded_thresholds():
    doc = open("docs/phase6_preregistration.md", encoding="utf-8").read()
    T = E.THRESHOLDS
    assert f"**<= {T['fvu_max']:.2f}**" in doc
    assert f"**<= {T['l0_max']} and >= {T['l0_min']}**" in doc
    assert f"**<= {T['dead_max']:.2f}**" in doc
    assert f"**>= {T['ce_recovered_min']:.2f}**" in doc
    assert (f"**FVU <= {T['sft_fvu_max']:.2f} and CE recovered >= {T['sft_ce_recovered_min']:.2f} on both**"
            in doc)
    assert E.MIN_EVAL_TOKENS == 500_000 and E.DEAD_SCAN_TOKENS == 10_000_000
    assert "500,000 tokens" in doc and "**10M**" in doc


def test_reconstruction_metrics_identity_and_mean_predictor():
    x = torch.randn(512, 64)
    r = E.reconstruction_metrics(identity_sae(), x)
    assert r["fvu"] < 1e-8 and r["l0"] == pytest.approx(64.0, abs=1.0) and sum(r["l0_hist"]) == 512
    r0 = E.reconstruction_metrics(mean_sae(x.mean(0)), x)
    assert r0["fvu"] == pytest.approx(1.0, abs=1e-6)       # a decoder of zeros predicts only the mean


def test_dead_fraction_counts_latents_that_cannot_fire():
    m = structured_model()
    tr = ShardLoader("data", "train")
    s = TopKSAE(64, 16, 4, seed=1)
    with torch.no_grad():
        s.b_enc[8:] = -1e9                              # these 8 can never be selected with a positive value
    r = E.dead_fraction(m, s, tr, LAYER, 1.0, tokens=2048, ctx=CTX, windows_per_round=16)
    assert r["scan_tokens"] >= 2048 and r["dead_fraction"] >= 0.5


def test_ce_recovered_is_one_for_a_perfect_sae_and_zero_for_the_mean_predictor():
    m = structured_model()
    va = ShardLoader("data", "validation")
    ids = va.batch_ids(5, 0, 24, CTX)
    inputs, targets = ids[:, :CTX], ids[:, 1:CTX + 1]
    acts = E.A.record_windows(m, inputs, LAYER)
    perfect = E.ce_recovered(m, identity_sae(), LAYER, 1.0, inputs, targets, chunk=8)
    assert perfect["ce_meanablate"] - perfect["ce_clean"] > 1e-3, perfect          # test is meaningful
    assert abs(perfect["ce_sae"] - perfect["ce_clean"]) < 1e-4
    assert perfect["ce_recovered"] == pytest.approx(1.0, abs=0.02)
    mean = acts.reshape(-1, 64).mean(0)
    base = E.ce_recovered(m, mean_sae(mean), LAYER, 1.0, inputs, targets, chunk=8)
    assert base["ce_recovered"] == pytest.approx(0.0, abs=1e-3)


def test_ce_recovered_respects_a_position_mask_and_scale():
    m = structured_model()
    ids = ShardLoader("data", "validation").batch_ids(5, 0, 8, CTX)
    inputs, targets = ids[:, :CTX], ids[:, 1:CTX + 1]
    mask = torch.zeros(inputs.shape, dtype=torch.bool)
    mask[:, 10:20] = True
    r = E.ce_recovered(m, identity_sae(), LAYER, 1.0, inputs, targets, mask, chunk=4)
    assert r["tokens"] == 8 * 10
    # the SAE sees scale*residual and its output is divided by the scale again: identity SAE stays perfect
    r2 = E.ce_recovered(m, identity_sae(), LAYER, 0.37, inputs, targets, mask, chunk=4)
    assert abs(r2["ce_sae"] - r2["ce_clean"]) < 1e-4


def test_verdict_applies_the_preregistered_rules():
    good = {"fvu": 0.10, "l0": 30.0, "dead_fraction": 0.05, "ce_recovered": 0.90}
    v = E.verdict(good)
    assert v["layer_passes_A1_A4"] and all(v[k] for k in ("A1_fvu", "A2_l0", "A3_dead", "A4_ce"))
    for key, val, rule in (("fvu", 0.16, "A1_fvu"), ("l0", 33.0, "A2_l0"), ("l0", 7.0, "A2_l0"),
                           ("dead_fraction", 0.11, "A3_dead"), ("ce_recovered", 0.84, "A4_ce")):
        bad = {**good, key: val}
        assert not E.verdict(bad)[rule] and not E.verdict(bad)["layer_passes_A1_A4"]
    t_ok = {"plain": {"fvu": 0.2, "ce_recovered": 0.85}, "template": {"fvu": 0.25, "ce_recovered": 0.80}}
    assert E.verdict({**good, "transfer": t_ok})["A5_transfer"]
    t_bad = {"plain": t_ok["plain"], "template": {"fvu": 0.26, "ce_recovered": 0.9}}
    assert not E.verdict({**good, "transfer": t_bad})["A5_transfer"]


def test_template_sequences_mask_the_response_and_transfer_metrics_run():
    tok = BPETokenizer.load("tokenizer/tok8192.json")
    inputs, targets, mask = E.template_sequences(tok, "data", 6, 128)
    assert inputs.shape == targets.shape == mask.shape and inputs.shape[0] == 6
    assert mask.any() and not mask.all()                       # prompt (and padding) excluded
    m = structured_model()
    s = identity_sae()
    cfg = load_config("configs/debug_cpu.yaml")
    inputs, targets, mask = E.template_sequences(tok, "data", 6, cfg["train"]["context_len"])
    r = E.ce_recovered(m, s, LAYER, 1.0, inputs, targets, mask, chunk=3)
    assert r["tokens"] == int(mask.sum()) and abs(r["ce_sae"] - r["ce_clean"]) < 1e-4
    f = E.template_fvu(m, s, LAYER, 1.0, inputs, mask, chunk=3)
    assert f["fvu"] < 1e-8


def test_full_evaluate_pipeline_on_the_debug_model_with_transfer(tmp_path):
    from interp.train_sae import TrainConfig, train
    torch.manual_seed(2)
    m = GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()
    tr, va = ShardLoader("data", "train"), ShardLoader("data", "validation")
    cfg = TrainConfig(layer=LAYER, n_features=128, k=4, lr=2e-3, warmup_steps=4, batch_tokens=256,
                      buffer_tokens=1024, epochs=2, ctx=CTX, norm_windows=8, chunk=8, eval_every=1,
                      ckpt_every=1, eval_windows=8)
    res = train(m, tr, va, cfg, str(tmp_path / "s"), log=lambda s: None)
    tok = BPETokenizer.load("tokenizer/tok8192.json")
    rep = E.evaluate(m, res["sae"], res["scale"], LAYER, va, tr, CTX, tok, "data", eval_tokens=4 * CTX,
                     dead_tokens=2048, sft_model=m, chunk=4, n_template=6)
    for k in ("fvu", "l0", "dead_fraction", "ce_recovered", "ce_clean", "transfer", "verdict", "eval_tokens"):
        assert k in rep
    assert rep["transfer"]["plain"]["fvu"] == pytest.approx(rep["fvu"], rel=1e-5)   # same model: same numbers
    assert math.isfinite(rep["fvu"]) and 0 <= rep["dead_fraction"] <= 1
    assert "A5_transfer" in rep["verdict"]
