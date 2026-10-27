"""Phase 6 / S2: the SAE training loop on the debug model (CPU)."""
import json
import os
from dataclasses import replace

import torch

from common.config import load_config
from interp.sae import load_sae
from interp.train_sae import TrainConfig, quick_eval, train
from model.gpt import GPT
from train.data_loader import ShardLoader

BASE = TrainConfig(layer=1, n_features=128, k=4, seed=0, lr=2e-3, warmup_steps=4, batch_tokens=256,
                   buffer_tokens=1024, epochs=4, ctx=64, dead_steps=6, aux_k=8, norm_windows=8,
                   chunk=8, log_every=2, eval_every=1, ckpt_every=1, eval_windows=8)


def setup():
    torch.manual_seed(2)
    m = GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()
    return m, ShardLoader("data", "train"), ShardLoader("data", "validation")


def run(tmp, cfg, name):
    m, tr, va = setup()
    return train(m, tr, va, cfg, str(tmp / name), log=lambda s: None)


def metrics(tmp, name):
    return [json.loads(l) for l in open(tmp / name / "metrics_sae.jsonl", encoding="utf-8")]


def test_training_runs_improves_fvu_and_writes_checkpoint_and_terminal_row(tmp_path):
    res = run(tmp_path, BASE, "a")
    rows = metrics(tmp_path, "a")
    evals = [r["eval"] for r in rows if "eval" in r]
    assert len(evals) == 4 and evals[-1]["fvu"] < evals[0]["fvu"]
    assert rows[-1]["terminal"] == "epochs" and rows[-1]["steps"] == 4 * (1024 // 256) == res["steps"]
    sae, meta = load_sae(str(tmp_path / "a" / "sae_latest.pt"))
    assert meta["layer"] == 1 and meta["act_scale"] == res["scale"] and meta["epochs_done"] == 4
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(128), atol=1e-5)
    assert os.path.exists(tmp_path / "a" / "ckpt.pt")
    assert all(0 <= e["alive_frac"] <= 1 and e["l0"] <= 4 + 1e-6 for e in evals)


def test_two_identical_runs_are_bit_identical(tmp_path):
    a, b = run(tmp_path, BASE, "a")["sae"], run(tmp_path, BASE, "b")["sae"]
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))


def test_resume_at_a_buffer_epoch_boundary_equals_an_uninterrupted_run(tmp_path):
    straight = run(tmp_path, BASE, "straight")["sae"]
    first = run(tmp_path, replace(BASE, stop_after_epoch=2), "cut")
    assert first["terminal"] == "stop_after_epoch" and first["steps"] == 2 * 4
    resumed = run(tmp_path, BASE, "cut")["sae"]              # same out_dir: resumes from ckpt.pt
    assert metrics(tmp_path, "cut")[-1]["terminal"] == "epochs"
    for (k, p), (_, q) in zip(straight.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(p, q), k


def test_aux_loss_path_runs_during_training(tmp_path):
    run(tmp_path, replace(BASE, n_features=1024, k=2, dead_steps=2, aux_k=4), "aux")   # k=2 of 1024: most latents die
    rows = [r for r in metrics(tmp_path, "aux") if "aux" in r]
    assert any(r["n_dead"] > 0 and r["aux"] > 0 for r in rows)


def test_quick_eval_reports_bounded_metrics():
    from interp.sae import TopKSAE
    sae = TopKSAE(8, 16, 4, seed=0)
    ev = quick_eval(sae, torch.randn(64, 8))
    assert set(ev) == {"fvu", "l0", "alive_frac"}
    assert ev["fvu"] >= 0 and 0 < ev["l0"] <= 4 and 0 < ev["alive_frac"] <= 1
