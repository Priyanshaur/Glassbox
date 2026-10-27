"""Phase 6 / S1: the activation pipeline (debug model, local smoke shards)."""
import numpy as np
import torch

from common.config import load_config
from interp import activations as A
from model.gpt import GPT

CTX = 64
LAYER = 1


def model():
    torch.manual_seed(5)
    return GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()


def loaders():
    return A.ShardLoader("data", "train"), A.ShardLoader("data", "validation")


def test_buffer_is_deterministic_and_epoch_dependent():
    m, (tr, _) = model(), loaders()
    a = A.fill_buffer(m, tr, LAYER, 3, 0, 1024, CTX, 1.0)
    b = A.fill_buffer(m, tr, LAYER, 3, 0, 1024, CTX, 1.0)
    c = A.fill_buffer(m, tr, LAYER, 3, 1, 1024, CTX, 1.0)
    assert a.shape == (1024, 64) and torch.equal(a, b) and not torch.equal(a, c)


def test_buffer_rows_equal_the_models_recorded_residual():
    m, (tr, _) = model(), loaders()
    buf = A.fill_buffer(m, tr, LAYER, 3, 0, 4 * CTX, CTX, 2.5)
    ids = A.draw_windows(tr, 3, 0, 4, CTX)
    with torch.no_grad():
        want = m(ids, record_layers=[LAYER]).recorded[LAYER].reshape(-1, 64) * 2.5
    assert torch.allclose(buf, want, atol=1e-6)


def test_chunk_size_does_not_change_activations():
    m, (tr, _) = model(), loaders()
    a = A.fill_buffer(m, tr, LAYER, 1, 0, 8 * CTX, CTX, 1.0, chunk=1)
    b = A.fill_buffer(m, tr, LAYER, 1, 0, 8 * CTX, CTX, 1.0, chunk=8)
    assert torch.allclose(a, b, atol=1e-5)


def test_epoch_batches_cover_each_row_once_deterministically():
    buf = torch.arange(100 * 3, dtype=torch.float32).reshape(100, 3)
    got = list(A.epoch_batches(buf, 16, 7, 2))
    assert len(got) == 6 and all(g.shape == (16, 3) for g in got)         # 96 of 100 rows, partial dropped
    rows = torch.cat(got)[:, 0]
    assert rows.unique().numel() == 96                                    # no row repeated
    again = torch.cat(list(A.epoch_batches(buf, 16, 7, 2)))
    assert torch.equal(torch.cat(got), again)
    assert not torch.equal(torch.cat(got), torch.cat(list(A.epoch_batches(buf, 16, 7, 3))))


def test_norm_scale_gives_mean_norm_sqrt_d():
    m, (tr, _) = model(), loaders()
    s = A.estimate_norm_scale(m, tr, LAYER, 0, 16, CTX)
    buf = A.fill_buffer(m, tr, LAYER, 0, 0, 16 * CTX, CTX, s)
    # the scale is estimated on seed/step 0 windows, the buffer on seed 0 epoch 0: same windows
    assert abs(float(buf.norm(dim=-1).mean()) - np.sqrt(64)) < 1e-3
    st = A.norm_stats(m, tr, LAYER, 0, 16, CTX)
    assert st["tokens"] == 16 * CTX and st["max_over_mean"] >= 1.0 and "pos0_mean_norm" in st


def test_eval_activations_come_from_the_validation_split_and_are_fixed():
    m, (_, va) = model(), loaders()
    a, ids = A.eval_activations(m, va, LAYER, 8, CTX, 1.0)
    b, _ = A.eval_activations(m, va, LAYER, 8, CTX, 1.0)
    assert a.shape == (8 * CTX, 64) and torch.equal(a, b) and ids.shape == (8, CTX)
