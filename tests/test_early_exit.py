"""GPT.forward(stop_after=i): the recorded activations must be BIT-identical to a full forward."""
import os

import pytest
import torch

from common.config import load_config
from model.gpt import GPT
from train.data_loader import ShardLoader

REAL = "checkpoints/glassbox-s3-final/weights_step12158.pt"


def _check(m, ids, layers):
    with torch.no_grad():
        full = m(ids, record_layers=layers)
        for L in layers:
            early = m(ids, record_layers=[L], stop_after=L)
            assert early.logits is None and set(early.recorded) == {L}
            assert torch.equal(early.recorded[L], full.recorded[L]), f"layer {L} differs"
            assert len(early.new_kvs) == L + 1


def test_early_exit_is_bit_identical_on_the_debug_model():
    torch.manual_seed(7)
    m = GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()
    ids = ShardLoader("data", "validation").batch_ids(5, 0, 6, 64)[:, :64]
    _check(m, ids, [0, 1, m.cfg["n_layers"] - 1] if "n_layers" in m.cfg else [0, 1])


def test_early_exit_default_path_is_unchanged():
    torch.manual_seed(7)
    m = GPT(load_config("configs/debug_cpu.yaml")["model"]).eval()
    ids = torch.randint(0, 100, (2, 16))
    with torch.no_grad():
        a, b = m(ids), m(ids, stop_after=None)
    assert torch.equal(a.logits, b.logits)


@pytest.mark.skipif(not os.path.exists(REAL), reason="real weights are not tracked in git")
def test_early_exit_is_bit_identical_on_the_real_weights_layer_4():
    cfg = load_config("configs/base_30m.yaml")
    m = GPT(cfg["model"])
    m.load_state_dict(torch.load(REAL, map_location="cpu", weights_only=True))
    m.eval()
    ids = ShardLoader(cfg["data"]["out_dir"], "validation").batch_ids(5, 0, 4, 512)[:, :512]
    _check(m, ids, [4])
