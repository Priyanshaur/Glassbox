"""Config system test: all YAMLs load and carry required keys incl. checkpoint block."""
import glob

from common.config import load_config, with_overrides

REQUIRED_TOP = {"project", "run_name", "seed", "model", "train", "checkpoint", "data", "logging"}
REQUIRED_MODEL = {"vocab_size", "n_layers", "d_model", "n_heads", "context_len"}
REQUIRED_CKPT = {"dir", "save_every", "keep_last_n"}


def test_all_configs_valid():
    paths = sorted(glob.glob("configs/*.yaml"))
    assert len(paths) >= 3, f"expected >=3 configs, found {paths}"
    for p in paths:
        cfg = load_config(p)
        assert REQUIRED_TOP <= set(cfg), f"{p} missing keys: {REQUIRED_TOP - set(cfg)}"
        assert REQUIRED_MODEL <= set(cfg["model"]), f"{p} model block incomplete"
        assert REQUIRED_CKPT <= set(cfg["checkpoint"]), f"{p} checkpoint block incomplete"


def test_overrides_reject_unknown_keys():
    cfg = load_config("configs/debug_cpu.yaml")
    try:
        with_overrides(cfg, not_a_key=1)
    except KeyError:
        return
    raise AssertionError("with_overrides should reject unknown keys")


def test_debug_is_tiny():
    cfg = load_config("configs/debug_cpu.yaml")
    assert cfg["model"]["n_layers"] == 2
    assert cfg["model"]["d_model"] == 64
    assert cfg["train"]["use_fp16"] is False
