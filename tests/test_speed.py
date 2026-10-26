"""bench/speed.py: statistics helpers and an end-to-end CPU smoke on the debug model."""
import json

import numpy as np

from bench import speed


def test_summarize_and_paired_ratio_ci_verdicts():
    s = speed.summarize([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s["median"] == 3.0 and s["min"] == 1.0 and s["max"] == 5.0 and s["p25"] == 2.0
    rng = np.random.default_rng(0)
    base = 1.0 + 0.01 * rng.standard_normal(10)
    pt, lo, hi, verdict = speed.paired_ratio_ci(2 * base, base)       # clearly faster
    assert 1.9 < pt < 2.1 and lo > 1 and verdict == "faster"
    pt, lo, hi, verdict = speed.paired_ratio_ci(base, 2 * base)       # clearly slower
    assert hi < 1 and verdict == "slower"
    noise = 1.0 + 0.3 * rng.standard_normal(10)
    assert speed.paired_ratio_ci(noise, noise.copy())[3] == "no measurable difference"


def test_speed_smoke_debug_model(tmp_path):
    out = str(tmp_path / "bench")
    rc = speed.main(["--config", "configs/debug_cpu.yaml", "--weights", "none", "--prompts", "2",
                     "--gen", "8", "--repeats", "2", "--warmup", "1", "--ns", "1,2",
                     "--grid", "full", "--long_len", "16", "--out_prefix", out])
    assert rc == 0
    r = json.load(open(out + "_raw.json"))
    assert r["params"]["gen"] == 8 and r["dropped_by_gate"] == []
    # every cache config ran, each with R=2 repeats of K=2 prompts and a [ttft, decode] pair
    names = r["raw_per_repeat_per_prompt_ttft_decode_s"]
    assert any(k.startswith("fp32/none/") for k in names) and any("int8_all/static" in k for k in names)
    for rp in names.values():
        assert len(rp) == 2 and len(rp[0]) == 2 and len(rp[0][0]) == 2
    gate = r["gate"]
    assert all(v["pass"] for v in gate.values()), gate
    assert any(c["comparison"].startswith("static vs tuple") for c in r["comparisons"])
    md = open(out + ".md", encoding="utf-8").read()
    assert "Correctness gate" in md and "speedup" in md and "Environment" in md


def test_speed_time_budget_models_filter_and_partial_dump(tmp_path):
    """--budget_s stops after >= 2 repeats and says so; --models fp32 drops int8;
    the per-repeat raw dump exists and the final dump replaces it."""
    out = str(tmp_path / "b")
    rc = speed.main(["--config", "configs/debug_cpu.yaml", "--weights", "none", "--prompts", "2",
                     "--gen", "8", "--repeats", "4", "--warmup", "1", "--ns", "1",
                     "--grid", "reduced", "--long_len", "16", "--models", "fp32",
                     "--budget_s", "0.0001", "--out_prefix", out])
    assert rc == 0
    r = json.load(open(out + "_raw.json"))
    assert "partial" not in r                                  # final dump replaced the partial one
    assert r["params"]["repeats"] == 2 and r["params"]["repeats_requested"] == 4
    names = r["raw_per_repeat_per_prompt_ttft_decode_s"]
    assert not any("int8" in k or "fp16" in k for k in names)
    assert all(len(v) == 2 for v in names.values())
    assert "stopped by the time budget" in open(out + ".md", encoding="utf-8").read()


def test_speed_fp16_path_exercised_on_cpu(tmp_path):
    """The fp16 config (static cache, fp32 sampling of fp16 logits, informational gate,
    fp16-vs-fp32 comparison) runs end to end; on a GPU it is the same code."""
    out = str(tmp_path / "h")
    rc = speed.main(["--config", "configs/debug_cpu.yaml", "--weights", "none", "--prompts", "2",
                     "--gen", "8", "--repeats", "2", "--warmup", "1", "--ns", "1,2",
                     "--grid", "reduced", "--long_len", "16", "--models", "fp32,fp16",
                     "--allow_cpu_fp16", "--out_prefix", out])
    assert rc == 0
    r = json.load(open(out + "_raw.json"))
    names = r["raw_per_repeat_per_prompt_ttft_decode_s"]
    assert "fp16/static/n=1/short" in names and "fp16/static/n=2/long" in names
    assert any(k.startswith("fp16|") for k in r["gate"]) and all(v["pass"] for v in r["gate"].values())
    assert any(c["comparison"].startswith("fp16 static vs fp32 static") for c in r["comparisons"])
