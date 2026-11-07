"""Writes notebooks/kaggle_steering.ipynb (probe or full steering run on a Kaggle T4). Cells use subprocess, not shell
magics, so the same cells run in a local Jupyter kernel for the pre-flight check (analysis/check_steering_notebook.py).
    python analysis/gen_steering_nb.py
"""
import json

MD0 = """# MiniGPT-Stack Phase 6 / C - SAE-feature STEERING (SFT model, layer 4)

Rules: `docs/phase6_preregistration.md` section C (unchanged) and `docs/steering_features.md` (features fixed before this run).
`MODE = "probe"` is a **timing probe only** (one feature, 3 prompts, all 19 conditions; labeled not-a-result; hard 20 min cap that
ends as an ERROR). `MODE = "full"` is the pre-registered run (5 features x 50 prompts x n=8, max_new 256) and ends with the
pre-registered verdict. SHA-256 gates on the SFT weights and the SAE stop the run on a mismatch.

**Attach three datasets:** `glassbox-dpo-init` (contains `sft_final_best.pt`), `glassbox-sae-l4` (contains `sae_latest.pt`),
`tinystories-shards-8192` (validation shard for the prompts)."""

PARAMS = '''MODE = "full"             # "probe" (timing only) or "full" (the pre-registered run)
PROBE_CAP_S = 20 * 60     # probe: past this the kernel raises -> ERROR (never COMPLETE)
FULL_CAP_S = int(5.0 * 3600)   # full: hard stop = ERROR; probe measured 3.4 h (extrapolated), resumable
SFT_BYTES, SFT_SHA = 125_893_429, "d043156ee2d23aed1336cb42c80efe8cd603c40421a469d4d9519d230cbc3a5c"
SAE_BYTES, SAE_SHA = 16_831_963, "bd9ec5e3192861b16b9976eb3050077e959debb611f051dc382c494e64a502e9"
import time
T0 = time.time()
assert MODE in ("probe", "full")
print("[steer] MODE", MODE, flush=True)'''

GPU = '''import torch
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", "| torch", torch.__version__)
DEVICE = "cuda"
assert torch.cuda.is_available(), "enable GPU (T4) in notebook settings"'''

CLONE = '''import os, subprocess, sys
REPO, ROOT = "https://github.com/Priyanshaur/Glassbox.git", "/kaggle/working/Glassbox"
if not os.path.exists(ROOT):
    subprocess.run(["git", "clone", REPO, ROOT], check=True)
else:
    subprocess.run(["git", "-C", ROOT, "pull", "--ff-only"], check=True)
os.chdir(ROOT); sys.path.insert(0, ROOT)
COMMIT = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
import inspect
from inference.generate import Generator
assert "inject" in inspect.signature(Generator.generate).parameters, "clone lacks the inject commit"
print("[steer] repo commit", COMMIT, "| Generator.generate has inject", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "regex", "pyyaml", "pytest", "tqdm"], check=True)'''

TESTS = '''import subprocess, sys
p = subprocess.run([sys.executable, "-m", "pytest", "tests/test_generate_inject.py", "tests/test_steering.py",
                    "tests/test_hooks.py", "-q"], capture_output=True, text=True)
print(p.stdout[-600:])
if p.returncode != 0:
    raise RuntimeError("[steer] fast tests FAILED; stopping")
print("[steer] fast tests: PASS", flush=True)'''

LOCATE = '''import glob, hashlib, os
def find(name, nbytes, sha):
    hits = glob.glob(INPUT_ROOT + "/**/" + name, recursive=True)
    assert hits, f"attach the dataset containing {name}"
    path = hits[0]
    assert os.path.getsize(path) == nbytes, f"{name}: size {os.path.getsize(path)} != {nbytes}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    print(f"[steer] {name} sha256 {h.hexdigest()} expected {sha} ->", "MATCH" if h.hexdigest() == sha else "MISMATCH", flush=True)
    assert h.hexdigest() == sha, f"{name} hash mismatch; stopping"
    return path
INPUT_ROOT = globals().get("INPUT_ROOT", "/kaggle/input")
SFT_PATH, SAE_PATH = find("sft_final_best.pt", SFT_BYTES, SFT_SHA), find("sae_latest.pt", SAE_BYTES, SAE_SHA)
if "SHARD_DIR" not in globals():
    metas = glob.glob(INPUT_ROOT + "/**/meta.json", recursive=True)
    assert metas, "attach the shards dataset"
    SHARD_DIR = os.path.dirname(metas[0])
print("[steer] shards", SHARD_DIR, sorted(os.listdir(SHARD_DIR))[:4], flush=True)'''

RUN = '''import subprocess, sys, time, os
OUT = globals().get("OUT", "/kaggle/working/steer_" + MODE)
args = [sys.executable, "-u", "analysis/steering.py", "gen", "--out_dir", OUT, "--sae", SAE_PATH, "--weights", SFT_PATH,
        "--data_dir", SHARD_DIR, "--device", DEVICE, "--commit", COMMIT,
        "--expect_sae_sha", SAE_SHA, "--expect_weights_sha", SFT_SHA]
if MODE == "probe":
    args += ["--features", "1672", "--n_prompts", str(globals().get("PROBE_PROMPTS", 3)), "--smoke"]
args += globals().get("EXTRA_ARGS", [])
cap = PROBE_CAP_S if MODE == "probe" else FULL_CAP_S
t = time.time()
try:
    rc = subprocess.run(args, timeout=cap).returncode
except subprocess.TimeoutExpired:
    raise RuntimeError(f"[steer] TRUNCATED: {MODE} hit the {cap}s cap before finishing; resumable from {OUT}/cells.jsonl (ERROR, not COMPLETE)")
RUN_S = time.time() - t
if rc != 0:
    raise RuntimeError(f"[steer] generation failed rc={rc}")
print(f"[steer] generation finished in {RUN_S:.0f}s", flush=True)'''

FINISH = '''import json
cells = sum(1 for _ in open(OUT + "/cells.jsonl"))
print(f"[steer] cells written: {cells} | {RUN_S / max(cells, 1):.2f} s/cell (includes the unsteered NLL scoring)")
if MODE == "probe":
    full_cells = (1 + 5 * 18) * 50
    print(f"[steer] PROBE ONLY - not a result. Full run = {full_cells} cells -> {full_cells * RUN_S / max(cells, 1) / 3600:.2f} GPU hours "
          f"(extrapolated from this probe; the probe has 19 conditions x 3 prompts, so it is representative)")
else:
    p = subprocess.run([sys.executable, "analysis/steering.py", "analyze", "--out_dir", OUT], capture_output=True, text=True)
    print(p.stdout[-3000:]); print(p.stderr[-800:])
    if p.returncode != 0:
        raise RuntimeError("[steer] analysis failed")
print("[steer] COMPLETE", MODE, flush=True)'''


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


cells = [md(MD0), code(PARAMS), md("## GPU check"), code(GPU), md("## Clone and tests"), code(CLONE), code(TESTS),
         md("## Locate and verify inputs"), code(LOCATE), md("## Generate"), code(RUN), md("## Summary"), code(FINISH)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/kaggle_steering.ipynb", "w", encoding="utf-8"), indent=1)
print("wrote notebooks/kaggle_steering.ipynb")
