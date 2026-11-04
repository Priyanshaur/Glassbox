"""(Needs ipykernel + nbclient.) Execute the S4 probe notebook's cells in a REAL Jupyter kernel (nbclient) on CPU with the debug
config. Only paths, device, config and sizes are swapped; clone/install/GPU-check/shards cells are
skipped (no network, no GPU). Everything else - flags, watchdog, SHA gate, tests, probes, estimate,
report, COMPLETE guard - runs as written."""
import hashlib
import json
import os
import sys

import nbformat
import torch
from nbclient import NotebookClient

OUT = sys.argv[1].replace("\\", "/")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.getcwd())
from common.config import load_config
from model.gpt import GPT

torch.manual_seed(0)
WDBG = OUT + "/base_debug.pt"
torch.save(GPT(load_config("configs/debug_cpu.yaml")["model"]).state_dict(), WDBG)
sha = hashlib.sha256(open(WDBG, "rb").read()).hexdigest()

src = json.load(open("notebooks/kaggle_sae_probe.ipynb", encoding="utf-8"))
cells = ["".join(c["source"]) for c in src["cells"]]
pick = lambda pre: next(c for c in cells if c.startswith(pre))
sub = lambda s: s.replace("/kaggle/working/", OUT + "/")

overrides = f'''
import torch
torch.cuda.synchronize = lambda *a, **k: None
CONFIG = "configs/debug_cpu.yaml"; DEVICE = "cpu"; LAYER = 1; N_FEATURES = 128; K = 4; LR = 2e-3
WARMUP = 4; BATCH_TOKENS = 256; BUFFER_TOKENS = 1024; EPOCHS = 6; CTX = 64; NORM_WINDOWS = 8
EVAL_WINDOWS = 8; EVAL_TOKENS = 512; DEAD_TOKENS = 2048
WEIGHTS_BYTES = {os.path.getsize(WDBG)}; WEIGHTS_SHA256 = "{sha}"; WDBG = "{WDBG}"
'''
inputs = pick("import glob, hashlib").replace(
    'hits = glob.glob("/kaggle/input/**/" + WEIGHTS_FILE, recursive=True)', "hits = [WDBG]")
order = [pick("# ---- PINNED"), overrides, inputs, pick("import subprocess, sys"),
         sub(pick("import json, numpy as np")), sub(pick("from interp.train_sae")),
         sub(pick("from interp import eval_sae")), sub(pick("import math")), pick("if COMPLETED")]
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(c) for c in order]
client = NotebookClient(nb, timeout=1200, kernel_name="python3", allow_errors=False)
try:
    client.execute()
    status = "SUCCEEDED"
except Exception as e:                                          # noqa: BLE001
    status = f"FAILED: {type(e).__name__}: {str(e)[-300:]}"
text = ""
for c in nb.cells:
    for o in c.get("outputs", []):
        text += (o.get("text", "") or "")
print("kernel run:", status)
for key in ("watchdog armed", "sha256 expected", "sha256 in-kernel", "weights file =", "fast tests",
            "scale (real", "best extraction", "stop rule", "(probe, not acceptance)", "COMPLETE"):
    for l in text.splitlines():
        if key in l:
            print("  ", l[:170])
            break
rep = json.load(open(OUT + "/probe_report.json")) if os.path.exists(OUT + "/probe_report.json") else {}
print("report keys:", sorted(rep)[:30])
print("hash_gate:", rep.get("hash_gate"), "| weights_sha256 logged:", rep.get("weights_sha256") == sha, "| watchdog:", rep.get("watchdog"))
sys.exit(0 if status == "SUCCEEDED" else 1)
