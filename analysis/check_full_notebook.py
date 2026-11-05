"""(Needs ipykernel + nbclient.) Execute the layer-4 FULL-RUN notebook's cells in a REAL Jupyter kernel on CPU with the
debug config, in one of two scenarios:

    python analysis/check_full_notebook.py <out_dir> complete    # 6 epochs  -> must end COMPLETE
    python analysis/check_full_notebook.py <out_dir> truncate    # limit hit -> must end as an ERROR, never COMPLETE

Only paths, device, config and sizes are swapped (clone/install/GPU-check/shards cells are skipped: no network, no
GPU). The flags, watchdog, SHA gate, fast tests, training, truncation rule, evaluation, report and COMPLETE guard
run as written. Exit code 0 means the scenario behaved as required."""
import hashlib
import json
import os
import sys

import nbformat
import torch
from nbclient import NotebookClient

OUT = sys.argv[1].replace("\\", "/")
MODE = sys.argv[2]
assert MODE in ("complete", "truncate")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.getcwd())
from common.config import load_config
from model.gpt import GPT

torch.manual_seed(0)
WDBG = OUT + "/base_debug.pt"
torch.save(GPT(load_config("configs/debug_cpu.yaml")["model"]).state_dict(), WDBG)
sha = hashlib.sha256(open(WDBG, "rb").read()).hexdigest()

src = json.load(open("notebooks/kaggle_sae_full.ipynb", encoding="utf-8"))
cells = ["".join(c["source"]) for c in src["cells"]]
pick = lambda pre: next(c for c in cells if c.startswith(pre))
sub = lambda s: s.replace("/kaggle/working/", OUT + "/")

max_hours = "1e-9" if MODE == "truncate" else "4.5"
overrides = f'''
import torch
torch.cuda.synchronize = lambda *a, **k: None
CONFIG = "configs/debug_cpu.yaml"; DEVICE = "cpu"; LAYER = 1; N_FEATURES = 128; K = 4; LR = 2e-3
WARMUP = 4; BATCH_TOKENS = 256; BUFFER_TOKENS = 1024; EPOCHS = 6; CTX = 64; NORM_WINDOWS = 8
EVAL_WINDOWS = 8; EVAL_EVERY = 2; CKPT_EVERY = 2; MAX_HOURS = {max_hours}
EVAL_TOKENS = 512; DEAD_TOKENS = 2048; COMMIT = "local"
WEIGHTS_BYTES = {os.path.getsize(WDBG)}; WEIGHTS_SHA256 = "{sha}"; WDBG = "{WDBG}"
'''
inputs = pick("import glob, hashlib").replace(
    'hits = glob.glob("/kaggle/input/**/" + WEIGHTS_FILE, recursive=True)', "hits = [WDBG]")
order = [pick("# ---- PINNED"), overrides, inputs, pick("import subprocess, sys"),
         sub(pick("import json, numpy as np")), sub(pick("import glob, json, shutil")),
         sub(pick("from interp import eval_sae")), sub(pick("RUN[\"status\"] = \"FINISHED")), pick("if COMPLETED")]
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(c) for c in order]
client = NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=False)
try:
    client.execute()
    status = "SUCCEEDED"
except Exception as e:                                          # noqa: BLE001
    status = f"FAILED: {type(e).__name__}: {str(e)[-400:]}"
text = ""
for c in nb.cells:
    for o in c.get("outputs", []):
        text += (o.get("text", "") or "")
has_complete = "[full] COMPLETE" in text
print("kernel run:", status)
print("COMPLETE printed:", has_complete)
for key in ("watchdog armed", "sha256 in-kernel", "fast tests", "best extraction", "training budget", "trained ",
            "VERDICT", "COMPLETE"):
    for l in text.splitlines():
        if key in l:
            print("  ", l[:190])
            break
rp = OUT + "/full_report.json"
rep = json.load(open(rp)) if os.path.exists(rp) else {}
print("report status:", rep.get("status"), "| epochs_done:", rep.get("training", {}).get("epochs_done"),
      "| ckpt kept:", os.path.exists(OUT + "/sae_full/ckpt.pt"))
if MODE == "complete":
    ok = status == "SUCCEEDED" and has_complete and rep.get("status") == "FINISHED_192_EPOCHS" \
        and rep["training"]["epochs_done"] == 6 and rep.get("hash_gate") == "PASS"
else:
    ok = status.startswith("FAILED") and "TRUNCATED" in status and not has_complete \
        and rep.get("status") == "TRUNCATED" and os.path.exists(OUT + "/sae_full/ckpt.pt")
print("SCENARIO", MODE, "->", "OK" if ok else "NOT AS REQUIRED")
sys.exit(0 if ok else 1)
