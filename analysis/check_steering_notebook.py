"""(Needs ipykernel + nbclient.) Runs the steering notebook's cells in a REAL Jupyter kernel on CPU with tiny sizes:
    python analysis/check_steering_notebook.py <out_dir> probe       # must end COMPLETE and print the GPU-hours extrapolation
    python analysis/check_steering_notebook.py <out_dir> truncate    # cap hit -> must raise (ERROR), never COMPLETE
Only paths, device and sizes are swapped (GPU check and clone cells are skipped); the SHA gates, subprocess, cap logic and
summary run as written. Exit code 0 means the scenario behaved as required."""
import json
import os
import sys

import nbformat
from nbclient import NotebookClient

OUT = sys.argv[1].replace("\\", "/")
MODE = sys.argv[2]
assert MODE in ("probe", "truncate")
os.makedirs(OUT, exist_ok=True)
src = json.load(open("notebooks/kaggle_steering.ipynb", encoding="utf-8"))
cells = ["".join(c["source"]) for c in src["cells"] if c["cell_type"] == "code"]
pick = lambda pre: next(c for c in cells if c.startswith(pre))
cap = "1" if MODE == "truncate" else "20*60"
over = f'''
import os, sys
INPUT_ROOT = "checkpoints"; SHARD_DIR = "data/full"; DEVICE = "cpu"; OUT = "{OUT}/steer"
PROBE_PROMPTS = 1; EXTRA_ARGS = ["--n", "2", "--max_new", "8"]
COMMIT = "local"; sys.path.insert(0, os.getcwd())
'''
params = pick("MODE =").replace("PROBE_CAP_S = 20 * 60", f"PROBE_CAP_S = {cap}")
order = [params, over, pick("import glob, hashlib"), pick("import subprocess, sys, time, os"), pick("import json\ncells")]
nb = nbformat.v4.new_notebook()
nb.cells = [nbformat.v4.new_code_cell(c) for c in order]
client = NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=False)
try:
    client.execute()
    status = "SUCCEEDED"
except Exception as e:                                          # noqa: BLE001
    status = f"FAILED: {type(e).__name__}: {str(e)[-300:]}"
text = "".join((o.get("text", "") or "") for c in nb.cells for o in c.get("outputs", []))
complete = "[steer] COMPLETE" in text
print("kernel run:", status)
print("COMPLETE printed:", complete)
for key in ("MATCH", "cells written", "PROBE ONLY"):
    for l in text.splitlines():
        if key in l:
            print("  ", l[:200]); break
if MODE == "probe":
    ok = status == "SUCCEEDED" and complete and "PROBE ONLY" in text and text.count("MATCH") >= 2
else:
    ok = status.startswith("FAILED") and "TRUNCATED" in status and not complete
print("SCENARIO", MODE, "->", "OK" if ok else "NOT AS REQUIRED")
sys.exit(0 if ok else 1)
