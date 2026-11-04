"""(Needs ipykernel + nbclient.) Run the probe notebook's REAL watchdog cell and COMPLETE guard in a real Jupyter kernel
(ipykernel via nbclient, the mechanism a notebook runner uses), with the cap shrunk to 4 s.

Case 1  hung stage         : cap fires while a cell sleeps -> kernel dies, run FAILS, no COMPLETE.
Case 2  normal finish      : COMPLETED set, watchdog disarmed -> COMPLETE printed, run succeeds.
Case 3  fallback (timer)   : faulthandler unusable -> the timer thread must end the run the same way.
"""
import json
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, DeadKernelError

NB = "notebooks/kaggle_sae_probe.ipynb"
src = json.load(open(NB, encoding="utf-8"))
cells = ["".join(c["source"]) for c in src["cells"]]
flags = next(c for c in cells if c.startswith("# ---- PINNED"))
guard = next(c for c in cells if c.startswith("if COMPLETED"))
assert "HARD_CAP_S = 30 * 60" in flags and "def arm_watchdog" in flags and "def disarm_watchdog" in flags
flags = flags.replace("HARD_CAP_S = 30 * 60", "HARD_CAP_S = 4")


def run(label, setup_extra, middle):
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(setup_extra + flags), nbformat.v4.new_code_cell(middle),
                nbformat.v4.new_code_cell(guard)]
    client = NotebookClient(nb, timeout=90, kernel_name="python3", allow_errors=False)
    t0 = time.time()
    outcome = "SUCCEEDED"
    try:
        client.execute()
    except DeadKernelError:
        outcome = "FAILED (DeadKernelError: the kernel process died)"
    except CellExecutionError as e:
        outcome = f"FAILED (CellExecutionError: {str(e).splitlines()[-1][:90]})"
    except Exception as e:                                      # noqa: BLE001
        outcome = f"FAILED ({type(e).__name__}: {str(e)[:90]})"
    text = ""
    for c in nb.cells:
        for o in c.get("outputs", []):
            text += (o.get("text", "") or "")
            if o.get("output_type") == "error":
                text += f"[error] {o.get('ename')}: {o.get('evalue')}\n"
    armed = [l for l in text.splitlines() if "watchdog armed" in l]
    print(f"--- {label}: run {outcome} after {time.time() - t0:.1f}s")
    print("    ", armed[0] if armed else "watchdog line missing")
    print("     COMPLETE printed:", "[probe] COMPLETE" in text)
    return outcome, text


o1, t1 = run("case 1: hung stage, cap 4s", "",
             "import time\nprint('stage start', flush=True)\ntime.sleep(25)\nprint('stage end')")
o2, t2 = run("case 2: normal finish", "",
             "COMPLETED = True\ndisarm_watchdog()\nprint('stage ok')")
break_fh = "import faulthandler\n\ndef _boom(*a, **k):\n    raise OSError('simulated')\n\nfaulthandler.dump_traceback_later = _boom\n"
o3, t3 = run("case 3: faulthandler unusable -> timer fallback", break_fh,
             "import time\nprint('stage start', flush=True)\ntime.sleep(25)\nprint('stage end')")
ok = (o1.startswith("FAILED") and "[probe] COMPLETE" not in t1 and "armed via faulthandler" in t1
      and o2 == "SUCCEEDED" and "[probe] COMPLETE" in t2
      and o3.startswith("FAILED") and "[probe] COMPLETE" not in t3 and "armed via timer" in t3)
print("WATCHDOG BEHAVES AS CLAIMED:", ok)
sys.exit(0 if ok else 1)
