"""Plot the pretraining loss curve from the training logs.

    pip install -r requirements-plots.txt
    python analysis/plot_loss.py                      # reads checkpoints/**/metrics.jsonl, writes docs/figures/pretrain_loss.png

Each `metrics.jsonl` line is one logged step: {"step", "loss", ...} and, every `val_every` steps, a "val" field. Runs shorter than
`--min_steps` logged steps' worth of training (the debug / smoke runs) are skipped. Nothing is smoothed except the optional
running mean drawn on top of the raw training loss.
"""
import argparse
import glob
import json
import os


def load_runs(pattern, min_last_step=1000):
    """{run name: {"step": [...], "loss": [...], "val_step": [...], "val": [...]}} for every metrics file whose last
    logged step is at least `min_last_step`. The run name is the file's parent folder."""
    runs = {}
    for path in sorted(glob.glob(pattern, recursive=True)):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        rows = [r for r in rows if "step" in r and "loss" in r]
        if not rows or max(r["step"] for r in rows) < min_last_step:
            continue
        rows.sort(key=lambda r: r["step"])
        runs[os.path.basename(os.path.dirname(path))] = {
            "step": [r["step"] for r in rows], "loss": [r["loss"] for r in rows],
            "val_step": [r["step"] for r in rows if "val" in r], "val": [r["val"] for r in rows if "val" in r]}
    return runs


def running_mean(xs, w):
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= w:
            s -= xs[i - w]
        out.append(s / min(i + 1, w))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="checkpoints/**/metrics.jsonl")
    ap.add_argument("--out", default="docs/figures/pretrain_loss.png")
    ap.add_argument("--min_last_step", type=int, default=1000)
    a = ap.parse_args(argv)
    runs = load_runs(a.pattern, a.min_last_step)
    if not runs:
        raise SystemExit(f"no metrics file matching {a.pattern} reaches step {a.min_last_step}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=130)
    for name, r in runs.items():
        ax.plot(r["step"], r["loss"], lw=0.8, alpha=0.45, label=f"{name}: training loss (every 50 steps)")
        ax.plot(r["step"], running_mean(r["loss"], 5), lw=1.6, label=f"{name}: running mean of 5 logs")
        ax.plot(r["val_step"], r["val"], "o-", ms=4, lw=1.2, label=f"{name}: validation loss (fixed windows)")
        ax.annotate(f"val {r['val'][-1]:.3f} at step {r['val_step'][-1]}", (r["val_step"][-1], r["val"][-1]),
                    textcoords="offset points", xytext=(-110, 14), fontsize=8)
    ax.set_xlabel("step (65,536 tokens per step)")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.set_yticks([1.2, 1.5, 2, 3, 5, 9])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title("Pretraining loss, 31.5M-parameter GPT on TinyStories")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    print(f"wrote {a.out} from {', '.join(runs)}")


if __name__ == "__main__":
    main()
