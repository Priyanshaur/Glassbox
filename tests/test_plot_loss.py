"""analysis/plot_loss.py: log parsing (no matplotlib needed) and the drawing itself (skipped without matplotlib)."""
import json

import pytest

from analysis import plot_loss as P


def _write(tmp_path, name, last_step):
    d = tmp_path / name
    d.mkdir()
    rows = [{"step": s, "loss": 9.0 / (1 + s / 100), **({"val": 9.0 / (1 + s / 100) + 0.01} if s % 1000 == 0 else {})}
            for s in range(0, last_step + 1, 50)]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_runs_keeps_long_runs_and_skips_short_ones(tmp_path):
    _write(tmp_path, "long", 3000)
    _write(tmp_path, "smoke", 200)
    runs = P.load_runs(str(tmp_path / "**" / "metrics.jsonl"), min_last_step=1000)
    assert list(runs) == ["long"]
    r = runs["long"]
    assert r["step"][0] == 0 and r["step"][-1] == 3000 and r["val_step"] == [0, 1000, 2000, 3000]
    assert len(r["loss"]) == len(r["step"]) and r["loss"][0] > r["loss"][-1]


def test_running_mean_matches_a_direct_computation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    got = P.running_mean(xs, 3)
    assert got[:3] == [1.0, 1.5, 2.0] and got[3:] == [3.0, 4.0, 5.0]


def test_main_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    _write(tmp_path, "long", 3000)
    out = tmp_path / "fig" / "loss.png"
    P.main(["--pattern", str(tmp_path / "**" / "metrics.jsonl"), "--out", str(out)])
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
