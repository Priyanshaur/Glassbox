"""analysis/a5_transfer.py: the pre-registered A5 verdict logic (fails if either input type fails; NaN never a pass)."""
import math

from analysis.a5_transfer import judge


def res(pf, pc, tf, tc):
    return {"plain": {"fvu": pf, "ce_recovered": pc}, "template": {"fvu": tf, "ce_recovered": tc}}


def test_passes_only_when_both_input_types_meet_both_thresholds():
    assert judge(res(0.25, 0.80, 0.25, 0.80))["A5_transfer"] is True            # boundaries are inclusive
    assert judge(res(0.26, 0.90, 0.10, 0.90))["A5_transfer"] is False           # plain FVU fails
    assert judge(res(0.10, 0.79, 0.10, 0.90))["A5_transfer"] is False           # plain CE fails
    assert judge(res(0.10, 0.90, 0.10, 0.79))["A5_transfer"] is False           # template CE fails
    assert judge(res(0.10, 0.90, 0.30, 0.90))["A5_transfer"] is False           # template FVU fails


def test_undefined_ce_recovered_is_undetermined_not_a_pass_or_silent_fail():
    out = judge(res(0.10, math.nan, 0.10, 0.90))
    assert out["A5_transfer"] is None and out["plain"]["ce_ok"] is None
    assert judge(res(0.30, math.nan, 0.10, 0.90))["A5_transfer"] is False       # a definite failure still fails


def test_measure_skips_saved_cells_and_saves_per_cell(monkeypatch):
    import analysis.a5_transfer as M
    calls = []
    monkeypatch.setattr(M, "_plain", lambda *a: (calls.append("plain"), a[8].__setitem__("plain", {"fvu": 0.1})))
    monkeypatch.setattr(M, "_template", lambda *a: (calls.append("template"), a[7].__setitem__("template", {"fvu": 0.2})))
    saves = []
    out = M.measure(None, None, 4, 1.0, ([], []), None, 1, lambda s: None, out={"plain": {"fvu": 0.9}},
                    on_cell=lambda: saves.append(1))
    assert calls == ["template"] and len(saves) == 1 and out["plain"]["fvu"] == 0.9   # saved cell untouched, one save
    calls.clear(); saves.clear()
    M.measure(None, None, 4, 1.0, ([], []), None, 1, lambda s: None, out={}, on_cell=lambda: saves.append(1))
    assert calls == ["plain", "template"] and len(saves) == 2
