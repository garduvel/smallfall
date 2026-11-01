"""Tests for src/cartpole_term/protective/compare.py (pure torch)."""

from __future__ import annotations

import math

import numpy as np
import torch

from _helpers import load_module

compare = load_module("protective/compare.py", "_protective_compare_under_test")

CRIT = torch.tensor([True, False])  # body 0 = trunk (critical), body 1 = leg


def _step(tr, *, force, torque, root_x, reward=-1.0, term=None, trunc=None):
    n = len(root_x)
    root = torch.tensor([[x, 0.0] for x in root_x])
    bodies = torch.stack([root, root + torch.tensor([0.3, 0.0])], dim=1)
    return tr.update(
        force_mag=torch.tensor(force, dtype=torch.float32),
        critical_mask=CRIT,
        torque=torch.tensor(torque, dtype=torch.float32),
        root_xy=root,
        body_xy=bodies,
        reward=torch.full((n,), reward),
        terminated=torch.tensor(term or [False] * n),
        truncated=torch.tensor(trunc or [False] * n),
    )


def _tracker(n):
    root = torch.zeros(n, 2)
    bodies = torch.stack([root, root + torch.tensor([0.3, 0.0])], dim=1)
    return compare.FirstFallTracker(root, bodies)


def test_records_peaks_only_during_first_fall():
    tr = _tracker(2)
    _step(tr, force=[[100, 5], [10, 50]], torque=[[3, -8], [1, 1]], root_x=[0.1, 0.1])
    ended = _step(tr, force=[[20, 5], [10, 60]], torque=[[1, 1], [2, 2]], root_x=[0.4, 0.2],
                  term=[True, False])
    assert ended.tolist() == [True, False]
    # env 0 is done: this huge post-fall reading must be ignored
    _step(tr, force=[[9999, 9999], [10, 70]], torque=[[99, 99], [2, 2]], root_x=[5.0, 0.3],
          trunc=[False, True])
    assert tr.all_done
    pf = tr.per_fall()
    assert pf["peak_force_critical_n"].tolist() == [100, 10]
    assert pf["peak_force_any_n"].tolist() == [100, 70]
    assert pf["peak_torque_nm"].tolist() == [8, 2]
    assert pf["steps"].tolist() == [2, 3]
    assert pf["settled"].tolist() == [1.0, 0.0]
    assert pf["return"].tolist() == [-2.0, -3.0]
    assert np.allclose(pf["root_travel_end_m"], [0.4, 0.3])
    # envelope includes the leg 0.3 m ahead of the root; growth excludes the initial 0.3
    assert np.allclose(pf["envelope_m"], [0.7, 0.6])
    assert np.allclose(pf["envelope_growth_m"], [0.4, 0.3])


def test_verdict():
    s = {
        "protective": {m: {"median": 1.0} for m in compare.PASS_METRICS},
        "walker": {m: {"median": 2.0} for m in compare.PASS_METRICS},
        "zero_action": {m: {"median": 1.5} for m in compare.PASS_METRICS},
    }
    assert compare.verdict(s)["pass"] is True
    s["zero_action"]["peak_torque_nm"]["median"] = 0.5
    v = compare.verdict(s)
    assert v["pass"] is False
    assert v["peak_torque_nm"]["beats_all_baselines"] is False


def test_summarize():
    s = compare.summarize({"x": np.array([1.0, 2.0, 3.0, 4.0])})
    assert s["x"]["median"] == 2.5 and s["x"]["max"] == 4.0
    assert math.isclose(s["x"]["mean"], 2.5)


def _summ(envelope, prot_any, settled=1.0):
    walker = {"peak_force_critical_n": {"median": 400.0}, "peak_force_any_n": {"median": 1000.0},
              "peak_torque_nm": {"median": 30.0}, "envelope_growth_m": {"median": 0.8}}
    prot = {"peak_force_critical_n": {"median": 0.0}, "peak_force_any_n": {"median": prot_any},
            "peak_torque_nm": {"median": 12.0}, "envelope_growth_m": {"median": envelope},
            "settled": {"mean": settled}}
    return {"protective": prot, "walker": walker}


def test_select_checkpoint_prefers_force_reducing_then_lowest_ground_covered():
    r = compare.select_checkpoint({
        "7_it120": _summ(0.30, 1150.0),    # least ground, but any-body force above the walker
        "8a_it100": _summ(0.40, 950.0),    # preferred
        "8b_it150": _summ(0.35, 900.0),    # preferred, less ground -> wins
        "8b_it200": _summ(0.20, 800.0, settled=0.90),  # does not settle -> not eligible
    })
    assert r["selected_label"] == "8b_it150"
    assert r["selected_is_eligible"] and r["selected_also_reduces_any_force"]
    assert [c["preferred"] for c in r["candidates"]] == [False, True, True, False]


def test_select_checkpoint_space_wins_when_nothing_reduces_force():
    r = compare.select_checkpoint({"7_it120": _summ(0.30, 1150.0), "8a_it100": _summ(0.40, 1100.0)})
    assert r["selected_label"] == "7_it120"
    assert r["selected_is_eligible"] is True and r["selected_also_reduces_any_force"] is False


def test_select_checkpoint_falls_back_and_says_so():
    r = compare.select_checkpoint({"a": _summ(0.9, 900.0), "b": _summ(0.85, 900.0), "c": _summ(0.1, 900.0, settled=0.5)})
    assert r["selected_label"] == "b" and r["selected_is_eligible"] is False
