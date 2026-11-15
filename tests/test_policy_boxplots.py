"""scripts/robot_policy_boxplots.py: Holm, paired tests, and the per-run layout."""

from __future__ import annotations

import numpy as np

from _helpers import load_script

pb = load_script("scripts/robot_policy_boxplots.py", "_policy_boxplots_under_test")


def test_holm_matches_hand_computation():
    assert np.allclose(pb.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def _run(seed, shift):
    rng = np.random.default_rng(seed)
    n = 300
    base = rng.random(n) + 0.5
    arm = lambda k: {"envelope_growth_m": base * k + 0.01 * rng.random(n), "peak_force_critical_n": np.zeros(n),  # noqa: E731
                     "peak_force_any_n": 1000 * base * k, "steps": 100 * base * k, "settled": np.ones(n)}
    return {"a": arm(0.6 - shift), "b": arm(0.6), "sf": arm(0.8), "walker": arm(2.0), "limp": arm(1.0)}


def test_new_policy_lower_than_reference_in_every_run():
    runs = {s: _run(s, 0.1) for s in (0, 1, 2)}
    arms = ["a", "b", "sf", "walker", "limp"]
    res = pb.analyse(runs, arms, ["a", "b"], "sf")
    rows = {(r["a"], r["b"]): r for r in res["pairwise_summary"]["envelope_growth_m"]}
    assert rows[("a", "sf")]["runs_a_lower"] == 3 and rows[("b", "sf")]["runs_a_lower"] == 3
    assert rows[("a", "b")]["runs_a_lower"] == 3            # new policies compared with each other
    assert rows[("a", "limp")]["runs_a_lower"] == 3
    assert len(res["pairs"]) == 2 + 2 + 1
    assert res["per_arm"]["walker"]["settled"]["mean"] == 1.0
