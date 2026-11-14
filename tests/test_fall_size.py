"""Fall-size measures (protective/fall_size.py) and the ablation that compares
policies across them (protective/fall_size_ablation.py)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from _helpers import load_module

fs = load_module("protective/fall_size.py", "_fall_size_under_test")
abl = load_module("protective/fall_size_ablation.py", "_fall_size_ablation_under_test")
ff = load_module("protective/compare.py", "_compare_for_size_test")


def _script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "robot_fall_size_ablation.py"
    spec = importlib.util.spec_from_file_location("_size_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- measures -----------------------------------------------------------------

def test_support_polygon_area_of_a_square():
    u = fs._directions(fs.HULL_DIRECTIONS, "cpu")
    pts = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    h = (pts @ u.T).amax(0)
    area = float(fs.support_polygon_area(h, u))
    assert 4.0 <= area <= 4.0 * 1.013 + 1e-6  # circumscribes the true hull, < 1.3% over


def test_support_polygon_area_of_a_circle():
    u = fs._directions(fs.HULL_DIRECTIONS, "cpu")
    h = torch.full((fs.HULL_DIRECTIONS,), 0.5)
    assert float(fs.support_polygon_area(h, u)) == pytest.approx(math.pi * 0.25, rel=0.01)


def _two_body_fall():
    """Env 0 slides 1 m along x over two steps; env 1 stays still."""
    root = torch.zeros(2, 2)
    body = torch.tensor([[[0.0, 0.0], [0.2, 0.0]], [[0.0, 0.0], [0.2, 0.0]]])
    t = fs.FallSizeTracker(root, body)
    no = torch.zeros(2, 1, dtype=torch.bool)
    for step, x in enumerate((0.5, 1.0)):
        r = torch.tensor([[x, 0.0], [0.0, 0.0]])
        b = body + r.unsqueeze(1)
        t.update(root_xy=r, body_xy=b, contact_xy=b[:, :1], in_contact=torch.tensor([[True], [False]]),
                 ended=torch.tensor([step == 1, step == 1]))
    return t


def test_tracker_measures_a_slide():
    t = _two_body_fall()
    assert t.all_done
    out = t.per_fall()
    assert tuple(out) == fs.SIZE_METRICS
    assert out["max_reach_m"] == pytest.approx([1.2, 0.2])
    assert out["envelope_growth_m"] == pytest.approx([1.0, 0.0])
    assert out["root_travel_end_m"] == pytest.approx([1.0, 0.0])
    assert out["root_travel_max_m"] == pytest.approx([1.0, 0.0])
    assert out["root_path_m"] == pytest.approx([1.0, 0.0])
    assert out["centroid_travel_end_m"] == pytest.approx([1.0, 0.0])
    assert out["contact_reach_m"] == pytest.approx([1.0, 0.0])
    # body 0 touched cells at x = 0.5 and 1.0 -> two 10 cm cells
    assert out["touched_area_m2"] == pytest.approx([0.02, 0.0])
    # a segment has no area; the hull area of a collinear slide is ~0
    assert out["swept_hull_area_m2"] == pytest.approx([0.0, 0.0], abs=1e-5)


def test_tracker_stops_after_the_first_fall():
    t = _two_body_fall()
    before = t.per_fall()
    far = torch.full((2, 2), 9.0)
    t.update(root_xy=far, body_xy=far.unsqueeze(1).repeat(1, 2, 1), contact_xy=far.unsqueeze(1),
             in_contact=torch.ones(2, 1, dtype=torch.bool), ended=torch.ones(2, dtype=torch.bool))
    after = t.per_fall()
    for k in before:
        assert after[k] == pytest.approx(before[k]), k


def test_ground_covered_matches_the_existing_tracker():
    g = torch.Generator().manual_seed(0)
    n, k = 5, 4
    root = torch.randn(n, 2, generator=g)
    body = root.unsqueeze(1) + 0.3 * torch.randn(n, k, 2, generator=g)
    old, new = ff.FirstFallTracker(root, body), fs.FallSizeTracker(root, body)
    for step in range(6):
        root = root + 0.1 * torch.randn(n, 2, generator=g)
        body = root.unsqueeze(1) + 0.3 * torch.randn(n, k, 2, generator=g)
        end = torch.full((n,), step == 5)
        old.update(force_mag=torch.zeros(n, k), critical_mask=torch.ones(k, dtype=torch.bool),
                   torque=torch.zeros(n, 3), root_xy=root, body_xy=body, reward=torch.zeros(n),
                   terminated=end, truncated=torch.zeros(n, dtype=torch.bool))
        new.update(root_xy=root, body_xy=body, contact_xy=body, in_contact=torch.zeros(n, k, dtype=torch.bool),
                   ended=end)
    assert new.per_fall()["envelope_growth_m"] == pytest.approx(old.per_fall()["envelope_growth_m"], abs=1e-6)


def test_script_metric_names_match_the_module():
    s = _script()
    assert s.SIZE_METRICS == fs.SIZE_METRICS
    assert s.AREA_METRICS == fs.AREA_METRICS


# --- ablation -----------------------------------------------------------------

M = ("a_m", "b_m", "c_m")


def _per_fall(rows: dict[str, tuple[float, float, float]], n=40, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 1.5, n)  # how big each fall is, shared by every policy
    return {p: {m: base * v + noise * rng.standard_normal(n) for m, v in zip(M, vals)} for p, vals in rows.items()}


def test_a_policy_smaller_on_everything_wins_every_method():
    pf = _per_fall({"limp": (1, 1, 1), "walker": (2, 2, 2), "small": (0.5, 0.5, 0.5)})
    rep = abl.analyse(pf, M, bootstrap=50)
    assert rep["A_dominance"]["undominated"] == ["small"]
    assert rep["B_mean_rank"]["winner"] == "small"
    assert rep["C_ratio_to_reference"]["winner"] == "small"
    assert rep["C_ratio_to_reference"]["scores"]["small"] == pytest.approx(0.5, rel=0.02)
    assert rep["C_ratio_to_reference"]["scores"]["limp"] == pytest.approx(1.0)
    assert rep["overall"] == {"methods_agree": True, "answer": ["small"],
                              "note": "mean rank, ratio to limp, and dominance agree"}
    assert rep["bootstrap"]["methods_agree_share"] == 1.0
    assert rep["mean_ranking_kendall_tau"] == pytest.approx(1.0)
    assert rep["pairwise"]["a_m"]["small|limp"]["smaller_share"] == 1.0
    assert rep["pairwise"]["a_m"]["small|limp"]["wilcoxon_p"] < 1e-6


def test_trade_off_gives_two_undominated_and_disagreeing_rankings():
    # x is small on a and b, y is very small on c only
    pf = _per_fall({"limp": (1, 1, 1.5), "x": (0.5, 0.5, 1.2), "y": (1.1, 1.1, 0.1)})
    rep = abl.analyse(pf, M, bootstrap=50)
    assert set(rep["A_dominance"]["undominated"]) == {"x", "y"}
    assert rep["smallest_by_metric"] == {"a_m": "x", "b_m": "x", "c_m": "y"}
    assert rep["mean_ranking_kendall_tau"] < 1.0
    # mean rank favours x (best on 2 of 3); the ratio favours y (one huge cut)
    assert rep["B_mean_rank"]["winner"] == "x"
    assert rep["C_ratio_to_reference"]["winner"] == "y"
    assert rep["overall"]["methods_agree"] is False
    assert set(rep["overall"]["answer"]) == {"x", "y"}


def test_ties_share_rank():
    x = np.array([[1.0, 2.0], [1.0, 1.0], [3.0, 0.0]])  # [P=3, F=2]
    r = abl._rank_rows(x)
    assert r[:, 0].tolist() == [1.5, 1.5, 3.0]
    assert r[:, 1].tolist() == [3.0, 2.0, 1.0]


def test_zero_valued_measure_does_not_blow_up_the_ratio():
    pf = _per_fall({"limp": (1, 1, 1), "z": (0, 0, 0)})
    rep = abl.analyse(pf, M, bootstrap=10)
    assert np.isfinite(rep["C_ratio_to_reference"]["scores"]["z"])
    assert rep["C_ratio_to_reference"]["winner"] == "z"


def test_scale_changes_medians_not_rankings():
    pf = _per_fall({"limp": (1, 1, 1), "walker": (2, 2, 2)})
    a = abl.analyse(pf, M, bootstrap=10)
    b = abl.analyse(pf, M, bootstrap=10, scale={"a_m": 0.5})
    assert b["medians"]["limp"]["a_m"] == pytest.approx(2 * a["medians"]["limp"]["a_m"])
    assert a["rank_by_metric"] == b["rank_by_metric"]


def test_unequal_fall_counts_and_missing_reference_fail_loudly():
    pf = _per_fall({"limp": (1, 1, 1), "w": (2, 2, 2)})
    pf["w"]["a_m"] = pf["w"]["a_m"][:-1]
    with pytest.raises(ValueError):
        abl.analyse(pf, M, bootstrap=0)
    with pytest.raises(ValueError):
        abl.analyse(_per_fall({"w": (1, 1, 1)}), M, bootstrap=0)


def test_exceedance_curve():
    assert abl.exceedance_curve(np.array([0.1, 0.5, 1.0, 2.0]), np.array([0.0, 0.5, 3.0])) == [1.0, 0.5, 0.0]


def test_script_reads_a_compare_npz(tmp_path):
    s = _script()
    rng = np.random.default_rng(1)
    arrays = {f"{p}__size__{m}": rng.uniform(0, 1, 20) * f for p, f in (("limp", 1), ("walker", 2), ("protective", 0.5))
              for m in s.SIZE_METRICS}
    arrays["walker__envelope_growth_m"] = np.zeros(20)  # other compare arrays are ignored
    np.savez(tmp_path / "c.npz", **arrays)
    pf = s.load_per_fall(str(tmp_path / "c.npz"))
    assert sorted(pf) == ["limp", "protective", "walker"]
    with pytest.raises(ValueError):
        s.load_per_fall(str(tmp_path / "c.npz"), ["limp", "nope"])
