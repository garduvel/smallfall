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


# --- reward term (fall_size_reward.py) -------------------------------------------

import sys  # noqa: E402
import types  # noqa: E402
from unittest import mock  # noqa: E402


def _reward_module():
    with mock.patch.dict(sys.modules, {"cartpole_term.protective.fall_size": fs}):
        return load_module("protective/fall_size_reward.py", "_fall_size_reward_under_test")


class _FakeEnv:
    """Two envs, three bodies (two watched for contact), random walks."""

    def __init__(self, seed=0):
        self.num_envs, self.device = 2, "cpu"
        self.g = torch.Generator().manual_seed(seed)
        self.root = torch.zeros(2, 3)
        self.robot = types.SimpleNamespace(body_names=("pelvis", "torso", "foot"), data=types.SimpleNamespace())
        self.sensor = types.SimpleNamespace(primary_names=("pelvis", "torso"), data=types.SimpleNamespace())
        self.scene = {"robot": self.robot, "contact": self.sensor}
        self.step()

    def step(self):
        self.root = self.root + 0.1 * torch.randn(2, 3, generator=self.g)
        body = self.root.unsqueeze(1) + 0.3 * torch.randn(2, 3, 3, generator=self.g)
        self.robot.data.root_link_pos_w = self.root.clone()
        self.robot.data.body_link_pos_w = body
        self.sensor.data.force = 5.0 * torch.randn(2, 2, 3, generator=self.g)


@pytest.mark.parametrize("measure", fs.SIZE_METRICS)
def test_reward_increments_sum_to_the_measure(measure):
    rw = _reward_module()
    env = _FakeEnv()
    term = rw.fall_size_increment(types.SimpleNamespace(params={"measure": measure}), env)
    # Mirror of what the term sees, recorded with the replay tracker.
    r = env.robot.data
    ref = None
    total = torch.zeros(2)
    for step in range(8):
        if step:
            env.step()
        out = term(env, measure=measure, sensor_name="contact")
        if ref is None:
            ref = fs.FallSizeTracker(r.root_link_pos_w[:, :2], r.body_link_pos_w[..., :2])
        body_xy = r.body_link_pos_w[..., :2]
        ref.update(root_xy=r.root_link_pos_w[:, :2], body_xy=body_xy, contact_xy=body_xy[:, :2],
                   in_contact=env.sensor.data.force.norm(dim=-1) > rw.CONTACT_N, ended=torch.zeros(2, dtype=torch.bool))
        total = total + out
    assert total.numpy() == pytest.approx(ref.per_fall()[measure], abs=1e-5)


def test_reward_restarts_after_reset():
    rw = _reward_module()
    env = _FakeEnv(seed=3)
    m = "root_path_m"
    term = rw.fall_size_increment(types.SimpleNamespace(params={"measure": m}), env)
    for _ in range(5):
        term(env, measure=m, sensor_name="contact")
        env.step()
    term.reset(env_ids=torch.tensor([1]))
    out = term(env, measure=m, sensor_name="contact")
    assert float(out[1]) == 0.0  # env 1: first step of a fresh fall, no path yet
    assert float(out[0]) > 0.0   # env 0 carries on


def test_reward_rejects_unknown_measure():
    rw = _reward_module()
    with pytest.raises(ValueError):
        rw.fall_size_increment(types.SimpleNamespace(params={"measure": "vibes"}), _FakeEnv())


# --- weight sizing (calibration.fall_size_weights) -------------------------------

cal = load_module("protective/calibration.py", "_calibration_for_size_test")


def test_fall_size_weights():
    sums = {"envelope_growth_m": 0.5, "root_path_m": 1.0, "touched_area_m2": 0.5 / 1000, "contact_reach_m": 0.0}
    out = cal.fall_size_weights(sums, -40.0)
    assert out["weights"]["envelope_growth_m"] == -40.0
    assert out["weights"]["root_path_m"] == pytest.approx(-20.0)
    assert out["terms"]["touched_area_m2"]["clamped"] and out["weights"]["touched_area_m2"] == pytest.approx(-40.0 * 50)
    assert out["terms"]["contact_reach_m"]["kept_base"] and out["weights"]["contact_reach_m"] == -40.0
    with pytest.raises(ValueError):
        cal.fall_size_weights({"root_path_m": 1.0}, -40.0)


def test_size_calibration_script_names_match():
    path = Path(__file__).resolve().parent.parent / "scripts" / "robot_calibrate_fall_size.py"
    spec = importlib.util.spec_from_file_location("_size_cal_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SIZE_METRICS == fs.SIZE_METRICS


def test_reward_tags_cover_every_measure():
    assert set(abl.REWARD_TAGS) == set(fs.SIZE_METRICS)
    assert len(set(abl.REWARD_TAGS.values())) == len(fs.SIZE_METRICS)
    assert all("_" not in t for t in abl.REWARD_TAGS.values())


def test_select_by_measure_uses_its_own_measure_and_settling():
    per_arm = {
        "it200": {"settled": np.ones(10), "size__root_path_m": np.full(10, 0.5)},
        "it250": {"settled": np.r_[np.ones(9), 0], "size__root_path_m": np.full(10, 0.1)},  # 90% settled
        "it300": {"settled": np.ones(10), "size__root_path_m": np.full(10, 0.3)},
    }
    out = abl.select_by_measure(per_arm, ["it200", "it250", "it300"], "root_path_m")
    assert out["selected"] == "it300" and out["selected_is_eligible"]
    for v in per_arm.values():
        v["settled"] = np.zeros(10)
    out = abl.select_by_measure(per_arm, ["it200", "it250", "it300"], "root_path_m")
    assert out["selected"] == "it250" and not out["selected_is_eligible"]


def test_trained_on_report():
    pf = _per_fall({"limp": (1, 1, 1), "walker": (2, 2, 2), "r_ground": (0.5, 0.9, 0.9), "r_reach": (0.6, 0.4, 0.8)})
    pf = {p: {mm: v[m] for m, mm in zip(M, ("envelope_growth_m", "max_reach_m", "root_path_m"))} for p, v in pf.items()}
    rep = abl.analyse(pf, ("envelope_growth_m", "max_reach_m", "root_path_m"), bootstrap=10)
    t = abl.trained_on_report(rep)
    assert t["num_trained"] == 2 and t["wins_own_measure"] == 2
    assert t["ratio_to_reference"]["r_ground"]["envelope_growth_m"] == pytest.approx(0.5, rel=0.05)
    assert abl.trained_on_report(abl.analyse(_per_fall({"limp": (1, 1, 1), "w": (2, 2, 2)}), M, bootstrap=0)) is None


def test_reward_ablation_notebook_tags_match():
    import ast
    src = (Path(__file__).resolve().parent.parent / "tools" / "notebook_builders" / "build_g1_reward_ablation_nb.py").read_text()
    i = src.index("TAGS = {")
    j = src.index("}", i)
    assert ast.literal_eval(src[i + len("TAGS = "):j + 1]) == abl.REWARD_TAGS


# --- round 2: safe-small selection ------------------------------------------------

def _arm(scale, settled=1.0, trunk=0.0, anyf=1000.0, n=30, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.5, 1.5, n)
    a = {f"size__{m}": base * scale for m in fs.SIZE_METRICS}
    a.update(settled=(np.arange(n) < round(settled * n)).astype(float),
             peak_force_critical_n=np.full(n, trunk), peak_force_any_n=np.full(n, anyf))
    return a


def test_size_score_is_ratio_to_limp():
    arms = {"limp": _arm(1.0), "x": _arm(0.5)}
    s = abl.per_fall_size_score(arms["x"], arms["limp"], abl.size_floors(arms))
    assert np.allclose(s, 0.5, rtol=0.03)


def test_select_safe_small_rules():
    arms = {"limp": _arm(1.0, trunk=1300, anyf=2500), "walker": _arm(2.0, settled=0.1, trunk=1700, anyf=2300),
            "small_unsafe": _arm(0.3, anyf=2600),          # any-body force above limp's
            "small_restless": _arm(0.4, settled=0.9),       # comes to rest too rarely
            "ok_big": _arm(0.8), "ok_small": _arm(0.6)}
    out = abl.select_safe_small(arms, ["small_unsafe", "small_restless", "ok_big", "ok_small"])
    assert out["selected"] == "ok_small" and out["selected_is_eligible"]
    assert out["size_score"] == pytest.approx(0.6, rel=0.03)
    rows = {r["label"]: r for r in out["candidates"]}
    assert not rows["small_unsafe"]["safe"] and rows["small_restless"]["safe"] and not rows["small_restless"]["eligible"]
    out = abl.select_safe_small(arms, ["small_unsafe", "small_restless"])
    assert out["selected"] == "small_unsafe" and not out["selected_is_eligible"]  # settles enough, flagged unsafe


def _load_script(name):
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"_{name[:-3]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_round2_choose():
    ch = _load_script("robot_round2_choose.py")
    c = lambda s, e=True: {"selected": "x", "selected_is_eligible": e, "size_score": s, "checkpoint": f"c{s}"}  # noqa: E731
    out = ch.choose({"ground_s0": c(0.80), "ground_s1": c(0.84), "mix_s0": c(0.70), "mix_s1": c(0.90),
                     "hull_s0": c(0.60, False), "hull_s1": c(0.61)})
    # hull is best on score but one run has no eligible checkpoint -> ranked last;
    # mix's mean (0.80) beats ground's (0.82)
    assert out["chosen_reward"] == "mix" and out["chosen_policy"] == "mix_s0"
    out = ch.choose({"ground_s0": c(0.80), "mix_s0": c(0.75), "mix_s1": c(0.76)})
    assert out["chosen_reward"] == "mix" and out["chosen_policy"] == "mix_s0"
    with pytest.raises(ValueError):
        ch.choose({"nolabel": c(0.5)})


def test_round2_summary():
    sm = _load_script("robot_round2_summary.py")
    arms = {"limp": _arm(1.0, trunk=1300, anyf=2500), "walker": _arm(2.0, settled=0.1, trunk=1700, anyf=2300),
            "protective": _arm(0.8), "mix_s0": _arm(0.6), "mix_s1": _arm(0.65), "ground_s0": _arm(0.8)}
    rng = np.random.default_rng(5)
    for m in fs.SIZE_METRICS:  # per-fall noise so the paired test has something to test
        arms["mix_s0"][f"size__{m}"] = arms["mix_s0"][f"size__{m}"] * rng.uniform(0.9, 1.1, 30)
    out = sm.summarize(arms, "mix_s0")
    assert out["check"]["pass"]
    assert out["by_reward"]["mix"]["runs"] == ["mix_s0", "mix_s1"]
    assert out["check"]["size_score_ratio_to_baseline"] == pytest.approx(0.75, rel=0.05)
    out = sm.summarize(arms, "ground_s0")
    assert not out["check"]["passes"]["smaller_than_smallfall"] and not out["check"]["pass"]
