"""Settle lock (protective/settle_lock.py) and the round-3 summary, without
mjlab: the mjlab action classes are stubbed for the module's import."""

from __future__ import annotations

import json
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from _helpers import load_module, load_script

STEP_DT = 0.02


@dataclass(kw_only=True)
class _JointPositionActionCfg:
    entity_name: str
    actuator_names: tuple = (".*",)
    scale: object = 1.0
    offset: object = 0.0
    clip: object = None
    preserve_order: bool = False
    use_default_offset: bool = True


class _JointPositionAction:
    """Just what SettleLockAction uses of mjlab's JointPositionAction."""

    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self._entity = env.scene[cfg.entity_name]
        self._target_ids = torch.arange(env.n_joints)
        self._processed_actions = torch.zeros(env.num_envs, env.n_joints)

    num_envs = property(lambda self: self._env.num_envs)
    device = property(lambda self: "cpu")
    action_dim = property(lambda self: self._env.n_joints)

    def process_actions(self, actions):
        self._processed_actions = actions * 0.5

    def reset(self, env_ids=None):
        pass


def _load():
    stubs = {
        "mjlab": types.ModuleType("mjlab"),
        "mjlab.envs": types.ModuleType("mjlab.envs"),
        "mjlab.envs.mdp": types.ModuleType("mjlab.envs.mdp"),
        "mjlab.envs.mdp.actions": types.SimpleNamespace(JointPositionAction=_JointPositionAction,
                                                        JointPositionActionCfg=_JointPositionActionCfg),
    }
    with mock.patch.dict(sys.modules, stubs):
        return load_module("protective/settle_lock.py", "_settle_lock_under_test")


sl = _load()


class _Env:
    def __init__(self, n=3, j=4):
        self.num_envs, self.n_joints, self.step_dt = n, j, STEP_DT
        self.episode_length_buf = torch.zeros(n, dtype=torch.long)
        self.extras = {"log": {}}
        d = types.SimpleNamespace(root_link_lin_vel_w=torch.ones(n, 3), root_link_ang_vel_w=torch.ones(n, 3),
                                  root_link_pos_w=torch.ones(n, 3), joint_pos_biased=torch.zeros(n, j))
        self.robot = types.SimpleNamespace(data=d)
        self.scene = {"robot": self.robot}

    def step(self, action):
        self.term.process_actions(action)
        self.episode_length_buf += 1


def _make(n=3, **kw):
    env = _Env(n)
    cfg = sl.SettleLockActionCfg(entity_name="robot", lock_after_s=0.1, still_hold_s=0.06, still_max_height_m=0.38, **kw)
    env.term = cfg.build(env)
    return env


def test_timer_locks_and_holds_the_pose_at_the_lock_step():
    env = _make()
    for k in range(5):   # lock_after_s 0.1 s = 5 steps: steps 0-4 free
        env.robot.data.joint_pos_biased = torch.full((3, 4), float(k))
        env.step(torch.ones(3, 4))
        assert not env.term.locked.any()
        assert torch.allclose(env.term._processed_actions, torch.full((3, 4), 0.5))
    env.robot.data.joint_pos_biased = torch.full((3, 4), 7.0)
    env.step(torch.ones(3, 4))
    assert env.term.locked.all() and env.term.locked_by_timer.all()
    assert (env.term.lock_step == 5).all()
    env.robot.data.joint_pos_biased = torch.full((3, 4), 9.0)   # keeps moving: target stays the lock pose
    env.step(torch.full((3, 4), 3.0))
    assert torch.allclose(env.term._processed_actions, torch.full((3, 4), 7.0))
    assert env.extras["log"]["Metrics/settle_locked_fraction"] == 1.0


def test_still_and_low_locks_early_only_when_low():
    env = _make()
    d = env.robot.data
    d.root_link_lin_vel_w = torch.zeros(3, 3)
    d.root_link_ang_vel_w = torch.zeros(3, 3)
    d.root_link_pos_w = torch.tensor([[0, 0, 0.2], [0, 0, 0.6], [0, 0, 0.2]], dtype=torch.float)
    d.root_link_ang_vel_w[2, 0] = 1.0            # env 2 still rotating
    for _ in range(3):                           # still_hold_s 0.06 s = 3 steps
        env.step(torch.zeros(3, 4))
    assert env.term.locked.tolist() == [True, False, False]
    assert not env.term.locked_by_timer[0]
    assert env.term.lock_step[0] == 2


def test_still_count_restarts_when_motion_resumes():
    out = dict(locked=torch.zeros(1, dtype=torch.bool), lock_target=torch.zeros(1, 2), lock_step=torch.full((1,), -1),
               by_timer=torch.zeros(1, dtype=torch.bool), still_count=torch.zeros(1, dtype=torch.long))
    speeds = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    for t, v in enumerate(speeds):
        r = sl.lock_update(**out, t=torch.tensor([t]), lin=torch.tensor([v]), ang=torch.tensor([0.0]),
                           height=torch.tensor([0.1]), joint_pos=torch.full((1, 2), float(t)), lock_steps=100,
                           still_steps=3, lin_max=0.3, ang_max=0.75, height_max=0.38)
        out = dict(zip(("locked", "lock_target", "lock_step", "by_timer", "still_count"), r))
        assert bool(out["locked"]) == (t == 5)
    assert torch.allclose(out["lock_target"], torch.full((1, 2), 5.0))


def test_reset_unlocks_only_the_reset_envs():
    env = _make()
    for _ in range(6):
        env.step(torch.zeros(3, 4))
    assert env.term.locked.all()
    env.term.reset(torch.tensor([1]))
    assert env.term.locked.tolist() == [True, False, True]
    assert env.term.lock_step.tolist() == [5, -1, 5]
    env.term.reset(None)
    assert not env.term.locked.any()


def test_cfg_copies_every_field_of_the_task_action():
    base = _JointPositionActionCfg(entity_name="robot", scale={"a": 0.3}, use_default_offset=True)
    cfg = sl.settle_lock_action_cfg(base, still_max_height_m=0.38, lock_after_s=2.0)
    assert isinstance(cfg, sl.SettleLockActionCfg)
    assert cfg.scale == {"a": 0.3} and cfg.entity_name == "robot" and cfg.use_default_offset
    assert cfg.lock_after_s == 2.0 and cfg.still_max_height_m == 0.38
    assert sl.STILL_MAX_HEIGHT_FRACTION * sl.STANDING_ROOT_HEIGHT_M["g1"] == pytest.approx(0.38)


# ---- round 3 summary / choice -------------------------------------------------

r3 = load_script("scripts/robot_round3_summary.py", "_round3_summary_under_test")
choose_mod = load_script("scripts/robot_round2_choose.py", "_round2_choose_for_r3")
FAM = r3.abl.FAMILY_MEASURES


def _arm(size, settled, trunk=0.0, anyf=1000.0, n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = {"settled": (rng.random(n) < settled).astype(float), "peak_force_critical_n": np.full(n, trunk),
         "peak_force_any_n": np.full(n, anyf)}
    for m in FAM:
        a[f"size__{m}"] = size * (1 + 0.1 * rng.random(n))
    return a


def _per_arm():
    return {"limp": _arm(1.0, 1.0, trunk=1200, anyf=2400, seed=1), "walker": _arm(2.0, 0.1, trunk=1500, anyf=2200, seed=2),
            "protective": _arm(0.9, 0.98, seed=3), "sf_lock": _arm(0.9, 1.0, seed=3),
            "r2s2": _arm(0.72, 0.978, seed=4), "r2s2_lock": _arm(0.72, 1.0, seed=4),
            "hullgrow_s0": _arm(0.70, 0.995, seed=5), "hullgrow_s1": _arm(0.74, 0.999, seed=6)}


def test_round3_check_and_lock_pairs():
    per_arm = _per_arm()
    out = r3.summarize(per_arm, "hullgrow_s0", {"sf_lock": {"locked_share": 0.2, "by_timer_share": 0.1, "lock_s_median": 2.0}})
    assert set(out["lock_pairs"]) == {"protective", "r2s2"}
    assert out["lock_pairs"]["protective"]["locked_share"] == 0.2
    c = out["check"]
    assert c["passes"]["comes_to_rest"] and c["passes"]["not_bigger_than_round2"] and c["pass"]
    assert "hullgrow" in out["by_reward"] and "r2s2" not in out["by_reward"]


def test_round3_fails_below_99_percent_rest():
    per_arm = _per_arm()
    per_arm["hullgrow_s0"] = _arm(0.70, 0.975, seed=5)
    c = r3.summarize(per_arm, "hullgrow_s0", {})["check"]
    assert not c["passes"]["comes_to_rest"] and not c["pass"]


def test_round3_fails_when_much_bigger_than_round2():
    per_arm = _per_arm()
    per_arm["hullgrow_s0"] = _arm(0.85, 0.999, seed=5)
    c = r3.summarize(per_arm, "hullgrow_s0", {})["check"]
    assert c["size_score_ratio_to_round2"] > 1.05 and not c["passes"]["not_bigger_than_round2"]


def test_choose_reads_round3_choice_files(tmp_path):
    for label, score in (("hullgrow_s0", 0.7), ("hullgrow_s1", 0.72), ("touched_s0", 0.69), ("touched_s1", 0.8)):
        (tmp_path / f"g1_r3sel_{label}_choice.json").write_text(json.dumps(
            {"selected": "protective", "selected_is_eligible": True, "size_score": score, "checkpoint": f"{label}.pt"}))
    out_file = tmp_path / "choice.json"
    with mock.patch.object(sys, "argv", ["x", "--choices", str(tmp_path / "g1_r3sel_*_choice.json"), "--output", str(out_file)]):
        choose_mod.main()
    out = json.loads(out_file.read_text())
    assert out["chosen_reward"] == "hullgrow" and out["chosen_policy"] == "hullgrow_s0"


def test_round3_notebook_labels():
    src = Path(__file__).resolve().parent.parent / "tools" / "notebook_builders" / "build_g1_round3_nb.py"
    text = src.read_text()
    rewards = dict(re.findall(r'"(\w+)": "(\w+_m2?)",', text))
    fs = load_module("protective/fall_size_ablation.py", "_fsa_for_r3")
    assert set(rewards) == {"hullgrow", "centroid", "hull", "touched"}
    sizes = ("envelope_growth_m", "max_reach_m", "root_travel_end_m", "root_travel_max_m", "root_path_m",
             "centroid_travel_end_m", "swept_hull_area_m2", "hull_growth_m2", "contact_reach_m", "touched_area_m2")
    assert set(rewards.values()) <= set(sizes)
    assert {fs.TAG_TO_MEASURE[t] for t in rewards} == set(rewards.values())
    # existing-policy labels must not look like round-3 runs (<reward>_s<seed>)
    for name in ("sf", "r2s1", "r2s2", "ablhullgrow", "ablcentroid", "ablhull", "abltouched"):
        for label in (name, f"{name}_lock"):
            assert not re.fullmatch(r"(.+)_s(\d+)", label)


def test_external_lock_clock_starts_at_the_switch_and_holds_via_the_action():
    n = 2
    lock = sl.ExternalSettleLock(n, 3, "cpu", STEP_DT, still_max_height_m=0.38, lock_after_s=0.1)  # 5 steps
    offset, scale = torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.5, 0.5, 0.5])
    flag_step = torch.tensor([10, -1])          # env 0 switches at step 10; env 1 never
    out = None
    for t in range(10, 16):
        active = flag_step >= 0
        pose = torch.full((n, 3), float(t))
        out = lock.apply(torch.zeros(n, 3), active=active, since=t - flag_step, lin=torch.ones(n), ang=torch.ones(n),
                         height=torch.ones(n), joint_pos_biased=pose, offset=offset, scale=scale)
        assert bool(lock.locked[0]) == (t >= 15) and not lock.locked[1]
    # held pose 15 written as an action: offset + scale * action == 15
    assert torch.allclose(offset + scale * out[0], torch.full((3,), 15.0))
    assert torch.allclose(out[1], torch.zeros(3))
    assert lock.locked_by_timer[0]
