"""The shared impulse push (src/cartpole_term/common/push.py) and the push
fields in RobotProfile. Runs without mjlab: the three mjlab names push.py
imports are stubbed while its functions run."""

from __future__ import annotations

import sys
import types
from unittest import mock

from _helpers import load_module

push = load_module("common/push.py", "_push_under_test")
profiles = load_module("common/robot_profile.py", "_profiles_under_test")


class _EventTermCfg:
    def __init__(self, func, mode, params):
        self.func, self.mode, self.params = func, mode, params


class _SceneEntityCfg:
    def __init__(self, name, body_names=()):
        self.name, self.body_names = name, body_names


def _stubs():
    impulse = object()
    return impulse, {
        "mjlab": types.ModuleType("mjlab"),
        "mjlab.envs": types.ModuleType("mjlab.envs"),
        "mjlab.envs.mdp": types.ModuleType("mjlab.envs.mdp"),
        "mjlab.envs.mdp.events": types.SimpleNamespace(apply_body_impulse=impulse),
        "mjlab.managers": types.ModuleType("mjlab.managers"),
        "mjlab.managers.event_manager": types.SimpleNamespace(EventTermCfg=_EventTermCfg),
        "mjlab.managers.scene_entity_config": types.SimpleNamespace(SceneEntityCfg=_SceneEntityCfg),
    }


def test_add_impulse_push_matches_the_go1_definition():
    impulse, stubs = _stubs()
    cfg = types.SimpleNamespace(events={"push_robot": "shipped"})
    with mock.patch.dict(sys.modules, stubs):
        push.add_impulse_push(cfg, "trunk", 150.0)
    term = cfg.events["wild_impulse"]
    assert cfg.events["push_robot"] == "shipped"  # the mild shipped push stays
    assert term.func is impulse and term.mode == "step"
    assert term.params["force_range"] == (-150.0, 150.0)
    assert term.params["torque_range"] == (-0.0, 0.0)
    assert term.params["duration_s"] == (0.1, 0.3)
    assert term.params["cooldown_s"] == (1.0, 3.0)
    assert term.params["asset_cfg"].name == "robot"
    assert term.params["asset_cfg"].body_names == ("trunk",)


def test_g1_push_is_the_go1_push_scaled_by_mass():
    go1, g1 = profiles.GO1, profiles.G1
    assert (go1.push_body, go1.impulse_force_n) == ("trunk", 150.0)
    assert g1.push_body == "torso_link"
    same_acceleration = go1.impulse_force_n * g1.mass_kg / go1.mass_kg  # 392.4 N
    assert abs(g1.impulse_force_n - same_acceleration) < 5.0


def test_g1_push_body_is_a_real_g1_body():
    every_body = {b for tier in profiles.G1.vulnerability_regions.values() for b in tier}
    assert profiles.G1.push_body in every_body
