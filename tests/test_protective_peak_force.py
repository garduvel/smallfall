"""Tests for the round-8 reward term peak_contact_force_growth
(src/cartpole_term/protective/rewards.py), runnable without mjlab/MuJoCo."""

from __future__ import annotations

import sys
import types
from unittest import mock

import torch

from _helpers import load_module


def _load():
    stubs = {
        "mjlab": types.ModuleType("mjlab"),
        "mjlab.envs": types.SimpleNamespace(ManagerBasedRlEnv=object),
        "mjlab.managers": types.ModuleType("mjlab.managers"),
        "mjlab.managers.reward_manager": types.SimpleNamespace(RewardTermCfg=object),
        "mjlab.managers.scene_entity_config": types.SimpleNamespace(SceneEntityCfg=lambda name: name),
        "mjlab.sensor": types.SimpleNamespace(ContactSensor=object),
    }
    with mock.patch.dict(sys.modules, stubs):
        return load_module("protective/rewards.py", "_protective_rewards_peak_under_test")


rewards = _load()


class _FakeEnv:
    def __init__(self, num_envs):
        self.num_envs, self.device = num_envs, "cpu"
        self.sensor = types.SimpleNamespace(data=types.SimpleNamespace(force=None))
        self.scene = {"contact": self.sensor}
        self.extras = {"log": {}}

    def forces(self, per_env_per_body_newtons):
        # [B, N] magnitudes -> [B, N, 3] vectors along z
        mag = torch.tensor(per_env_per_body_newtons, dtype=torch.float32)
        self.sensor.data.force = torch.stack([torch.zeros_like(mag), torch.zeros_like(mag), mag], dim=-1)


def _term(env):
    return rewards.peak_contact_force_growth(cfg=types.SimpleNamespace(params={}), env=env)


def test_episode_sum_is_the_peak_any_body_force():
    env = _FakeEnv(2)
    term = _term(env)
    total = torch.zeros(2)
    for step in ([[100, 50], [0, 0]], [[80, 900], [300, 10]], [[400, 20], [250, 0]], [[0, 1200], [0, 0]]):
        env.forces(step)
        total += term(env, sensor_name="contact")
    assert total.tolist() == [1200.0, 300.0]


def test_no_charge_below_the_running_peak():
    env = _FakeEnv(1)
    term = _term(env)
    env.forces([[500, 0]])
    assert term(env, sensor_name="contact").tolist() == [500.0]
    env.forces([[499, 499]])
    assert term(env, sensor_name="contact").tolist() == [0.0]


def test_reset_is_per_env():
    env = _FakeEnv(2)
    term = _term(env)
    env.forces([[500, 0], [700, 0]])
    term(env, sensor_name="contact")
    term.reset(env_ids=torch.tensor([0]))
    env.forces([[200, 0], [200, 0]])
    assert term(env, sensor_name="contact").tolist() == [200.0, 0.0]
    term.reset()
    assert term(env, sensor_name="contact").tolist() == [200.0, 200.0]


def test_safe_after_inference_mode():
    env = _FakeEnv(2)
    with torch.inference_mode():
        term = _term(env)
        env.forces([[10, 0], [10, 0]])
        term(env, sensor_name="contact")
    term.reset(env_ids=torch.tensor([1]))  # must not raise outside inference mode
    env.forces([[30, 0], [30, 0]])
    assert term(env, sensor_name="contact").tolist() == [20.0, 30.0]
    assert "Metrics/peak_contact_force_growth_mean" in env.extras["log"]
