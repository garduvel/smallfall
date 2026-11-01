"""Tests for src/cartpole_term/common/perturbation.py, with mjlab's two push
events replaced by minimal fakes that keep the same per-env state fields."""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
import torch

from _helpers import load_module


class _FakeAsset:
    def __init__(self):
        self.cleared = []

    def write_external_wrench_to_sim(self, forces, torques, env_ids, body_ids):
        assert not forces.any() and not torques.any()
        self.cleared.append(env_ids.tolist())


class _FakeImpulse:
    """Same state fields as mjlab 1.6.0's apply_body_impulse; its tick starts
    an impulse for every env whose cooldown has run out."""

    def __init__(self, n):
        self._asset, self._body_ids, self._num_bodies, self._device = _FakeAsset(), [0], 1, "cpu"
        self._active = torch.zeros(n, dtype=torch.bool)
        self._time_remaining = torch.zeros(n)
        self._interval_time_left = torch.zeros(n)
        self.started = []

    def __call__(self, env, env_ids, *args, **kwargs):
        eligible = (~self._active) & (self._interval_time_left <= 0)
        self.started.append(eligible.nonzero().squeeze(-1).tolist())
        self._active = self._active | eligible


_pushed = []


def _fake_push(env, env_ids, velocity_range, **kwargs):
    _pushed.append(env_ids.tolist())


def _load():
    events = types.SimpleNamespace(apply_body_impulse=_FakeImpulse, push_by_setting_velocity=_fake_push)
    stubs = {
        "mjlab": types.ModuleType("mjlab"),
        "mjlab.envs": types.ModuleType("mjlab.envs"),
        "mjlab.envs.mdp": types.ModuleType("mjlab.envs.mdp"),
        "mjlab.envs.mdp.events": events,
    }
    with mock.patch.dict(sys.modules, stubs):
        return load_module("common/perturbation.py", "_perturbation_under_test")


pert = _load()


def test_impulse_cut_and_never_restarted_for_down_envs():
    term = pert.apply_body_impulse_until_down(3)
    env = types.SimpleNamespace()
    term(env, None)                       # no mask yet: all three start an impulse
    assert term.started[-1] == [0, 1, 2]
    env.perturbations_off = torch.tensor([False, True, False])
    term(env, None)
    assert term._asset.cleared == [[1]]   # env 1's impulse in progress is zeroed at once
    assert term.started[-1] == []         # and it is not restarted
    term._active[:] = False               # impulses expire; cooldowns of envs 0 and 2 run out
    term._interval_time_left[[0, 2]] = 0.0
    term(env, None)
    assert term.started[-1] == [0, 2]


def test_velocity_push_skips_down_envs():
    _pushed.clear()
    env = types.SimpleNamespace(perturbations_off=torch.tensor([True, False, True, False]))
    pert.push_by_setting_velocity_until_down(env, torch.tensor([0, 1, 3]), {"x": (-1, 1)})
    assert _pushed == [[1, 3]]
    pert.push_by_setting_velocity_until_down(env, torch.tensor([0, 2]), {"x": (-1, 1)})
    assert _pushed == [[1, 3]]            # everyone down: mjlab's push is not called at all
    pert.push_by_setting_velocity_until_down(types.SimpleNamespace(), torch.tensor([0]), {"x": (-1, 1)})
    assert _pushed[-1] == [0]             # no mask on the env: unchanged behaviour


def test_cfg_swap_is_strict():
    ev = lambda f: types.SimpleNamespace(func=f)  # noqa: E731
    cfg = types.SimpleNamespace(events={"wild_impulse": ev(_FakeImpulse), "push_robot": ev(_fake_push)})
    pert.stop_perturbations_when_down(cfg)
    assert cfg.events["wild_impulse"].func is pert.apply_body_impulse_until_down
    assert cfg.events["push_robot"].func is pert.push_by_setting_velocity_until_down
    with pytest.raises(KeyError):
        pert.stop_perturbations_when_down(types.SimpleNamespace(events={"push_robot": ev(_fake_push)}))
