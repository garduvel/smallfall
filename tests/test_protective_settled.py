"""Tests for the protective task's `settled` termination (src/cartpole_term/
protective/terminations.py), runnable without mjlab/MuJoCo: the three mjlab
names the module imports are stubbed only for the duration of its import."""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
import torch

from _helpers import load_module

STEP_DT = 0.02
HOLD_S = 0.1  # -> 5 steps at STEP_DT
LIN_T = 0.2
ANG_T = 0.5


def _load_settled_cls():
    class _SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    stubs = {
        "mjlab": types.ModuleType("mjlab"),
        "mjlab.envs": types.SimpleNamespace(ManagerBasedRlEnv=object),
        "mjlab.managers": types.ModuleType("mjlab.managers"),
        "mjlab.managers.scene_entity_config": types.SimpleNamespace(SceneEntityCfg=_SceneEntityCfg),
        "mjlab.managers.termination_manager": types.SimpleNamespace(TerminationTermCfg=object),
    }
    with mock.patch.dict(sys.modules, stubs):
        module = load_module("protective/terminations.py", "_protective_terminations_under_test")
    return module.settled, _SceneEntityCfg


settled_cls, SceneEntityCfg = _load_settled_cls()


class _FakeEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.device = "cpu"
        self.step_dt = STEP_DT
        self.extras = {"log": {}}
        self.data = types.SimpleNamespace(
            root_link_lin_vel_w=torch.zeros(num_envs, 3),
            root_link_ang_vel_w=torch.zeros(num_envs, 3),
        )
        self.scene = {"robot": types.SimpleNamespace(data=self.data)}

    def set_speeds(self, lin: list[float], ang: list[float]) -> None:
        self.data.root_link_lin_vel_w = torch.tensor([[v, 0.0, 0.0] for v in lin])
        self.data.root_link_ang_vel_w = torch.tensor([[0.0, 0.0, w] for w in ang])


def _make(num_envs: int = 1, hold_s: float = HOLD_S):
    env = _FakeEnv(num_envs)
    params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "linear_threshold_mps": LIN_T,
        "angular_threshold_rad_s": ANG_T,
        "hold_s": hold_s,
    }
    term = settled_cls(cfg=types.SimpleNamespace(params=params), env=env)
    return env, term, params


def _call(env, term, params) -> torch.Tensor:
    return term(env, **params)


def test_hold_steps_derived_from_step_dt():
    _, term, _ = _make(hold_s=0.5)
    assert term._hold_steps == 25


def test_fires_only_after_full_hold():
    env, term, params = _make()
    env.set_speeds([0.0], [0.0])
    results = [bool(_call(env, term, params)[0]) for _ in range(5)]
    assert results == [False, False, False, False, True]


def test_single_spike_restarts_the_hold():
    env, term, params = _make()
    env.set_speeds([0.0], [0.0])
    for _ in range(4):
        assert not _call(env, term, params)[0]
    env.set_speeds([1.0], [0.0])
    assert not _call(env, term, params)[0]
    env.set_speeds([0.0], [0.0])
    results = [bool(_call(env, term, params)[0]) for _ in range(5)]
    assert results == [False, False, False, False, True]


@pytest.mark.parametrize("lin,ang", [(0.1, 0.9), (0.5, 0.1)])
def test_both_speeds_must_be_below_threshold(lin, ang):
    env, term, params = _make()
    env.set_speeds([lin], [ang])
    assert not any(bool(_call(env, term, params)[0]) for _ in range(20))


def test_envs_are_independent_and_reset_is_per_env():
    env, term, params = _make(num_envs=2)
    env.set_speeds([0.0, 0.0], [0.0, 0.0])
    for _ in range(5):
        out = _call(env, term, params)
    assert out.tolist() == [True, True]
    term.reset(env_ids=torch.tensor([1]))
    out = _call(env, term, params)
    assert out.tolist() == [True, False]


def test_reset_after_inference_mode_step_does_not_raise():
    """Reproduces the round-4 crash path: the frozen-eval runner steps the eval
    env under torch.inference_mode, then resets it OUTSIDE inference mode, and
    mjlab's TerminationManager.reset passes env_ids=slice(None)."""
    env, term, params = _make(num_envs=3)
    env.set_speeds([0.0] * 3, [0.0] * 3)
    with torch.inference_mode():
        for _ in range(3):
            _call(env, term, params)
    assert term._below_count.is_inference()

    # The old in-place reset raised exactly here.
    with pytest.raises(RuntimeError, match="[Ii]nference"):
        term._below_count[slice(None)] = 0

    term.reset(env_ids=slice(None))
    assert term._below_count.tolist() == [0, 0, 0]
    term.reset(env_ids=torch.tensor([0]))
    term.reset(env_ids=None)
    assert _call(env, term, params).tolist() == [False, False, False]
