"""Tests for the protective task's log-only space metrics
(src/cartpole_term/protective/space.py), runnable without mjlab/MuJoCo."""

from __future__ import annotations

import math
import sys
import types
from unittest import mock

import torch

from _helpers import load_module


def _load():
    class _SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    stubs = {
        "mjlab": types.ModuleType("mjlab"),
        "mjlab.envs": types.SimpleNamespace(ManagerBasedRlEnv=object),
        "mjlab.managers": types.ModuleType("mjlab.managers"),
        "mjlab.managers.metrics_manager": types.SimpleNamespace(MetricsTermCfg=object),
        "mjlab.managers.scene_entity_config": types.SimpleNamespace(SceneEntityCfg=_SceneEntityCfg),
    }
    with mock.patch.dict(sys.modules, stubs):
        return load_module("protective/space.py", "_protective_space_under_test")


metrics = _load()


class _FakeEnv:
    """Robot = root at body index 0 plus extra bodies at fixed xy offsets."""

    def __init__(self, num_envs: int, offsets: list[tuple[float, float]]):
        self.num_envs = num_envs
        self.device = "cpu"
        self._offsets = torch.tensor(offsets)
        self.data = types.SimpleNamespace()
        self.scene = {"robot": types.SimpleNamespace(data=self.data)}
        self.place([(0.0, 0.0)] * num_envs)

    def place(self, root_xy: list[tuple[float, float]]) -> None:
        root = torch.tensor(root_xy)
        z = torch.zeros(self.num_envs, 1)
        self.data.root_link_pos_w = torch.cat([root, z], dim=-1)
        bodies = root.unsqueeze(1) + self._offsets.unsqueeze(0)
        self.data.body_link_pos_w = torch.cat([bodies, torch.zeros(*bodies.shape[:2], 1)], dim=-1)


def _term(env, of):
    return metrics.distance_from_fall_start(cfg=types.SimpleNamespace(params={"of": of}), env=env)


def test_origin_captured_on_first_step_after_reset_not_at_reset():
    env = _FakeEnv(1, [(0.0, 0.0)])
    term = _term(env, "root")
    env.place([(5.0, 5.0)])  # stale pre-reset pose visible at reset time
    term.reset(env_ids=slice(None))
    env.place([(1.0, 0.0)])  # first post-reset step
    assert term(env, of="root").item() == 0.0
    env.place([(4.0, 4.0)])
    assert math.isclose(term(env, of="root").item(), 5.0, rel_tol=1e-6)


def test_bodies_distance_is_farthest_body_from_start():
    env = _FakeEnv(1, [(0.0, 0.0), (0.3, 0.0), (0.0, -0.1)])
    term = _term(env, "bodies")
    env.place([(0.0, 0.0)])
    assert math.isclose(term(env, of="bodies").item(), 0.3, rel_tol=1e-6)
    env.place([(1.0, 0.0)])
    assert math.isclose(term(env, of="bodies").item(), 1.3, rel_tol=1e-6)


def test_reset_is_per_env_and_safe_after_inference_mode():
    env = _FakeEnv(2, [(0.0, 0.0)])
    term = _term(env, "root")
    with torch.inference_mode():
        term(env, of="root")
    env.place([(1.0, 0.0), (1.0, 0.0)])
    term.reset(env_ids=torch.tensor([1]))  # outside inference mode
    out = term(env, of="root")
    assert out.tolist() == [1.0, 0.0]


def test_body_spread():
    env = _FakeEnv(1, [(-0.2, 0.0), (0.2, 0.0)])
    assert math.isclose(metrics.body_spread(env).item(), 0.2, rel_tol=1e-6)
    env2 = _FakeEnv(1, [(0.0, 0.0), (0.0, 0.0)])
    assert metrics.body_spread(env2).item() == 0.0


def _growth(env):
    return metrics.fall_envelope_growth(cfg=types.SimpleNamespace(params={}), env=env)


def test_envelope_growth_telescopes_to_final_envelope_minus_initial_extent():
    env = _FakeEnv(1, [(0.0, 0.0), (0.3, 0.0)])
    term = _growth(env)
    total = 0.0
    for root_x in [0.0, 0.2, 0.1, 0.5, 0.4]:  # moves out, back, further, back
        env.place([(root_x, 0.0)])
        total += term(env).item()
    # final envelope = 0.5 + 0.3 = 0.8; initial extent 0.3 is not penalized
    assert math.isclose(total, 0.5, rel_tol=1e-6)


def test_envelope_growth_first_step_after_reset_is_zero():
    env = _FakeEnv(2, [(0.0, 0.0), (0.3, 0.0)])
    term = _growth(env)
    assert term(env).tolist() == [0.0, 0.0]
    env.place([(1.0, 0.0), (1.0, 0.0)])
    assert torch.allclose(term(env), torch.tensor([1.0, 1.0]))
    term.reset(env_ids=torch.tensor([0]))
    env.place([(3.0, 0.0), (3.0, 0.0)])
    out = term(env).tolist()
    assert out[0] == 0.0  # new episode: new origin, nothing penalized yet
    assert math.isclose(out[1], 2.0, rel_tol=1e-6)


def test_envelope_growth_safe_after_inference_mode():
    env = _FakeEnv(1, [(0.0, 0.0)])
    term = _growth(env)
    with torch.inference_mode():
        term(env)
        env.place([(0.5, 0.0)])
        term(env)
    term.reset(env_ids=slice(None))
    env.place([(0.7, 0.0)])
    assert term(env).item() == 0.0
