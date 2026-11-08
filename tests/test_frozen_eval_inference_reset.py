"""FrozenEvalRunner._run_eval_rollout must reset the eval env inside
torch.inference_mode (DEVIATIONS.md, 2026-11-08: the G1's contact-sensor
history buffer crashed the second frozen-eval pass). Runs without mjlab:
the three imports runner.py needs are stubbed for the duration of its load."""

from __future__ import annotations

import sys
import types
from unittest import mock

import torch

from _helpers import load_module


def _load_runner_cls():
    stubs = {
        "mjlab": types.ModuleType("mjlab"),
        "mjlab.envs": types.SimpleNamespace(ManagerBasedRlEnv=object),
        "mjlab.rl": types.SimpleNamespace(MjlabOnPolicyRunner=object, RslRlVecEnvWrapper=object),
        "rsl_rl": types.ModuleType("rsl_rl"),
        "rsl_rl.utils": types.SimpleNamespace(check_nan=lambda *a: None),
    }
    with mock.patch.dict(sys.modules, stubs):
        return load_module("runner.py", "_runner_under_test").FrozenEvalRunner


class _SensorLikeEnv:
    """Mimics mjlab's ContactSensor history: reset() zeroes a buffer in place,
    step() reallocates it (so it becomes an inference tensor under the
    rollout's inference mode). Every episode ends after two steps."""

    num_envs = 4
    max_episode_length = 5

    def __init__(self):
        self.buf = torch.zeros(self.num_envs)
        self.t = 0

    def seed(self, seed):
        pass

    def reset(self):
        self.buf[:] = 0.0  # raises on an inference tensor outside inference mode
        self.t = 0
        return torch.zeros(self.num_envs, 3), {}

    def step(self, actions):
        self.t += 1
        self.buf = torch.ones(self.num_envs)
        done = torch.full((self.num_envs,), self.t >= 2)
        extras = {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool),
                  "log": {"Episode_Termination/fell_over": 1.0 if self.t >= 2 else 0.0}}
        return torch.zeros(self.num_envs, 3), torch.ones(self.num_envs), done, extras


def _runner(cls):
    r = cls.__new__(cls)
    r._eval_env = _SensorLikeEnv()
    r.device = "cpu"
    r._extra_scalar_metric_fn = None
    r.alg = types.SimpleNamespace(actor=lambda obs, stochastic_output=False: torch.zeros(obs.shape[0], 1))
    return r


def test_two_eval_rollouts_in_a_row_do_not_crash():
    r = _runner(_load_runner_cls())
    first = r._run_eval_rollout(seed=100, stochastic_output=False)   # "mean" pass
    second = r._run_eval_rollout(seed=100, stochastic_output=True)   # "sampled" pass: crashed before the fix
    assert first["mean_episode_length"] == second["mean_episode_length"] == 2.0
    assert first["termination_cause_breakdown"] == {"fell_over": 1.0}


def test_the_fake_env_reproduces_the_original_error():
    env = _SensorLikeEnv()
    with torch.inference_mode():
        env.step(None)
    try:
        env.reset()
    except RuntimeError as e:
        assert "inference tensor" in str(e).lower()
    else:
        raise AssertionError("expected the in-place reset to fail outside inference mode")
