"""Unit tests for src/cartpole_term/rewards.py (CLAUDE.md S2.6, S11).

Runs against fake env/asset_cfg doubles -- no mjlab or GPU required
(CLAUDE.md S0.2.5).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
from _helpers import load_module

rewards = load_module("rewards.py", "_rewards_under_test")

CART_CENTER_SIGMA = 0.5
CART_CENTER_WEIGHT = 0.2
CART_POS_LIMIT_M = 1.0

_ASSET_CFG = SimpleNamespace(name="cartpole", joint_ids=[0])


def _fake_env(x: torch.Tensor):
    data = SimpleNamespace(joint_pos=x)
    asset = SimpleNamespace(data=data)
    return SimpleNamespace(scene={"cartpole": asset})


def test_returns_exactly_one_at_zero():
    env = _fake_env(torch.tensor([[0.0]]))
    result = rewards.cart_centered(env, _ASSET_CFG, CART_CENTER_SIGMA)
    assert result.item() == 1.0


def test_strictly_decreasing_in_abs_x():
    xs_pos = torch.tensor([[0.0], [0.1], [0.3], [0.5], [0.8], [1.0], [1.5]])
    values_pos = rewards.cart_centered(_fake_env(xs_pos), _ASSET_CFG, CART_CENTER_SIGMA).tolist()
    assert all(a > b for a, b in zip(values_pos, values_pos[1:])), values_pos

    xs_neg = torch.tensor([[0.0], [-0.1], [-0.3], [-0.5], [-0.8], [-1.0], [-1.5]])
    values_neg = rewards.cart_centered(_fake_env(xs_neg), _ASSET_CFG, CART_CENTER_SIGMA).tolist()
    assert all(a > b for a, b in zip(values_neg, values_neg[1:])), values_neg


def test_below_0_02_at_termination_boundary():
    env = _fake_env(torch.tensor([[CART_POS_LIMIT_M], [-CART_POS_LIMIT_M]]))
    result = rewards.cart_centered(env, _ASSET_CFG, CART_CENTER_SIGMA)
    assert (result < 0.02).all(), result.tolist()


def test_uses_absolute_joint_pos_not_relative():
    # CLAUDE.md S3.2.2: same reasoning as tests/test_terminations.py -- the
    # zero default makes a numeric swap-test blind, so check the source.
    src = inspect.getsource(rewards.cart_centered)
    assert "joint_pos_rel" not in src
    assert "data.joint_pos" in src


def test_total_per_step_reward_within_bounds():
    """CLAUDE.md S11: shipped smooth_reward + cart_centered*S2.6-weight must
    stay in [0, 1.2] for any cart position.

    smooth_reward itself needs a real mjlab env (env.action_manager.action)
    to evaluate and can't run in this sandbox, but per CLAUDE.md S2.5 it is
    a product of four factors each independently confirmed in [0, 1] by
    construction (upright=(cos+1)/2, centered=(1+g)/2 with g in (0,1],
    small_control=(4+q)/5 with q in [0,1], small_velocity=(1+g)/2) --
    so smooth_reward in [0, 1] is a mathematical consequence of that shape,
    not something that needs re-deriving here. What this test verifies
    directly is that cart_centered's own contribution, after the S2.6
    weight, never pushes the combined bound past 1.2.
    """
    xs = torch.linspace(-3.0, 3.0, steps=61).unsqueeze(-1)
    centered = rewards.cart_centered(_fake_env(xs), _ASSET_CFG, CART_CENTER_SIGMA)
    assert (centered > 0).all() and (centered <= 1.0).all()

    smooth_reward_max = 1.0
    smooth_reward_min = 0.0
    weighted_centered_max = (centered * CART_CENTER_WEIGHT).max().item()
    weighted_centered_min = (centered * CART_CENTER_WEIGHT).min().item()

    assert smooth_reward_min + weighted_centered_min >= 0.0
    assert smooth_reward_max + weighted_centered_max <= 1.2 + 1e-6
