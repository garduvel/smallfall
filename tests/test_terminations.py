"""Unit tests for src/cartpole_term/terminations.py (CLAUDE.md S3.2, S11).

Runs against fake env/asset_cfg doubles -- no mjlab or GPU required
(CLAUDE.md S0.2.5).
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import torch
from _helpers import load_module

terminations = load_module("terminations.py", "_terminations_under_test")

POLE_ANGLE_LIMIT_RAD = 0.20943951023931953  # 12 degrees
CART_POS_LIMIT_M = 1.0

_ASSET_CFG = SimpleNamespace(name="cartpole", joint_ids=[0])


def _fake_env(joint_pos: torch.Tensor):
    """joint_pos: [num_envs, 1] tensor for the single selected joint."""
    data = SimpleNamespace(joint_pos=joint_pos)
    asset = SimpleNamespace(data=data)
    return SimpleNamespace(scene={"cartpole": asset})


def test_pole_angle_within_bounds_is_false():
    theta = torch.tensor([[0.0], [0.1]])
    env = _fake_env(theta)
    result = terminations.pole_angle_out_of_bounds(env, _ASSET_CFG, POLE_ANGLE_LIMIT_RAD)
    assert result.tolist() == [False, False]


def test_pole_angle_out_of_bounds_is_true():
    theta = torch.tensor([[POLE_ANGLE_LIMIT_RAD + 0.01], [-(POLE_ANGLE_LIMIT_RAD + 0.01)]])
    env = _fake_env(theta)
    result = terminations.pole_angle_out_of_bounds(env, _ASSET_CFG, POLE_ANGLE_LIMIT_RAD)
    assert result.tolist() == [True, True]


def test_pole_angle_wraps_unlimited_hinge():
    # hinge_1 is unlimited: raw angle grows without bound. A raw value of
    # 2*pi is physically upright (wrapped angle 0) and must NOT be flagged,
    # despite the large raw magnitude -- this is the CLAUDE.md S3.2.1 bug
    # class ("unwrapped comparison... silently passes on balance").
    theta = torch.tensor(
        [[2 * math.pi], [2 * math.pi + POLE_ANGLE_LIMIT_RAD + 0.01]]
    )
    env = _fake_env(theta)
    result = terminations.pole_angle_out_of_bounds(env, _ASSET_CFG, POLE_ANGLE_LIMIT_RAD)
    assert result.tolist() == [False, True]


def test_cart_pos_within_bounds_is_false():
    x = torch.tensor([[0.0], [0.99]])
    env = _fake_env(x)
    result = terminations.cart_pos_out_of_bounds(env, _ASSET_CFG, CART_POS_LIMIT_M)
    assert result.tolist() == [False, False]


def test_cart_pos_out_of_bounds_is_true():
    x = torch.tensor([[1.01], [-1.5]])
    env = _fake_env(x)
    result = terminations.cart_pos_out_of_bounds(env, _ASSET_CFG, CART_POS_LIMIT_M)
    assert result.tolist() == [True, True]


def test_uses_absolute_joint_pos_not_relative():
    # CLAUDE.md S3.2.2 / S12#12: must read `asset.data.joint_pos` (absolute),
    # never a `joint_pos_rel`-style attribute. The balance task's zero
    # default makes the two numerically identical, so a numeric test
    # wouldn't catch a swap -- CLAUDE.md explicitly asks for a test that
    # "fails if someone swaps in the relative term", so this checks the
    # source directly.
    src_pole = inspect.getsource(terminations.pole_angle_out_of_bounds)
    src_cart = inspect.getsource(terminations.cart_pos_out_of_bounds)
    assert "joint_pos_rel" not in src_pole
    assert "joint_pos_rel" not in src_cart
    assert "data.joint_pos" in src_pole
    assert "data.joint_pos" in src_cart
