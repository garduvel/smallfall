"""Failure terminations for the cartpole-term task (CLAUDE.md S3.2).

Both functions read `asset.data.joint_pos` -- the ABSOLUTE joint position --
never `joint_pos_rel`. The balance task's default joint positions are both
0.0, so the two are numerically identical here, which is exactly why using
the relative term by mistake would pass unnoticed (CLAUDE.md S3.2.2, S12#12).

mjlab types (Entity, SceneEntityCfg, ManagerBasedRlEnv) are imported only
under TYPE_CHECKING: with `from __future__ import annotations`, annotations
are never evaluated at runtime, so these functions work against any
duck-typed object exposing the same `.scene[name].data.joint_pos` /
`.joint_ids` shape -- e.g. in tests/test_terminations.py, which runs without
mjlab or a GPU installed (CLAUDE.md S0.2.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def pole_angle_out_of_bounds(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",
    limit_rad: float,
) -> torch.Tensor:
    """True where the wrapped pole angle exceeds +/- limit_rad.

    hinge_1 is an unlimited hinge: its raw joint_pos does not wrap and grows
    without bound as the pole spins, so it must be wrapped into (-pi, pi]
    before comparing against the limit.
    """
    asset: "Entity" = env.scene[asset_cfg.name]
    theta = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    theta_wrapped = torch.atan2(torch.sin(theta), torch.cos(theta))
    return theta_wrapped.abs() > limit_rad


def cart_pos_out_of_bounds(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",
    limit_m: float,
) -> torch.Tensor:
    """True where the absolute cart position exceeds +/- limit_m."""
    asset: "Entity" = env.scene[asset_cfg.name]
    x = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    return x.abs() > limit_m
