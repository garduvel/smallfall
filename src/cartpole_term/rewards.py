"""Added reward term: explicit cart centering (CLAUDE.md S2.6).

Additive and NOT potential-based (Ng, Harada & Russell 1999) -- this can
shift the optimal policy toward the centre rather than merely accelerate
convergence to the shipped optimum. That is deliberate (CLAUDE.md S2.6,
S13 item 4). Do not convert this to a potential-based form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.scene_entity_config import SceneEntityCfg


def cart_centered(
    env: "ManagerBasedRlEnv",
    asset_cfg: "SceneEntityCfg",
    sigma: float,
) -> torch.Tensor:
    """Gaussian centering kernel on absolute cart position. Returns (0, 1]."""
    asset: "Entity" = env.scene[asset_cfg.name]
    x = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)  # absolute, not _rel
    return torch.exp(-(x / sigma) ** 2)
