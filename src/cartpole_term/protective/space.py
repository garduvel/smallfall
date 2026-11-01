""""Space taken up by the fall" for the protective task: log-only metrics
(mjlab MetricsTermCfg terms -- no weight, no reward effect, reported per
episode as Episode_Metrics/*) and, from round 7, one reward term
(`fall_envelope_growth`). Not literature-sourced: no published fall reward
penalizes space directly.

(A) Ground covered: horizontal distance from the fall-start root position,
    either of the farthest body (`bodies`, reduce="max" -> radius of the
    circle, centred on the fall start, that the whole fall stayed inside)
    or of the root (`root`, reduce="last" -> start-to-rest travel).
(B) Sprawl: RMS horizontal distance of the bodies from their own centroid.

The fall-start point is captured on the FIRST step after reset, not inside
reset(): mjlab's reset events (the fall-state injection) run before the
metrics manager's reset, but body positions come from derived `xpos`,
which is stale until the next forward() -- reading it in reset() would
return the pre-reset terminal pose. The first post-reset step is 20 ms
after injection (~4 cm at the bank's ~2 m/s median horizontal speed).

Per-env buffers are updated out of place: they can become inference
tensors under the frozen-eval rollout's torch.inference_mode (see
terminations.py / DEVIATIONS.md).
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class distance_from_fall_start:
    """Horizontal distance from the fall-start root xy, of either the
    farthest body (`of="bodies"`) or the root itself (`of="root"`)."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        of = cfg.params.get("of", "bodies")
        if of not in ("bodies", "root"):
            raise ValueError(f"distance_from_fall_start: of must be 'bodies' or 'root', got {of!r}")
        self._origin_xy = torch.zeros(env.num_envs, 2, device=env.device)
        self._needs_origin = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._needs_origin = torch.ones_like(self._needs_origin)
        else:
            needs = self._needs_origin.clone()
            needs[env_ids] = True
            self._needs_origin = needs

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        of: str = "bodies",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        root_xy = asset.data.root_link_pos_w[:, :2]
        self._origin_xy = torch.where(self._needs_origin.unsqueeze(-1), root_xy, self._origin_xy)
        self._needs_origin = torch.zeros_like(self._needs_origin)
        if of == "root":
            return torch.norm(root_xy - self._origin_xy, dim=-1)
        body_xy = asset.data.body_link_pos_w[..., :2]
        return torch.norm(body_xy - self._origin_xy.unsqueeze(1), dim=-1).amax(dim=-1)


def body_spread(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
    """RMS horizontal distance of the robot's bodies from their centroid (m)."""
    body_xy = env.scene[asset_cfg.name].data.body_link_pos_w[..., :2]
    centered = body_xy - body_xy.mean(dim=1, keepdim=True)
    return centered.square().sum(dim=-1).mean(dim=-1).sqrt()


class fall_envelope_growth(distance_from_fall_start):
    """Reward term: per-step INCREASE in the fall envelope (farthest body's
    horizontal distance from the fall-start root xy). Increments telescope,
    so an episode's raw sum is its final envelope minus the robot's own
    extent on the first post-reset step (which is not penalized). Chosen
    over a per-step distance penalty because its episode total is exactly
    "how much extra clear radius this fall needed" (round 6: sprawl was
    constant at ~0.22 m, while ground covered varied ~2x between falls).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._running_max = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        fresh = self._needs_origin
        dist = super().__call__(env, of="bodies", asset_cfg=asset_cfg)
        prev_max = torch.where(fresh, dist, self._running_max)
        growth = (dist - prev_max).clamp(min=0.0)
        self._running_max = torch.maximum(prev_max, dist)
        if hasattr(env, "extras"):
            env.extras.setdefault("log", {})["Metrics/fall_envelope_growth_mean"] = growth.mean()
        return growth
