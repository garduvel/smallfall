"""A fall-size measure as a per-step reward (or log-only metric) term, for
the reward ablation (DEVIATIONS.md 2026-11-13): one SmallFall policy per
measure in protective/fall_size.py.

Each step the term returns the INCREASE of the chosen measure since the
previous step, so an episode's raw sum is exactly that measure for the fall
(as the existing fall_envelope_growth term does for ground covered). For
"end" measures (root / centroid travel at the end) the increase can be
negative: moving back toward the start is paid back. The fall start is
captured on the first step after reset, as in space.py (positions are stale
inside reset()).

No mjlab import, so it is unit-tested locally with a fake env.
"""

from __future__ import annotations

import torch

from cartpole_term.protective.fall_size import SIZE_METRICS, FallSizeTracker

CONTACT_N = 1.0  # same threshold as the compare replay


class fall_size_increment:
    def __init__(self, cfg, env):
        measure = cfg.params.get("measure")
        if measure not in SIZE_METRICS:
            raise ValueError(f"fall_size_increment: unknown measure {measure!r}; one of {SIZE_METRICS}")
        n = env.num_envs
        self._tracker = None
        self._contact_idx = None
        self._fresh = torch.ones(n, dtype=torch.bool, device=env.device)
        self._prev = torch.zeros(n, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._fresh = torch.ones_like(self._fresh)
        else:
            fresh = self._fresh.clone()
            fresh[env_ids] = True
            self._fresh = fresh

    def __call__(self, env, measure: str, sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
        asset = env.scene[asset_name]
        sensor = env.scene[sensor_name]
        root_xy = asset.data.root_link_pos_w[:, :2]
        body_xy = asset.data.body_link_pos_w[..., :2]
        if self._contact_idx is None:
            names = list(asset.body_names)
            self._contact_idx = torch.tensor([names.index(b) for b in sensor.primary_names], device=root_xy.device)
        if self._tracker is None:
            self._tracker = FallSizeTracker(root_xy, body_xy)
        fresh = self._fresh
        self._tracker.begin(fresh, root_xy, body_xy)
        self._fresh = torch.zeros_like(fresh)
        self._tracker.update(
            root_xy=root_xy, body_xy=body_xy, contact_xy=body_xy[:, self._contact_idx],
            in_contact=torch.norm(sensor.data.force, dim=-1) > CONTACT_N,
            ended=torch.zeros_like(fresh),
        )
        total = self._tracker.totals()[measure]
        prev = torch.where(fresh, torch.zeros_like(self._prev), self._prev)
        self._prev = total
        return total - prev
