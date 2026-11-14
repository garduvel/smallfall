"""Ten ways to measure how small a fall is (metric ablation, DEVIATIONS.md
2026-11-13). Pure torch so it is unit-tested locally; the compare script
feeds it the same per-step readings as FirstFallTracker.

Every measure is per fall, in the horizontal plane, relative to the fall's
start (the injected state), and LOWER = SMALLER. Body positions are the
robot's body-frame origins (body_link_pos_w), as in FirstFallTracker; a body's
geometry around that point is not included.

  envelope_growth_m      farthest any body gets from the start root position,
                         minus how far the farthest body already was at the
                         start (the project's primary measure, unchanged)
  max_reach_m            farthest any body gets from the start root position
                         (the clear radius the fall needs)
  root_travel_end_m      root's straight-line distance from start to end
  root_travel_max_m      root's largest distance from start at any time
  root_path_m            length of the root's path (counts back-and-forth)
  centroid_travel_end_m  how far the mean of all body positions moved,
                         start to end
  swept_hull_area_m2     area of the convex hull of every body position over
                         the whole fall (floor area the fall passes over)
  hull_growth_m2         swept_hull_area_m2 minus the start pose's hull area
  contact_reach_m        farthest a body touching the ground gets from the
                         start root position (0 if none ever touches)
  touched_area_m2        floor area touched: 10 cm grid cells that held a
                         ground-touching body at some step, times cell area
"""

from __future__ import annotations

import math

import numpy as np
import torch

SIZE_METRICS = (
    "envelope_growth_m",
    "max_reach_m",
    "root_travel_end_m",
    "root_travel_max_m",
    "root_path_m",
    "centroid_travel_end_m",
    "swept_hull_area_m2",
    "hull_growth_m2",
    "contact_reach_m",
    "touched_area_m2",
)
AREA_METRICS = ("swept_hull_area_m2", "hull_growth_m2", "touched_area_m2")

HULL_DIRECTIONS = 32  # support directions; the polygon over-estimates a hull's area by < 1.3%
GRID_CELL_M = 0.10
GRID_HALF_M = 3.0  # positions beyond +-3 m are counted in the edge cell


def _directions(n: int, device) -> torch.Tensor:
    ang = torch.arange(n, device=device, dtype=torch.float32) * (2 * math.pi / n)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # [D, 2]


def support_polygon_area(h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Area of the polygon {p : u_i . p <= h_i for all i}, for evenly spaced
    unit directions u [D, 2] and support values h [..., D] of a point set (so
    the polygon is that set's convex hull, slightly enlarged). Vertices are
    the intersections of neighbouring support lines; shoelace formula."""
    u_next = torch.roll(u, -1, dims=0)
    h_next = torch.roll(h, -1, dims=-1)
    det = u[:, 0] * u_next[:, 1] - u[:, 1] * u_next[:, 0]  # sin(2 pi / D), > 0
    x = (h * u_next[:, 1] - h_next * u[:, 1]) / det
    y = (u[:, 0] * h_next - u_next[:, 0] * h) / det
    x2, y2 = torch.roll(x, -1, dims=-1), torch.roll(y, -1, dims=-1)
    return 0.5 * (x * y2 - x2 * y).sum(-1).abs()


class FallSizeTracker:
    """Per-fall size measures for each env's first fall (same windowing as
    FirstFallTracker: call right after env.reset(); update() after every
    step; an env stops recording once its episode ends)."""

    def __init__(self, root_xy: torch.Tensor, body_xy: torch.Tensor):
        n, dev = root_xy.shape[0], root_xy.device
        self.origin_xy = root_xy.clone()
        rel = body_xy - self.origin_xy.unsqueeze(1)
        self.u = _directions(HULL_DIRECTIONS, dev)
        self.initial_extent = rel.norm(dim=-1).amax(-1)
        self.reach = self.initial_extent.clone()
        self.support = (rel @ self.u.T).amax(dim=1)  # [N, D]
        self.initial_hull = support_polygon_area(self.support, self.u)
        self.centroid0 = body_xy.mean(1)
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.root_end, self.root_max, self.root_path, self.centroid_end, self.contact_reach = z(), z(), z(), z(), z()
        self.prev_root = root_xy.clone()
        self.cells_per_side = int(round(2 * GRID_HALF_M / GRID_CELL_M))
        self.touched = torch.zeros(n, self.cells_per_side ** 2, dtype=torch.bool, device=dev)
        self.active = torch.ones(n, dtype=torch.bool, device=dev)

    def begin(self, mask: torch.Tensor, root_xy: torch.Tensor, body_xy: torch.Tensor) -> None:
        """Start a fresh fall for the envs in `mask` (origin = their current
        position) and make them active. Out of place throughout: the buffers
        can be inference tensors (see space.py's module docstring)."""
        m = mask
        m1 = m.unsqueeze(-1)
        origin = torch.where(m1, root_xy, self.origin_xy)
        rel = body_xy - origin.unsqueeze(1)
        extent = rel.norm(dim=-1).amax(-1)
        support = (rel @ self.u.T).amax(dim=1)
        self.origin_xy = origin
        self.initial_extent = torch.where(m, extent, self.initial_extent)
        self.reach = torch.where(m, extent, self.reach)
        self.support = torch.where(m1, support, self.support)
        self.initial_hull = torch.where(m, support_polygon_area(support, self.u), self.initial_hull)
        self.centroid0 = torch.where(m1, body_xy.mean(1), self.centroid0)
        zero = torch.zeros_like(self.root_end)
        for name in ("root_end", "root_max", "root_path", "centroid_end", "contact_reach"):
            setattr(self, name, torch.where(m, zero, getattr(self, name)))
        self.prev_root = torch.where(m1, root_xy, self.prev_root)
        self.touched = self.touched & ~m1
        self.active = self.active | m

    def update(self, *, root_xy: torch.Tensor, body_xy: torch.Tensor, contact_xy: torch.Tensor,
               in_contact: torch.Tensor, ended: torch.Tensor) -> None:
        """root_xy [N,2]; body_xy [N,K,2] every body; contact_xy [N,C,2] the
        bodies watched for ground contact, with in_contact [N,C] bool;
        ended [N] bool (terminated | truncated) -- this step is recorded, then
        the env stops."""
        a = self.active
        rel = body_xy - self.origin_xy.unsqueeze(1)

        def keep_max(old, new):
            return torch.where(a, torch.maximum(old, new), old)

        root_d = (root_xy - self.origin_xy).norm(dim=-1)
        self.reach = keep_max(self.reach, rel.norm(dim=-1).amax(-1))
        self.support = torch.where(a.unsqueeze(-1), torch.maximum(self.support, (rel @ self.u.T).amax(1)), self.support)
        self.root_max = keep_max(self.root_max, root_d)
        self.root_end = torch.where(a, root_d, self.root_end)
        self.root_path = torch.where(a, self.root_path + (root_xy - self.prev_root).norm(dim=-1), self.root_path)
        self.prev_root = root_xy.clone()
        self.centroid_end = torch.where(a, (body_xy.mean(1) - self.centroid0).norm(dim=-1), self.centroid_end)

        crel = contact_xy - self.origin_xy.unsqueeze(1)
        cd = torch.where(in_contact, crel.norm(dim=-1), torch.zeros_like(in_contact, dtype=crel.dtype))
        self.contact_reach = keep_max(self.contact_reach, cd.amax(-1) if cd.shape[1] else torch.zeros_like(root_d))
        if crel.shape[1]:
            idx = ((crel + GRID_HALF_M) / GRID_CELL_M).floor().long().clamp(0, self.cells_per_side - 1)
            flat = idx[..., 0] * self.cells_per_side + idx[..., 1]  # [N, C]
            self.touched = self.touched | self._scatter_or(flat, in_contact & a.unsqueeze(-1))
        self.active = a & ~ended

    def _scatter_or(self, flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Cells `flat` [N, C] where `mask` is True, as an [N, cells] bool
        grid. Writing the same True twice is harmless, so duplicates are fine."""
        rows = torch.arange(flat.shape[0], device=flat.device).unsqueeze(-1).expand_as(flat)
        grid = torch.zeros(self.touched.shape, dtype=torch.bool, device=flat.device)
        return grid.index_put((rows[mask], flat[mask]), torch.ones((), dtype=torch.bool, device=flat.device))

    @property
    def all_done(self) -> bool:
        return not bool(self.active.any())

    def per_fall(self) -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in self.totals().items()}

    def totals(self) -> dict[str, torch.Tensor]:
        """Each measure so far, per env (tensors, on the env's device)."""
        hull = support_polygon_area(self.support, self.u)
        out = {
            "envelope_growth_m": self.reach - self.initial_extent,
            "max_reach_m": self.reach,
            "root_travel_end_m": self.root_end,
            "root_travel_max_m": self.root_max,
            "root_path_m": self.root_path,
            "centroid_travel_end_m": self.centroid_end,
            "swept_hull_area_m2": hull,
            "hull_growth_m2": hull - self.initial_hull,
            "contact_reach_m": self.contact_reach,
            "touched_area_m2": self.touched.sum(-1).float() * GRID_CELL_M ** 2,
        }
        assert tuple(out) == SIZE_METRICS
        return out
