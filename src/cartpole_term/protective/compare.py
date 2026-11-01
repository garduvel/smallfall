"""Per-fall bookkeeping for scripts/go1_protective_compare.py (Gate 5's pass
test): replay identical injected fall states under several policies and
compare peak contact force, peak torque, and ground covered. Pure torch (no
mjlab import) so it is unit-tested locally; the script feeds it tensors read
from the env after every step.

Only each env's FIRST episode is recorded. The env must run with
auto_reset=False, so post-step readings are the step's true final state
rather than an already-injected next fall.
"""

from __future__ import annotations

import numpy as np
import torch

PER_FALL_FIELDS = (
    "peak_force_any_n",
    "peak_force_critical_n",
    "peak_torque_nm",
    "mean_torque_sq",
    "envelope_m",
    "envelope_growth_m",
    "root_travel_end_m",
    "steps",
    "settled",
    "return",
)


class FirstFallTracker:
    def __init__(self, root_xy: torch.Tensor, body_xy: torch.Tensor, start_active: bool = True):
        """Call right after env.reset(): positions are current there (reset()
        runs forward()), so the origin is the injected state itself.
        start_active=False: no env is recorded until begin() opens its window
        (the Gate 6 hand-off, where each env's window starts at its own step)."""
        n = root_xy.shape[0]
        dev = root_xy.device
        self.origin_xy = root_xy.clone()
        self.initial_extent = self._farthest(body_xy)
        self.active = torch.full((n,), start_active, dtype=torch.bool, device=dev)
        z = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.peak_force_any, self.peak_force_critical, self.peak_torque = z(), z(), z()
        self.torque_sq_sum, self.ret, self.root_travel_end = z(), z(), z()
        self.envelope = self.initial_extent.clone()
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.settled = torch.zeros(n, dtype=torch.bool, device=dev)

    def begin(self, mask: torch.Tensor, root_xy: torch.Tensor, body_xy: torch.Tensor) -> None:
        """Open a fresh recording window for the envs in `mask`, with the
        origin at their current position. Out of place throughout."""
        m = mask & ~self.active
        self.origin_xy = torch.where(m.unsqueeze(-1), root_xy, self.origin_xy)
        extent = self._farthest(body_xy)
        self.initial_extent = torch.where(m, extent, self.initial_extent)
        self.envelope = torch.where(m, extent, self.envelope)
        zero = torch.zeros_like(self.ret)
        for name in ("peak_force_any", "peak_force_critical", "peak_torque", "torque_sq_sum", "ret", "root_travel_end"):
            setattr(self, name, torch.where(m, zero, getattr(self, name)))
        self.steps = torch.where(m, torch.zeros_like(self.steps), self.steps)
        self.settled = torch.where(m, torch.zeros_like(self.settled), self.settled)
        self.active = self.active | m

    def _farthest(self, body_xy: torch.Tensor) -> torch.Tensor:
        return torch.norm(body_xy - self.origin_xy.unsqueeze(1), dim=-1).amax(dim=-1)

    def update(
        self,
        *,
        force_mag: torch.Tensor,  # [N, B] per-body net ground-contact force (N)
        critical_mask: torch.Tensor,  # [B] bool
        torque: torch.Tensor,  # [N, A] actuator force
        root_xy: torch.Tensor,  # [N, 2]
        body_xy: torch.Tensor,  # [N, K, 2]
        reward: torch.Tensor,  # [N]
        terminated: torch.Tensor,  # [N] bool -- `settled` fired
        truncated: torch.Tensor,  # [N] bool -- 150-step cap
    ) -> torch.Tensor:
        """Record one step for envs still in their first fall. Returns the
        [N] bool mask of envs whose first fall ended on this step."""
        a = self.active
        crit = force_mag[:, critical_mask]
        crit_max = crit.amax(dim=-1) if crit.shape[1] > 0 else torch.zeros_like(reward)

        def keep_max(old, new):
            return torch.where(a, torch.maximum(old, new), old)

        self.peak_force_any = keep_max(self.peak_force_any, force_mag.amax(dim=-1))
        self.peak_force_critical = keep_max(self.peak_force_critical, crit_max)
        self.peak_torque = keep_max(self.peak_torque, torque.abs().amax(dim=-1))
        self.envelope = keep_max(self.envelope, self._farthest(body_xy))
        self.torque_sq_sum = torch.where(a, self.torque_sq_sum + torque.square().sum(-1), self.torque_sq_sum)
        self.ret = torch.where(a, self.ret + reward, self.ret)
        self.steps = torch.where(a, self.steps + 1, self.steps)
        self.root_travel_end = torch.where(
            a, torch.norm(root_xy - self.origin_xy, dim=-1), self.root_travel_end
        )
        ended = a & (terminated | truncated)
        self.settled = torch.where(ended, terminated, self.settled)
        self.active = a & ~ended
        return ended

    @property
    def all_done(self) -> bool:
        return not bool(self.active.any())

    def per_fall(self) -> dict[str, np.ndarray]:
        steps = self.steps.clamp(min=1).float()
        out = {
            "peak_force_any_n": self.peak_force_any,
            "peak_force_critical_n": self.peak_force_critical,
            "peak_torque_nm": self.peak_torque,
            "mean_torque_sq": self.torque_sq_sum / steps,
            "envelope_m": self.envelope,
            "envelope_growth_m": self.envelope - self.initial_extent,
            "root_travel_end_m": self.root_travel_end,
            "steps": self.steps.float(),
            "settled": self.settled.float(),
            "return": self.ret,
        }
        return {k: v.detach().cpu().numpy() for k, v in out.items()}


def summarize(per_fall: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    summary = {}
    for k, v in per_fall.items():
        summary[k] = {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)),
            "max": float(np.max(v)),
        }
    return summary


# Lower is better for all of these; the protective policy passes Gate 5 if its
# median beats every baseline's median on each.
PASS_METRICS = ("peak_force_critical_n", "peak_force_any_n", "peak_torque_nm", "envelope_growth_m")


def verdict(summaries: dict[str, dict], candidate: str = "protective") -> dict:
    result = {}
    for m in PASS_METRICS:
        cand = summaries[candidate][m]["median"]
        vs = {arm: s[m]["median"] for arm, s in summaries.items() if arm != candidate}
        result[m] = {
            "candidate_median": cand,
            "baseline_medians": vs,
            "beats_all_baselines": all(cand < b for b in vs.values()),
        }
    result["pass"] = all(r["beats_all_baselines"] for r in result.values() if isinstance(r, dict))
    return result


# --- Gate 6/7: live hand-off bookkeeping (scripts/go1_handoff_eval.py) -------

EVENT_NONE, EVENT_FLAG, EVENT_MISSED_FALL = 0, 1, 2
PHASE_WALKING, PHASE_WINDOW, PHASE_DONE = 0, 1, 2


class HandoffTrials:
    """Per-env state for one live trial per env (each env's first episode).

    Timeline per env: WALKING (walker in control) until either the predictor
    flags (prob > threshold, scored on the walker's own obs+action exactly as
    in go1_collect_flagged_states.py), or a real fall happens unflagged
    (missed). That opens the env's measurement WINDOW, recorded by a
    FirstFallTracker; in the handoff arm a flag also latches the protective
    policy on for good. The window closes when the robot has been still for
    `hold_steps` consecutive steps (same thresholds as the protective task's
    `settled` termination), after `window_steps`, or at the env time limit.
    Envs with no event within `walk_steps` end as EVENT_NONE.

    Call order per env step t: pre_step (before choosing actions; returns the
    latch mask), env.step, post_step. Pure torch so it is unit-tested locally.
    """

    def __init__(self, *, handoff: bool, walk_steps: int, window_steps: int, hold_steps: int,
                 lin_threshold: float, ang_threshold: float,
                 root_xy: torch.Tensor, body_xy: torch.Tensor,
                 down_height: torch.Tensor | None = None):
        n, dev = root_xy.shape[0], root_xy.device
        self.handoff = handoff
        self.walk_steps, self.window_steps, self.hold_steps = walk_steps, window_steps, hold_steps
        self.lin_threshold, self.ang_threshold = lin_threshold, ang_threshold
        self.tracker = FirstFallTracker(root_xy, body_xy, start_active=False)
        long = lambda v: torch.full((n,), v, dtype=torch.long, device=dev)  # noqa: E731
        self.phase = long(PHASE_WALKING)
        self.event = long(EVENT_NONE)
        self.flag_step, self.fall_step, self.window_start = long(-1), long(-1), long(-1)
        self.window_len, self.settle_count = long(0), long(0)
        self.latched = torch.zeros(n, dtype=torch.bool, device=dev)
        # "Down" = on the ground: the fall termination has fired, or the root
        # is below `down_height` (per env; None disables the height test).
        # Latches for the rest of the trial. The script turns perturbations
        # off for down envs (common/perturbation.py).
        self.down = torch.zeros(n, dtype=torch.bool, device=dev)
        self.down_height = down_height
        self.prob_at_flag = torch.full((n,), float("nan"), device=dev)
        self.root_pose_at_start = torch.full((n, 7), float("nan"), device=dev)

    def pre_step(self, t: int, prob: torch.Tensor, threshold: float, root_xy: torch.Tensor,
                 body_xy: torch.Tensor, root_pose: torch.Tensor) -> torch.Tensor:
        new = (self.phase == PHASE_WALKING) & (prob > threshold)
        if t >= self.walk_steps:
            new = torch.zeros_like(new)
        self._open(new, EVENT_FLAG, t, root_xy, body_xy, root_pose)
        self.flag_step = torch.where(new, t, self.flag_step)
        self.prob_at_flag = torch.where(new, prob, self.prob_at_flag)
        if self.handoff:
            before = self.latched
            self.latched = self.latched | new
            assert not bool((before & ~self.latched).any()), "a latched env switched back"
        return self.latched

    def post_step(self, t: int, *, fell: torch.Tensor, time_out: torch.Tensor, force_mag: torch.Tensor,
                  critical_mask: torch.Tensor, torque: torch.Tensor, root_xy: torch.Tensor,
                  body_xy: torch.Tensor, reward: torch.Tensor, lin_speed: torch.Tensor,
                  ang_speed: torch.Tensor, root_pose: torch.Tensor,
                  root_z: torch.Tensor | None = None) -> None:
        low = (root_z < self.down_height) if (root_z is not None and self.down_height is not None) else torch.zeros_like(fell)
        self.down = self.down | fell | low
        first_fall = fell & (self.fall_step < 0) & (self.phase != PHASE_DONE)
        self.fall_step = torch.where(first_fall, t, self.fall_step)

        in_window = self.phase == PHASE_WINDOW
        still = (lin_speed < self.lin_threshold) & (ang_speed < self.ang_threshold)
        self.settle_count = torch.where(in_window & still, self.settle_count + 1, torch.zeros_like(self.settle_count))
        self.window_len = torch.where(in_window, self.window_len + 1, self.window_len)
        settled_now = in_window & (self.settle_count >= self.hold_steps)
        cap_now = in_window & ((self.window_len >= self.window_steps) | time_out)
        ended = self.tracker.update(
            force_mag=force_mag, critical_mask=critical_mask, torque=torque, root_xy=root_xy,
            body_xy=body_xy, reward=reward, terminated=settled_now, truncated=cap_now,
        )
        assert bool(((ended & ~in_window) == 0).all()), "tracker ended an env outside its window"
        self.phase = torch.where(ended, PHASE_DONE, self.phase)

        # A real fall the predictor never flagged: the walker stays in
        # control in BOTH arms; the window opens on the post-fall state.
        missed = (self.phase == PHASE_WALKING) & fell
        self._open(missed, EVENT_MISSED_FALL, t + 1, root_xy, body_xy, root_pose)

        if t >= self.walk_steps - 1:
            self.phase = torch.where(self.phase == PHASE_WALKING, PHASE_DONE, self.phase)

    def _open(self, mask, event, t, root_xy, body_xy, root_pose):
        self.event = torch.where(mask, event, self.event)
        self.window_start = torch.where(mask, t, self.window_start)
        self.root_pose_at_start = torch.where(mask.unsqueeze(-1), root_pose, self.root_pose_at_start)
        self.phase = torch.where(mask, PHASE_WINDOW, self.phase)
        self.tracker.begin(mask, root_xy, body_xy)

    @property
    def all_done(self) -> bool:
        return bool((self.phase == PHASE_DONE).all())

    def results(self) -> dict[str, np.ndarray]:
        out = {k: v for k, v in self.tracker.per_fall().items()}
        for name in ("event", "flag_step", "fall_step", "window_start", "window_len",
                     "prob_at_flag", "root_pose_at_start", "latched", "down"):
            out[name] = getattr(self, name).detach().cpu().numpy()
        return out


# --- Round 8: mechanical checkpoint selection (scripts/go1_select_protective_checkpoint.py)

MIN_SETTLED_RATE = 0.99
# User's stated priority (2026-10-03): the main goal is less space taken up by
# the fall; force "should be reduced a bit" but may stay large.
PRIMARY_METRIC = "envelope_growth_m"
REQUIRED_METRICS = ("envelope_growth_m", "peak_force_critical_n", "peak_torque_nm")
SECONDARY_METRIC = "peak_force_any_n"


def select_checkpoint(summaries_by_label: dict[str, dict], baseline: str = "walker",
                      candidate: str = "protective") -> dict:
    """Pick one checkpoint from compare.py summaries on a SELECTION bank
    (never the test bank). Labels identify a checkpoint (e.g. "8b_it150").
    Rule, fixed before any round-8 data existed (DEVIATIONS.md, amended):
      eligible  = settled rate >= 0.99 AND candidate median < baseline median
                  on ground covered, trunk/hip force, and peak torque;
      preferred = eligible AND candidate median < baseline median on
                  any-body peak force (the "force reduced a bit" goal);
      choose, from the preferred checkpoints if there are any, otherwise from
      the eligible ones, the one with the LOWEST median ground covered
      (ties -> first label in sorted order).
    If none is eligible, fall back to the lowest ground covered among
    checkpoints with settled >= 0.99 (then among all), flagged
    eligible=False so the caller reports it rather than hiding it."""
    rows = []
    for label in sorted(summaries_by_label):
        s = summaries_by_label[label]
        beats = {m: s[candidate][m]["median"] < s[baseline][m]["median"] for m in PASS_METRICS}
        settled = s[candidate]["settled"]["mean"]
        eligible = settled >= MIN_SETTLED_RATE and all(beats[m] for m in REQUIRED_METRICS)
        rows.append({
            "label": label,
            "settled": settled,
            "beats_baseline": beats,
            "envelope_growth_median": s[candidate][PRIMARY_METRIC]["median"],
            "any_force_median": s[candidate][SECONDARY_METRIC]["median"],
            "eligible": eligible,
            "preferred": eligible and beats[SECONDARY_METRIC],
        })
    pool = ([r for r in rows if r["preferred"]] or [r for r in rows if r["eligible"]]
            or [r for r in rows if r["settled"] >= MIN_SETTLED_RATE] or rows)
    best = min(pool, key=lambda r: (r["envelope_growth_median"], r["label"]))
    return {"selected_label": best["label"], "selected_is_eligible": best["eligible"],
            "selected_also_reduces_any_force": best["preferred"], "candidates": rows}
