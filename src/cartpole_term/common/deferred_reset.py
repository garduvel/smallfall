"""Deferred-reset wrapper (Gate 5/6 infrastructure, multi-robot plan --
the original design plan). See DEVIATIONS.md for the full
writeup; summary here.

mjlab's default behaviour (`ManagerBasedRlEnvCfg.auto_reset=True`) resets a
terminated env's physics WITHIN the same `step()` call that detected the
termination, before returning -- so the terminal state (the robot mid-fall,
or at the exact instant a fall condition first became true) is never
observable by a caller; it's overwritten before the return value exists.
That makes it structurally impossible to ever train or evaluate a
protective/recovery policy (Gate 5), whose entire job is to act on states
that only exist AFTER the nominal policy has already failed.

Confirmed against the real installed-version source this session (mjlab
1.6.0, fetched from mujocolab/mjlab@main,
src/mjlab/envs/manager_based_rl_env.py) -- supersedes this project's
earlier "unconfirmed ManagerBasedRlEnvCfg.auto_reset field" note
(DEVIATIONS.md, S6.2.8 entry): `auto_reset` is a real, documented
ManagerBasedRlEnvCfg field. With `auto_reset=False`, `step()` skips the
automatic per-env reset entirely -- the returned observation for a
done env is the true terminal state, and nothing in mjlab's own step()
body treats a still-terminated env specially beyond one bookkeeping
flag (`_manual_reset_pending`), so physics keeps advancing normally for
it on every subsequent `step()` call.

The one obstacle: mjlab's own `step()` raises `RuntimeError` if ANY env
has `_manual_reset_pending=True` (set for every env that was done last
step) when `step()` is called again -- a safety rail that forces prompt
resets under the library's own intended manual-reset usage pattern. This
wrapper deliberately bypasses that rail for envs the caller has
explicitly asked to keep running past termination (`hold(env_ids)`), by
clearing their `_manual_reset_pending` bit directly instead of calling
`reset(env_ids=...)`. This IS a deviation from mjlab's documented usage
pattern (its own docstring says auto_reset=False is "intended for users
running their own training loop... that handles the reset between
steps" -- this wrapper deliberately sometimes does NOT reset). Logged in
DEVIATIONS.md.

**Known timing tradeoff, accepted:** for envs the caller does NOT hold,
this wrapper mimics auto_reset=True's own external behaviour (done envs
come back reset) but the reset itself happens at the TOP of the NEXT
`step()` call rather than synchronously inside the call that detected
termination. Concretely: the action a caller supplies for an env on the
step immediately following its (unheld) termination is chosen against
that env's terminal-state observation, not a fresh post-reset
observation -- one step of a policy "reacting" to a state it's about to
be yanked out of, discarded once the reset actually lands. For a
dedicated recovery/protective-policy env where every terminated env is
deliberately held (the intended use case), this tradeoff never arises:
there is no "unheld" env in that pipeline. It only matters for mixed
use (some envs held, most not), and is called out here rather than
hidden.
"""

from __future__ import annotations

import torch


class DeferredResetEnv:
    """Wraps a raw `ManagerBasedRlEnv` (or the RslRlVecEnvWrapper around
    one) built with `cfg.auto_reset = False`. Any env never passed to
    `hold()` behaves exactly like ordinary auto_reset=True from the
    caller's point of view (done -> reset, one wrapper-step later -- see
    module docstring). An env passed to `hold(env_ids)` right after it
    terminates/truncates keeps running, under whatever `actions` the
    caller feeds it next, until the caller calls `release(env_ids)`.
    """

    def __init__(self, raw_env):
        cfg = getattr(raw_env, "cfg", None)
        if cfg is None or getattr(cfg, "auto_reset", True):
            raise ValueError(
                "DeferredResetEnv requires an env built with cfg.auto_reset = False -- "
                "with the shipped default (True), mjlab resets terminated envs "
                "internally before this wrapper ever sees them, so holding is impossible."
            )
        self.raw_env = raw_env
        self.num_envs = raw_env.num_envs
        self.held = torch.zeros(self.num_envs, dtype=torch.bool, device=raw_env.device)

    def reset(self, *, seed=None, env_ids=None, options=None):
        obs, extras = self.raw_env.reset(seed=seed, env_ids=env_ids, options=options)
        if env_ids is None:
            self.held.zero_()
        else:
            self.held[env_ids] = False
        return obs, extras

    def step(self, actions: torch.Tensor):
        """Resolve any envs left pending-reset from the previous call
        (see module docstring's timing tradeoff), then step. Returns the
        raw env's own 5-tuple (obs, rewards, terminated, truncated,
        extras) unchanged -- callers decide which of THIS call's
        newly-terminated/truncated envs to hold via `hold(env_ids)`
        before the next `step()` call."""
        pending = self.raw_env._manual_reset_pending
        pending_ids = pending.nonzero(as_tuple=False).squeeze(-1)
        if len(pending_ids) > 0:
            still_held_mask = self.held[pending_ids]
            unheld_pending_ids = pending_ids[~still_held_mask]
            if len(unheld_pending_ids) > 0:
                self.raw_env.reset(env_ids=unheld_pending_ids)  # clears their pending bit internally
            held_pending_ids = pending_ids[still_held_mask]
            if len(held_pending_ids) > 0:
                pending[held_pending_ids] = False  # bypass: no physical reset, just clear the guard bit

        return self.raw_env.step(actions)

    def hold(self, env_ids: torch.Tensor) -> None:
        """Mark env_ids to keep running past termination instead of being
        reset on the NEXT step() call. Call this using the SAME step's
        `terminated`/`truncated` output that just flagged env_ids."""
        if len(env_ids) > 0:
            self.held[env_ids] = True

    def release(self, env_ids: torch.Tensor) -> None:
        """End a held recovery/continuation episode for env_ids: reset
        them now, immediately (not deferred to the next step)."""
        if len(env_ids) > 0:
            self.held[env_ids] = False
            self.raw_env.reset(env_ids=env_ids)
