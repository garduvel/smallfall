"""Termination term for the protective "small-fall" task: end the episode
once the robot has genuinely come to rest, instead of only on a fixed
step-count timeout.

**Why this exists (user-reported, confirmed against our own logged data):**
the original design fixed the episode at 10 steps / 0.2s, reasoned (see
env_cfg.py's prior docstring) from SafeFall's own 40-step/200Hz protective
episode, time-scaled to Go1's 50Hz. But this project's OWN measured
post-fall dynamics (outputs/go1_flagged_states_seed0.npz's `post_fall_obs`,
collected by scripts/go1_collect_flagged_states.py) already showed mean
base angular speed under the UNCONTROLLED nominal policy still elevated at
step 28 of a 30-step window -- i.e. the fixed 10-step cutoff was ending
episodes mid-fall, before the robot had settled, exactly the "terminates
while still in the falling motion" symptom reported. Watching the trained
policy's video surfaced the same thing the numbers already implied.

**Literature check, done before writing this (not guessed):** FRASA (Zhu
et al., arXiv:2410.08655, an end-to-end RL fall-recovery/stand-up agent for
humanoids) does NOT gate its episode length on velocity at all -- its own
episodes terminate on unsafe states (excessive trunk pitch / pitch-rate) or
a fixed max duration, same coarse shape this project already had. What
FRASA DOES contribute is its *task-completion* criterion, confirmed by
direct fetch of the paper: recovery is only declared successful once "the
error in the state vector components ... must remain below 5 degrees for
0.5 seconds" -- a SUSTAINED-DURATION threshold check, not an instantaneous
single-step crossing. That structure (below-threshold, HELD for a fixed
duration, not just touched once) is what `settled` below reuses -- adapted
to root linear+angular velocity (this task has no target pose to track
against, unlike FRASA's stand-up controller, so "has it stopped moving" is
the directly-measurable analogue of "has it stabilized"). The 0.5s hold
duration is FRASA's own literal number, time-scaled to Go1's confirmed 50Hz
(env.step_dt, not hardcoded -- reusable unmodified for G1 at its own Hz per
the multi-robot plan). The two velocity thresholds themselves are NOT from
FRASA -- that paper never states a velocity number (confirmed: it checks
state-vector position error, not speed) -- they are this project's own
calibration. See DEVIATIONS.md for both the "what's cited" / "what's
invented" split and the numbers chosen.

Registered with `time_out=False`: reaching `settled` is a genuine terminal
state (once at rest, under this task's reward there is nothing further to
gain, so no value bootstrap should occur), distinct from the plain
step-count `time_out` term (kept, from the base velocity-task cfg,
unmodified) which now serves only as a safety cap for the rare episode that
never settles inside the enlarged budget -- see env_cfg.py's
PROTECTIVE_EPISODE_LENGTH_S.
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class settled:
    """terminated=True once root linear AND angular speed have both stayed
    below their thresholds for `hold_s` seconds running (see module
    docstring for the FRASA-derived sustained-duration structure). A
    per-env consecutive-steps-below-threshold counter, reset to 0 the
    instant either speed exceeds its threshold, so a single mid-bounce dip
    near zero can't fire this early -- confirmed via mjlab's own class-term
    protocol (mjlab/managers/manager_base.py: a term whose `func` is a
    class gets `cfg.func(cfg=cfg, env=env)` instantiated once, and its
    `.reset(env_ids)` called by the manager on every env reset) that this
    is the correct place to keep that per-env state, not a module global.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
        params = cfg.params
        self._linear_threshold: float = params["linear_threshold_mps"]
        self._angular_threshold: float = params["angular_threshold_rad_s"]
        hold_s: float = params["hold_s"]
        self._hold_steps = max(1, round(hold_s / env.step_dt))
        self._below_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # Out-of-place on purpose: the counter can be an inference tensor (env
        # built under torch.inference_mode by the frozen-eval runner), and an
        # in-place write to it outside inference mode raises -- the same bug
        # that hit ContactSensor history_length (see DEVIATIONS.md).
        if env_ids is None:
            self._below_count = torch.zeros_like(self._below_count)
        else:
            counts = self._below_count.clone()
            counts[env_ids] = 0
            self._below_count = counts

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        linear_threshold_mps: float = 0.2,
        angular_threshold_rad_s: float = 0.5,
        hold_s: float = 0.5,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        linear_speed = torch.norm(asset.data.root_link_lin_vel_w, dim=-1)
        angular_speed = torch.norm(asset.data.root_link_ang_vel_w, dim=-1)

        below = (linear_speed < self._linear_threshold) & (angular_speed < self._angular_threshold)
        self._below_count = torch.where(
            below, self._below_count + 1, torch.zeros_like(self._below_count)
        )
        is_settled = self._below_count >= self._hold_steps

        env.extras["log"]["Metrics/settled_linear_speed_mean"] = linear_speed.mean()
        env.extras["log"]["Metrics/settled_angular_speed_mean"] = angular_speed.mean()
        env.extras["log"]["Metrics/settled_fraction"] = is_settled.float().mean()
        return is_settled
