"""Reward terms for the protective "small-fall" task (Gate 5 of
~/.claude/plans/snug-booping-gosling.md). See env_cfg.py's module
docstring for the full grounding/citation writeup; summary here.

**Citation correction, made this session:** earlier code (and the plan
this project inherited) attributed a "1000/1/0.5 per-body-vulnerability
tier multiplier" to arXiv:2512.01336. A direct fetch of that paper's real
Table I this session found NO such multiplier there -- that 1000/1/0.5
figure belongs to SafeFall (arXiv:2511.18509), a DIFFERENT paper. What
arXiv:2512.01336's Table I actually reports is a simpler, complete,
directly-usable 2-category contact-force split plus real weights for
several other terms this task also uses:

  root linear velocity:        -1.03125
  contact force, critical body: -0.02
  contact force, all bodies:    -0.001
  action rate:                  -0.00667

`vulnerability_weighted_contact_force` now implements THAT scheme
directly -- two categories (critical vs. all-other), with -0.02/-0.001 as
the literal, final per-Newton weights (no separate outer "blend then
rescale" step needed, unlike the earlier SafeFall-tier-multiplier design
this replaced). Discovered because the first two rounds of empirical
rebalancing (see DEVIATIONS.md) kept finding new dominant terms without
resolving the underlying "reward stays flat" problem -- consulting the
literature directly, as this project's own standing rule requires for
anything new/uncertain, surfaced a citation error AND a complete,
better-grounded weight set in one pass, rather than continuing to guess.

**Still uses the INSTANTANEOUS per-step contact force, not SafeFall's
0.3-peak/0.7-average-over-substeps blend** -- that blend needed mjlab's
ContactSensorCfg `history_length>0` (a rolling substep buffer), which hit
a confirmed real bug on the first actual GPU run of this task: mjlab's
own `ContactSensor.reset()` does `buf[env_ids] = 0.0` on that history
buffer, and whatever inference-mode context this project's runner
constructs the eval env under makes that buffer a permanent PyTorch
"inference tensor" that can never be written to in-place again outside
inference mode -- `RuntimeError: Inplace update to inference tensor
outside InferenceMode is not allowed.` Not fixable here (mjlab is
site-packages, never edited -- CLAUDE.md S12#18). Moot now anyway, since
this term no longer follows SafeFall's blend design at all.

`root_linear_velocity_penalty`'s weight (-1.03125) IS a literal number
from arXiv:2512.01336's Table I, confirmed this session -- reused as-is,
with the same caveat every "adapted, not independently calibrated for
Go1" number in this project carries (see DEVIATIONS.md).

`action_rate_l2` is NOT reimplemented here -- reused directly from
mjlab.envs.mdp's generic, already-shipped function, but now at
arXiv:2512.01336's own cited weight (-0.00667) rather than Go1's walking-
task weight, which two rounds of measurement showed was still 5-15x too
large for this task's own action-rate magnitude (see DEVIATIONS.md).
`joint_pos_limits` also stays a thin logging wrapper around mjlab's
generic function, at Go1's own shipped weight -- arXiv:2512.01336 reports
a DoF-position-limit term too (-29) but its raw quantity's definition
isn't confirmed comparable to mjlab's own `joint_pos_limits` formula, so
that number is NOT blindly copied (same reasoning that caught the
action-rate and contact-force mismatches in the first place). Wired in
env_cfg.py, not duplicated here.

`logged_joint_torques_l2` (round 5) wraps mjlab's generic torque penalty;
its weight was sized from our own measured raw value (see env_cfg.py). A
round-4 `settled_bonus` term was removed: it could never fire under PPO's
exploration noise (see DEVIATIONS.md).
"""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class vulnerability_weighted_contact_force:
    """Per-body contact-force penalty, arXiv:2512.01336 Table I's real
    2-category scheme (critical vs. all other bodies) -- see module
    docstring for the citation correction this replaced. Precomputes the
    per-primary weight vector once at construction (mirrors mjlab's own
    `feet_swing_height` class-reward-term pattern,
    mjlab/tasks/velocity/mdp/rewards.py) instead of rebuilding it from a
    dict lookup every step.

    The per-category values ARE the final Newtons->reward weights (no
    separate outer RewardTermCfg.weight multiplier needed -- pass
    weight=1.0 for this term in env_cfg.py).

    Params (via RewardTermCfg.params):
      sensor_name: a ContactSensorCfg with fields including "force",
        reduce="netforce" (one net wrench per primary body), NO
        history_length (see module docstring's bug writeup).
      body_names_by_category: {"critical": (...), "non_critical": (...)}.
        Every name the sensor resolves to (sensor.primary_names) MUST
        appear in exactly one category; raises loudly at construction if
        not (never silently skips or zero-weights an unclassified body).
      category_weights: e.g. {"critical": -0.02, "non_critical": -0.001}
        (arXiv:2512.01336 Table I, literal cited numbers).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        params = cfg.params
        sensor_name: str = params["sensor_name"]
        body_names_by_category: dict[str, tuple[str, ...]] = params["body_names_by_category"]
        category_weights: dict[str, float] = params["category_weights"]

        sensor: ContactSensor = env.scene[sensor_name]
        weights = []
        for name in sensor.primary_names:
            category = next((c for c, names in body_names_by_category.items() if name in names), None)
            if category is None:
                raise KeyError(
                    f"vulnerability_weighted_contact_force: body '{name}' (resolved by sensor "
                    f"'{sensor_name}') is not listed in any category "
                    f"{list(body_names_by_category)} -- every primary the sensor tracks must be "
                    "classified; add it to RobotProfile.vulnerability_regions or the sensor's "
                    "primary pattern is over-broad."
                )
            weights.append(category_weights[category])
        self._weight_per_primary = torch.tensor(weights, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        body_names_by_category: dict[str, tuple[str, ...]],
        category_weights: dict[str, float],
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        data = sensor.data
        assert data.force is not None, (
            f"vulnerability_weighted_contact_force: sensor '{sensor_name}' has no force field "
            "-- add 'force' to its ContactSensorCfg.fields."
        )
        force_mag = torch.norm(data.force, dim=-1)  # [B, N] -- instantaneous, this policy step
        # NOTE: weight_per_primary is already negative (a real reward
        # contribution, not a magnitude to be scaled by an outer weight) --
        # the caller's RewardTermCfg.weight for this term must be 1.0.
        penalty = (force_mag * self._weight_per_primary.unsqueeze(0)).sum(dim=-1)
        # Diagnostic only -- mjlab's own reward functions log this way (e.g.
        # soft_landing's Metrics/landing_force_mean); added specifically to
        # answer "is this term contributing anything, or is
        # root_linear_velocity drowning it out" without guessing from the
        # total reward alone.
        env.extras["log"]["Metrics/vuln_contact_force_raw_mean"] = force_mag.mean()
        env.extras["log"]["Metrics/vuln_contact_force_raw_max"] = force_mag.max()
        env.extras["log"]["Metrics/vuln_contact_force_penalty_mean"] = penalty.mean()
        return penalty


def root_linear_velocity_penalty(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize root linear speed (arXiv:2512.01336 Table I, weight
    -1.03125 -- applied by the caller's RewardTermCfg, not baked in here). A fall that
    ends with the robot still moving fast is a fall that hasn't been
    arrested; this rewards actually slowing down, not just softening
    contact."""
    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_link_lin_vel_w
    penalty = torch.sum(torch.square(lin_vel), dim=-1)
    env.extras["log"]["Metrics/root_lin_vel_sq_mean"] = penalty.mean()
    return penalty


def logged_action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Thin logging wrapper around mjlab.envs.mdp.action_rate_l2 -- can't
    add logging inside that function itself (site-packages, never edited,
    CLAUDE.md S12#18), so this calls it unchanged and logs its mean
    alongside the other three terms for the same before/after-total-reward
    comparison."""
    from mjlab.envs.mdp import action_rate_l2

    value = action_rate_l2(env)
    env.extras["log"]["Metrics/action_rate_l2_mean"] = value.mean()
    return value


def logged_joint_pos_limits(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Thin logging wrapper around mjlab.envs.mdp.joint_pos_limits -- same
    reasoning as logged_action_rate_l2 above."""
    from mjlab.envs.mdp import joint_pos_limits

    value = joint_pos_limits(env, asset_cfg=asset_cfg)
    env.extras["log"]["Metrics/joint_pos_limits_mean"] = value.mean()
    return value


def logged_joint_torques_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Thin logging wrapper around mjlab.envs.mdp.joint_torques_l2 (sum of
    squared actuator_force) -- same reasoning as logged_action_rate_l2."""
    from mjlab.envs.mdp import joint_torques_l2

    value = joint_torques_l2(env, asset_cfg=asset_cfg)
    env.extras["log"]["Metrics/joint_torques_l2_mean"] = value.mean()
    return value


class peak_contact_force_growth:
    """Round 8 reward term: per-step INCREASE in the episode's running peak
    ground-contact force (N) on any single body. Increments telescope, so an
    episode's raw sum is exactly its peak any-body force -- the Gate 5/7
    metric `peak_force_any_n` that round 7 failed (the per-Newton force term
    above charges non-critical bodies only -0.001/N per step, so one hard
    leg impact was nearly free). Same construction as space.py's
    fall_envelope_growth. Our own term, not literature-sourced.

    Like the rest of this file it sees the INSTANTANEOUS force at each policy
    step (50 Hz), not substep peaks (the history_length bug, module
    docstring); the evaluation metric is sampled the same way.

    Per-env state is updated out of place (inference-tensor bug class, see
    terminations.py / DEVIATIONS.md).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self._running_max = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._running_max = torch.zeros_like(self._running_max)
        else:
            running_max = self._running_max.clone()
            running_max[env_ids] = 0.0
            self._running_max = running_max

    def __call__(self, env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        assert sensor.data.force is not None, (
            f"peak_contact_force_growth: sensor '{sensor_name}' has no force field"
        )
        peak_now = torch.norm(sensor.data.force, dim=-1).amax(dim=-1)  # [B] hardest-hit body this step
        growth = (peak_now - self._running_max).clamp(min=0.0)
        self._running_max = torch.maximum(self._running_max, peak_now)
        if hasattr(env, "extras"):
            env.extras.setdefault("log", {})["Metrics/peak_contact_force_growth_mean"] = growth.mean()
        return growth
