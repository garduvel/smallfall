"""Protective "small-fall" task for Go1 (Gate 5 of the multi-robot plan --
the original design plan). A new, short, fixed-length mjlab
task whose reward makes a falling robot land compact and low-damage,
instead of tumbling uncontrolled -- trained with the same mjlab+RSL-RL PPO
stack as every other policy in this project (CLAUDE.md S12#1, extended to
this phase-2 task: no new algorithm, no custom rollout loop).

Built ON TOP of Go1's own already-registered velocity-walking task cfg
(`cartpole_term.go1.env_cfg.go1_velocity_term_env_cfg`) -- same scene,
robot, action space (12-dim joint position targets), and actor observation
LAYOUT (48-dim: base_lin_vel+base_ang_vel+projected_gravity+joint_pos+
joint_vel+actions+command), so a trained protective policy shares its
observation contract with the nominal walking policy exactly, with zero
remapping needed if a future Gate 6 hands control between the two mid-
rollout. Only the parts about SURVIVING A FALL WELL are new; everything
about velocity TRACKING and the mild everyday push_robot event is removed.

**Design pivot from the plan's own stated Gate 5 approach, logged fully in
DEVIATIONS.md:** the plan chose a "rollout-then-handoff" initial-state
mechanism specifically because literal state injection was believed to
need "an unconfirmed mjlab 'set full state directly' API". That API
(`Entity.write_root_link_pose_to_sim` / `write_root_link_velocity_to_sim` /
`write_joint_state_to_sim`) is real and confirmed this session by reading
the actual mjlab source (mujocolab/mjlab@main,
src/mjlab/entity/entity.py) -- so this task uses literal state injection
instead (see events.py), which is simpler, needs no custom PPO training
loop, and is MORE faithful to SafeFall's own real design (its Stage II
literally inits episodes from predictor-flagged unsafe states) than
rollout-then-handoff ever was.

Grounding for the concrete choices below, confirmed against real mjlab
source this session (not guessed):

- `ContactSensorCfg`/`ContactMatch` (src/mjlab/sensor/contact_sensor.py):
  `mode="body"` primaries matched directly against the SAME body names
  already live-confirmed in `RobotProfile.GO1.vulnerability_regions`
  (common/robot_profile.py) -- no geom-name-suffix guessing needed.
  `secondary=ContactMatch(mode="body", pattern="terrain")` isolates
  ground-impact force from self-collision, the identical pattern Go1's
  OWN shipped `feet_ground_cfg` already uses for its 4-foot sensor.
- Go1's FLAT variant (the one this whole project trains on,
  `Mjlab-Velocity-Flat-Unitree-Go1`) explicitly REMOVES every per-body-
  group ground-contact sensor the ROUGH variant ships (confirmed by
  reading `unitree_go1_flat_env_cfg`'s own `remove_sensors` set in
  mjlab/tasks/velocity/config/go1/env_cfgs.py) -- this task adds its OWN
  full-body sensor back; it does not exist on the base task.
- `history_length` is deliberately LEFT AT ITS DEFAULT (disabled) on this
  sensor -- setting it (needed for SafeFall's own peak/average-over-
  substeps blend) hit a confirmed real bug on the first GPU run:
  mjlab's own `ContactSensor.reset()` does an in-place write on the
  history buffer that becomes a permanently-unwritable PyTorch "inference
  tensor" under this project's runner's construction pattern
  (`RuntimeError: Inplace update to inference tensor outside
  InferenceMode is not allowed`). No other sensor in this project has
  ever set history_length>0, so this code path had never been exercised
  before. Not fixable here (mjlab is site-packages); the reward instead
  uses the instantaneous per-step force (see rewards.py). See
  DEVIATIONS.md.
- Episode length -- **revised, see DEVIATIONS.md.** The original design
  fixed this at 10 steps (0.2s), time-scaled from SafeFall's 40-step/200Hz
  protective episode. That number is now understood to have been a
  misreading of this project's OWN already-collected evidence, not a
  correct cross-check: `outputs/go1_flagged_states_seed0.npz`'s
  `post_fall_obs` shows mean base angular speed under the UNCONTROLLED
  nominal policy *still elevated at step 28* of a 30-step window -- i.e.
  NOT concentrated in the first 10 steps at all. A fixed 10-step episode
  was ending training rollouts (and, visibly, the demo video) mid-fall,
  before the robot had actually come to rest -- exactly the "terminates
  while still in the falling motion" symptom reported after watching the
  video. `PROTECTIVE_EPISODE_LENGTH_S` is now a generous safety-cap
  BUDGET, not the expected episode length -- most episodes now end early
  via the new `settled` termination (terminations.py) once the robot
  genuinely stops moving. Settling is a PASS CRITERION measured on the
  deterministic policy in frozen eval (share of eval episodes that never
  settle = mean-action `censored_fraction`), not a training reward: under
  PPO's exploration noise the robot never holds still, so a settling bonus
  can't fire during training (round 4, see DEVIATIONS.md). See
  terminations.py's module docstring for the literature grounding
  (FRASA, arXiv:2410.08655) of the sustained-duration settling check.
"""

from __future__ import annotations

TASK_ID = "Mjlab-Go1-Protective-Fall"
DEFAULT_STATE_BANK_FILE = "outputs/go1_fall_state_bank_seed0.npz"
# Per-robot task ids. The Go1's is unchanged; the G1's was added for the
# humanoid port (DEVIATIONS.md, 2026-11-13).
TASK_IDS = {"go1": TASK_ID, "g1": "Mjlab-G1-Protective-Fall"}
DEFAULT_STATE_BANK_FILES = {"go1": DEFAULT_STATE_BANK_FILE, "g1": "outputs/g1_flag_state_bank_seed5.npz"}
# Safety-cap BUDGET, not the expected episode length -- see module
# docstring. 3.0s = 150 steps @ Go1's confirmed 50Hz gives ample headroom
# past step 28, where our own post_fall_obs data showed angular speed
# still elevated under the old 10-step cutoff. Most episodes should end
# well before this via the `settled` termination below.
PROTECTIVE_EPISODE_LENGTH_S = 3.0
# G1: the Go1's 3 s scaled by sqrt(start height) (0.76 m pelvis vs 0.278 m
# trunk, mjlab's init_state), since fall time grows with sqrt(height):
# 3.0 x 1.65 = 4.96 -> 5.0 s. Still a safety cap; `settled` ends episodes.
PROTECTIVE_EPISODE_LENGTH_S_BY_ROBOT = {"go1": PROTECTIVE_EPISODE_LENGTH_S, "g1": 5.0}

# --- Settling (see terminations.py's module docstring for the FRASA
# (arXiv:2410.08655) sustained-duration design this reuses). The hold
# duration (SETTLED_HOLD_S) is FRASA's own literal number (0.5s, its
# stand-up completion criterion's hold duration); the two velocity
# thresholds are NOT from that paper (it checks state-vector position
# error, not speed, and gives no velocity number at all) -- this
# project's own calibration, log as such in DEVIATIONS.md. 0.2 m/s / 0.5
# rad/s are common "quasi-static / at rest" cutoffs in legged-robot
# control, not tuned against this task's own data yet.
LINEAR_SETTLED_THRESHOLD_MPS = 0.2
ANGULAR_SETTLED_THRESHOLD_RAD_S = 0.5
SETTLED_HOLD_S = 0.5
# Round 5: sized from round 4's measured raw joint_torques_l2 (~200, rising)
# so it contributes ~-0.1/step, on par with action_rate. arXiv:2512.01336's
# "large torque magnitude" weight (-2.9e-3) would give ~-0.58/step and become
# the largest term, and its raw quantity isn't confirmed to match mjlab's
# sum-of-squared-actuator-force, so it is not copied.
JOINT_TORQUES_WEIGHT = -5e-4
# Round 7: penalize ground covered by the fall (space.py's
# fall_envelope_growth; the raw episode sum = envelope growth in metres).
# mjlab's Episode_Reward = weight x mean raw value per step over the 150-step
# episode. Round 6 measured envelope ~0.77 m under training noise; with the
# robot's own ~0.3 m extent unpenalized, growth ~0.47 m -> mean ~0.0031/step,
# so -30 gives ~-0.1, on par with contact force (-0.16). Our own term and
# calibration, not literature-sourced. Sprawl is NOT penalized: round 6 showed
# it constant at ~0.22 m (the robot's own size).
FALL_ENVELOPE_WEIGHT = -30.0
# Round 8: penalize the episode's peak any-body contact force (rewards.py's
# peak_contact_force_growth; the raw episode sum = that peak in Newtons).
# Round 7 failed Gates 5 and 7 on exactly this metric (median ~1100-1200 N,
# not below the walker). -0.02 is the same per-Newton value as the critical-
# body force weight, applied once to the peak instead of every step: a 1100 N
# peak costs 1100 * 0.02 * dt(0.02) = -0.44 of return, the same order as the
# envelope term (-0.28) and the per-step contact term. Our own calibration.
PEAK_CONTACT_FORCE_WEIGHT = -0.02

# THIRD ROUND -- see DEVIATIONS.md for rounds 1 and 2 (the SafeFall-tier-
# multiplier design this replaced, and the empirical rebalancing that kept
# finding new dominant terms without fixing the underlying flat reward).
# A direct fetch of arXiv:2512.01336's real Table I surfaced a citation
# error (the "1000/1/0.5" tier multiplier belongs to SafeFall, not this
# paper) AND a complete, better-grounded weight set for several terms at
# once. Contact force is now arXiv:2512.01336's own real 2-category split
# -- critical body vs. all other bodies -- with these as the FINAL,
# already-scaled per-Newton weights (no separate outer multiplier; the
# contact-force RewardTermCfg.weight in this file is 1.0, a pass-through).
CONTACT_FORCE_WEIGHTS = {"critical": -0.02, "non_critical": -0.001}
# arXiv:2512.01336 Table I, reused as-is (adapted, not independently
# calibrated for Go1's own mass/units -- see DEVIATIONS.md).
ROOT_LINEAR_VELOCITY_WEIGHT = -1.03125
# arXiv:2512.01336 Table I's own real number for this exact term --
# replaces two rounds of guessing (Go1's borrowed -0.1, then an empirically
# -reduced -0.02) that both left this term dominant: cutting the weight
# without knowing the paper's real value let PPO's entropy bonus inflate
# the RAW action-rate magnitude (measured action std 0.70 -> 1.85 between
# those two attempts) by roughly the same factor the weight shrank,
# leaving its contribution to the total reward unchanged either way. If
# -0.00667 still doesn't shrink the raw magnitude appropriately, that
# would point at entropy_coef (an actual PPO hyperparameter, not touched
# without the user's go-ahead) rather than this weight.
ACTION_RATE_WEIGHT = -0.00667
# Round 4: raised from Go1's borrowed walking weight (-1.0). arXiv:2512.01336's
# own out-of-limit DoF-position weight (-29) is by far the largest in its
# Table I (~28x its root-linear-velocity weight), i.e. joint-limit
# avoidance is a primary driver of their "controlled" falls; at -1.0 ours
# contributed only ~-0.3/step vs root_linear_velocity's ~-4.2. -29 is still
# NOT copied (raw-quantity definition not confirmed comparable); instead
# sized from our own round-3 measurement (raw joint_pos_limits_mean ~0.33
# late in training): -10 * 0.33 ~= -3.3/step, comparable to, not dominating,
# root_linear_velocity. See DEVIATIONS.md.
JOINT_POS_LIMITS_WEIGHT = -10.0

# The Go1 reward weights by term name. The G1 starts from these and re-sizes
# the four Go1-calibrated terms (CALIBRATED_TERMS) by the Go1's own rule, see
# protective/calibration.py; overrides come in through `weight_overrides`.
GO1_REWARD_WEIGHTS = {
    "root_linear_velocity": ROOT_LINEAR_VELOCITY_WEIGHT,
    "action_rate_l2": ACTION_RATE_WEIGHT,
    "joint_pos_limits": JOINT_POS_LIMITS_WEIGHT,
    "joint_torques_l2": JOINT_TORQUES_WEIGHT,
    "fall_envelope_growth": FALL_ENVELOPE_WEIGHT,
    "peak_contact_force_growth": PEAK_CONTACT_FORCE_WEIGHT,
}


def protective_contact_sensor_cfg(robot: str = "go1"):
    """Per-body robot-vs-terrain contact sensor over every vulnerability
    body. Shared by the protective task and the Gate 6 hand-off env
    (scripts/go1_handoff_eval.py) so both measure force identically."""
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    from cartpole_term.common.robot_profile import get_profile

    profile = get_profile(robot)
    all_vulnerability_bodies = tuple(
        name for names in profile.vulnerability_regions.values() for name in names
    )
    return ContactSensorCfg(
        name="protective_body_ground_contact",
        primary=ContactMatch(mode="body", pattern=all_vulnerability_bodies, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        # history_length deliberately omitted (default: disabled) -- see
        # module docstring's bug writeup / DEVIATIONS.md.
    )


def go1_protective_fall_env_cfg(
    play: bool = False,
    state_bank_file: str = DEFAULT_STATE_BANK_FILE,
    peak_force_weight: float = PEAK_CONTACT_FORCE_WEIGHT,
):
    """The Go1 protective task, unchanged (see protective_fall_env_cfg)."""
    return protective_fall_env_cfg(
        "go1", play=play, state_bank_file=state_bank_file, peak_force_weight=peak_force_weight
    )


def protective_fall_env_cfg(
    robot: str = "go1",
    play: bool = False,
    state_bank_file: str | None = None,
    peak_force_weight: float | None = None,
    weight_overrides: dict[str, float] | None = None,
    fall_size_measure: str = "envelope_growth_m",
    fall_size_weight: float | None = None,
    log_fall_sizes: bool = False,
):
    """Protective-fall task cfg for `robot`, built on its velocity task.
    For the Go1 with no overrides it is exactly the round-8 task.

    Reward ablation (DEVIATIONS.md 2026-11-13): `fall_size_measure` picks the
    fall-size reward term (protective/fall_size.py). The default,
    envelope_growth_m, keeps the original fall_envelope_growth term
    untouched; any other measure replaces it with a `fall_size` term
    (fall_size_reward.py) whose weight is `fall_size_weight` (default: the
    fall_envelope_growth weight). `log_fall_sizes` adds every measure as a
    log-only Episode_Metrics/fallsize_<measure> (episode sum), used to size
    the weights.

    `weight_overrides`: {reward term name: weight}, applied over
    GO1_REWARD_WEIGHTS (the G1's calibrated weights come in here).
    `peak_force_weight`, if given, wins over both for that one term.

    Original Go1 notes:  `state_bank_file` is baked into the
    reset event's params at cfg-construction time (see events.py) --
    callers that need a non-default bank build their own env_cfg_factory
    closure over this function rather than relying on package-level
    registration (mirrors scripts/go1_wild_push_predictor.py's
    build_impulse_env_cfg_factory pattern for the same reason: a fixed
    module-level `register_mjlab_task` call can't take a per-run CLI arg).

    `peak_force_weight`: weight of round 8's peak-impact term. 0.0 drops the
    term, which is exactly the round-7 reward (inference never reads rewards,
    so any checkpoint loads under either setting).
    """
    from cartpole_term.common.backend import get_robot_backend
    from cartpole_term.common.robot_profile import get_profile

    profile = get_profile(robot)
    if state_bank_file is None:
        state_bank_file = DEFAULT_STATE_BANK_FILES[robot]
    weights = dict(GO1_REWARD_WEIGHTS)
    unknown = set(weight_overrides or {}) - set(weights)
    if unknown:
        raise ValueError(f"unknown reward terms in weight_overrides: {sorted(unknown)}")
    weights.update(weight_overrides or {})
    if peak_force_weight is not None:
        weights["peak_contact_force_growth"] = peak_force_weight
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from cartpole_term.protective.events import inject_fall_state
    from cartpole_term.protective.rewards import (
        logged_action_rate_l2,
        logged_joint_pos_limits,
        logged_joint_torques_l2,
        peak_contact_force_growth,
        root_linear_velocity_penalty,
        vulnerability_weighted_contact_force,
    )
    from cartpole_term.protective.space import body_spread, distance_from_fall_start, fall_envelope_growth
    from cartpole_term.protective.terminations import settled

    cfg = get_robot_backend(robot).env_cfg_factory(play=play)

    cfg.episode_length_s = PROTECTIVE_EPISODE_LENGTH_S_BY_ROBOT[robot]

    # No ongoing perturbation needed -- this task starts ALREADY mid-fall
    # (via the reset event below), it doesn't need an additional push.
    cfg.events.pop("push_robot", None)

    # Replace the default standing-pose reset with literal fall-state
    # injection (see events.py's module docstring for the full pivot
    # rationale). Overwrites whatever reset events the base cfg shipped
    # (reset_base/reset_robot_joints) -- this task's whole premise is that
    # the robot never starts in the default standing pose.
    cfg.events["reset_fall_state"] = EventTermCfg(
        func=inject_fall_state,
        mode="reset",
        params={"state_bank_path": state_bank_file},
    )
    for stale_reset in ("reset_base", "reset_robot_joints"):
        cfg.events.pop(stale_reset, None)

    # A fall that's already underway shouldn't end the episode early on
    # orientation grounds -- only `time_out` (kept, from the base cfg, now
    # a safety cap only) and the new `settled` term below decide when a
    # protective episode ends.
    cfg.terminations.pop("fell_over", None)

    # New: end the episode once the robot has genuinely come to rest
    # (sustained-duration velocity check, not the old fixed step count) --
    # see terminations.py's module docstring and DEVIATIONS.md.
    cfg.terminations["settled"] = TerminationTermCfg(
        func=settled,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "linear_threshold_mps": LINEAR_SETTLED_THRESHOLD_MPS,
            "angular_threshold_rad_s": ANGULAR_SETTLED_THRESHOLD_RAD_S,
            "hold_s": SETTLED_HOLD_S,
        },
    )

    protective_contact_cfg = protective_contact_sensor_cfg(robot)
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (protective_contact_cfg,)

    # Replace ALL velocity-tracking / gait rewards -- irrelevant here, the
    # robot isn't trying to walk. Regularization terms (action_rate,
    # joint_pos_limits) are reused directly from mjlab's generic
    # envs.mdp module at Go1's OWN already-shipped weights, not
    # reimplemented (see rewards.py's module docstring) -- wrapped with a
    # thin logging pass-through (logged_action_rate_l2/
    # logged_joint_pos_limits) so all four terms' raw, per-step means show
    # up under Metrics/* during training, for directly answering "which
    # term dominates" instead of guessing from the summed reward.
    # arXiv:2512.01336's own 2-category split (critical vs. all) -- collapse
    # RobotProfile's 3-tier vulnerability_regions (high/medium/low) into
    # {"critical": high, "non_critical": medium+low}. "high" (trunk) is the
    # one region that maps onto the paper's own "critical body" framing;
    # medium+low (hips/thighs/calves) become the paper's "all other bodies".
    contact_categories = {
        "critical": profile.vulnerability_regions["high"],
        "non_critical": profile.vulnerability_regions["medium"] + profile.vulnerability_regions["low"],
    }
    cfg.rewards = {
        "vulnerability_contact_force": RewardTermCfg(
            func=vulnerability_weighted_contact_force,
            weight=1.0,  # pass-through -- category_weights below are already the final per-Newton weights
            params={
                "sensor_name": protective_contact_cfg.name,
                "body_names_by_category": contact_categories,
                "category_weights": CONTACT_FORCE_WEIGHTS,
            },
        ),
        "root_linear_velocity": RewardTermCfg(
            func=root_linear_velocity_penalty,
            weight=weights["root_linear_velocity"],
            params={},
        ),
        "action_rate_l2": RewardTermCfg(func=logged_action_rate_l2, weight=weights["action_rate_l2"], params={}),
        "joint_pos_limits": RewardTermCfg(func=logged_joint_pos_limits, weight=weights["joint_pos_limits"], params={}),
        "joint_torques_l2": RewardTermCfg(func=logged_joint_torques_l2, weight=weights["joint_torques_l2"], params={}),
        "fall_envelope_growth": RewardTermCfg(func=fall_envelope_growth, weight=weights["fall_envelope_growth"], params={}),
    }
    if fall_size_measure != "envelope_growth_m" or log_fall_sizes:
        from cartpole_term.protective.fall_size import SIZE_METRICS
        from cartpole_term.protective.fall_size_reward import fall_size_increment

        if fall_size_measure not in SIZE_METRICS:
            raise ValueError(f"unknown fall_size_measure {fall_size_measure!r}; one of {SIZE_METRICS}")
        size_params = lambda m: {"measure": m, "sensor_name": protective_contact_cfg.name}  # noqa: E731
    if fall_size_measure != "envelope_growth_m":
        del cfg.rewards["fall_envelope_growth"]
        w = weights["fall_envelope_growth"] if fall_size_weight is None else fall_size_weight
        cfg.rewards["fall_size"] = RewardTermCfg(func=fall_size_increment, weight=w, params=size_params(fall_size_measure))
    if weights["peak_contact_force_growth"] != 0.0:
        cfg.rewards["peak_contact_force_growth"] = RewardTermCfg(
            func=peak_contact_force_growth,
            weight=weights["peak_contact_force_growth"],
            params={"sensor_name": protective_contact_cfg.name},
        )

    # Log-only "space" metrics (no reward effect). See space.py.
    cfg.metrics = dict(cfg.metrics or {})
    cfg.metrics.update({
        "fall_envelope_radius": MetricsTermCfg(
            func=distance_from_fall_start, params={"of": "bodies"}, reduce="max"
        ),
        "root_travel_at_end": MetricsTermCfg(
            func=distance_from_fall_start, params={"of": "root"}, reduce="last"
        ),
        "body_spread_at_end": MetricsTermCfg(func=body_spread, params={}, reduce="last"),
        "body_spread_mean": MetricsTermCfg(func=body_spread, params={}, reduce="mean"),
    })
    if log_fall_sizes:
        cfg.metrics.update({f"fallsize_{m}": MetricsTermCfg(func=fall_size_increment, params=size_params(m), reduce="sum")
                            for m in SIZE_METRICS})

    return cfg
