"""Demo-only command-distribution overrides for the multi-robot phase-2
pipeline (see DEVIATIONS.md's "multi-robot scaffolding" entry).

`force_forward_only_commands` is applied ONLY in video/visual-check
scripts (scripts/robot_watch_episodes.py, and later
scripts/robot_record_demo_video.py) -- NEVER in training, push-tuning,
frozen-dataset collection, or predictor training. Those must keep the
task's real, shipped command distribution unchanged (same reason
scripts/go1_watch_episodes.py already pins spawn yaw for its own demo env
only, never for training/collection/eval).

Why this is safe and not a guess: mjlab's UniformVelocityCommand
(src/mjlab/tasks/velocity/mdp/velocity_command.py, read directly this
session) already ships a purpose-built mechanism for exactly this --
`rel_forward_envs` ("Fraction of environments that receive forward-only
commands (positive lin_vel_x, zero lin_vel_y and ang_vel_z). ... important
for stair climbing"). Its resampling code applies the forward override
LAST, after heading/standing/world-frame flags are set, so `rel_forward_envs
=1.0` alone is not quite enough on its own: `_update_command` (called every
step, not just at resample time) still re-derives yaw from heading error
for any env flagged `is_heading_env`, and zeroes the whole command for any
env flagged `is_standing_env` -- both would silently undo the forward-only
override for a fraction of envs on later steps if left at their shipped
values. Setting `rel_heading_envs=0.0` and `rel_standing_envs=0.0`
alongside `rel_forward_envs=1.0` closes both gaps. `rel_world_envs`
defaults to 0.0 and isn't overridden by either shipped robot config, so it
isn't touched here.
"""

from __future__ import annotations

COMMAND_TERM_NAME = "twist"  # confirmed live via inspect_robot_task.py for both robots
# The command_vel curriculum term (mdp.commands_vel, velocity_env_cfg.py)
# re-asserts cfg.ranges.lin_vel_x/lin_vel_y/ang_vel_z from
# env.common_step_counter on ITS OWN schedule (every step, confirmed via
# reading curriculums.py's real source), silently overwriting any manual
# range override set beforehand. A fresh demo env sits at step 0, so it
# keeps re-imposing the shipped stage-0 range (-1.0,1.0) no matter what a
# caller sets ranges.lin_vel_x to first. Must be disabled, not fought
# after the fact, for a --commanded-speed pin to actually stick.
CURRICULUM_TERM_NAME = "command_vel"

# Absolute highest forward speed the shipped command_vel curriculum ever
# asks for (velocity_env_cfg.py's final velocity_stages entry: {"step":
# 10000*24, "lin_vel_x": (-2.0, 3.0)}), read directly this session -- not
# invented. A FRESH demo env always restarts that curriculum from its
# step-0 stage (lin_vel_x=(-1.0,1.0)) regardless of which checkpoint's
# weights are loaded, since the curriculum tracks the env's own live
# step counter, not something the checkpoint carries -- so reaching this
# speed in a demo requires pinning ranges.lin_vel_x directly, not just
# relying on natural resampling. The policy has had very little actual
# training time at this exact speed (only from iteration 10,000 onward),
# so struggling or falling at this pinned speed is a real, informative
# result, not a bug.
MAX_FORWARD_COMMAND_LIN_VEL_X = 3.0


def force_forward_only_commands(cfg, fixed_speed: float | None = None) -> None:
    """Mutates `cfg.commands["twist"]` in place so every env gets a
    positive-only, zero-lateral, zero-yaw velocity command every episode.
    Raises loudly if the command term isn't named "twist" -- do not
    silently no-op if mjlab's naming changes.

    fixed_speed: if given, ALSO pins `ranges.lin_vel_x` to exactly this
    value (a degenerate (fixed_speed, fixed_speed) range), so every
    forward command samples the same speed instead of the usual random
    draw within the current range, AND disables the command_vel
    curriculum term (see CURRICULUM_TERM_NAME's comment above) so it
    can't silently overwrite that pin on the next step. Composes cleanly
    with the rel_forward_envs mechanism below: UniformVelocityCommand's
    forward-env override does `vel_command_b[fwd_ids, 0] =
    vel_command_b[fwd_ids, 0].abs().clamp(min=0.3)`, which is a no-op on an
    already-fixed positive value. None (default) leaves the natural
    forward-only sampled range AND the curriculum untouched.
    """
    if COMMAND_TERM_NAME not in cfg.commands:
        raise RuntimeError(
            f"Expected a {COMMAND_TERM_NAME!r} command term on this env cfg but found none "
            f"(keys: {list(cfg.commands.keys())}) -- mjlab's velocity-command wiring has changed "
            "since this override was written. Fix COMMAND_TERM_NAME in "
            "cartpole_term/common/commands.py or drop the override; do not silently skip it."
        )
    twist_cfg = cfg.commands[COMMAND_TERM_NAME]
    twist_cfg.rel_forward_envs = 1.0
    twist_cfg.rel_heading_envs = 0.0
    twist_cfg.rel_standing_envs = 0.0
    if fixed_speed is not None:
        twist_cfg.ranges.lin_vel_x = (fixed_speed, fixed_speed)
        removed = cfg.curriculum.pop(CURRICULUM_TERM_NAME, None)
        if removed is None:
            print(
                f"[commands] WARNING: no {CURRICULUM_TERM_NAME!r} curriculum term found to disable "
                f"(keys: {list(cfg.curriculum.keys())}) -- if mjlab's naming changed, the fixed_speed "
                "pin may still get silently overwritten by whatever curriculum term now controls "
                "ranges.lin_vel_x. Check CURRICULUM_TERM_NAME in cartpole_term/common/commands.py."
            )
