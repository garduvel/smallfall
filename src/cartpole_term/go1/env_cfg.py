"""Go1 velocity-flat task discovery + registration wiring (phase-2, outside
CLAUDE.md's cartpole-only spec -- see DEVIATIONS.md).

Unlike cartpole, mjlab's shipped Go1 flat-velocity task already ships its own
fall termination (bad_orientation, ~70 degrees, confirmed via
scripts/inspect_go1_task.py) and a tracking reward -- there is no cartpole-
style custom terminations.py/rewards.py to write here. This module's only
job is registration wiring: find mjlab's actual Go1 flat-velocity task id and
reuse its env cfg UNMODIFIED (never editing site-packages, same rule as
CLAUDE.md S3.3/S12#18), then register a second task id pointed at the same
cfg plus cartpole_term.go1.runner.Go1FrozenEvalRunner for the frozen-eval
hook (CLAUDE.md S6, generalized in cartpole_term/runner.py).

The source task id is discovered by pattern match against
mjlab.tasks.registry.list_tasks(), never hardcoded as a guessed literal
(CLAUDE.md S0.2: never invent an API/name that hasn't been confirmed against
the installed version). Run scripts/inspect_go1_task.py first on the GPU
machine if this raises -- it performs this same search read-only and prints
every registered task id so the pattern can be corrected here if mjlab's
actual naming differs from what was assumed from public docs/source review.
"""

from __future__ import annotations

TASK_ID = "Mjlab-Go1-Velocity-Term"


def discover_source_task_id() -> str:
    """mjlab's shipped Go1 flat-velocity task id, found by pattern rather
    than assumed. Thin wrapper around the shared multi-robot discovery
    helper (cartpole_term.common.discovery), generalized this session for
    the G1 humanoid extension (see DEVIATIONS.md's "multi-robot
    scaffolding" entry) -- kept as a named function here, unchanged from
    every existing call site's point of view (go1/rl_cfg.py,
    go1/runner.py), rather than forcing callers to import RobotProfile
    directly. Must resolve to the exact same result as before this edit
    (Mjlab-Velocity-Flat-Unitree-Go1) -- verified by re-running
    scripts/inspect_go1_task.py as a regression check."""
    from cartpole_term.common.discovery import discover_task_id
    from cartpole_term.common.robot_profile import GO1

    return discover_task_id(GO1)


# mjlab's own Go1 flat-velocity config ships `air_time` (mdp.feet_air_time --
# rewards a foot for spending time in the air between contacts, i.e. a
# proper alternating gait rhythm) at weight 0.0 -- present but inactive
# (confirmed by reading src/mjlab/tasks/velocity/config/go1/env_cfgs.py on
# GitHub, mujocolab/mjlab@main). This is the ONE deliberate reward deviation
# from mjlab's shipped Go1 config in this whole project (see DEVIATIONS.md):
# the value 0.25 is NOT invented -- it's IsaacLab's own weight for the
# equivalent term on the equivalent flat Go1 velocity task, cited by mjlab's
# own maintainers in mujocolab/mjlab GitHub Discussion #220 (a direct
# mjlab-vs-IsaacLab comparison for this exact task). Motivation: without any
# reward for gait rhythm, a policy that shuffles/slides its feet sideways
# can satisfy the velocity-tracking reward just as well as a proper trot,
# and that is what was observed after 1000 iterations at weight 0.0.
GO1_AIR_TIME_WEIGHT = 0.25


def go1_velocity_term_env_cfg(play: bool = False):
    """mjlab's shipped Go1 flat-velocity env cfg, with one reward-weight
    override (GO1_AIR_TIME_WEIGHT, see above) -- everything else (dynamics,
    terminations, all other reward terms/weights, observations, commands,
    domain randomization) is exactly as mjlab ships it.

    play=True is not yet wired to a distinct play cfg (mjlab's registry API
    for retrieving a task's *play* cfg specifically, as opposed to its
    training cfg, wasn't confirmed by scripts/inspect_go1_task.py at the time
    this was written -- see that script's "mjlab.tasks.registry -- public
    names" dump). Only play=False is used anywhere in the training/frozen-
    eval/predictor pipeline (CLAUDE.md S6.2.2's rule -- never use the play
    cfg for frozen eval -- applies here too), so this gap only matters once
    Go1 video recording is implemented; raising loudly here rather than
    silently substituting the training cfg for a real play cfg.
    """
    from mjlab.tasks.registry import load_env_cfg

    if play:
        raise NotImplementedError(
            "go1_velocity_term_env_cfg(play=True) not implemented yet -- "
            "mjlab's play-cfg loader API for a non-cartpole task wasn't "
            "confirmed. Check outputs/go1_task_inspection.json's "
            "'mjlab.tasks.registry -- public names' section and wire this up "
            "before attempting Go1 video recording. See DEVIATIONS.md."
        )
    cfg = load_env_cfg(discover_source_task_id())
    if "air_time" not in cfg.rewards:
        raise RuntimeError(
            "Expected an 'air_time' reward term on the discovered Go1 task but "
            f"found none (keys: {list(cfg.rewards.keys())}) -- mjlab's Go1 "
            "reward config has changed since this override was written. Fix "
            "GO1_AIR_TIME_WEIGHT's target key or drop the override; do not "
            "silently skip it. See DEVIATIONS.md."
        )
    cfg.rewards["air_time"].weight = GO1_AIR_TIME_WEIGHT
    return cfg
