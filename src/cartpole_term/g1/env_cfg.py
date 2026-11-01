"""G1 humanoid velocity-flat task discovery + registration wiring (phase-2,
see DEVIATIONS.md's "multi-robot scaffolding" entry). Mirrors
cartpole_term.go1.env_cfg exactly, generalized via the shared
RobotProfile/discovery mechanism (cartpole_term.common) instead of a
second hand-copied regex -- this is the direct fix for the audit finding
that Go1's original discover_source_task_id used a naive, non-word-bounded
regex that would not have generalized safely to a second robot.

mjlab's shipped G1 flat-velocity task ships the same air_time reward term
at weight 0.0 as Go1 does (confirmed via GitHub source review this
session), so the same IsaacLab-sourced override Go1 already applies (see
cartpole_term.go1.env_cfg's GO1_AIR_TIME_WEIGHT comment and DEVIATIONS.md
for the full citation) is declared for G1 too, in
RobotProfile.reward_overrides -- not re-derived, not assumed identical
without checking: this file's own loud RuntimeError guard below re-confirms
the term still exists on the actually-installed mjlab version before ever
mutating it, and scripts/inspect_robot_task.py --robot g1 independently
confirms the shipped weight is still 0.0 before that override is applied
in a real training run.

G1's exact obs/action widths are NOT hardcoded here -- see
RobotProfile.expected_obs_dim/expected_action_dim in
cartpole_term/common/robot_profile.py, which start as None and are filled
in only after scripts/inspect_robot_task.py --robot g1 has run live on the
GPU machine (CLAUDE.md S0.2: never invent an API/value that hasn't been
confirmed against the installed version).
"""

from __future__ import annotations

TASK_ID = "Mjlab-G1-Velocity-Term"


def discover_source_task_id() -> str:
    """G1's real mjlab-shipped source task id, found via the shared
    multi-robot discovery helper (never a guessed literal)."""
    from cartpole_term.common.discovery import discover_task_id
    from cartpole_term.common.robot_profile import G1

    return discover_task_id(G1)


def g1_velocity_term_env_cfg(play: bool = False):
    """mjlab's shipped G1 flat-velocity env cfg, with the reward-weight
    overrides declared in RobotProfile.reward_overrides applied -- see
    module docstring. Everything else (dynamics, terminations, all other
    reward terms/weights, observations, commands, domain randomization
    incl. the push_robot event) is exactly as mjlab ships it, same rule
    Go1's own env cfg follows.

    play=True is not implemented, same unresolved gap as Go1's own
    play_env_cfg (see cartpole_term.go1.env_cfg's docstring for why) --
    raising loudly here rather than silently substituting the training
    cfg. Only play=False is used anywhere in the training/frozen-eval/
    predictor pipeline.
    """
    from cartpole_term.common.robot_profile import G1
    from mjlab.tasks.registry import load_env_cfg

    if play:
        raise NotImplementedError(
            "g1_velocity_term_env_cfg(play=True) not implemented yet -- "
            "mjlab's play-cfg loader API for this task wasn't confirmed. "
            "Run scripts/inspect_robot_task.py --robot g1 and wire this up "
            "before attempting G1 video recording. See DEVIATIONS.md."
        )
    cfg = load_env_cfg(discover_source_task_id())
    for reward_name, weight in G1.reward_overrides.items():
        if reward_name not in cfg.rewards:
            raise RuntimeError(
                f"Expected a {reward_name!r} reward term on the discovered G1 task "
                f"but found none (keys: {list(cfg.rewards.keys())}) -- mjlab's G1 "
                "reward config has changed since this override was written. Fix "
                "RobotProfile.reward_overrides in cartpole_term/common/robot_profile.py "
                "or drop the override; do not silently skip it. See DEVIATIONS.md."
            )
        cfg.rewards[reward_name].weight = weight
    return cfg
