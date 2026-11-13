"""The fall-inducing push, for any robot in RobotProfile: mjlab's
apply_body_impulse on the profile's push body, layered on top of the
task's own mild push_robot event.

Moved here from scripts/go1_wild_push_predictor.py (2026-11-13) so the push
sweep, the dataset collection and the videos share one definition for every
robot. For the Go1 the result is identical to the old factory: same event
name, same params, body "trunk".

The event must be added to the cfg before ManagerBasedRlEnv is built:
apply_body_impulse is a stateful class instantiated at env-build time, and
mjlab's EventManager deep-copies its cfg then (DEVIATIONS.md).
"""

from __future__ import annotations

DEFAULT_IMPULSE_TORQUE_N = 0.0  # pure linear force at the body, no added spin
DEFAULT_IMPULSE_DURATION_S = (0.1, 0.3)
DEFAULT_IMPULSE_COOLDOWN_S = (1.0, 3.0)  # matches push_robot's shipped interval_range_s
EVENT_NAME = "wild_impulse"


def add_impulse_push(cfg, push_body: str, force_n: float, torque_n: float = DEFAULT_IMPULSE_TORQUE_N,
                     duration_s=DEFAULT_IMPULSE_DURATION_S, cooldown_s=DEFAULT_IMPULSE_COOLDOWN_S):
    """Adds the impulse event to an env cfg in place and returns the cfg."""
    from mjlab.envs.mdp.events import apply_body_impulse
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    cfg.events[EVENT_NAME] = EventTermCfg(
        func=apply_body_impulse,
        mode="step",
        params={
            "force_range": (-force_n, force_n),
            "torque_range": (-torque_n, torque_n),
            "duration_s": duration_s,
            "cooldown_s": cooldown_s,
            "asset_cfg": SceneEntityCfg("robot", body_names=(push_body,)),
        },
    )
    return cfg


def impulse_env_cfg_factory(robot: str, force_n: float, torque_n: float = DEFAULT_IMPULSE_TORQUE_N,
                            duration_s=DEFAULT_IMPULSE_DURATION_S, cooldown_s=DEFAULT_IMPULSE_COOLDOWN_S):
    """An env_cfg_factory(play=False) for `robot` with the impulse push added."""
    from cartpole_term.common.backend import get_robot_backend
    from cartpole_term.common.robot_profile import get_profile

    profile = get_profile(robot)
    if profile.push_body is None:
        raise ValueError(f"RobotProfile {robot!r} has no push_body set")
    backend = get_robot_backend(robot)

    def factory(play: bool = False):
        cfg = backend.env_cfg_factory(play=play)
        return add_impulse_push(cfg, profile.push_body, force_n, torque_n, duration_s, cooldown_s)

    return factory
