"""Gate 1 prep: dump the real mjlab API surface needed to implement
terminations.py, rewards.py, and env_cfg.py, so nothing in this repo has to
guess a signature (CLAUDE.md S0.2). Read-only -- does not modify anything.

Run once, on the GPU machine, from the repo root:

    python scripts/inspect_mjlab.py

Paste the full output back.
"""

from __future__ import annotations

import inspect


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_source(obj, label: str) -> None:
    section(label)
    try:
        print(inspect.getsource(obj))
    except (OSError, TypeError) as e:
        print(f"(source unavailable: {e})")


def main() -> None:
    import mjlab.tasks.cartpole.cartpole_env_cfg as cec

    print_source(cec, "FULL SOURCE: mjlab/tasks/cartpole/cartpole_env_cfg.py")
    section("FILE PATH")
    print(inspect.getfile(cec))

    from mjlab.tasks import registry

    section("SIGNATURE: register_mjlab_task")
    print(inspect.signature(registry.register_mjlab_task))
    print_source(registry.register_mjlab_task, "SOURCE: register_mjlab_task")

    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    for cls in (TerminationTermCfg, RewardTermCfg, SceneEntityCfg):
        print_source(cls, f"SOURCE: {cls.__module__}.{cls.__qualname__}")

    import mjlab.envs.mdp as mdp

    section("mjlab.envs.mdp -- public names")
    print([n for n in dir(mdp) if not n.startswith("_")])

    for name in ("time_out", "joint_pos_rel", "joint_vel_rel"):
        if hasattr(mdp, name):
            print_source(getattr(mdp, name), f"SOURCE: mjlab.envs.mdp.{name}")
        else:
            print(f"\nmjlab.envs.mdp has no attribute {name!r}")

    from mjlab.envs import ManagerBasedRlEnv

    section("SIGNATURE: ManagerBasedRlEnv.__init__")
    print(inspect.signature(ManagerBasedRlEnv.__init__))
    section("ManagerBasedRlEnv -- public attributes/methods")
    print([n for n in dir(ManagerBasedRlEnv) if not n.startswith("_")])

    section("SIGNATURE: ManagerBasedRlEnv.step (if present)")
    if hasattr(ManagerBasedRlEnv, "step"):
        print(inspect.signature(ManagerBasedRlEnv.step))


if __name__ == "__main__":
    main()
