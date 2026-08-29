"""Prep for Gate 4 (collect_frozen.py) and later video gates. Read-only.

1. mjlab.managers.termination_manager.TerminationManager's real source --
   need to know if it caches per-term, per-env results accessibly, since
   post-step joint queries are contaminated by auto-reset (confirmed in
   scripts/inspect_reset_behavior.py) and extras["log"] only gives
   aggregate per-step counts, not per-env attribution -- but
   collect_frozen.py needs per-TRANSITION termination_cause (CLAUDE.md S7.3).
2. mjlab.utils.wrappers.VideoRecorder -- needed for policy_final.mp4
   (CLAUDE.md S10 item 1).
3. How to reach the raw mujoco model/data from a ManagerBasedRlEnv, for
   scripts/record_video.py's manual mujoco.Renderer overlay path (S10 item 2).

Run once, on the GPU machine, from the repo root:

    python scripts/inspect_video_and_termmgr.py

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
    from mjlab.managers.termination_manager import TerminationManager

    print_source(TerminationManager, "FULL SOURCE: mjlab.managers.termination_manager.TerminationManager")

    from mjlab.utils.wrappers import VideoRecorder

    print_source(VideoRecorder, "FULL SOURCE: mjlab.utils.wrappers.VideoRecorder")

    section("mjlab.sim -- public names")
    import mjlab.sim as sim_mod

    print([n for n in dir(sim_mod) if not n.startswith("_")])

    from mjlab.envs import ManagerBasedRlEnv

    section("ManagerBasedRlEnv -- FULL source (small enough, and settles auto_reset/sim/render questions at once)")
    try:
        print(inspect.getsource(ManagerBasedRlEnv))
    except (OSError, TypeError) as e:
        print(f"(source unavailable: {e})")

    section("ManagerBasedRlEnvCfg -- FULL source (need: does it have auto_reset?)")
    from mjlab.envs import ManagerBasedRlEnvCfg

    try:
        print(inspect.getsource(ManagerBasedRlEnvCfg))
    except (OSError, TypeError) as e:
        print(f"(source unavailable: {e})")


if __name__ == "__main__":
    main()
