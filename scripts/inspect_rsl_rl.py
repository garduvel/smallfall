"""Gate 3 prep, part 2: dump rsl_rl.runners.on_policy_runner.OnPolicyRunner's
real source (CLAUDE.md S6). MjlabOnPolicyRunner subclasses this without
overriding learn(), so this is where the actual training loop -- and the
correct insertion point for a periodic frozen-eval hook -- lives.
Read-only.

Run once, on the GPU machine, from the repo root:

    python scripts/inspect_rsl_rl.py

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
    from rsl_rl.runners.on_policy_runner import OnPolicyRunner

    print_source(OnPolicyRunner, "FULL SOURCE: rsl_rl.runners.on_policy_runner.OnPolicyRunner")

    section("OnPolicyRunner -- public attributes/methods")
    print([a for a in dir(OnPolicyRunner) if not a.startswith("_")])

    try:
        from rsl_rl.algorithms.ppo import PPO

        print_source(PPO, "FULL SOURCE: rsl_rl.algorithms.ppo.PPO")
    except Exception as e:  # noqa: BLE001 -- diagnostic only
        print(f"\n(PPO import/getsource failed: {e})")


if __name__ == "__main__":
    main()
