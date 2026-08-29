"""Gate 3 prep: dump the real RSL-RL/mjlab runner API surface needed to
implement src/cartpole_term/runner.py's frozen-eval hook (CLAUDE.md S6).
Read-only -- does not modify or train anything.

Run once, on the GPU machine, from the repo root:

    python scripts/inspect_runner.py

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
    import mjlab.scripts.train as train_script

    print_source(train_script, "FULL SOURCE: mjlab/scripts/train.py")

    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    print_source(MjlabOnPolicyRunner, "FULL SOURCE: mjlab.rl.MjlabOnPolicyRunner")
    print_source(RslRlVecEnvWrapper, "FULL SOURCE: mjlab.rl.RslRlVecEnvWrapper")

    section("MjlabOnPolicyRunner -- MRO (to find the real learn()/save()/load() owner)")
    print(MjlabOnPolicyRunner.__mro__)

    from rsl_rl.modules import EmpiricalNormalization

    section("SIGNATURE: rsl_rl.modules.EmpiricalNormalization.__init__")
    print(inspect.signature(EmpiricalNormalization.__init__))
    print_source(EmpiricalNormalization, "SOURCE: rsl_rl.modules.EmpiricalNormalization")

    try:
        from mjlab.tasks.registry import _TaskCfg

        section("SOURCE: mjlab.tasks.registry._TaskCfg")
        print(inspect.getsource(_TaskCfg))
    except Exception as e:  # noqa: BLE001 -- purely diagnostic, never fatal
        print(f"\n(_TaskCfg introspection skipped: {e})")


if __name__ == "__main__":
    main()
