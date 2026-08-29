"""Gate 1 check: a short scripted rollout with random actions should
terminate well before the 1000-step time-out cap, with both `pole_angle` and
`cart_pos` failure causes firing (CLAUDE.md S11 gate 1).

Cause attribution uses extras["log"]["Episode_Termination/<name>"], not
joint_pos read after step() returns: scripts/inspect_reset_behavior.py
confirmed ManagerBasedRlEnv.step() auto-resets terminated envs in place by
default, so joint_pos read post-step is already the *next* episode's state,
not the failure state. The termination manager's own per-step log dict is
used instead -- it's exactly the CLAUDE.md S3.2 cause-breakdown mechanism,
already wired up by mjlab.

Run on the GPU machine, from the repo root:

    python scripts/check_terminations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

import cartpole_term  # noqa: F401,E402 -- side effect: registers the task
from cartpole_term.env_cfg import TASK_ID  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

NUM_ENVS = 256
MAX_STEPS = 1000  # the time-out cap itself; gate 1 needs failures well before this
ACTION_DIM = 1  # single actuated joint: slider
CAUSES = ("pole_angle", "cart_pos", "time_out")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = NUM_ENVS

    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()

    totals = {c: 0.0 for c in CAUSES}
    first_step = {c: None for c in CAUSES}
    total_terminated_steps = 0
    total_truncated_steps = 0

    for step in range(MAX_STEPS):
        actions = torch.empty(NUM_ENVS, ACTION_DIM, device=device).uniform_(-1.0, 1.0)
        _, _, terminated, truncated, extras = env.step(actions)

        log = extras.get("log", {})
        for cause in CAUSES:
            val = float(log.get(f"Episode_Termination/{cause}", 0.0))
            totals[cause] += val
            if val > 0 and first_step[cause] is None:
                first_step[cause] = step

        total_terminated_steps += int(terminated.sum())
        total_truncated_steps += int(truncated.sum())

    print(f"Ran {MAX_STEPS} steps x {NUM_ENVS} envs, iid uniform random actions in [-1, 1].")
    print(f"Total failure terminations (terminated=True, summed over steps): {total_terminated_steps}")
    print(f"Total time-outs (truncated=True, summed over steps): {total_truncated_steps}")
    print("Termination-cause breakdown (extras['log'], summed over steps):")
    for cause in CAUSES:
        suffix = f" (first at step {first_step[cause]})" if first_step[cause] is not None else " (never)"
        print(f"  {cause}: {totals[cause]:.0f}{suffix}")

    if total_terminated_steps == 0:
        raise AssertionError(
            f"No failure terminations occurred in {MAX_STEPS} steps x {NUM_ENVS} "
            "envs under random actions -- expected frequent early failure, not "
            "survival (CLAUDE.md S11 gate 1)."
        )
    if totals["pole_angle"] == 0:
        raise AssertionError("pole_angle termination never fired.")
    if totals["cart_pos"] == 0:
        raise AssertionError(
            "cart_pos termination never fired under iid random actions across "
            f"{NUM_ENVS} envs x {MAX_STEPS} steps. An inverted pendulum is "
            "unstable at its upright equilibrium, so open-loop random actions "
            "may simply tip the pole (pole_angle) faster than the cart can "
            "drift to the rail -- a real dynamics finding, not necessarily a "
            "bug. Report this back rather than loosening the check."
        )

    print("\nGate 1 check PASSED: both termination causes fired, well before the 1000-step cap.")


if __name__ == "__main__":
    main()
