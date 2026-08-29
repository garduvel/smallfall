"""One-time check, empirical rather than read-from-source: does
ManagerBasedRlEnv.step() auto-reset an env internally on the same call
where it terminates, or does the terminal state stay visible until an
explicit reset()?

This determines how scripts/check_terminations.py (and later, the labelling
pipeline) must read joint state to attribute *why* an env terminated. Guessing
wrong here would make cause-attribution silently meaningless instead of
loudly wrong, so it's checked by directly observing behaviour rather than by
reading source and assuming.

Also exercises, end to end: task registration, the custom terminations, and
env construction on the actual mjlab version installed -- so if any of
env_cfg.py / terminations.py / __init__.py have a wiring bug, it surfaces
here as a real error instead of downstream.

Run once, on the GPU machine, from the repo root:

    python scripts/inspect_reset_behavior.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

import cartpole_term  # noqa: F401,E402 -- side effect: registers the task
from cartpole_term.env_cfg import CART_POS_LIMIT_M, TASK_ID  # noqa: E402
from cartpole_term.terminations import cart_pos_out_of_bounds  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    section("SOURCE: ManagerBasedRlEnv.step")
    try:
        print(inspect.getsource(ManagerBasedRlEnv.step))
    except (OSError, TypeError) as e:
        print(f"(unavailable: {e})")

    section("SOURCE: ManagerBasedRlEnv.reset")
    try:
        print(inspect.getsource(ManagerBasedRlEnv.reset))
    except (OSError, TypeError) as e:
        print(f"(unavailable: {e})")

    section(f"Loading registered task: {TASK_ID}")
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = 8
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.reset()
    print("env constructed and reset OK")

    section("Instance attributes containing 'manager'")
    print([a for a in dir(env) if "manager" in a.lower()])

    cart_cfg = SceneEntityCfg("cartpole", joint_names=("slider",))

    section("Driving the cart at max force in one direction until cart_pos fires")
    found = False
    for step in range(200):
        actions = torch.ones(8, 1, device=device)  # max control effort, one direction
        _, _, terminated, truncated, extras = env.step(actions)
        if terminated.any():
            x = env.scene["cartpole"].data.joint_pos[:, cart_cfg.joint_ids].squeeze(-1)
            print(f"step {step}: terminated={terminated.tolist()}")
            print(f"cart x read from env.scene AFTER step() returns: {x.tolist()}")
            print(
                "cart_pos_out_of_bounds() evaluated on that post-step state: "
                f"{cart_pos_out_of_bounds(env, cart_cfg, CART_POS_LIMIT_M).tolist()}"
            )
            print(f"extras keys: {list(extras.keys())}")
            for k, v in extras.items():
                print(f"  extras[{k!r}] = {v!r}")
            found = True
            break
    if not found:
        raise AssertionError(
            "No termination within 200 steps at full force -- unexpected; "
            "cart_pos_out_of_bounds may not be wired correctly."
        )

    print("\nInterpretation guide:")
    print(
        "- If |x| for a terminated env is > "
        f"{CART_POS_LIMIT_M} (i.e. cart_pos_out_of_bounds is True there): "
        "no auto-reset inside step() -- terminal state is visible, post-step "
        "querying in check_terminations.py is safe as originally written."
    )
    print(
        "- If |x| is near 0 instead (i.e. cart_pos_out_of_bounds is False "
        "there despite terminated=True): step() auto-resets internally -- "
        "cause attribution must come from `extras` or a manager attribute, "
        "not from querying env.scene after step() returns."
    )


if __name__ == "__main__":
    main()
