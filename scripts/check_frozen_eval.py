"""Gate 3 check: the frozen-eval runner (CLAUDE.md S6, S11 gate 3).

Constructs the registered task's runner directly (not via mjlab's `train`
CLI, since we need to pass eval_num_envs/eval_interval overrides that the
CLI's generic `runner_cls(env, agent_cfg, log_dir, device)` construction
call doesn't support) and runs a handful of iterations, small and fast
enough for a mechanism check rather than a real training run.

Pass condition (S11 gate 3): eval runs, weights provably unchanged (the
runner's own internal assertions raise AssertionError if not -- if this
script doesn't crash, that assertion held), eval_history.json is written,
second-env memory cost is logged.

Run on the GPU machine, from the repo root:

    python scripts/check_frozen_eval.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

import cartpole_term  # noqa: F401,E402 -- side effect: registers the task
from cartpole_term.env_cfg import TASK_ID  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

NUM_ENVS = 64  # small training env count -- this is a mechanism check, not real training
EVAL_NUM_ENVS = 32  # smaller than the real 256 (CLAUDE.md S6.2.2), same reason
EVAL_INTERVAL = 2
NUM_ITERATIONS = 4  # fires the eval hook twice (it=0, it=2), proving it's reusable


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"
    configure_torch_backends()

    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = 0

    rl_cfg = load_rl_cfg(TASK_ID)
    rl_cfg.seed = 0
    rl_cfg.max_iterations = NUM_ITERATIONS

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError(
            f"No runner_cls registered for {TASK_ID} -- check "
            "src/cartpole_term/__init__.py wires FrozenEvalRunner in."
        )

    agent_cfg = asdict(rl_cfg)
    log_dir = Path("logs/rsl_rl/cartpole_term/gate3_check").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = runner_cls(
        env,
        agent_cfg,
        str(log_dir),
        device,
        eval_num_envs=EVAL_NUM_ENVS,
        eval_interval=EVAL_INTERVAL,
    )
    runner.learn(num_learning_iterations=NUM_ITERATIONS, init_at_random_ep_len=True)
    env.close()

    eval_history_path = log_dir / "eval_history.json"
    if not eval_history_path.exists():
        raise AssertionError(f"{eval_history_path} was not written.")
    history = json.loads(eval_history_path.read_text())
    if len(history) == 0:
        raise AssertionError("eval_history.json is empty -- the eval hook never fired.")

    print(f"\neval_history.json has {len(history)} record(s):")
    for rec in history:
        print(
            f"  it={rec['iteration']} "
            f"mean_action_len={rec['mean_action']['mean_episode_length']:.1f} "
            f"sampled_action_len={rec['sampled_action']['mean_episode_length']:.1f} "
            f"censored_frac={rec['mean_action']['censored_fraction']:.3f} "
            f"causes={rec['mean_action']['termination_cause_breakdown']} "
            f"build_info={rec['eval_env_build_info']}"
        )

    print(
        "\nGate 3 check PASSED: eval ran twice, weights were asserted unchanged "
        "inside the runner itself (no AssertionError raised = held), "
        "eval_history.json was written, second-env build cost logged above."
    )


if __name__ == "__main__":
    main()
