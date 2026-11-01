"""PPO training for Mjlab-Go1-Velocity-Term, with the same frozen in-training
evaluation mechanism as cartpole (CLAUDE.md S6, reused via
cartpole_term.runner.FrozenEvalRunner / cartpole_term.go1.runner.
Go1FrozenEvalRunner -- see DEVIATIONS.md for why this is phase-2, outside
CLAUDE.md's cartpole-only spec).

Unlike scripts/train.py, there is no reward/termination override wiring in
THIS file -- mjlab's shipped Go1 flat-velocity task's termination and reward
are used almost entirely unmodified, with exactly one weight override
(air_time, 0.0 -> 0.25, IsaacLab's own value for the equivalent term --
see cartpole_term/go1/env_cfg.py's GO1_AIR_TIME_WEIGHT and DEVIATIONS.md)
applied inside go1_velocity_term_env_cfg() itself, not here. This script's
only job is: build the env, attach Go1FrozenEvalRunner, run learn(), same
eval-every-save_interval cadence as cartpole so every saved checkpoint has
a measurement.

This is the FIRST Go1 training run -- Go1's own struggling/target/competent/
fluent operating-point windows (the CLAUDE.md S7.1 analogue) don't exist yet
and are deliberately NOT guessed here; scripts/go1_select_checkpoints.py
picks them by inspecting this run's actual eval_history.json curve, same
"measured, not guessed" principle as the rest of this project.

Run on the GPU machine, from the repo root:

    python scripts/go1_train.py --smoke                          # tiny end-to-end check
    python scripts/go1_train.py --seeds 0
    python scripts/go1_train.py --seeds 0 --env.scene.num-envs 2048   # if memory-bound
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

import cartpole_term.go1  # noqa: F401,E402 -- side effect: registers the task
from cartpole_term.cli import (  # noqa: E402
    SMOKE_EVAL_INTERVAL,
    SMOKE_EVAL_NUM_ENVS,
    SMOKE_MAX_ITERATIONS,
    SMOKE_NUM_ENVS,
)
from cartpole_term.go1.env_cfg import TASK_ID  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.os import dump_yaml  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402


def train_one_seed(args: argparse.Namespace, seed: int) -> Path:
    device = "cuda"
    configure_torch_backends()

    env_cfg = load_env_cfg(TASK_ID)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = SMOKE_NUM_ENVS if args.smoke else args.env_scene_num_envs

    rl_cfg = load_rl_cfg(TASK_ID)
    rl_cfg.seed = seed
    if args.smoke:
        rl_cfg.max_iterations = SMOKE_MAX_ITERATIONS
    elif args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations

    eval_num_envs = SMOKE_EVAL_NUM_ENVS if args.smoke else args.eval_num_envs
    eval_interval = SMOKE_EVAL_INTERVAL if args.smoke else args.eval_interval
    rl_cfg.save_interval = eval_interval  # CLAUDE.md S6.2.1's reasoning, same as cartpole

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_seed{seed}"
    log_dir = (Path("logs/rsl_rl") / rl_cfg.experiment_name / run_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))
    resolved = vars(args).copy()
    resolved["seed"] = seed
    (log_dir / "resolved_args.json").write_text(
        __import__("json").dumps(resolved, indent=2, default=str)
    )

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError(f"No runner_cls registered for {TASK_ID} -- check src/cartpole_term/go1/__init__.py.")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[go1-train] CUDA memory before learn(): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    runner = runner_cls(
        env,
        asdict(rl_cfg),
        str(log_dir),
        device,
        eval_num_envs=eval_num_envs,
        eval_interval=eval_interval,
        early_stop_patience=args.early_stop_patience,
    )
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[go1-train] CUDA memory after learn(): {torch.cuda.memory_allocated() / 1e9:.2f} GB "
              f"(peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)")

    print(f"[go1-train] seed={seed} done. Logs in {log_dir}")
    return log_dir


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4, same requirement for Go1).")

    parser = argparse.ArgumentParser(description="Train Mjlab-Go1-Velocity-Term (phase-2, see DEVIATIONS.md).")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--env.scene.num-envs", dest="env_scene_num_envs", type=int, default=4096)
    parser.add_argument("--eval.num-envs", dest="eval_num_envs", type=int, default=256)
    parser.add_argument("--eval.interval", dest="eval_interval", type=int, default=10)
    parser.add_argument("--max-iterations", dest="max_iterations", type=int, default=None)
    parser.add_argument("--early-stop-patience", dest="early_stop_patience", type=int, default=None)
    args = parser.parse_args()

    for seed in args.seeds:
        train_one_seed(args, seed)


if __name__ == "__main__":
    main()
