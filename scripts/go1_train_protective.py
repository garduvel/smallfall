"""PPO training for Mjlab-Go1-Protective-Fall (Gate 5, see DEVIATIONS.md).
Adapted directly from scripts/go1_train.py's already-validated structure --
same frozen-eval mechanism (CLAUDE.md S6, via ProtectiveFrozenEvalRunner),
same "no custom rollout loop, no PPO hyperparameter tuning" discipline.
This IS an ordinary from-scratch PPO training script -- no checkpoint is
loaded, and no custom step-by-step control-switching logic is needed, because
the realistic "already falling" start state comes entirely from the task's
own reset event (literal state injection from a pre-collected fall-state
bank, see cartpole_term/protective/events.py) rather than from anything
this script has to orchestrate itself.

Prerequisite: outputs/go1_fall_state_bank_seed0.npz must exist --
produced by `python scripts/go1_collect_flagged_states.py --capture-state-bank
...` (already run for the lead-time measurement; re-run with this flag
added, or point --state-bank-file at wherever it landed).

Run on the GPU machine, from the repo root:

    python scripts/go1_train_protective.py --smoke
    python scripts/go1_train_protective.py --seeds 0
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

# Must precede `import torch`: mjlab's @torch.jit.script math utils crash on
# this Colab image (missing libnvrtc-builtins.so.13.0). See DEVIATIONS.md.
os.environ.setdefault("PYTORCH_JIT", "0")
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

import cartpole_term.protective  # noqa: F401,E402 -- side effect: registers the task
from cartpole_term.cli import (  # noqa: E402
    SMOKE_EVAL_INTERVAL,
    SMOKE_EVAL_NUM_ENVS,
    SMOKE_MAX_ITERATIONS,
    SMOKE_NUM_ENVS,
)
from cartpole_term.protective.env_cfg import TASK_ID  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.os import dump_yaml  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402


def train_one_seed(args: argparse.Namespace, seed: int) -> Path:
    device = "cuda"
    configure_torch_backends()

    if not Path(args.state_bank_file).exists():
        raise RuntimeError(
            f"Fall-state bank not found: {args.state_bank_file}. Produce it with "
            "`python scripts/go1_collect_flagged_states.py --capture-state-bank "
            "--checkpoint-file checkpoints/model_9999.pt` (or point --state-bank-file "
            "at wherever it landed) before training the protective policy -- this task "
            "has no meaningful reset behavior without it."
        )

    from cartpole_term.protective.env_cfg import PEAK_CONTACT_FORCE_WEIGHT, go1_protective_fall_env_cfg

    peak_force_weight = PEAK_CONTACT_FORCE_WEIGHT if args.peak_force_weight is None else args.peak_force_weight

    def env_cfg_factory(play: bool = False):
        # One factory for the training env AND the frozen-eval env, so the
        # eval measures the policy on the same bank and reward it trains on.
        return go1_protective_fall_env_cfg(
            play=play, state_bank_file=args.state_bank_file, peak_force_weight=peak_force_weight
        )

    env_cfg = env_cfg_factory(play=False)
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
    rl_cfg.save_interval = eval_interval

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_seed{seed}"
    log_dir = (Path("logs/rsl_rl") / rl_cfg.experiment_name / run_name).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))
    resolved = vars(args).copy()
    resolved["seed"] = seed
    resolved["peak_force_weight"] = peak_force_weight  # the value actually used (CLI None -> env_cfg default)
    (log_dir / "resolved_args.json").write_text(
        __import__("json").dumps(resolved, indent=2, default=str)
    )

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID)
    if runner_cls is None:
        raise RuntimeError(f"No runner_cls registered for {TASK_ID} -- check src/cartpole_term/protective/__init__.py.")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[protective-train] CUDA memory before learn(): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    runner = runner_cls(
        env,
        asdict(rl_cfg),
        str(log_dir),
        device,
        eval_num_envs=eval_num_envs,
        eval_interval=eval_interval,
        early_stop_patience=args.early_stop_patience,
        env_cfg_factory=env_cfg_factory,
    )
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[protective-train] CUDA memory after learn(): {torch.cuda.memory_allocated() / 1e9:.2f} GB "
              f"(peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)")

    print(f"[protective-train] seed={seed} done. Logs in {log_dir}")
    return log_dir


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")

    parser = argparse.ArgumentParser(description="Train Mjlab-Go1-Protective-Fall (Gate 5, see DEVIATIONS.md).")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--state-bank-file", dest="state_bank_file", type=str, default="outputs/go1_fall_state_bank_seed0.npz")
    parser.add_argument(
        "--peak-force-weight", dest="peak_force_weight", type=float, default=None,
        help="weight of the round-8 peak-impact reward term (default: env_cfg.PEAK_CONTACT_FORCE_WEIGHT). "
        "0 reproduces the round-7 reward.",
    )
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
