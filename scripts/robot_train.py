"""PPO training with frozen in-training evaluation, generalized for any
robot in cartpole_term.common.robot_profile.PROFILES (Gate 1 of the
multi-robot inflicted-failure predictor + protective "small-fall" policy
plan -- phase-2, outside CLAUDE.md's cartpole-only spec, see
DEVIATIONS.md).

Generalizes scripts/go1_train.py via
cartpole_term.common.backend.get_robot_backend.

An earlier version of this file also ported scripts/train.py's
bracketed-window auto-densify + seed-perturbation retry mechanism
(competence-window-based auto-retrain). REMOVED: this design has no
competence-window consumer (see the plan's "REVISED" Gate 1 -- single
checkpoint, no windowed selection), and the mechanism actively misfired
in practice: Go1's episode length jumps from ~110 to ~900+ within the
very first --eval.interval 1000 gap, which reads as "every window from
target through fluent got skipped," triggering a full, unwanted retrain
at eval_interval=1 (evaluating every single iteration) immediately after
a real training run finished. See DEVIATIONS.md for the incident. Do not
re-add this mechanism to this file.

Push perturbation is NEVER touched here -- training always uses mjlab's
shipped, unmodified push_robot config for both robots (an explicit,
user-confirmed choice; the tuned/harsher push used for
collection/protective-training/the combined controller lives in Gate 2+
scripts only, never merged into this file's env cfg). Do not add a
push-override flag to this script.

Per Gate 1: do NOT artificially cap --max-iterations short (the earlier
exploratory 100-iteration Go1 run's own documented caveat --
locomotion quality was poor specifically because of that cap,
NEXT_STEPS_GO1.md). Left unset, mjlab's own shipped rl_cfg.max_iterations
is used as-is -- live-confirmed this session as 10,000 for Go1, 30,000
for G1 (a real difference, not assumed equal).

Run on the GPU machine, from the repo root, Go1 first per the plan's
rollout order:

    python scripts/robot_train.py --robot go1 --smoke
    python scripts/robot_train.py --robot go1 --seeds 0
    python scripts/robot_train.py --robot go1 --seeds 0 --early-stop-patience 5
    python scripts/robot_train.py --robot g1 --seeds 0   # only after Go1's pipeline is validated end to end
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from cartpole_term.cli import (  # noqa: E402
    SMOKE_EVAL_INTERVAL,
    SMOKE_EVAL_NUM_ENVS,
    SMOKE_MAX_ITERATIONS,
    SMOKE_NUM_ENVS,
)
from cartpole_term.common.backend import get_robot_backend  # noqa: E402
from cartpole_term.common.robot_profile import get_profile  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.utils.os import dump_yaml  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402


def train_one_seed(args: argparse.Namespace, seed: int) -> Path:
    device = "cuda"
    configure_torch_backends()

    profile = get_profile(args.robot)
    backend = get_robot_backend(args.robot)

    env_cfg = backend.env_cfg_factory(play=False)
    env_cfg.seed = seed
    env_cfg.scene.num_envs = SMOKE_NUM_ENVS if args.smoke else args.env_scene_num_envs

    rl_cfg = backend.rl_cfg_factory()
    rl_cfg.seed = seed
    if args.smoke:
        rl_cfg.max_iterations = SMOKE_MAX_ITERATIONS
    elif args.max_iterations is not None:
        rl_cfg.max_iterations = args.max_iterations
    # else: leave mjlab's own shipped max_iterations as-is (10,000 Go1 /
    # 30,000 G1, live-confirmed) -- Gate 1's "run to the real shipped
    # default, don't cap short" rule.

    eval_num_envs = SMOKE_EVAL_NUM_ENVS if args.smoke else args.eval_num_envs
    eval_interval = SMOKE_EVAL_INTERVAL if args.smoke else args.eval_interval
    rl_cfg.save_interval = eval_interval  # CLAUDE.md S6.2.1's reasoning, unchanged

    resuming = getattr(args, "resume_from", None) is not None
    if resuming:
        # Continue writing into the SAME run directory the checkpoint came
        # from, rather than starting a fresh timestamped one -- keeps
        # tensorboard/eval_history.json/resolved_args.json as one
        # continuous record of the whole training run, resumed segments
        # included.
        log_dir = Path(args.resume_from).resolve().parent
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_seed{seed}"
        log_dir = (Path("logs/rsl_rl") / rl_cfg.experiment_name / run_name).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
        dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))
        resolved = vars(args).copy()
        resolved["seed"] = seed
        (log_dir / "resolved_args.json").write_text(json.dumps(resolved, indent=2, default=str))

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(f"[robot-train] robot={args.robot} CUDA memory before learn(): {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    runner = backend.runner_cls(
        env,
        asdict(rl_cfg),
        str(log_dir),
        device,
        eval_num_envs=eval_num_envs,
        eval_interval=eval_interval,
        early_stop_patience=args.early_stop_patience,
    )

    if resuming:
        # load_cfg=None -> rsl_rl's PPO.load() loads actor+critic+optimizer
        # AND sets current_learning_iteration from the checkpoint's saved
        # "iter" field (confirmed via rsl_rl's real source,
        # leggedrobotics/rsl_rl -- OnPolicyRunner.load()/PPO.load()) --
        # a genuine training resume, not the actor-only load
        # build_runner_from_checkpoint uses for inference-only eval/collection.
        existing_history_path = log_dir / "eval_history.json"
        if existing_history_path.exists():
            # FrozenEvalRunner.__init__ always starts _eval_history empty;
            # without this, the first eval call during the resumed segment
            # would overwrite eval_history.json and silently lose every
            # record from before the resume.
            runner._eval_history = json.loads(existing_history_path.read_text())
        runner.load(args.resume_from, load_cfg=None, strict=True, map_location=device)
        remaining = rl_cfg.max_iterations - runner.current_learning_iteration
        print(
            f"[robot-train] robot={args.robot} resumed from {args.resume_from} at "
            f"iteration {runner.current_learning_iteration}, target {rl_cfg.max_iterations} "
            f"-> {remaining} more iterations."
        )
        if remaining <= 0:
            print(
                f"[robot-train] robot={args.robot}: checkpoint is already at or past the "
                f"target iteration ({runner.current_learning_iteration} >= {rl_cfg.max_iterations}) -- nothing to do."
            )
            env.close()
            return log_dir
        runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    else:
        runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        print(
            f"[robot-train] robot={args.robot} CUDA memory after learn(): "
            f"{torch.cuda.memory_allocated() / 1e9:.2f} GB "
            f"(peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)"
        )

    print(f"[robot-train] robot={args.robot} seed={seed} done. Logs in {log_dir}")
    return log_dir


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")

    parser = argparse.ArgumentParser(description="Train a robot's velocity-term task (phase-2, see DEVIATIONS.md).")
    parser.add_argument("--robot", required=True, choices=("go1", "g1"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--env.scene.num-envs", dest="env_scene_num_envs", type=int, default=4096)
    parser.add_argument("--eval.num-envs", dest="eval_num_envs", type=int, default=256)
    parser.add_argument("--eval.interval", dest="eval_interval", type=int, default=10)
    parser.add_argument(
        "--max-iterations", dest="max_iterations", type=int, default=None,
        help="TOTAL target iteration count (not an increment) -- also used with --resume-from to "
        "compute how many more iterations to run. Leave unset to run to mjlab's own shipped "
        "default (10,000 Go1 / 30,000 G1, live-confirmed) -- Gate 1's rule, don't cap short.",
    )
    parser.add_argument(
        "--resume-from", dest="resume_from", type=str, default=None,
        help="Path to a checkpoint (e.g. a model_5000.pt from an earlier --max-iterations 5000 "
        "run) to resume training FROM, continuing to --max-iterations total (actor+critic+"
        "optimizer+iteration count all restored -- a genuine PPO resume, not the actor-only "
        "load used elsewhere for inference). Continues writing into that checkpoint's existing "
        "run directory rather than starting a new one.",
    )
    parser.add_argument(
        "--early-stop-patience", dest="early_stop_patience", type=int, default=None,
        help="Stop once this many CONSECUTIVE frozen evals report the policy has plateaued.",
    )
    args = parser.parse_args()

    for seed in args.seeds:
        train_one_seed(args, seed)


if __name__ == "__main__":
    main()
