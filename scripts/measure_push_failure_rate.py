"""Gate 2 of the multi-robot inflicted-failure predictor + protective
"small-fall" policy plan (phase-2, see DEVIATIONS.md; plan
summarized there).

Sweeps a grid of candidate push-perturbation configs against ONE real,
fully-trained checkpoint (Gate 1's output -- no competence sweep, see the
plan's "REVISED" Gate 1-4 entries) and measures the resulting
`terminated`-fraction per config, stochastic actions throughout (matching
this project's established "label under the behavior policy" rule --
scripts/go1_collect_frozen.py's collect_one does the same). Prints every
swept point and does NOT auto-pick a winner -- same "let the user see the
tradeoff" posture as scripts/go1_horizon_sweep.py, because the right
severity here is a judgment call informed by Gate 4's downstream
AUPRC-vs-lead-time results, not something to decide from failure rate
alone (see the plan's Gate 2 "new consideration" note: a push strong
enough to fail the policy almost instantly could still give a
high-AUPRC/near-zero-lead-time predictor, which is not actually useful for
Gate 5/6's handoff).

This config is used ONLY at collection/protective-training/combined-
controller time (Gate 3+) -- NEVER merged into robot_train.py's env cfg.
Training always uses mjlab's shipped, unmodified push_robot config; this
script's whole job is finding what to escalate it to for failure-inducing
data collection, per the user's explicit choice to keep training and
collection dynamics separate here (a deliberate, logged departure from
CLAUDE.md S6.2.2's usual "never let eval/collection dynamics differ from
training" rule elsewhere in this project).

Escalation path, in order, before inventing anything new (both mechanisms
already exist in mjlab/envs/mdp/events.py, confirmed via GitHub source
this session -- not new code):
  1. `velocity_range` scaling on the shipped push_robot event
     (mdp.push_by_setting_velocity) -- an instantaneous, mass-independent
     velocity kick, fully tunable via plain config, no code change needed.
  2. `apply_body_impulse` -- a stateful, torque-capable transient force,
     if velocity kicks alone can't clear the target band even at a sane
     harsh setting.

Run on the GPU machine, from the repo root, after scripts/robot_train.py:

    python scripts/measure_push_failure_rate.py --robot go1 \\
        --checkpoint-file logs/rsl_rl/go1_velocity_term/<run>/model_9999.pt
    python scripts/measure_push_failure_rate.py --robot go1 --checkpoint-file <path> --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch  # noqa: E402

from cartpole_term.cli import SMOKE_NUM_ENVS  # noqa: E402
from cartpole_term.common.backend import get_robot_backend  # noqa: E402
from cartpole_term.common.robot_profile import get_profile  # noqa: E402

TARGET_BAND = (0.10, 0.40)  # invented, unsourced -- see module docstring + DEVIATIONS.md
SAFETY_BAND = (0.02, 0.90)  # if EVERY swept config falls outside this, escalate/widen the grid, don't silently keep one
SMOKE_EPISODES = 32
DEFAULT_EPISODES = 500

# Multiplicative scale factors applied to the shipped push_robot
# velocity_range (every axis scaled together, direction/shape preserved).
# 1.0 = mjlab's stock config (already active during Gate 1 training).
DEFAULT_VELOCITY_SCALES = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]

# apply_body_impulse escalation grid, used only if every velocity_range
# scale above fails to clear TARGET_BAND[0]. Units: mjlab's own (N for
# force -- see mjlab.envs.mdp.events.apply_body_impulse's real signature,
# confirmed via GitHub source, before wiring this up for real).
DEFAULT_IMPULSE_FORCE_GRID = [50.0, 100.0, 200.0, 400.0]


def scaled_velocity_range(base_velocity_range: dict[str, tuple[float, float]], scale: float) -> dict[str, tuple[float, float]]:
    return {axis: (lo * scale, hi * scale) for axis, (lo, hi) in base_velocity_range.items()}


def run_one_config(
    checkpoint_path: str,
    profile,
    backend,
    seed: int,
    episodes: int,
    device: str,
    mechanism: str,
    config_params: dict,
) -> dict:
    from cartpole_term.runner import build_runner_from_checkpoint

    # CRITICAL: mjlab's EventManager does `self.cfg = deepcopy(cfg)` at
    # __init__ time (confirmed via mjlab/managers/event_manager.py's real
    # source) -- any push_robot param override MUST be applied to the env
    # cfg BEFORE ManagerBasedRlEnv construction, inside the factory
    # itself. Mutating an already-built env's `.cfg.events[...]`
    # afterward (the previous version of this function) has NO EFFECT on
    # the live simulation -- confirmed the hard way: it produced
    # bit-for-bit identical failure rates across four different "scales".
    if mechanism == "velocity":
        def env_cfg_factory(play: bool = False):
            cfg = backend.env_cfg_factory(play=play)
            cfg.events["push_robot"].params["velocity_range"] = config_params["velocity_range"]
            return cfg
    elif mechanism == "body_impulse":
        raise NotImplementedError(
            "mechanism='body_impulse' escalation not wired up yet -- velocity_range scaling "
            f"(tried up to {DEFAULT_VELOCITY_SCALES[-1]}x) didn't clear the target band. "
            "Confirm mjlab.envs.mdp.apply_body_impulse's real signature/params live before wiring "
            "this in (see module docstring) rather than guessing its call shape. See DEVIATIONS.md."
        )
    else:
        raise ValueError(f"Unknown mechanism {mechanism!r}")

    runner = build_runner_from_checkpoint(
        checkpoint_path,
        seed=seed,
        eval_num_envs=SMOKE_NUM_ENVS if episodes <= SMOKE_EPISODES else min(episodes, 1024),
        device=device,
        runner_cls=backend.runner_cls,
        env_cfg_factory=env_cfg_factory,
        task_id=backend.task_id,
        extra_scalar_metric_fn=None,
    )
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    env = runner._build_eval_env()
    raw_env = env.unwrapped

    num_envs = raw_env.scene.num_envs
    env.seed(seed)
    obs, _ = env.reset()
    obs = obs.to(device)

    terminated_count = 0
    truncated_count = 0
    episode_count = 0
    t0 = time.time()
    with torch.inference_mode():
        while episode_count < episodes:
            actions = runner.alg.actor(obs, stochastic_output=True)
            next_obs, _rewards, dones, extras = env.step(actions)
            dones_cpu = dones.bool().cpu()
            time_outs = extras.get("time_outs")
            truncated_cpu = time_outs.bool().cpu() if time_outs is not None else torch.zeros_like(dones_cpu)
            terminated_cpu = dones_cpu & ~truncated_cpu

            terminated_count += int(terminated_cpu.sum().item())
            truncated_count += int(truncated_cpu.sum().item())
            episode_count += int(dones_cpu.sum().item())

            obs = next_obs.to(device)
    wall_time = time.time() - t0

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during push measurement"

    env.close()

    total_episodes = terminated_count + truncated_count
    failure_rate = terminated_count / total_episodes if total_episodes > 0 else 0.0
    return {
        "mechanism": mechanism,
        "config_params": config_params,
        "num_envs": num_envs,
        "episodes_measured": total_episodes,
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "failure_rate": failure_rate,
        "wall_time_s": wall_time,
    }


# Go1 reference for the impulse sweep: falls per 1,000 robot-steps in the five
# 150 N predictor collections of the multi-seed run (4096 robots x 262 steps
# from reset each; 1.03-1.09, mean 1.07). Measured the same way below.
GO1_150N_FALLS_PER_1000_STEPS = 1.07
IMPULSE_SWEEP_STEPS = 262
IMPULSE_SWEEP_NUM_ENVS = 4096


def run_impulse_config(checkpoint_path: str, robot: str, backend, force_n: float, seed: int,
                       num_envs: int, steps: int, device: str) -> dict:
    """Falls per 1,000 robot-steps under the impulse push at `force_n`, over
    `steps` steps from reset with `num_envs` robots (fallen robots reset and
    keep counting), i.e. the same exposure as the predictor collection."""
    from cartpole_term.common.push import impulse_env_cfg_factory
    from cartpole_term.runner import build_runner_from_checkpoint

    runner = build_runner_from_checkpoint(
        checkpoint_path, seed=seed, eval_num_envs=num_envs, device=device,
        runner_cls=backend.runner_cls, env_cfg_factory=impulse_env_cfg_factory(robot, force_n),
        task_id=backend.task_id, extra_scalar_metric_fn=None,
    )
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()
    env = runner._build_eval_env()
    t0 = time.time()
    falls = 0
    with torch.inference_mode():
        env.seed(seed)
        obs, _ = env.reset()
        obs = obs.to(device)
        for _ in range(steps):
            actions = runner.alg.actor(obs, stochastic_output=True)
            obs, _r, dones, extras = env.step(actions)
            obs = obs.to(device)
            time_outs = extras.get("time_outs")
            truncated = time_outs.bool() if time_outs is not None else torch.zeros_like(dones.bool())
            falls += int((dones.bool() & ~truncated).sum().item())
    for name, before in actor_snapshot.items():
        assert torch.equal(before, runner.alg.actor.state_dict()[name]), f"actor.{name} changed"
    env.close()
    robot_steps = num_envs * steps
    return {"force_n": force_n, "num_envs": num_envs, "steps": steps, "falls": falls,
            "robot_steps": robot_steps, "falls_per_1000_steps": 1000 * falls / robot_steps,
            "wall_time_s": time.time() - t0}


def impulse_sweep(args, profile, backend, device: str) -> None:
    forces = args.impulse_forces or [profile.impulse_force_n]
    rows = []
    for f in forces:
        print(f"[push-tuning] robot={args.robot} impulse +-{f} N on {profile.push_body}: measuring...")
        row = run_impulse_config(args.checkpoint_file, args.robot, backend, f, args.seed,
                                 args.num_envs, args.steps, device)
        rows.append(row)
        print(f"[push-tuning]   {f:.0f} N: {row['falls']} falls in {row['robot_steps']:,} robot-steps = "
              f"{row['falls_per_1000_steps']:.2f} per 1,000 (Go1 at 150 N: {GO1_150N_FALLS_PER_1000_STEPS})")
    chosen = next((r for r in rows if r["force_n"] == profile.impulse_force_n), None)
    verdict = None
    if chosen is not None:
        ratio = chosen["falls_per_1000_steps"] / GO1_150N_FALLS_PER_1000_STEPS
        verdict = {"force_n": profile.impulse_force_n, "ratio_to_go1": ratio, "within_quarter_to_4x": 0.25 <= ratio <= 4.0}
        print(f"[push-tuning] chosen force {profile.impulse_force_n} N: {ratio:.2f}x the Go1's fall rate "
              f"-> {'within' if verdict['within_quarter_to_4x'] else 'OUTSIDE'} the pre-set 0.25x-4x check (DEVIATIONS.md)")
    out = Path("outputs") / f"{profile.name}_impulse_sweep_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"robot": profile.name, "push_body": profile.push_body,
                               "checkpoint_path": args.checkpoint_file,
                               "go1_150n_falls_per_1000_steps": GO1_150N_FALLS_PER_1000_STEPS,
                               "results": rows, "chosen_force_check": verdict}, indent=2))
    print(f"[push-tuning] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", required=True, choices=("go1", "g1"))
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--velocity-scales", dest="velocity_scales", type=float, nargs="+", default=DEFAULT_VELOCITY_SCALES,
        help="Multiplicative scale factors applied to mjlab's shipped push_robot velocity_range.",
    )
    parser.add_argument("--mechanism", choices=("velocity", "impulse"), default="velocity",
                        help="'impulse': falls per 1,000 robot-steps under the dataset push at each --impulse-forces.")
    parser.add_argument("--impulse-forces", dest="impulse_forces", type=float, nargs="+", default=None,
                        help="impulse only, N per axis; default the robot's RobotProfile.impulse_force_n")
    parser.add_argument("--num-envs", dest="num_envs", type=int, default=IMPULSE_SWEEP_NUM_ENVS, help="impulse only")
    parser.add_argument("--steps", type=int, default=IMPULSE_SWEEP_STEPS, help="impulse only")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4, same requirement for this phase-2 work).")
    device = "cuda"

    episodes = SMOKE_EPISODES if args.smoke else (args.episodes or DEFAULT_EPISODES)

    profile = get_profile(args.robot)
    backend = get_robot_backend(args.robot)

    if args.mechanism == "impulse":
        if args.smoke:
            args.num_envs, args.steps = SMOKE_NUM_ENVS, 20
        impulse_sweep(args, profile, backend, device)
        return

    base_env_cfg = backend.env_cfg_factory(play=False)
    base_velocity_range = dict(base_env_cfg.events["push_robot"].params["velocity_range"])
    print(f"[push-tuning] robot={args.robot} shipped push_robot.velocity_range: {base_velocity_range}")

    results: list[dict] = []
    for scale in args.velocity_scales:
        config_params = {"velocity_range": scaled_velocity_range(base_velocity_range, scale)}
        print(f"[push-tuning] robot={args.robot} velocity_range scale={scale}x: measuring ({episodes} episodes)...")
        row = run_one_config(
            args.checkpoint_file, profile, backend, args.seed, episodes, device,
            mechanism="velocity", config_params=config_params,
        )
        row["scale"] = scale
        results.append(row)
        print(
            f"[push-tuning] scale={scale}x: failure_rate={row['failure_rate']:.3f} "
            f"({row['terminated_count']}/{row['episodes_measured']} terminated), "
            f"wall_time={row['wall_time_s']:.1f}s"
        )

    in_target_band = [r for r in results if TARGET_BAND[0] <= r["failure_rate"] <= TARGET_BAND[1]]
    in_safety_band = [r for r in results if SAFETY_BAND[0] <= r["failure_rate"] <= SAFETY_BAND[1]]
    if not in_safety_band:
        raise RuntimeError(
            f"Every swept velocity_range scale landed outside the safety band {SAFETY_BAND} -- "
            f"results: {[(r['scale'], r['failure_rate']) for r in results]}. Widen "
            "--velocity-scales or escalate to apply_body_impulse (see module docstring) -- "
            "do not silently keep a degenerate config. See DEVIATIONS.md."
        )
    if not in_target_band:
        print(
            f"[push-tuning] WARNING: no swept scale landed inside the target band {TARGET_BAND}, "
            f"though {len(in_safety_band)} landed inside the wider safety band {SAFETY_BAND}. "
            "Consider widening --velocity-scales for a finer grid near the crossing."
        )

    out_path = Path("outputs") / f"{profile.name}_push_tuning_seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"robot": profile.name, "checkpoint_path": args.checkpoint_file, "results": results}, indent=2))
    print(f"\n[push-tuning] wrote {out_path}")
    print(
        "[push-tuning] per the plan's Gate 2 note: carry at least a low/mid/high severity forward into "
        "Gate 3 (scripts/robot_collect_frozen.py), not just whichever clears the target band fastest -- "
        "this lets Gate 4 show the AUPRC-vs-lead-time tradeoff across severities instead of hiding it."
    )


if __name__ == "__main__":
    main()
