"""Frozen-policy dataset collection for Go1 (phase-2 analogue of CLAUDE.md
S7.3/S11 gate 8 -- see DEVIATIONS.md). Same mechanism as
scripts/collect_frozen.py's collect_one(): loads a checkpoint, keeps
sampling stochastically (the label is defined under the behaviour policy),
freezes weights (asserted unchanged at the end), and labels each transition
via cartpole_term.predictor.labeling.TerminationLabeler exactly as before.

The one real difference from cartpole's collect_one(): failure-cause
attribution doesn't hardcode "pole_angle"/"cart_pos". It discovers every
termination term registered with time_out=False on the actual env cfg (any
such term maps to `terminated`, never `truncated` -- CLAUDE.md S3.2.3's rule,
which is a general mjlab convention, not cartpole-specific) and reads
termination_manager.get_term(name) for each of those, same per-env per-step
attribution mechanism scripts/inspect_video_and_termmgr.py originally
confirmed for cartpole. For Go1 this is expected to resolve to a single
cause ("fell_over" or similar -- confirmed by scripts/inspect_go1_task.py),
but nothing here assumes that count.

Run on the GPU machine, from the repo root, after
scripts/go1_select_checkpoints.py has produced outputs/go1_selected_checkpoints.json:

    python scripts/go1_collect_frozen.py --smoke
    python scripts/go1_collect_frozen.py --seeds 0 --operating-point target
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term.go1  # noqa: F401,E402
from cartpole_term.cli import SMOKE_COLLECT_TRANSITIONS, SMOKE_NUM_ENVS  # noqa: E402
from cartpole_term.predictor.labeling import TerminationLabeler  # noqa: E402

TARGET_TRANSITIONS = 1_000_000
TARGET_TERMINATED_EPISODES = 1_000
COLLECT_NUM_ENVS = 4096
SMOKE_TARGET_TERMINATED_EPISODES = 20
MAX_STEPS_SAFETY_CAP = 2_000_000
COLLECT_SEED_OFFSET = 300


def collect_one(
    checkpoint_path: str,
    seed: int,
    operating_point: str,
    num_envs: int,
    target_transitions: int,
    target_terminated: int,
    horizon: int,
    device: str,
) -> tuple[dict, dict]:
    from cartpole_term.go1.runner import build_go1_runner_from_checkpoint

    runner = build_go1_runner_from_checkpoint(checkpoint_path, seed=seed, eval_num_envs=num_envs, device=device)
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    env = runner._build_eval_env()
    raw_env = env.unwrapped

    cfg_terminations = raw_env.cfg.terminations
    failure_cause_names = [
        name for name, term_cfg in cfg_terminations.items() if not getattr(term_cfg, "time_out", False)
    ]
    if not failure_cause_names:
        raise RuntimeError(
            "No non-time_out termination terms found on this env cfg -- nothing can ever "
            "emit label=1. Check cartpole_term/go1/env_cfg.py wiring against "
            "outputs/go1_task_inspection.json."
        )
    print(f"[go1-collect] failure-cause terms: {failure_cause_names}")

    labeler = TerminationLabeler(num_envs=num_envs, horizon=horizon)

    env.seed(seed + COLLECT_SEED_OFFSET)
    obs, _ = env.reset()
    obs = obs.to(device)

    next_episode_id = num_envs
    episode_id = np.arange(num_envs, dtype=np.int64)
    step_in_episode = np.zeros(num_envs, dtype=np.int64)
    cur_episode_len = np.zeros(num_envs, dtype=np.int64)
    episode_lengths: list[int] = []

    records: list[dict] = []
    terminated_episode_count = 0
    total_steps = 0
    total_pushed = 0

    t0 = time.time()
    with torch.inference_mode():
        while len(records) < target_transitions or terminated_episode_count < target_terminated:
            if total_steps > MAX_STEPS_SAFETY_CAP:
                print(
                    f"[go1-collect] SAFETY CAP hit ({MAX_STEPS_SAFETY_CAP} steps) before reaching "
                    f"targets (transitions={len(records)}/{target_transitions}, "
                    f"terminated_episodes={terminated_episode_count}/{target_terminated}). Stopping."
                )
                break

            actions = runner.alg.actor(obs, stochastic_output=True)
            obs_actor_np = obs["actor"].detach().cpu().numpy()
            action_np = actions.detach().cpu().numpy()

            next_obs, _rewards, dones, extras = env.step(actions)
            dones_cpu = dones.bool().cpu()
            time_outs = extras.get("time_outs")
            truncated_cpu = time_outs.bool().cpu() if time_outs is not None else torch.zeros_like(dones_cpu)
            terminated_cpu = dones_cpu & ~truncated_cpu

            cause_hits = {name: raw_env.termination_manager.get_term(name).cpu() for name in failure_cause_names}

            payloads = [
                {
                    "obs": obs_actor_np[i],
                    "action": action_np[i],
                    "env_id": i,
                    "episode_id": int(episode_id[i]),
                    "step_in_episode": int(step_in_episode[i]),
                }
                for i in range(num_envs)
            ]
            total_pushed += num_envs
            emitted = labeler.step(payloads, terminated_cpu.tolist(), truncated_cpu.tolist())
            for sample in emitted:
                cause = ""
                if sample.label == 1:
                    env_id = sample.env_id
                    hit_names = sorted(name for name, hits in cause_hits.items() if bool(hits[env_id]))
                    cause = "+".join(hit_names) if hit_names else "unknown"
                records.append(
                    {
                        **sample.payload,
                        "label": sample.label,
                        "steps_to_termination": (
                            sample.steps_to_termination if sample.steps_to_termination is not None else -1
                        ),
                        "termination_cause": cause,
                    }
                )

            terminated_episode_count += int(terminated_cpu.sum().item())
            total_steps += 1

            cur_episode_len += 1
            step_in_episode += 1
            done_np = dones_cpu.numpy()
            if done_np.any():
                idx = np.nonzero(done_np)[0]
                episode_lengths.extend(cur_episode_len[idx].tolist())
                cur_episode_len[idx] = 0
                step_in_episode[idx] = 0
                episode_id[idx] = next_episode_id + np.arange(len(idx))
                next_episode_id += len(idx)

            obs = next_obs.to(device)

    wall_time = time.time() - t0

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during frozen-policy collection"

    pending_at_end = sum(labeler.pending_counts())
    terminal_step_discard_count = terminated_episode_count
    censored_count = total_pushed - len(records) - pending_at_end - terminal_step_discard_count

    obs_arr = np.stack([r["obs"] for r in records]).astype(np.float32)
    action_arr = np.stack([r["action"] for r in records]).astype(np.float32)
    label_arr = np.array([r["label"] for r in records], dtype=np.int64)
    env_id_arr = np.array([r["env_id"] for r in records], dtype=np.int64)
    episode_id_arr = np.array([r["episode_id"] for r in records], dtype=np.int64)
    step_in_episode_arr = np.array([r["step_in_episode"] for r in records], dtype=np.int64)
    steps_to_term_arr = np.array([r["steps_to_termination"] for r in records], dtype=np.int64)
    cause_arr = np.array([r["termination_cause"] for r in records], dtype="U32")

    dataset = {
        "obs": obs_arr,
        "action": action_arr,
        "label": label_arr,
        "env_id": env_id_arr,
        "episode_id": episode_id_arr,
        "step_in_episode": step_in_episode_arr,
        "steps_to_termination": steps_to_term_arr,
        "termination_cause": cause_arr,
    }

    base_rate = float(label_arr.mean()) if len(label_arr) > 0 else 0.0
    cause_counts = {c: int((cause_arr == c).sum()) for c in sorted(set(cause_arr.tolist())) if c}
    lengths_arr = np.array(episode_lengths, dtype=np.int64)
    sidecar = {
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "operating_point": operating_point,
        "collect_seed": seed + COLLECT_SEED_OFFSET,
        "num_envs": num_envs,
        "horizon": horizon,
        "obs_dim": int(obs_arr.shape[1]) if len(obs_arr) else None,
        "action_dim": int(action_arr.shape[1]) if len(action_arr) else None,
        "failure_cause_terms": failure_cause_names,
        "wall_time_s": wall_time,
        "total_env_steps": total_steps,
        "total_transitions_pushed": total_pushed,
        "labelled_transitions": len(records),
        "base_rate": base_rate,
        "censored_step_count": int(censored_count),
        "terminal_step_discard_count": int(terminal_step_discard_count),
        "pending_at_end_discarded": int(pending_at_end),
        "episode_count": len(episode_lengths),
        "terminated_episode_count": terminated_episode_count,
        "termination_cause_counts": cause_counts,
        "episode_length_mean": float(lengths_arr.mean()) if len(lengths_arr) else None,
        "episode_length_median": float(np.median(lengths_arr)) if len(lengths_arr) else None,
        "episode_length_histogram": (
            {str(edge): int(count) for edge, count in zip(*np.histogram(lengths_arr, bins=10)[::-1], strict=False)}
            if len(lengths_arr)
            else {}
        ),
        "hit_transitions_target": len(records) >= target_transitions,
        "hit_terminated_episodes_target": terminated_episode_count >= target_terminated,
        "hit_safety_cap": total_steps > MAX_STEPS_SAFETY_CAP,
    }

    return dataset, sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a frozen-policy labelled dataset for Go1 (phase-2, see DEVIATIONS.md).")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--operating-point", dest="operating_point", choices=("struggling", "target", "competent", "fluent"), default="target")
    parser.add_argument("--predictor.horizon", dest="predictor_horizon", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    selected_path = Path("outputs/go1_selected_checkpoints.json")
    if not selected_path.exists():
        raise FileNotFoundError(f"{selected_path} not found -- run scripts/go1_select_checkpoints.py first.")
    selected = json.loads(selected_path.read_text())

    target_transitions = SMOKE_COLLECT_TRANSITIONS if args.smoke else TARGET_TRANSITIONS
    target_terminated = SMOKE_TARGET_TERMINATED_EPISODES if args.smoke else TARGET_TERMINATED_EPISODES
    num_envs = SMOKE_NUM_ENVS if args.smoke else COLLECT_NUM_ENVS

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_key = str(seed)
        if seed_key not in selected:
            raise KeyError(f"seed {seed} not found in {selected_path} -- run go1_select_checkpoints.py for it.")
        if args.operating_point not in selected[seed_key]:
            raise KeyError(f"operating point '{args.operating_point}' not found for seed {seed}")

        entry = selected[seed_key][args.operating_point]
        checkpoint_path = entry["checkpoint_path"]
        print(f"[go1-collect] seed={seed} point={args.operating_point} checkpoint={checkpoint_path}")

        dataset, sidecar = collect_one(
            checkpoint_path=checkpoint_path,
            seed=seed,
            operating_point=args.operating_point,
            num_envs=num_envs,
            target_transitions=target_transitions,
            target_terminated=target_terminated,
            horizon=args.predictor_horizon,
            device=device,
        )

        npz_path = outputs_dir / f"go1_frozen_{args.operating_point}_seed{seed}.npz"
        np.savez_compressed(npz_path, **dataset)
        json_path = outputs_dir / f"go1_frozen_{args.operating_point}_seed{seed}.json"
        json_path.write_text(json.dumps(sidecar, indent=2))

        print(
            f"[go1-collect] seed={seed} point={args.operating_point}: "
            f"{sidecar['labelled_transitions']} labelled transitions, "
            f"obs_dim={sidecar['obs_dim']} action_dim={sidecar['action_dim']} "
            f"base_rate={sidecar['base_rate']:.3f}, "
            f"terminated_episodes={sidecar['terminated_episode_count']}, "
            f"wall_time={sidecar['wall_time_s']:.1f}s"
        )
        print(f"  wrote {npz_path}")
        print(f"  wrote {json_path}")


if __name__ == "__main__":
    main()
