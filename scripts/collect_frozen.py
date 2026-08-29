"""Frozen-policy dataset collection (CLAUDE.md S7.3, S11 gate 8).

Loads a checkpoint (selected via scripts/select_checkpoints.py), sets eval
mode but keeps sampling STOCHASTICALLY -- CLAUDE.md S7.3: the label is
defined under the behaviour policy, so the behaviour policy is what runs.
Weights are frozen and asserted unchanged at the end (no process_env_step or
optimizer step ever happens during collection).

Cause attribution reads env.unwrapped.termination_manager.get_term(name)
immediately after each step() call: confirmed via
scripts/inspect_video_and_termmgr.py to be the per-env, per-term boolean
cache set inside step() BEFORE the auto-reset block runs, so -- unlike a
post-step joint_pos query, which scripts/inspect_reset_behavior.py showed is
contaminated by auto-reset -- this correctly reflects each env's own cause
on its own termination step.

Run on the GPU machine, from the repo root, after
scripts/select_checkpoints.py has produced outputs/selected_checkpoints.json:

    python scripts/collect_frozen.py                              # all seeds, target point
    python scripts/collect_frozen.py --smoke                       # S11 gate 4 tiny check
    python scripts/collect_frozen.py --seeds 0 --operating-point struggling
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term  # noqa: F401,E402
from cartpole_term.cli import SMOKE_COLLECT_TRANSITIONS, SMOKE_NUM_ENVS, build_parser  # noqa: E402
from cartpole_term.predictor.labeling import TerminationLabeler  # noqa: E402
from cartpole_term.runner import build_runner_from_checkpoint  # noqa: E402

TARGET_TRANSITIONS = 1_000_000
TARGET_TERMINATED_EPISODES = 1_000
COLLECT_NUM_ENVS = 4096
SMOKE_TARGET_TERMINATED_EPISODES = 20  # 1000 is absurd against a 5000-transition smoke target
MAX_STEPS_SAFETY_CAP = 2_000_000  # engineering guard against a runaway loop, not a spec requirement
COLLECT_SEED_OFFSET = 300  # distinct from train (seed) and select's re-measure (seed+200)


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
    runner = build_runner_from_checkpoint(
        checkpoint_path, seed=seed, eval_num_envs=num_envs, device=device
    )
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    env = runner._build_eval_env()  # identical dynamics to training, CLAUDE.md S6.2.2
    raw_env = env.unwrapped

    labeler: TerminationLabeler = TerminationLabeler(num_envs=num_envs, horizon=horizon)

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
                    f"[collect] SAFETY CAP hit ({MAX_STEPS_SAFETY_CAP} steps) before "
                    f"reaching targets (transitions={len(records)}/{target_transitions}, "
                    f"terminated_episodes={terminated_episode_count}/{target_terminated}). "
                    "Stopping and reporting what was collected (CLAUDE.md S7.3)."
                )
                break

            actions = runner.alg.actor(obs, stochastic_output=True)  # S7.3: sample, don't take the mean
            obs_actor_np = obs["actor"].detach().cpu().numpy()
            action_np = actions.detach().cpu().numpy()

            next_obs, _rewards, dones, extras = env.step(actions)
            dones_cpu = dones.bool().cpu()
            time_outs = extras.get("time_outs")
            truncated_cpu = (
                time_outs.bool().cpu() if time_outs is not None else torch.zeros_like(dones_cpu)
            )
            terminated_cpu = dones_cpu & ~truncated_cpu

            pole_hit = raw_env.termination_manager.get_term("pole_angle").cpu()
            cart_hit = raw_env.termination_manager.get_term("cart_pos").cpu()

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
                    if pole_hit[env_id] and cart_hit[env_id]:
                        cause = "both"
                    elif pole_hit[env_id]:
                        cause = "pole_angle"
                    elif cart_hit[env_id]:
                        cause = "cart_pos"
                records.append(
                    {
                        **sample.payload,
                        "label": sample.label,
                        "steps_to_termination": (
                            sample.steps_to_termination
                            if sample.steps_to_termination is not None
                            else -1
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
    # Every terminated episode silently drops exactly one entry: the fresh,
    # age-0 push from its own terminal step, which never falls in the
    # emitted [1, horizon] positive range (CLAUDE.md S7.4: "the terminal
    # step itself is not a training example"). That is routine and
    # provable from TerminationLabeler.step()'s structure -- NOT the same
    # thing as truncation-censoring (S7.3's "censored" concept: steps whose
    # future is unknown because a time-out cut the episode short). Keep the
    # two separate rather than reporting one conflated "censored" number.
    terminal_step_discard_count = terminated_episode_count
    censored_count = total_pushed - len(records) - pending_at_end - terminal_step_discard_count

    obs_arr = np.stack([r["obs"] for r in records]).astype(np.float32)
    action_arr = np.stack([r["action"] for r in records]).astype(np.float32)
    label_arr = np.array([r["label"] for r in records], dtype=np.int64)
    env_id_arr = np.array([r["env_id"] for r in records], dtype=np.int64)
    episode_id_arr = np.array([r["episode_id"] for r in records], dtype=np.int64)
    step_in_episode_arr = np.array([r["step_in_episode"] for r in records], dtype=np.int64)
    steps_to_term_arr = np.array([r["steps_to_termination"] for r in records], dtype=np.int64)
    cause_arr = np.array([r["termination_cause"] for r in records], dtype="U16")

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
    cause_counts = {
        c: int((cause_arr == c).sum()) for c in ("pole_angle", "cart_pos", "both") if (cause_arr == c).any()
    }
    lengths_arr = np.array(episode_lengths, dtype=np.int64)
    sidecar = {
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "operating_point": operating_point,
        "collect_seed": seed + COLLECT_SEED_OFFSET,
        "num_envs": num_envs,
        "horizon": horizon,
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
            {
                str(edge): int(count)
                for edge, count in zip(
                    *np.histogram(lengths_arr, bins=10)[::-1], strict=False
                )
            }
            if len(lengths_arr)
            else {}
        ),
        "hit_transitions_target": len(records) >= target_transitions,
        "hit_terminated_episodes_target": terminated_episode_count >= target_terminated,
        "hit_safety_cap": total_steps > MAX_STEPS_SAFETY_CAP,
    }

    return dataset, sidecar


def main() -> None:
    parser = build_parser(
        ("collect",), description="Collect a frozen-policy labelled dataset (CLAUDE.md S7.3)."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    selected_path = Path("outputs/selected_checkpoints.json")
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{selected_path} not found -- run scripts/select_checkpoints.py first."
        )
    selected = json.loads(selected_path.read_text())

    target_transitions = SMOKE_COLLECT_TRANSITIONS if args.smoke else TARGET_TRANSITIONS
    target_terminated = SMOKE_TARGET_TERMINATED_EPISODES if args.smoke else TARGET_TERMINATED_EPISODES
    num_envs = SMOKE_NUM_ENVS if args.smoke else COLLECT_NUM_ENVS

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        seed_key = str(seed)
        if seed_key not in selected:
            raise KeyError(f"seed {seed} not found in {selected_path} -- run select_checkpoints.py for it.")
        if args.operating_point not in selected[seed_key]:
            raise KeyError(f"operating point '{args.operating_point}' not found for seed {seed}")

        entry = selected[seed_key][args.operating_point]
        checkpoint_path = entry["checkpoint_path"]
        print(f"[collect] seed={seed} point={args.operating_point} checkpoint={checkpoint_path}")

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

        npz_path = outputs_dir / f"frozen_{args.operating_point}_seed{seed}.npz"
        np.savez_compressed(npz_path, **dataset)
        json_path = outputs_dir / f"frozen_{args.operating_point}_seed{seed}.json"
        json_path.write_text(json.dumps(sidecar, indent=2))

        print(
            f"[collect] seed={seed} point={args.operating_point}: "
            f"{sidecar['labelled_transitions']} labelled transitions, "
            f"base_rate={sidecar['base_rate']:.3f}, "
            f"terminated_episodes={sidecar['terminated_episode_count']}, "
            f"censored={sidecar['censored_step_count']}, "
            f"wall_time={sidecar['wall_time_s']:.1f}s"
        )
        print(f"  wrote {npz_path}")
        print(f"  wrote {json_path}")


if __name__ == "__main__":
    main()
