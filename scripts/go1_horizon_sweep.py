"""Phase-2 tool (outside CLAUDE.md's cartpole-only spec, see DEVIATIONS.md):
sweep the termination-predictor's prediction horizon for a single Go1
checkpoint, to answer a concrete question -- is the current default horizon
(16 steps = 0.32s at Go1's confirmed 50 Hz) actually near the ceiling of how
far in advance a fall can be predicted, or would a longer horizon buy more
lead time (in seconds) without giving up accuracy?

This is NOT a re-run of the gated struggling/target/competent/fluent
pipeline (CLAUDE.md S7.1's analogue) -- it fixes ONE checkpoint and varies
ONLY the label horizon, everything else (architecture, loss, optimizer,
splitting, metrics) identical across every point in the sweep, via the same
already-tested functions the real pipeline uses:

  - `go1_collect_frozen.collect_one` for rollout + labelling (imported
    unmodified -- it already takes `horizon` as a plain int argument, no
    changes needed there).
  - `go1_eval_predictor.evaluate_one_dataset` for splitting, training,
    and every metric (imported unmodified -- same TerminationPredictor,
    same BCE/Adam/early-stopping, same AUPRC/AUROC/lead-time functions
    used everywhere else in this project).

Each horizon gets its own fresh rollout (not a relabelling of one shared
rollout) because `collect_one`'s stopping condition and the ring-buffer
labelling logic are both horizon-dependent, and reusing the tested,
unmodified function is worth more here than the wall-clock saved by
relabelling in place -- collection is cheap regardless
(~15-20s per 1M transitions per DEVIATIONS.md's gate-8 timings; this script
defaults to a much lighter probe than that, see PROBE_* below, since the
point is a quick comparison, not a final headline number).

Default horizons (steps -> seconds at Go1's STEP_SECONDS=0.02):
4->0.08, 8->0.16, 16->0.32 (current default), 24->0.48, 40->0.80,
60->1.20, 80->1.60, 120->2.40. Chosen to bracket the current default on
both sides and reach past 1.6s, not to privilege any particular answer.

This script does NOT pick a winner. It prints a comparison table and
writes outputs/go1_horizon_sweep_seed{N}.json; the choice of which horizon
to adopt is a judgment call over the accuracy/lead-time tradeoff, made
after looking at the numbers -- not something to automate here (same
"measured, not guessed, and not auto-decided" spirit as
scripts/go1_select_checkpoints.py's mechanical-but-reported approach).

Usage, from the repo root, on the GPU machine:

    python scripts/go1_horizon_sweep.py --checkpoint-file logs/rsl_rl/go1_velocity_term/<run>/model_999.pt --label target
    python scripts/go1_horizon_sweep.py --checkpoint-file <path> --label competent --horizon-steps 8 16 32 64
    python scripts/go1_horizon_sweep.py --checkpoint-file <path> --label target --full   # gate-8-sized collection, slower, for a final call
    python scripts/go1_horizon_sweep.py --smoke --checkpoint-file <path>  # ~1 min sanity check of the sweep mechanics

Pass a distinct --label per checkpoint (e.g. "target", "competent") when
comparing more than one operating point for the same --seed -- it tags
every intermediate dataset/model file and the final
outputs/go1_horizon_sweep_{label}_seed{N}.json, so two runs never collide.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term.go1  # noqa: F401,E402
from cartpole_term.cli import SMOKE_COLLECT_TRANSITIONS, SMOKE_NUM_ENVS  # noqa: E402
from go1_collect_frozen import COLLECT_NUM_ENVS, TARGET_TERMINATED_EPISODES, TARGET_TRANSITIONS, collect_one  # noqa: E402
from go1_eval_predictor import STEP_SECONDS, evaluate_one_dataset  # noqa: E402

DEFAULT_HORIZON_STEPS = [4, 8, 16, 24, 40, 60, 80, 120]
SMOKE_TARGET_TERMINATED_EPISODES = 10

# Lightweight probe defaults -- enough transitions for a stable AUPRC/lead-time
# comparison across horizons without paying the full gate-8 collection cost
# 8 times over. Mirrors the original (pre-"raised to full") lightweight probe
# noted in DEVIATIONS.md's eval_predictor_generalization.py entry. Pass --full
# to use the same targets go1_collect_frozen.py's gated collection uses, once
# a shortlist of horizons is worth confirming rigorously.
PROBE_NUM_ENVS = 1024
PROBE_TARGET_TRANSITIONS = 50_000
PROBE_TARGET_TERMINATED = 100


def run_sweep(args: argparse.Namespace, device: str) -> dict:
    if args.smoke:
        num_envs = SMOKE_NUM_ENVS
        target_transitions = SMOKE_COLLECT_TRANSITIONS
        target_terminated = SMOKE_TARGET_TERMINATED_EPISODES
    elif args.full:
        num_envs = COLLECT_NUM_ENVS
        target_transitions = TARGET_TRANSITIONS
        target_terminated = TARGET_TERMINATED_EPISODES
    else:
        num_envs = PROBE_NUM_ENVS
        target_transitions = PROBE_TARGET_TRANSITIONS
        target_terminated = PROBE_TARGET_TERMINATED

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for horizon in args.horizon_steps:
        tag = f"hsweep_{args.label}_h{horizon:04d}"
        print(f"[go1-hsweep] horizon={horizon} steps ({horizon * STEP_SECONDS:.2f}s): collecting...")
        t0 = time.time()
        dataset, sidecar = collect_one(
            checkpoint_path=args.checkpoint_file,
            seed=args.seed,
            operating_point=tag,
            num_envs=num_envs,
            target_transitions=target_transitions,
            target_terminated=target_terminated,
            horizon=horizon,
            device=device,
        )
        npz_path = outputs_dir / f"go1_frozen_{tag}_seed{args.seed}.npz"
        json_path = outputs_dir / f"go1_frozen_{tag}_seed{args.seed}.json"
        np.savez_compressed(npz_path, **dataset)
        json_path.write_text(json.dumps(sidecar, indent=2))
        print(
            f"[go1-hsweep] horizon={horizon}: collected {sidecar['labelled_transitions']} transitions, "
            f"base_rate={sidecar['base_rate']:.3f}, terminated_episodes={sidecar['terminated_episode_count']}, "
            f"wall_time={sidecar['wall_time_s']:.1f}s -- training predictor..."
        )

        metrics = evaluate_one_dataset(npz_path, json_path, seed=args.seed, hidden=args.hidden, device="cpu")
        total_wall_s = time.time() - t0

        row = {
            "horizon_steps": horizon,
            "horizon_seconds": horizon * STEP_SECONDS,
            "collection_base_rate": sidecar["base_rate"],
            "test_base_rate": metrics["split_base_rates"]["test"],
            "auprc": metrics["mlp"]["auprc"],
            "auprc_baseline": metrics["mlp"]["auprc_baseline"],
            "auroc": metrics["mlp"]["auroc"],
            "median_lead_steps": metrics["mlp_lead_time"]["median_lead_steps"],
            "median_lead_seconds": metrics["mlp_lead_time"]["median_lead_seconds"],
            "num_detected_episodes": metrics["mlp_lead_time"]["num_detected_episodes"],
            "num_episodes_with_positives": metrics["mlp_lead_time"]["num_episodes_with_positives"],
            "collected_transitions": sidecar["labelled_transitions"],
            "terminated_episodes": sidecar["terminated_episode_count"],
            "collection_wall_time_s": sidecar["wall_time_s"],
            "total_wall_time_s": total_wall_s,
            "npz_path": str(npz_path),
        }
        results.append(row)
        print(
            f"[go1-hsweep] horizon={horizon} ({row['horizon_seconds']:.2f}s): "
            f"AUPRC={row['auprc']:.4f} (baseline={row['auprc_baseline']:.4f}) AUROC={row['auroc']:.4f} "
            f"median_lead={row['median_lead_steps']:.1f} steps ({row['median_lead_seconds']:.2f}s)"
        )

    return {
        "checkpoint_path": args.checkpoint_file,
        "seed": args.seed,
        "step_seconds": STEP_SECONDS,
        "resolved_args": vars(args),
        "results": results,
    }


def print_summary_table(results: list[dict]) -> None:
    header = (
        f"{'steps':>6} {'sec':>6} {'coll.base%':>10} {'test.base%':>10} "
        f"{'AUPRC':>7} {'AUPRC_base':>10} {'AUROC':>7} {'lead(steps)':>11} {'lead(s)':>8}"
    )
    print("\n[go1-hsweep] summary (sorted by horizon):")
    print(header)
    print("-" * len(header))
    for row in sorted(results, key=lambda r: r["horizon_steps"]):
        print(
            f"{row['horizon_steps']:>6} {row['horizon_seconds']:>6.2f} "
            f"{100 * row['collection_base_rate']:>9.2f}% {100 * row['test_base_rate']:>9.2f}% "
            f"{row['auprc']:>7.4f} {row['auprc_baseline']:>10.4f} {row['auroc']:>7.4f} "
            f"{row['median_lead_steps']:>11.1f} {row['median_lead_seconds']:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep the Go1 termination-predictor's label horizon for one fixed checkpoint (phase-2, see DEVIATIONS.md).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Rollout/split/train seed (not a training seed).")
    parser.add_argument(
        "--label", type=str, default="sweep",
        help="Tags every intermediate file (dataset/model) and the final output JSON, e.g. "
        "'target' or 'competent' -- REQUIRED to be distinct across multiple runs against different "
        "checkpoints for the same --seed, or one run's files silently overwrite the previous one's.",
    )
    parser.add_argument("--horizon-steps", dest="horizon_steps", type=int, nargs="+", default=DEFAULT_HORIZON_STEPS)
    parser.add_argument(
        "--predictor-hidden", dest="predictor_hidden", type=str, default="128,128",
        help="Matches CLAUDE.md S8.2's (128,128) default; kept identical across every horizon in the sweep.",
    )
    parser.add_argument("--smoke", action="store_true", help="Tiny sanity pass: confirms the sweep runs end to end, not a real comparison.")
    parser.add_argument(
        "--full", action="store_true",
        help="Use the same collection targets go1_collect_frozen.py's gated pipeline uses "
        "(1M transitions / 1000 terminated episodes / 4096 envs) instead of the lightweight probe default. Slower.",
    )
    args = parser.parse_args()
    h1, h2 = (int(p) for p in args.predictor_hidden.split(","))
    args.hidden = (h1, h2)

    if args.smoke and args.full:
        raise ValueError("--smoke and --full are mutually exclusive.")
    if len(set(args.horizon_steps)) != len(args.horizon_steps):
        raise ValueError(f"--horizon-steps has duplicates: {args.horizon_steps}")
    if any(h < 1 for h in args.horizon_steps):
        raise ValueError(f"--horizon-steps must all be >= 1, got {args.horizon_steps}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4, same requirement for Go1).")
    device = "cuda"

    output = run_sweep(args, device)
    print_summary_table(output["results"])

    out_path = Path("outputs") / f"go1_horizon_sweep_{args.label}_seed{args.seed}.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[go1-hsweep] wrote {out_path}")
    print("[go1-hsweep] upload this file back for a decision on which horizon to adopt.")


if __name__ == "__main__":
    main()
