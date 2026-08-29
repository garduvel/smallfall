"""Mechanical checkpoint selection at three operating points (CLAUDE.md
S7.1, S7.2, S11 gate 7).

Selection is mechanical, not eyeballed: for each seed and each operating
point, choose the EARLIEST checkpoint whose frozen-eval mean episode length
(under the deterministic mean action) falls inside the point's tolerance
window. Earliest, not closest -- if the metric is non-monotonic, the early
crossing is on the improving limb and is the reproducible choice. Then
re-measure that checkpoint with a fresh eval seed (train_seed + 200) and
report THAT number, not the selection measurement (S7.2.4: selecting and
reporting on the same measurement is winner's-curse biased).

Run on the GPU machine, from the repo root, after scripts/train.py has
completed for the seeds you want to select:

    python scripts/select_checkpoints.py               # default seeds 0 1 2
    python scripts/select_checkpoints.py --seeds 0
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# name -> (target mean episode length, tolerance), CLAUDE.md S7.1
OPERATING_POINTS: dict[str, tuple[float, float]] = {
    "struggling": (80.0, 15.0),
    "target": (150.0, 25.0),
    "competent": (400.0, 60.0),
}

REMEASURE_SEED_OFFSET = 200  # CLAUDE.md S7.2.4


def select_from_history(history: list[dict], target: float, tolerance: float) -> dict | None:
    """Earliest record (by iteration) whose mean_action mean episode length
    falls inside [target - tolerance, target + tolerance], or None."""
    lo, hi = target - tolerance, target + tolerance
    for record in sorted(history, key=lambda r: r["iteration"]):
        length = record["mean_action"]["mean_episode_length"]
        if lo <= length <= hi:
            return record
    return None


def select_any_for_smoke(history: list[dict]) -> dict | None:
    """--smoke fallback: a 5-iteration run never approaches any real
    operating-point window (80/150/400 steps), so requiring the real
    windows would make the smoke test fail for a reason unrelated to
    whether the pipeline actually works. Just pick the last available
    checkpoint -- gate 4 only needs a valid checkpoint path to flow through
    collect_frozen.py and eval_predictor.py, not a representative one."""
    if not history:
        return None
    return sorted(history, key=lambda r: r["iteration"])[-1]


def find_latest_run_dir(experiment_name: str, seed: int, log_root: Path) -> Path:
    pattern = str(log_root / experiment_name / f"*_seed{seed}")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No training run directory found matching {pattern}. Run "
            "scripts/train.py for this seed first."
        )
    return Path(matches[-1])


def main() -> None:
    import torch

    from cartpole_term.cli import build_parser
    from cartpole_term.runner import build_runner_from_checkpoint

    parser = build_parser(
        ("select",), description="Select frozen-eval checkpoints at three operating points (CLAUDE.md S7.2)."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required (CLAUDE.md S0.2 / S5.4) for the re-measurement step."
        )
    device = "cuda"

    log_root = Path("logs/rsl_rl")
    experiment_name = "cartpole_term"
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    out_path = outputs_dir / "selected_checkpoints.json"
    # Merge with whatever's already on disk rather than clobbering it: this
    # script is routinely invoked once per seed-batch across separate
    # sessions (e.g. --seeds 0 today, --seeds 1 2 next), and an unconditional
    # overwrite would silently drop already-selected seeds' entries. Found
    # via a real Gate 10 run -- see DEVIATIONS.md.
    all_results: dict[str, dict] = json.loads(out_path.read_text()) if out_path.exists() else {}
    for seed in args.seeds:
        run_dir = find_latest_run_dir(experiment_name, seed, log_root)
        eval_history_path = run_dir / "eval_history.json"
        if not eval_history_path.exists():
            raise FileNotFoundError(
                f"{eval_history_path} not found -- did training run to completion?"
            )
        history = json.loads(eval_history_path.read_text())

        seed_results: dict[str, dict] = {}
        for point_name, (target, tolerance) in OPERATING_POINTS.items():
            if args.smoke:
                selected = select_any_for_smoke(history)
            else:
                selected = select_from_history(history, target, tolerance)
            if selected is None:
                available = [
                    (r["iteration"], r["mean_action"]["mean_episode_length"]) for r in history
                ]
                raise RuntimeError(
                    f"No checkpoint for seed={seed} qualifies for operating point "
                    f"'{point_name}' (target={target} +/- {tolerance}). Available "
                    f"checkpoints (iteration, mean_episode_length): {available}. "
                    "Not silently substituting -- CLAUDE.md S7.2.3."
                )

            iteration = selected["iteration"]
            checkpoint_path = run_dir / f"model_{iteration}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Selected checkpoint {checkpoint_path} does not exist on disk "
                    f"(eval_history.json says iteration {iteration} qualified, but "
                    "save_interval and eval_interval may have drifted apart)."
                )

            remeasure_seed = seed + REMEASURE_SEED_OFFSET
            runner = build_runner_from_checkpoint(
                str(checkpoint_path), seed=seed, eval_num_envs=256, device=device
            )
            remeasurement = runner.evaluate(seed=remeasure_seed, stochastic_output=False)

            seed_results[point_name] = {
                "seed": seed,
                "iteration": iteration,
                "checkpoint_path": str(checkpoint_path),
                "selection_measurement": selected["mean_action"],
                "remeasurement_seed": remeasure_seed,
                "remeasurement": remeasurement,
            }
            print(
                f"[select] seed={seed} point={point_name} it={iteration} "
                f"selection_len={selected['mean_action']['mean_episode_length']:.1f} "
                f"remeasured_len={remeasurement['mean_episode_length']:.1f}"
            )

        all_results[str(seed)] = seed_results
        # Write after each seed, not just at the end: if a later seed in
        # this batch raises (e.g. a window genuinely not crossed), this
        # seed's already-computed selection isn't lost.
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"  wrote {out_path} (seed {seed} done)")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
