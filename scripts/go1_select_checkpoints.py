"""Mechanical checkpoint selection for Go1, at operating points analogous to
CLAUDE.md S7.1/S7.2 (phase-2, see DEVIATIONS.md). Same overall mechanism as
cartpole (earliest checkpoint whose frozen-eval episode length falls inside a
window, then re-measured with a fresh eval seed (seed+200) and THAT number
reported, not the selection measurement -- S7.2.4's winner's-curse point
applies identically here) -- reused via direct import of
scripts/select_checkpoints.py's find_latest_run_dir/select_any_for_smoke.

ONE deliberate divergence from cartpole, found empirically on the first real
run (see DEVIATIONS.md for the full writeup): selection is done on
`sampled_action.mean_episode_length`, NOT `mean_action.mean_episode_length`
like cartpole. Go1's action space is a joint-position offset from a default
standing pose (use_default_offset=True); the DETERMINISTIC mean-action eval
therefore reads near-max survival (~740+/1000) from iteration 0 onward --
"hold the default pose and do nothing" is a fairly stable standing behaviour
for a quadruped regardless of whether the policy has learned to track
commanded velocity (confirmed by Episode_Reward/track_linear_velocity
staying ~0 while mean_action_len was already high). This makes mean_action
useless as a competence axis here: it never passes through low values, so
struggling/target/competent windows are literally unreachable. The
STOCHASTIC sampled-action eval doesn't have this problem (full exploration
noise makes "do nothing" fall over immediately) and is, not coincidentally,
the exact same distribution scripts/go1_collect_frozen.py always samples
under anyway -- so it's the more relevant axis for this project's actual
purpose, not just a workaround. Both selection AND the S7.2.4 re-measurement
use stochastic_output=True for this reason (cartpole's re-measurement uses
mean/deterministic; that would be inconsistent with a sampled-action
selection criterion and would make every re-measured number read ~950+
regardless of which operating point was picked).

Go1's own struggling/target/competent/fluent windows are still provisional --
the numbers below (as of the first real 100-iteration run on seed 0) are
educated updates, not final. Pass --struggling/--target/--competent/--fluent
to override further if a run's actual sampled_action curve doesn't land in
sensible places.

Run on the GPU machine, from the repo root, after scripts/go1_train.py:

    python scripts/go1_select_checkpoints.py --seeds 0
    python scripts/go1_select_checkpoints.py --seeds 0 --target 300 40   # override after inspecting the curve
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for select_checkpoints import below

from select_checkpoints import (  # noqa: E402
    find_latest_run_dir,
    select_any_for_smoke,
)

SELECTION_FIELD = "sampled_action"  # see module docstring -- NOT mean_action, unlike cartpole


def select_from_history_by_field(history: list[dict], target: float, tolerance: float, field: str) -> dict | None:
    """Same "earliest checkpoint in [target-tolerance, target+tolerance]"
    logic as scripts/select_checkpoints.py's select_from_history, just keyed
    by an arbitrary eval_history.json top-level field instead of a hardcoded
    "mean_action" -- see SELECTION_FIELD / module docstring."""
    lo, hi = target - tolerance, target + tolerance
    for record in sorted(history, key=lambda r: r["iteration"]):
        length = record[field]["mean_episode_length"]
        if lo <= length <= hi:
            return record
    return None


# name -> (target, tolerance). A re-run of the SAME nominal seed (0) showed
# real run-to-run curve variance (GPU-parallel PPO isn't perfectly
# deterministic even at a fixed seed -- see DEVIATIONS.md): run 1 had
# it=10:196.1, run 2 had it=10:174.7 with it=15 already at 252.4, skipping
# clean over a narrow 196+-15 window entirely. Point-matched windows fit to
# one run's exact values don't reproduce. Fixed by making the four windows a
# CONTIGUOUS PARTITION of the whole [90, 1000] range instead of narrow
# point-matches: [90,140]/[140,300]/[300,850]/[850,1000], sized off both
# runs' curves (it=0 always ~110-115; the pre-jump region spans roughly
# 140-300 across runs; the post-jump climb spans roughly 300-850; the
# plateau is consistently ~850-960). Any curve shape that rises from ~110 to
# ~900+ crosses all four with no gap between them, so a specific run's exact
# iteration-to-value mapping no longer matters.
DEFAULT_OPERATING_POINTS: dict[str, tuple[float, float]] = {
    "struggling": (115.0, 25.0),   # [90, 140]
    "target": (220.0, 80.0),       # [140, 300]
    "competent": (575.0, 275.0),   # [300, 850]
    "fluent": (925.0, 75.0),       # [850, 1000]
}

REMEASURE_SEED_OFFSET = 200


def main() -> None:
    from cartpole_term.go1.env_cfg import TASK_ID  # noqa: F401 -- documents which task this targets
    from cartpole_term.go1.runner import build_go1_runner_from_checkpoint

    parser = argparse.ArgumentParser(description="Select frozen-eval checkpoints for Go1 (phase-2, see DEVIATIONS.md).")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    for point_name in DEFAULT_OPERATING_POINTS:
        parser.add_argument(f"--{point_name}", type=float, nargs=2, metavar=("TARGET", "TOLERANCE"), default=None)
    args = parser.parse_args()

    operating_points = dict(DEFAULT_OPERATING_POINTS)
    for point_name in DEFAULT_OPERATING_POINTS:
        override = getattr(args, point_name)
        if override is not None:
            operating_points[point_name] = tuple(override)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for the re-measurement step (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    log_root = Path("logs/rsl_rl")
    experiment_name = "go1_velocity_term"
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    out_path = outputs_dir / "go1_selected_checkpoints.json"
    all_results: dict[str, dict] = json.loads(out_path.read_text()) if out_path.exists() else {}

    for seed in args.seeds:
        run_dir = find_latest_run_dir(experiment_name, seed, log_root)
        eval_history_path = run_dir / "eval_history.json"
        if not eval_history_path.exists():
            raise FileNotFoundError(f"{eval_history_path} not found -- did go1_train.py run to completion?")
        history = json.loads(eval_history_path.read_text())

        seed_results: dict[str, dict] = {}
        for point_name, (target, tolerance) in operating_points.items():
            selected = (
                select_any_for_smoke(history)
                if args.smoke
                else select_from_history_by_field(history, target, tolerance, SELECTION_FIELD)
            )
            if selected is None:
                available = [(r["iteration"], r[SELECTION_FIELD]["mean_episode_length"]) for r in history]
                raise RuntimeError(
                    f"No checkpoint for seed={seed} qualifies for operating point '{point_name}' "
                    f"(target={target} +/- {tolerance}). Available (iteration, {SELECTION_FIELD}.mean_episode_length): "
                    f"{available}. Pass --{point_name} TARGET TOLERANCE to adjust the window for this "
                    "task's actual eval curve -- see this script's module docstring."
                )

            iteration = selected["iteration"]
            checkpoint_path = run_dir / f"model_{iteration}.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Selected checkpoint {checkpoint_path} does not exist on disk.")

            # stochastic_output=True, NOT cartpole's deterministic
            # re-measurement -- see module docstring: selection is on the
            # sampled-action distribution here, so the re-measurement must
            # match it to be a meaningful re-measurement of the SAME thing.
            remeasure_seed = seed + REMEASURE_SEED_OFFSET
            runner = build_go1_runner_from_checkpoint(str(checkpoint_path), seed=seed, eval_num_envs=256, device=device)
            remeasurement = runner.evaluate(seed=remeasure_seed, stochastic_output=True)

            seed_results[point_name] = {
                "seed": seed,
                "iteration": iteration,
                "checkpoint_path": str(checkpoint_path),
                "selection_field": SELECTION_FIELD,
                "selection_measurement": selected[SELECTION_FIELD],
                "remeasurement_seed": remeasure_seed,
                "remeasurement": remeasurement,
            }
            print(
                f"[go1-select] seed={seed} point={point_name} it={iteration} "
                f"selection_len={selected[SELECTION_FIELD]['mean_episode_length']:.1f} "
                f"remeasured_len={remeasurement['mean_episode_length']:.1f}"
            )

        all_results[str(seed)] = seed_results
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"  wrote {out_path} (seed {seed} done)")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
