"""Re-size the protective reward weights for a robot from a short calibration
training run (G1 port, DEVIATIONS.md 2026-11-13). The rule is in
src/cartpole_term/protective/calibration.py.

The calibration run is scripts/go1_train_protective.py --robot <robot> with
the Go1 weights (no --weights-json) for at least 11 iterations. This reads its
TensorBoard events, averages each sized term's Episode_Reward over
iterations 0-10, and writes the new weights as JSON for --weights-json.

    python scripts/robot_calibrate_protective.py --run-dir logs/rsl_rl/g1_protective_fall/<run> \\
        --output outputs/g1_protective_weights.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import importlib.util  # noqa: E402

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

_PROTECTIVE = Path(__file__).resolve().parent.parent / "src" / "cartpole_term" / "protective"


def _load(name: str):
    """Load a protective module by path: going through the cartpole_term
    package would import mjlab, which this step does not need."""
    spec = importlib.util.spec_from_file_location(f"_protective_{name}", _PROTECTIVE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cal = _load("calibration")
CALIBRATED_TERMS = _cal.CALIBRATED_TERMS
CALIBRATION_ITERATIONS = _cal.CALIBRATION_ITERATIONS
GO1_REFERENCE_CONTRIBUTIONS = _cal.GO1_REFERENCE_CONTRIBUTIONS
calibrate = _cal.calibrate


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    base = _load("env_cfg").GO1_REWARD_WEIGHTS

    acc = EventAccumulator(args.run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    lo, hi = CALIBRATION_ITERATIONS
    measured = {}
    for term in CALIBRATED_TERMS:
        tag = f"Episode_Reward/{term}"
        if tag not in acc.Tags()["scalars"]:
            print(f"[calibrate] WARNING: {tag} not logged; keeping the Go1 weight")
            continue
        vals = [e.value for e in acc.Scalars(tag) if lo <= e.step <= hi]
        # The Go1 reference has all 11; tolerate a few iterations with no
        # finished episode (nothing logged) rather than failing.
        if len(vals) < 6:
            raise RuntimeError(f"[calibrate] {tag}: only {len(vals)} values in iterations {lo}-{hi}; "
                               "run the calibration training for at least 11 iterations")
        measured[term] = sum(vals) / len(vals)

    out = calibrate(measured, base)
    out["run_dir"] = args.run_dir
    out["iterations"] = [lo, hi]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"{'term':28s}{'Go1 contrib':>13s}{'measured':>12s}{'factor':>9s}{'Go1 weight':>13s}{'new weight':>13s}")
    for term, r in out["terms"].items():
        got = "missing" if r["measured_contribution"] is None else f"{r['measured_contribution']:.4f}"
        note = " (clamped)" if r["clamped"] else (" (kept)" if r["kept_base"] else "")
        print(f"{term:28s}{GO1_REFERENCE_CONTRIBUTIONS[term]:13.4f}{got:>12s}{r['factor']:9.3f}"
              f"{r['base_weight']:13.3g}{r['weight']:13.3g}{note}")
    print(f"[calibrate] wrote {args.output}")


if __name__ == "__main__":
    main()
