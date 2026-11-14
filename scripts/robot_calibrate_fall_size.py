"""Size one fall-size reward weight per measure for the reward ablation
(DEVIATIONS.md 2026-11-13). The rule is fall_size_weights() in
src/cartpole_term/protective/calibration.py.

Input: a short training run made with --log-fall-sizes (every measure
logged as Episode_Metrics/fallsize_<measure>, its episode total) and the
robot's calibrated weights (robot_calibrate_protective.py output), whose
fall_envelope_growth weight is the anchor.

    python scripts/robot_calibrate_fall_size.py --run-dir logs/rsl_rl/g1_protective_fall/sizecal \\
        --weights-json outputs/g1_protective_weights.json --output outputs/g1_fall_size_weights.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

_PROTECTIVE = Path(__file__).resolve().parent.parent / "src" / "cartpole_term" / "protective"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_protective_{name}", _PROTECTIVE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cal = _load("calibration")
# fall_size.py needs torch; the names are repeated here and checked by a test.
SIZE_METRICS = (
    "envelope_growth_m", "max_reach_m", "root_travel_end_m", "root_travel_max_m", "root_path_m",
    "centroid_travel_end_m", "swept_hull_area_m2", "hull_growth_m2", "contact_reach_m", "touched_area_m2",
)


def read_sums(run_dir: str) -> dict[str, float]:
    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    acc.Reload()
    lo, hi = _cal.CALIBRATION_ITERATIONS
    sums = {}
    for m in SIZE_METRICS:
        tag = f"Episode_Metrics/fallsize_{m}"
        if tag not in acc.Tags()["scalars"]:
            raise RuntimeError(f"[size-calibrate] {tag} not logged: train with --log-fall-sizes")
        vals = [e.value for e in acc.Scalars(tag) if lo <= e.step <= hi]
        if len(vals) < 6:
            raise RuntimeError(f"[size-calibrate] {tag}: only {len(vals)} values in iterations {lo}-{hi}")
        sums[m] = sum(vals) / len(vals)
    return sums


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--weights-json", required=True, help="robot_calibrate_protective.py output")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    envelope_weight = json.loads(Path(args.weights_json).read_text())["weights"]["fall_envelope_growth"]
    out = _cal.fall_size_weights(read_sums(args.run_dir), envelope_weight)
    out["run_dir"] = args.run_dir
    out["iterations"] = list(_cal.CALIBRATION_ITERATIONS)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"{'measure':24s}{'episode total':>15s}{'factor':>9s}{'weight':>12s}")
    for m, r in out["terms"].items():
        note = " (clamped)" if r["clamped"] else (" (kept)" if r["kept_base"] else "")
        print(f"{m:24s}{r['episode_sum']:15.4f}{r['factor']:9.3f}{r['weight']:12.4g}{note}")
    print(f"[size-calibrate] wrote {args.output}")


if __name__ == "__main__":
    main()
