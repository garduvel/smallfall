"""Which policy falls smallest overall, under ten ways of measuring fall size
(metric ablation, DEVIATIONS.md 2026-11-13). CPU only: reads the per-fall
arrays a compare replay saved (scripts/go1_protective_compare.py writes
size__<metric> for every policy, including any --extra-policy).

    python scripts/robot_fall_size_ablation.py --compare-npz outputs/g1_size_ablation.npz \\
        --robot g1 --output outputs/g1_size_ablation_report.json

The measures are in src/cartpole_term/protective/fall_size.py, the three
combining methods and the overall rule in fall_size_ablation.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src" / "cartpole_term"


def _load(rel: str, name: str):
    """By path: the package import would pull in mjlab, which this step does not need."""
    spec = importlib.util.spec_from_file_location(name, _SRC / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


abl = _load("protective/fall_size_ablation.py", "_fall_size_ablation")
# fall_size.py imports torch; only its metric names are needed here.
SIZE_METRICS = (
    "envelope_growth_m", "max_reach_m", "root_travel_end_m", "root_travel_max_m", "root_path_m",
    "centroid_travel_end_m", "swept_hull_area_m2", "hull_growth_m2", "contact_reach_m", "touched_area_m2",
)
AREA_METRICS = ("swept_hull_area_m2", "hull_growth_m2", "touched_area_m2")
START_HEIGHT_M = {"go1": 0.278, "g1": 0.76}  # robot_profile / DEVIATIONS.md
SHORT = {
    "envelope_growth_m": "ground covered", "max_reach_m": "max reach", "root_travel_end_m": "root end",
    "root_travel_max_m": "root max", "root_path_m": "root path", "centroid_travel_end_m": "centroid end",
    "swept_hull_area_m2": "hull area", "hull_growth_m2": "hull growth", "contact_reach_m": "contact reach",
    "touched_area_m2": "touched area",
}


def load_per_fall(npz_path: str, policies=None) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(npz_path)
    found = sorted({k.split("__", 1)[0] for k in data.files if "__size__" in k})
    if not found:
        raise ValueError(f"{npz_path} has no size__ arrays: re-run the compare replay with the current code")
    policies = list(policies or found)
    missing = [f"{p}__size__{m}" for p in policies for m in SIZE_METRICS if f"{p}__size__{m}" not in data.files]
    if missing:
        raise ValueError(f"{npz_path} is missing {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return {p: {m: data[f"{p}__size__{m}"] for m in SIZE_METRICS} for p in policies}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare-npz", required=True)
    p.add_argument("--robot", choices=tuple(START_HEIGHT_M), required=True)
    p.add_argument("--policies", nargs="*", default=None, help="default: every policy in the npz")
    p.add_argument("--reference", default="limp")
    p.add_argument("--bootstrap", type=int, default=abl.BOOTSTRAP_RESAMPLES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    per_fall = load_per_fall(args.compare_npz, args.policies)
    h = START_HEIGHT_M[args.robot]
    scale = {m: (h * h if m in AREA_METRICS else h) for m in SIZE_METRICS}
    rep = abl.analyse(per_fall, SIZE_METRICS, reference=args.reference, bootstrap=args.bootstrap,
                      seed=args.seed, scale=scale)
    radii = np.round(np.arange(0.0, 3.01, 0.05), 2)
    rep["reach_curves"] = {"radii_m": radii.tolist(),
                           **{f"{pol}|{m}": abl.exceedance_curve(per_fall[pol][m], radii)
                              for pol in rep["policies"] for m in ("max_reach_m", "contact_reach_m")}}
    rep["trained_on"] = abl.trained_on_report(rep)
    rep["config"] = vars(args) | {"start_height_m": h, "scale_note": "medians are divided by start height "
                                  "(areas by its square); rankings do not depend on it"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rep, indent=2))

    pols = rep["policies"]
    w = max(12, *(len(x) + 2 for x in pols))
    print(f"\n[size] {rep['num_falls']} falls; medians in robot heights (areas in heights^2); lower = smaller")
    print(f"{'measure':16s}" + "".join(f"{x:>{w}s}" for x in pols) + "   smallest")
    for m in SIZE_METRICS:
        print(f"{SHORT[m]:16s}" + "".join(f"{rep['medians'][x][m]:{w}.3f}" for x in pols)
              + f"   {rep['smallest_by_metric'][m]}")
    print(f"\n[size] do the measures rank the policies the same way? mean Kendall tau "
          f"{rep['mean_ranking_kendall_tau']:.2f} (1 = identical rankings)")
    firsts = sorted(set(rep["smallest_by_metric"].values()))
    print(f"[size] policies that come out smallest on at least one measure: {firsts}")
    print(f"\n[size] A. dominance: not beaten on every measure by anyone: {rep['A_dominance']['undominated']}")
    print(f"[size] B. mean rank (1 = always smallest): "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(rep['B_mean_rank']['scores'].items(), key=lambda t: t[1])))
    print(f"[size] C. size relative to {args.reference} (geometric mean): "
          + ", ".join(f"{k} {v:.2f}" for k, v in sorted(rep['C_ratio_to_reference']['scores'].items(), key=lambda t: t[1])))
    o, b = rep["overall"], rep["bootstrap"]
    print(f"\n[size] OVERALL: {o['answer']} ({o['note']})")
    print(f"[size] resampling the falls {b['resamples']}x: B picks "
          + ", ".join(f"{k} {v:.0%}" for k, v in b["B_win_share"].items() if v > 0)
          + "; C picks " + ", ".join(f"{k} {v:.0%}" for k, v in b["C_win_share"].items() if v > 0)
          + f"; all three agree {b['methods_agree_share']:.0%}")
    t = rep["trained_on"]
    if t:
        cols = list(SIZE_METRICS)
        print(f"\n[size] REWARD ABLATION: median size relative to {args.reference} "
              f"(rows = the measure the policy was trained on; * = its own measure)")
        print(f"{'trained on':14s}" + "".join(f"{SHORT[m][:10]:>11s}" for m in cols))
        for pol, row in t["ratio_to_reference"].items():
            own = t["own_measure"][pol]["measure"]
            print(f"{pol:14s}" + "".join(f"{row[m]:10.2f}{'*' if m == own else ' '}" for m in cols))
        print(f"\n[size] policies smallest (among the trained ones) on the measure they were trained on: "
              f"{t['wins_own_measure']} of {t['num_trained']}")
        for pol, r in t["own_measure"].items():
            if r["rank_among_trained"] != 1:
                print(f"  {pol}: rank {r['rank_among_trained']} of {r['of']} on its own measure "
                      f"(smallest there: {r['smallest_trained_on_it']})")
    print(f"[size] wrote {args.output}")


if __name__ == "__main__":
    main()
