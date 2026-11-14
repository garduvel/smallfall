"""Round 2 (G1) result on the test bank (DEVIATIONS.md 2026-11-14). CPU only.

Input: the test-bank compare npz (arms: the original SmallFall as
`protective`, every round-2 run's chosen checkpoint as <reward>_s<seed>,
walker, limp) and the round-2 choice file (which reward and run were
chosen on the selection bank, before the test).

Reports per policy: size score vs limp, ground covered, rest rate, trunk and
any-body force, safe; per reward: mean and spread over its runs; and the
pre-registered check for the chosen policy against the original SmallFall
(paired per fall).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_S = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_select_safe_small", _S / "robot_select_safe_small.py")
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)
abl = sel.abl

HEIGHT_M = 0.76


def summarize(per_arm, chosen: str, baseline: str = "protective") -> dict:
    floors = abl.size_floors(per_arm)
    w, ref = abl.safety_row(per_arm["walker"]), abl.safety_row(per_arm["limp"])
    rows, per_fall = {}, {}
    for a in per_arm:
        s = abl.per_fall_size_score(per_arm[a], per_arm["limp"], floors)
        per_fall[a] = s
        r = abl.safety_row(per_arm[a]) | {"size_score": float(np.exp(np.mean(np.log(s))))}
        r["safe"] = r["trunk_force_median"] <= w["trunk_force_median"] and r["any_force_median"] < ref["any_force_median"]
        rows[a] = r
    groups = {}
    for a in per_arm:
        m = re.fullmatch(r"(.+)_s(\d+)", a)
        if m:
            groups.setdefault(m.group(1), []).append(a)
    by_reward = {g: {"runs": sorted(v), "size_score_mean": float(np.mean([rows[a]["size_score"] for a in v])),
                     "size_score_min": float(min(rows[a]["size_score"] for a in v)),
                     "size_score_max": float(max(rows[a]["size_score"] for a in v)),
                     "settled_mean": float(np.mean([rows[a]["settled"] for a in v]))} for g, v in groups.items()}
    d = np.log(per_fall[chosen]) - np.log(per_fall[baseline])
    c, b = rows[chosen], rows[baseline]
    check = {
        "chosen": chosen, "baseline": baseline,
        "size_score_ratio_to_baseline": float(np.exp(np.mean(d))),
        "smaller_share_per_fall": float((d < 0).mean()),
        "wilcoxon_p": float(wilcoxon(per_fall[chosen], per_fall[baseline]).pvalue),
        "ground_covered_wilcoxon_p": float(wilcoxon(per_arm[chosen]["size__envelope_growth_m"],
                                                    per_arm[baseline]["size__envelope_growth_m"]).pvalue),
    }
    check["passes"] = {
        "smaller_than_smallfall": check["size_score_ratio_to_baseline"] < 1 and check["wilcoxon_p"] < 0.05,
        "smaller_than_limp": c["size_score"] < 1,
        "ground_covered_below_limp_and_walker": c["ground_covered_median"] < min(ref["ground_covered_median"],
                                                                                 w["ground_covered_median"]),
        "comes_to_rest": c["settled"] >= abl.SAFE_MIN_SETTLED,
        "safe": c["safe"],
    }
    check["pass"] = all(check["passes"].values())
    return {"policies": rows, "by_reward": by_reward, "check": check}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-npz", required=True)
    p.add_argument("--choice-json", required=True, help="round-2 choice: {'chosen_policy': '<reward>_s<seed>'}")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = np.load(args.test_npz)
    arms = sorted({k.split("__")[0] for k in data.files if "__size__" in k})
    per_arm = sel.load_arms(args.test_npz, arms)
    chosen = json.loads(Path(args.choice_json).read_text())["chosen_policy"]
    out = summarize(per_arm, chosen)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"\n{'policy':16s}{'size score':>11s}{'ground (H)':>11s}{'at rest':>9s}{'trunk N':>9s}{'any N':>8s}  safe")
    for a, r in sorted(out["policies"].items(), key=lambda t: t[1]["size_score"]):
        mark = "  <- chosen" if a == chosen else ""
        print(f"{a:16s}{r['size_score']:11.3f}{r['ground_covered_median'] / HEIGHT_M:11.3f}{r['settled']:9.3f}"
              f"{r['trunk_force_median']:9.0f}{r['any_force_median']:8.0f}  {r['safe']}{mark}")
    print("\nper reward (its runs):")
    for g, r in out["by_reward"].items():
        print(f"  {g:10s} size score mean {r['size_score_mean']:.3f} (runs {r['size_score_min']:.3f}-{r['size_score_max']:.3f}),"
              f" at rest {r['settled_mean']:.3f}")
    c = out["check"]
    print(f"\nchosen {chosen} vs original SmallFall: size score x{c['size_score_ratio_to_baseline']:.3f}, "
          f"smaller on {c['smaller_share_per_fall']:.0%} of falls, p={c['wilcoxon_p']:.2g}")
    for k, v in c["passes"].items():
        print(f"  {k:38s} {'PASS' if v else 'FAIL'}")
    print(f"ROUND 2 VERDICT: {'PASS' if c['pass'] else 'FAIL'}")
    print(f"[round2] wrote {args.output}")


if __name__ == "__main__":
    main()
