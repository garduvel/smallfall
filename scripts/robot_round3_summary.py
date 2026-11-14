"""Round 3 (G1, settle lock) result on the test bank (DEVIATIONS.md
2026-11-14). CPU only.

Input: the test-bank compare npz and the round-3 choice file. Arms:
  protective            the original SmallFall, no lock
  walker, limp
  <reward>_s<seed>      round-3 runs, trained and replayed with the lock
  <name> / <name>_lock  pairs: an existing policy without / with the lock
                        added at test time only (no retraining); the
                        original SmallFall's pair is protective / sf_lock
  r2s2                  round 2's chosen policy (hullgrow_s2), no lock

Reports every policy (size score vs limp, ground covered, rest rate, forces,
safe, share of falls that needed the lock), each reward's runs, the effect
of adding the lock to each existing policy, and the pre-registered check
for the chosen round-3 policy.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

_S = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_round2_summary", _S / "robot_round2_summary.py")
r2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r2)
sel, abl = r2.sel, r2.abl

MIN_SETTLED = 0.99
ROUND2_POLICY = "r2s2"
NONINFERIOR_RATIO = 1.05


def lock_rows(npz, arms) -> dict[str, dict[str, float]]:
    out = {}
    for a in arms:
        if f"{a}__locked" in npz.files:
            locked = npz[f"{a}__locked"]
            t = npz[f"{a}__lock_s"][locked > 0]
            out[a] = {"locked_share": float(locked.mean()), "by_timer_share": float(npz[f"{a}__locked_by_timer"].mean()),
                      "lock_s_median": float(np.median(t)) if len(t) else None}
    return out


def summarize(per_arm, chosen: str, locks: dict) -> dict:
    base = r2.summarize(per_arm, chosen, baseline="protective")
    rows = base["policies"]
    for a, r in rows.items():
        r.update(locks.get(a, {"locked_share": 0.0, "by_timer_share": 0.0, "lock_s_median": None}))
    floors = abl.size_floors(per_arm)
    score = {a: abl.per_fall_size_score(per_arm[a], per_arm["limp"], floors) for a in per_arm}
    pairs = {}
    for a in per_arm:
        u = {"sf_lock": "protective"}.get(a, a[:-5])  # the original SmallFall's unlocked arm is `protective`
        if a.endswith("_lock") and u in per_arm:
            d = np.log(score[a]) - np.log(score[u])
            pairs[u] = {"settled": rows[u]["settled"], "settled_lock": rows[a]["settled"],
                        "size_score": rows[u]["size_score"], "size_score_lock": rows[a]["size_score"],
                        "size_ratio_lock_to_none": float(np.exp(np.mean(d))),
                        "locked_share": rows[a]["locked_share"],
                        "any_force": rows[u]["any_force_median"], "any_force_lock": rows[a]["any_force_median"]}
    c = base["check"]
    d = np.log(score[chosen]) - np.log(score[ROUND2_POLICY])
    c["size_score_ratio_to_round2"] = float(np.exp(np.mean(d)))
    c["round2_wilcoxon_p"] = float(wilcoxon(score[chosen], score[ROUND2_POLICY]).pvalue)
    c["passes"]["comes_to_rest"] = rows[chosen]["settled"] >= MIN_SETTLED
    c["passes"]["not_bigger_than_round2"] = c["size_score_ratio_to_round2"] <= NONINFERIOR_RATIO
    c["pass"] = all(c["passes"].values())
    c["rule"] = (f"rest >= {MIN_SETTLED}; safe; smaller than the original SmallFall (paired, p < 0.05) and limp; "
                 f"ground covered below limp and walker; size score <= {NONINFERIOR_RATIO} x round 2's {ROUND2_POLICY}")
    return base | {"lock_pairs": pairs}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-npz", required=True)
    p.add_argument("--choice-json", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = np.load(args.test_npz)
    arms = sorted({k.split("__")[0] for k in data.files if "__size__" in k})
    for need in ("protective", "walker", "limp", ROUND2_POLICY):
        if need not in arms:
            raise KeyError(f"{args.test_npz} has no arm {need!r}")
    per_arm = sel.load_arms(args.test_npz, arms)
    chosen = json.loads(Path(args.choice_json).read_text())["chosen_policy"]
    out = summarize(per_arm, chosen, lock_rows(data, arms))
    Path(args.output).write_text(json.dumps(out, indent=2))

    H = r2.HEIGHT_M
    print(f"\n{'policy':18s}{'size score':>11s}{'ground (H)':>11s}{'at rest':>9s}{'locked':>8s}{'trunk N':>9s}{'any N':>8s}  safe")
    for a, r in sorted(out["policies"].items(), key=lambda t: t[1]["size_score"]):
        print(f"{a:18s}{r['size_score']:11.3f}{r['ground_covered_median'] / H:11.3f}{r['settled']:9.3f}{r['locked_share']:8.0%}"
              f"{r['trunk_force_median']:9.0f}{r['any_force_median']:8.0f}  {r['safe']}{'  <- chosen' if a == chosen else ''}")
    print("\nper reward (round-3 runs, trained with the lock):")
    for g, r in out["by_reward"].items():
        print(f"  {g:10s} size score mean {r['size_score_mean']:.3f} (runs {r['size_score_min']:.3f}-{r['size_score_max']:.3f}),"
              f" at rest {r['settled_mean']:.3f}")
    print("\nadding the lock at test time only (no retraining):")
    for u, r in out["lock_pairs"].items():
        print(f"  {u:16s} at rest {r['settled']:.3f} -> {r['settled_lock']:.3f}; size x{r['size_ratio_lock_to_none']:.3f}; "
              f"lock used in {r['locked_share']:.0%}; any-body force {r['any_force']:.0f} -> {r['any_force_lock']:.0f} N")
    c = out["check"]
    print(f"\nchosen {chosen}: vs original SmallFall size x{c['size_score_ratio_to_baseline']:.3f} (p={c['wilcoxon_p']:.2g}); "
          f"vs round 2's {ROUND2_POLICY} x{c['size_score_ratio_to_round2']:.3f} (p={c['round2_wilcoxon_p']:.2g})")
    for k, v in c["passes"].items():
        print(f"  {k:38s} {'PASS' if v else 'FAIL'}")
    print(f"ROUND 3 VERDICT: {'PASS' if c['pass'] else 'FAIL'}")
    print(f"[round3] wrote {args.output}")


if __name__ == "__main__":
    main()
