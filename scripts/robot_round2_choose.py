"""Round 2 (G1): choose the reward and the one policy, from the selection
bank only, BEFORE the test (DEVIATIONS.md 2026-11-14).

Rule: each run already has its checkpoint (robot_select_safe_small.py).
Rewards whose runs all have an eligible checkpoint rank first; within a
group, the lowest mean size score over the reward's runs wins. The chosen
policy is that reward's run with the lowest size score (eligible first).

    python scripts/robot_round2_choose.py --choices "outputs/g1_r2sel_*_choice.json" --output outputs/g1_r2_choice.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path


def choose(choices: dict[str, dict]) -> dict:
    """choices: {'<reward>_s<seed>': select_safe_small output}."""
    groups: dict[str, list[str]] = {}
    for run in choices:
        m = re.fullmatch(r"(.+)_s(\d+)", run)
        if not m:
            raise ValueError(f"run label {run!r} is not <reward>_s<seed>")
        groups.setdefault(m.group(1), []).append(run)
    rewards = []
    for g, runs in sorted(groups.items()):
        scores = [choices[r]["size_score"] for r in runs]
        rewards.append({"reward": g, "runs": sorted(runs), "all_eligible": all(choices[r]["selected_is_eligible"] for r in runs),
                        "mean_size_score": sum(scores) / len(scores)})
    best = min(rewards, key=lambda r: (not r["all_eligible"], r["mean_size_score"], r["reward"]))
    run = min(best["runs"], key=lambda r: (not choices[r]["selected_is_eligible"], choices[r]["size_score"], r))
    return {"chosen_reward": best["reward"], "chosen_policy": run, "checkpoint": choices[run]["checkpoint"],
            "rewards": rewards, "runs": {r: {k: choices[r][k] for k in ("selected", "selected_is_eligible", "size_score", "checkpoint")}
                                         for r in sorted(choices)}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--choices", required=True, help="glob of g1_r2sel_<reward>_s<seed>_choice.json")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    files = sorted(glob.glob(args.choices))
    if not files:
        raise FileNotFoundError(args.choices)
    choices = {}
    for f in files:
        m = re.search(r"r2sel_(.+)_choice\.json$", f)
        choices[m.group(1)] = json.loads(Path(f).read_text())
    out = choose(choices)
    Path(args.output).write_text(json.dumps(out, indent=2))
    for r in sorted(out["rewards"], key=lambda r: r["mean_size_score"]):
        print(f"  {r['reward']:10s} mean size score {r['mean_size_score']:.3f} over {len(r['runs'])} runs"
              f"{'' if r['all_eligible'] else '  (a run has no eligible checkpoint)'}")
    print(f"[choose] reward {out['chosen_reward']}, policy {out['chosen_policy']} -> {out['checkpoint']}")


if __name__ == "__main__":
    main()
