"""Pick the checkpoint of a policy trained with one fall-size measure, judged
by that same measure on a selection bank (reward ablation, DEVIATIONS.md
2026-11-13; rule: select_by_measure in protective/fall_size_ablation.py).

Input: a compare replay npz where the candidates are arms (the
--protective-checkpoint arm plus --extra-policy arms). Each candidate is
given as ARM=CHECKPOINT so the output names the chosen file.

    python scripts/robot_select_by_measure.py --compare-npz outputs/g1_rsel_rootpath.npz \\
        --measure root_path_m --candidate protective=ckpt_300.pt --candidate it200=ckpt_200.pt \\
        --output outputs/g1_rsel_rootpath_choice.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parent.parent / "src" / "cartpole_term" / "protective" / "fall_size_ablation.py"
_spec = importlib.util.spec_from_file_location("_fall_size_ablation", _P)
abl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abl)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare-npz", required=True)
    p.add_argument("--measure", required=True, choices=tuple(abl.REWARD_TAGS))
    p.add_argument("--candidate", action="append", required=True, metavar="ARM=CHECKPOINT")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cands = dict(c.split("=", 1) for c in args.candidate)
    data = np.load(args.compare_npz)
    per_arm = {}
    for arm in cands:
        keys = (f"{arm}__settled", f"{arm}__size__{args.measure}")
        missing = [k for k in keys if k not in data.files]
        if missing:
            raise KeyError(f"{args.compare_npz} lacks {missing}")
        per_arm[arm] = {"settled": data[keys[0]], f"size__{args.measure}": data[keys[1]]}
    out = abl.select_by_measure(per_arm, list(cands), args.measure)
    out["checkpoint"] = cands[out["selected"]]
    out["checkpoints"] = cands
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    for r in out["candidates"]:
        print(f"  {r['label']:>12s}: came to rest {r['settled']:.3f}, median {args.measure} {r['median']:.4f}"
              f"{'' if r['eligible'] else '  (comes to rest < 99%)'}")
    print(f"[select] {args.measure}: chose {out['selected']} -> {out['checkpoint']}"
          f"{'' if out['selected_is_eligible'] else '  (NOT eligible: none comes to rest in 99% of falls)'}")


if __name__ == "__main__":
    main()
