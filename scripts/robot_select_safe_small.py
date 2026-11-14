"""Round 2 (G1) checkpoint choice for one training run: the smallest fall
that is still safe (rule: select_safe_small in
src/cartpole_term/protective/fall_size_ablation.py; DEVIATIONS.md
2026-11-14). CPU only, reads a selection-bank compare npz whose arms are the
candidates plus walker and limp.

    python scripts/robot_select_safe_small.py --compare-npz outputs/g1_r2sel_mix_s0.npz \\
        --candidate protective=ckpt_600.pt --candidate it300=ckpt_300.pt ... --output choice.json
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

FIELDS = ("settled", "peak_force_critical_n", "peak_force_any_n")


def load_arms(npz_path: str, arms) -> dict[str, dict[str, np.ndarray]]:
    data = np.load(npz_path)
    out = {}
    for a in arms:
        keys = [f"{a}__{f}" for f in FIELDS] + [f"{a}__size__{m}" for m in abl.FAMILY_MEASURES]
        missing = [k for k in keys if k not in data.files]
        if missing:
            raise KeyError(f"{npz_path} lacks {missing[:4]}")
        out[a] = {k.split("__", 1)[1]: data[k] for k in keys}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare-npz", required=True)
    p.add_argument("--candidate", action="append", required=True, metavar="ARM=CHECKPOINT")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    cands = dict(c.split("=", 1) for c in args.candidate)
    per_arm = load_arms(args.compare_npz, [*cands, "walker", "limp"])
    out = abl.select_safe_small(per_arm, list(cands))
    out["checkpoint"] = cands[out["selected"]]
    out["checkpoints"] = cands
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"  {'candidate':>11s}{'size score':>12s}{'at rest':>9s}{'trunk N':>9s}{'any N':>8s}  safe")
    for r in out["candidates"]:
        print(f"  {r['label']:>11s}{r['size_score']:12.3f}{r['settled']:9.3f}{r['trunk_force_median']:9.0f}"
              f"{r['any_force_median']:8.0f}  {r['safe']}{'' if r['eligible'] else '  (not eligible)'}")
    print(f"[select] chose {out['selected']} -> {out['checkpoint']} (size score {out['size_score']:.3f})"
          f"{'' if out['selected_is_eligible'] else '  NOT ELIGIBLE (fallback)'}")


if __name__ == "__main__":
    main()
