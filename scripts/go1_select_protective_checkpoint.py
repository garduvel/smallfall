"""Round 8: mechanically pick one protective checkpoint from several
scripts/go1_protective_compare.py result files produced on a SELECTION bank.
The rule lives in src/cartpole_term/protective/compare.py (select_checkpoint,
unit-tested) and was fixed before any round-8 data existed (DEVIATIONS.md).
The selected checkpoint is then tested once on a different, fresh bank.

Usage:
    python scripts/go1_select_protective_checkpoint.py \
        --compare-glob "outputs/round8_select_*.json" \
        --output outputs/round8_selected_checkpoint.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cartpole_term.protective.compare import select_checkpoint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare-glob", required=True, help='e.g. "outputs/round8_select_*.json"; the part matched by * is the checkpoint label')
    p.add_argument("--output", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(args.compare_glob))
    if not files:
        raise FileNotFoundError(f"[select] no files match {args.compare_glob}")
    prefix = Path(args.compare_glob.split("*")[0]).name
    summaries = {}
    for f in files:
        label = Path(f).stem[len(prefix):]  # e.g. "8b_it150"
        summaries[label] = json.loads(Path(f).read_text())["summaries"]

    result = select_checkpoint(summaries)
    result["compare_files"] = files
    Path(args.output).write_text(json.dumps(result, indent=2))

    print(f"{'label':>12s}{'settled':>9s}{'ground m':>10s}{'any-body N':>12s}  beats walker on (crit, any, torque, ground)  eligible  preferred")
    for r in result["candidates"]:
        b = r["beats_baseline"]
        flags = "  ".join(str(b[k])[0] for k in ("peak_force_critical_n", "peak_force_any_n", "peak_torque_nm", "envelope_growth_m"))
        print(f"{r['label']:>12s}{r['settled']:9.3f}{r['envelope_growth_median']:10.3f}{r['any_force_median']:12.1f}  "
              f"{flags:42s}  {str(r['eligible']):8s}  {r['preferred']}")
    print(f"[select] SELECTED {result['selected_label']} (eligible: {result['selected_is_eligible']}, "
          f"also reduces any-body force: {result['selected_also_reduces_any_force']})")
    print(f"[select] wrote {args.output}")


if __name__ == "__main__":
    main()
