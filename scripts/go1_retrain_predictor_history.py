"""New, standalone script (phase-2, see DEVIATIONS.md): retrain the Go1
wild-push termination predictor with history-stacked inputs (CLAUDE.md
S8.1's `--predictor.history H`, never actually implemented anywhere in this
project until now -- scripts/eval_predictor.py raises NotImplementedError
for history != 1 to this day) on top of the ALREADY-COLLECTED
1M-plus-transition frozen dataset (outputs/go1_wildpush_frozen_seed0.npz) --
no new GPU collection needed. Reason for building this: the live lead-time
measurement (scripts/go1_collect_flagged_states.py) found a genuine
worst-case problem -- roughly a fifth of real falls give the H=1 (single-
frame) predictor barely more warning than the fall itself takes to happen.
A single frame can't distinguish "about to fall fast" from "about to
recover" until very late because it has no access to a trend; stacking the
last H frames gives the model that trend directly (mirrors STATE-NAV's own
gait-cycle aggregation, the reference paper CLAUDE.md's whole predictor
design defers to).

History-stacking implementation: src/cartpole_term/predictor/history.py,
unit-tested in-sandbox (tests/test_history_stacking.py, 6/6 pass, no GPU
needed) before ever touching this GPU-only script -- same "verify what you
can locally first" discipline as CLAUDE.md's own labelling tests.

Reuses, unmodified: cartpole_term.predictor.history.build_history_stacked_
dataset (new, tested), go1_wild_push_predictor.train_and_evaluate_predictor
(existing, now takes an optional `tag` so this doesn't overwrite the locked
H=1 predictor's output files -- see that function's docstring).

Default history=8 (160ms at Go1's 50Hz): grounded in the locked push
mechanism's own `duration_s=(0.1, 0.3)` range (5-15 steps) rather than
guessed -- 8 sits inside that range, long enough to see a push's onset
developing, short enough to stay a "recent trend" window rather than a full
gait cycle (which this project doesn't need here; the goal is catching a
fast-developing fall early, not periodic gait context).

Run on the GPU machine, from the repo root, after
scripts/go1_wild_push_predictor.py has already produced
outputs/go1_wildpush_frozen_seed0.npz (no need to re-run collection):

    python scripts/go1_retrain_predictor_history.py
    python scripts/go1_retrain_predictor_history.py --history 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from cartpole_term.predictor.history import build_history_stacked_dataset  # noqa: E402
from go1_wild_push_predictor import PREDICTOR_HIDDEN, train_and_evaluate_predictor  # noqa: E402

DEFAULT_HISTORY = 8  # 160ms at 50Hz -- see module docstring for the push-duration grounding
DEFAULT_FROZEN_NPZ = "outputs/go1_wildpush_frozen_seed0.npz"
DEFAULT_FROZEN_JSON = "outputs/go1_wildpush_frozen_seed0.json"
STEP_SECONDS = 0.02  # Go1: decimation=4, timestep=0.005 -> 50 Hz


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen-npz", dest="frozen_npz", type=str, default=DEFAULT_FROZEN_NPZ)
    parser.add_argument("--frozen-json", dest="frozen_json", type=str, default=DEFAULT_FROZEN_JSON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history", type=int, default=DEFAULT_HISTORY)
    parser.add_argument("--predictor-hidden", dest="predictor_hidden", type=str, default=",".join(str(h) for h in PREDICTOR_HIDDEN))
    args = parser.parse_args()

    h1, h2 = (int(p) for p in args.predictor_hidden.split(","))
    hidden = (h1, h2)

    print(f"[retrain-history] loading {args.frozen_npz} ...")
    npz = np.load(args.frozen_npz)
    dataset = {k: npz[k] for k in npz.files}

    # The .json sidecar (collection metadata: base rate, episode counts,
    # wall-clock time, etc.) is informational only -- everything actually
    # needed to train/evaluate lives in the .npz. Optional: an earlier
    # notebook's download cell never pulled this file, so it may not exist
    # locally at all.
    frozen_json_path = Path(args.frozen_json)
    if frozen_json_path.exists():
        collection_sidecar = json.loads(frozen_json_path.read_text())
    else:
        print(f"[retrain-history] {args.frozen_json} not found -- proceeding without it (informational only).")
        collection_sidecar = {"note": f"collection sidecar ({args.frozen_json}) was not available"}
    print(
        f"[retrain-history] loaded {len(dataset['label'])} transitions, "
        f"base_rate={float(dataset['label'].mean()):.4f}, history={args.history}"
    )

    stacked = build_history_stacked_dataset(dataset, history=args.history)
    n_dropped = len(dataset["label"]) - len(stacked["label"])
    print(
        f"[retrain-history] history-stacked: {len(stacked['label'])} rows kept, "
        f"{n_dropped} dropped (needed a frame before their own episode's start), "
        f"input_dim={stacked['obs'].shape[1] + stacked['action'].shape[1]}"
    )

    metrics = train_and_evaluate_predictor(
        stacked,
        sidecar={**collection_sidecar, "note": "history-stacked derivative of the original collection; see 'history' field"},
        seed=args.seed,
        hidden=hidden,
        step_seconds=STEP_SECONDS,
        tag=f"_history{args.history}",
    )
    metrics["history"] = args.history
    metrics["rows_dropped_for_history_boundary"] = n_dropped

    outputs_dir = Path("outputs")
    metrics_path = outputs_dir / f"go1_wildpush_metrics_history{args.history}_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(
        f"[retrain-history] history={args.history}: test_base_rate={metrics['split_base_rates']['test']:.4f} "
        f"AUPRC={metrics['mlp']['auprc']:.4f} (baseline={metrics['mlp']['auprc_baseline']:.4f}) "
        f"AUROC={metrics['mlp']['auroc']:.4f} "
        f"precision={metrics['mlp']['precision']:.4f} recall={metrics['mlp']['recall']:.4f} "
        f"median_lead={metrics['mlp_lead_time']['median_lead_steps']:.1f} steps "
        f"({metrics['mlp_lead_time']['median_lead_seconds']:.2f}s)"
    )
    print("[retrain-history] compare against the H=1 baseline in outputs/go1_wildpush_metrics_seed0.json:")
    baseline = json.loads(Path("outputs/go1_wildpush_metrics_seed0.json").read_text())
    print(
        f"[retrain-history] history=1  (baseline): AUPRC={baseline['mlp']['auprc']:.4f} "
        f"AUROC={baseline['mlp']['auroc']:.4f} precision={baseline['mlp']['precision']:.4f} "
        f"recall={baseline['mlp']['recall']:.4f} "
        f"median_lead={baseline['mlp_lead_time']['median_lead_steps']:.1f} steps "
        f"({baseline['mlp_lead_time']['median_lead_seconds']:.2f}s)"
    )
    print(f"  wrote {metrics_path}")
    print(f"  wrote {metrics['model_path']}")


if __name__ == "__main__":
    main()
