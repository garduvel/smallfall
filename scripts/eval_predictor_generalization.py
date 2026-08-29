"""Phase-2 tool (not part of CLAUDE.md's gated spec): test an already-trained,
frozen predictor's generalization against policy checkpoints it never trained
on -- every checkpoint saved during one training run, or checkpoints from a
completely different run altogether.

This is deliberately NOT online predictor training. The predictor's weights
are loaded once and never updated; nothing here feeds back into any policy
(CLAUDE.md S1's online-training scope-out and S12#4/#6 still apply -- this
script only adds inference-time *evaluation* of an existing predictor against
a *policy* checkpoint it has never seen). "Alongside a completely new
training run" means: after scripts/train.py has written its checkpoints to
logs/rsl_rl/cartpole_term/<run_dir>/model_*.pt (every --eval.interval
iterations per CLAUDE.md S5.3), point this script at that run_dir with
--log-dir. It evaluates the frozen predictor against each saved checkpoint in
turn, purely by rollout + inference; it does not hook into the training loop
itself, so there's no risk of the predictor influencing training (S12#6).

For each checkpoint: collect a fresh, smaller frozen dataset with
collect_frozen.collect_one (same rollout/labelling code as CLAUDE.md S7.3 --
sampled/stochastic actions, weight-freeze asserted), then run the ALREADY-
TRAINED predictor on 100% of it in inference mode only (no fit, no split --
none of this data was used to train the predictor, so there is no leakage to
guard against). Reports the same core metrics as eval_predictor.py.

Usage, from the repo root, on the GPU machine:

    # Every checkpoint saved during this run so far:
    python scripts/eval_predictor_generalization.py \\
        --predictor-weights outputs/predictor_target_seed0.pt \\
        --log-dir logs/rsl_rl/cartpole_term/2026-08-22_05-44-34_seed0

    # One-off, an explicit checkpoint from anywhere (e.g. a brand-new run):
    python scripts/eval_predictor_generalization.py \\
        --predictor-weights outputs/predictor_target_seed0.pt \\
        --checkpoint-file logs/rsl_rl/cartpole_term/<new_run_dir>/model_120.pt

    --smoke shrinks the rollout so this runs in under a minute for a sanity
    check (same spirit as CLAUDE.md S5.5's --smoke, not the same numbers --
    this script is outside that spec).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term  # noqa: F401,E402
from collect_frozen import collect_one  # noqa: E402
from eval_predictor import (  # noqa: E402
    base_rate,
    build_features,
    classification_metrics,
    feature_screening,
    median_detection_lead_time,
)

# Same targets as collect_frozen.py (CLAUDE.md S7.3) -- collection itself is
# cheap here (~15-20s per 1M transitions, per the gate-8 timings recorded in
# DEVIATIONS.md/README.md), so there's no reason for a per-checkpoint probe
# to be lighter-weight/less rigorous than the original gated datasets.
DEFAULT_TRANSITIONS = 1_000_000
DEFAULT_TERMINATED_EPISODES = 1_000
DEFAULT_NUM_ENVS = 4096
SMOKE_TRANSITIONS = 2_000
SMOKE_TERMINATED_EPISODES = 10
SMOKE_NUM_ENVS = 64
DEFAULT_HORIZON = 16
GEN_SEED_OFFSET = 700  # distinct from train/select/collect's own seed offsets

CHECKPOINT_ITER_RE = re.compile(r"model_(\d+)\.pt$")


def discover_checkpoints(log_dir: Path) -> list[Path]:
    ckpts = sorted(
        log_dir.glob("model_*.pt"),
        key=lambda p: int(CHECKPOINT_ITER_RE.search(p.name).group(1)),
    )
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt checkpoints found under {log_dir}")
    return ckpts


def checkpoint_iteration(path: Path) -> int:
    m = CHECKPOINT_ITER_RE.search(path.name)
    if not m:
        raise ValueError(f"can't parse iteration number out of checkpoint filename: {path.name}")
    return int(m.group(1))


def parse_hidden(spec: str) -> tuple[int, int]:
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"--predictor-hidden must be 'H1,H2' (two ints), got {spec!r}")
    return int(parts[0]), int(parts[1])


def load_frozen_predictor(weights_path: Path, history: int = 1, hidden: tuple[int, int] = (128, 128)):
    from cartpole_term.predictor.model import TerminationPredictor

    model = TerminationPredictor(history=history, hidden=hidden)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def evaluate_predictor_on_checkpoint(
    model,
    checkpoint_path: str,
    seed: int,
    num_envs: int,
    target_transitions: int,
    target_terminated: int,
    horizon: int,
    threshold: float,
    device: str,
) -> dict:
    dataset, sidecar = collect_one(
        checkpoint_path=checkpoint_path,
        seed=seed,
        operating_point="generalization_probe",
        num_envs=num_envs,
        target_transitions=target_transitions,
        target_terminated=target_terminated,
        horizon=horizon,
        device=device,
    )

    x = np.concatenate([dataset["obs"], dataset["action"]], axis=1).astype(np.float32)
    y = dataset["label"].astype(np.float32)

    with torch.no_grad():
        probs = torch.sigmoid(model(torch.as_tensor(x)).squeeze(-1)).numpy()

    metrics = classification_metrics(y, probs, threshold)
    lead_time = median_detection_lead_time(
        dataset["episode_id"], dataset["steps_to_termination"], y, probs, threshold
    )

    features = build_features(dataset["obs"], dataset["action"])
    screening = feature_screening(features, y.astype(np.int64))

    per_cause: dict[str, dict] = {}
    for cause in ("pole_angle", "cart_pos"):
        mask = (dataset["termination_cause"] == cause) | (y == 0)
        if not (dataset["termination_cause"] == cause).any():
            per_cause[cause] = {"note": "no positives with this cause in this probe", "n_positive": 0}
            continue
        y_c, p_c = y[mask], probs[mask]
        per_cause[cause] = {
            "n_positive": int((dataset["termination_cause"] == cause).sum()),
            "auprc": None,
        }
        if len(np.unique(y_c)) > 1:
            from sklearn.metrics import average_precision_score

            per_cause[cause]["auprc"] = float(average_precision_score(y_c, p_c))
        per_cause[cause]["lead_time"] = median_detection_lead_time(
            dataset["episode_id"][mask], dataset["steps_to_termination"][mask], y[mask], probs[mask], threshold
        )

    return {
        "checkpoint_path": checkpoint_path,
        "collection_sidecar": sidecar,
        "metrics": metrics,
        "lead_time": lead_time,
        "feature_screening": screening,
        "per_cause_breakdown": per_cause,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an already-trained, frozen termination predictor's generalization "
            "against policy checkpoints it never trained on (phase-2 tool, outside CLAUDE.md's "
            "gated spec -- see this file's module docstring)."
        )
    )
    parser.add_argument("--predictor-weights", type=str, required=True, help="Path to a saved predictor .pt (e.g. outputs/predictor_target_seed0.pt).")
    parser.add_argument("--predictor-history", type=int, default=1)
    parser.add_argument("--predictor-hidden", type=str, default="128,128", help="'H1,H2' hidden widths matching how --predictor-weights was trained (default 128,128, the CLAUDE.md S8.2 spec width; pass e.g. 256,256 for scripts/train_multi_checkpoint_predictor.py's default-sized network).")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--checkpoint-file", type=str, help="Evaluate against exactly one policy checkpoint.")
    target_group.add_argument("--log-dir", type=str, help="Evaluate against every model_*.pt checkpoint under this training run's log directory.")
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed for the generalization probe (independent of the predictor's own training seed).")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-transitions", type=int, default=None)
    parser.add_argument("--num-terminated-episodes", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for F1/precision/recall/confusion/lead-time. No val split is fit here (the predictor is frozen and this data is unseen), so this must be passed explicitly rather than refit -- default 0.5.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path. Default: outputs/predictor_generalization_<predictor-weights-stem>_<tag>.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for rollout collection (matches CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    num_envs = args.num_envs or (SMOKE_NUM_ENVS if args.smoke else DEFAULT_NUM_ENVS)
    target_transitions = args.num_transitions or (SMOKE_TRANSITIONS if args.smoke else DEFAULT_TRANSITIONS)
    target_terminated = args.num_terminated_episodes or (
        SMOKE_TERMINATED_EPISODES if args.smoke else DEFAULT_TERMINATED_EPISODES
    )

    weights_path = Path(args.predictor_weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"predictor weights not found: {weights_path}")
    model = load_frozen_predictor(weights_path, history=args.predictor_history, hidden=parse_hidden(args.predictor_hidden))

    if args.checkpoint_file:
        checkpoints = [Path(args.checkpoint_file)]
        tag = Path(args.checkpoint_file).stem
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            raise FileNotFoundError(f"log dir not found: {log_dir}")
        checkpoints = discover_checkpoints(log_dir)
        tag = log_dir.name
        print(f"[generalize] found {len(checkpoints)} checkpoints under {log_dir}")

    seed = args.seed + GEN_SEED_OFFSET

    results = []
    for ckpt_path in checkpoints:
        t0 = time.time()
        iteration = checkpoint_iteration(ckpt_path) if CHECKPOINT_ITER_RE.search(ckpt_path.name) else None
        print(f"[generalize] checkpoint={ckpt_path} (iteration={iteration}) ...")
        record = evaluate_predictor_on_checkpoint(
            model=model,
            checkpoint_path=str(ckpt_path),
            seed=seed,
            num_envs=num_envs,
            target_transitions=target_transitions,
            target_terminated=target_terminated,
            horizon=args.horizon,
            threshold=args.threshold,
            device=device,
        )
        record["iteration"] = iteration
        results.append(record)
        m = record["metrics"]
        print(
            f"  base_rate={m['base_rate']:.4f} auprc={m['auprc']} auroc={m['auroc']} "
            f"episode_length_mean={record['collection_sidecar']['episode_length_mean']} "
            f"({time.time() - t0:.1f}s)"
        )

    out_path = Path(args.out) if args.out else Path("outputs") / f"predictor_generalization_{weights_path.stem}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "predictor_weights": str(weights_path),
                "predictor_history": args.predictor_history,
                "predictor_hidden": args.predictor_hidden,
                "probe_seed": seed,
                "threshold": args.threshold,
                "num_envs": num_envs,
                "target_transitions": target_transitions,
                "target_terminated_episodes": target_terminated,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"[generalize] wrote {out_path}")


if __name__ == "__main__":
    main()
