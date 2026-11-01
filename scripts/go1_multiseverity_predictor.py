"""New script (phase-2, outside CLAUDE.md's cartpole-only spec, see
DEVIATIONS.md): collect labelled Go1 failure data at SEVERAL push
severities and train ONE termination predictor on the pooled result,
instead of a single fixed severity.

Why this, specifically, over the other two options considered
(gentler single push; history-stacked input): this is the one that
matches the reference literature's own methodology most directly.
SafeFall (arXiv:2511.18509) builds its training data from a combination
of MULTIPLE failure factors (sensor noise, external pushes at more than
one magnitude, foot slip, foot trip, latency, dynamics mismatch) rather
than one fixed condition -- the whole point being that a classifier/
policy trained on a single stressor level doesn't generalize to the
range of real disturbances it'll actually face. Same idea, applied here
to just the one factor already wired up (push severity): pool several
severities instead of picking one and hoping it's representative.

Also directly answers this project's own earlier open question (see
DEVIATIONS.md's Gate 2/3/4 entries): does predictor quality trade off
against push severity? Reports a PER-SEVERITY breakdown on the test set,
not just the pooled number, so that's visible rather than hidden -- same
"don't silently pool and dilute" discipline this project already applied
once before (scripts/train_unified_predictor.py's per-source breakdown,
prior session).

Reuses, unmodified:
  - scripts/go1_wild_push_predictor.py's `collect_under_wild_push` (the
    already-fixed, already-verified collection function -- correctly
    applies the push-scale override BEFORE env construction, per the
    EventManager-deepcopy bug fixed there) and `build_features_generic`/
    `feature_screening_generic`.
  - cartpole_term.predictor.{model.TerminationPredictor, train.train_predictor,
    split.split_episodes} (same architecture/loss/optimizer as every
    other predictor in this project).
  - scripts/eval_predictor.py's generic metric functions.

NEW in this file: the pooling logic (per-source episode_id offsetting,
so combining datasets can't silently collide unrelated episodes under
the same integer id -- same fix already applied once before in
scripts/train_unified_predictor.py) and the per-severity test breakdown.

Run on the GPU machine, from the repo root, after producing a checkpoint
(scripts/go1_train_full.ipynb or scripts/robot_train.py):

    python scripts/go1_multiseverity_predictor.py --checkpoint-file model_9999.pt
    python scripts/go1_multiseverity_predictor.py --checkpoint-file model_9999.pt --push-scales 4 16 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from cartpole_term.cli import SMOKE_COLLECT_TRANSITIONS, SMOKE_NUM_ENVS  # noqa: E402
from eval_predictor import (  # noqa: E402
    base_rate,
    best_threshold_by_f1,
    classification_metrics,
    make_inputs,
    median_detection_lead_time,
    split_dataset_by_split_fn,
)
from go1_wild_push_predictor import (  # noqa: E402
    PREDICTOR_HORIZON,
    build_features_generic,
    collect_under_wild_push,
    feature_screening_generic,
)

# Spread deliberately wide, spanning what's already been measured this
# session (16x -> ~10% base rate) plus a gentler and a harsher tier, per
# the "combine multiple severities" literature approach above. These are
# starting points, not final -- the script prints each tier's REAL
# measured base rate as it collects, same "measured, not guessed"
# discipline as everywhere else here.
DEFAULT_PUSH_SCALES = (4.0, 16.0, 32.0)
SEVERITY_LABELS = ("low", "medium", "high")  # paired with DEFAULT_PUSH_SCALES by rank, not a hard mapping if custom scales are passed

TARGET_TRANSITIONS_PER_SEVERITY = 1_000_000
TARGET_TERMINATED_PER_SEVERITY = 1_000
COLLECT_NUM_ENVS = 4096
PREDICTOR_HIDDEN = (128, 128)
SMOKE_TARGET_TERMINATED_EPISODES = 20

# Same fix already applied once before in scripts/train_unified_predictor.py
# (prior session): collect_one/collect_under_wild_push number episodes
# starting from 0 independently per call, so pooling without an offset
# would let unrelated episodes from different severities collide under
# the same integer id -- not a train/test leakage risk, but it would
# silently merge unrelated episodes' per-episode aggregates. Not a guess;
# same constant/reasoning as that prior fix.
EPISODE_ID_OFFSET_PER_SOURCE = 10_000_000


def severity_label(index: int, scale: float) -> str:
    if index < len(SEVERITY_LABELS):
        return SEVERITY_LABELS[index]
    return f"tier{index}"


def concat_datasets_with_offset(datasets: list[dict], labels: list[str]) -> dict:
    offset_datasets = []
    source_labels_per_row = []
    for i, ds in enumerate(datasets):
        offset_ds = dict(ds)
        offset_ds["episode_id"] = ds["episode_id"] + i * EPISODE_ID_OFFSET_PER_SOURCE
        offset_datasets.append(offset_ds)
        source_labels_per_row.append(np.full(len(ds["label"]), labels[i], dtype="U16"))

    pooled = {}
    for key in datasets[0].keys():
        pooled[key] = np.concatenate([ds[key] for ds in offset_datasets], axis=0)
    pooled["severity_source"] = np.concatenate(source_labels_per_row, axis=0)
    return pooled


def train_and_evaluate_pooled(dataset: dict, seed: int, hidden: tuple[int, int], step_seconds: float) -> dict:
    from cartpole_term.predictor.model import TerminationPredictor
    from cartpole_term.predictor.split import split_episodes
    from cartpole_term.predictor.train import train_predictor

    train, val, test = split_dataset_by_split_fn(dataset, split_episodes, seed=seed)
    for name, split in (("train", train), ("val", val), ("test", test)):
        if len(split["label"]) == 0:
            raise ValueError(f"{name} split is empty -- dataset too small.")

    x_train, y_train = make_inputs(train), train["label"].astype(np.float32)
    x_val, y_val = make_inputs(val), val["label"].astype(np.float32)
    x_test, y_test = make_inputs(test), test["label"].astype(np.float32)
    obs_action_dim = x_train.shape[1]

    torch.manual_seed(seed)
    model = TerminationPredictor(history=1, hidden=hidden, obs_action_dim=obs_action_dim)
    result = train_predictor(model, x_train, y_train, x_val, y_val, device="cpu", seed=seed)
    model.eval()

    model_save_path = Path("outputs") / f"go1_multiseverity_predictor_seed{seed}.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    with torch.no_grad():
        p_val_mlp = torch.sigmoid(model(torch.as_tensor(x_val)).squeeze(-1)).numpy()
        p_test_mlp = torch.sigmoid(model(torch.as_tensor(x_test)).squeeze(-1)).numpy()
    threshold_mlp = best_threshold_by_f1(y_val, p_val_mlp)
    mlp_metrics = classification_metrics(y_test, p_test_mlp, threshold_mlp)
    mlp_metrics["best_val_epoch"] = result.best_epoch
    mlp_metrics["best_val_auprc"] = result.best_val_auprc
    mlp_lead_time = median_detection_lead_time(
        test["episode_id"], test["steps_to_termination"], y_test, p_test_mlp, threshold_mlp, step_seconds=step_seconds
    )

    baselines: dict[str, dict] = {}
    baselines["constant_negative"] = classification_metrics(y_test, np.zeros_like(y_test), threshold=0.5)
    train_base_rate = base_rate(y_train)
    baselines["base_rate_constant"] = classification_metrics(
        y_test, np.full_like(y_test, train_base_rate), threshold=train_base_rate
    )
    logreg_all = LogisticRegression(max_iter=1000)
    logreg_all.fit(x_train, y_train)
    p_val_logreg = logreg_all.predict_proba(x_val)[:, 1]
    p_test_logreg = logreg_all.predict_proba(x_test)[:, 1]
    t_logreg = best_threshold_by_f1(y_val, p_val_logreg)
    baselines["logistic_regression_all_inputs"] = classification_metrics(y_test, p_test_logreg, t_logreg)

    features_train = build_features_generic(train["obs"], train["action"])
    screening = feature_screening_generic(features_train, y_train)
    best_feature_name = max(
        (n for n, r in screening.items() if r["auc_roc"] is not None), key=lambda n: screening[n]["auc_roc"], default=None
    )

    # THE point of pooling: report per-severity-source breakdown, not just
    # the pooled number -- does this actually help every regime, or just
    # dilute toward the middle? Same discipline as
    # scripts/train_unified_predictor.py's per-source breakdown.
    per_source: dict[str, dict] = {}
    for source in sorted(set(test["severity_source"].tolist())):
        source_mask = test["severity_source"] == source
        y_s, p_s = y_test[source_mask], p_test_mlp[source_mask]
        lead_s = median_detection_lead_time(
            test["episode_id"][source_mask], test["steps_to_termination"][source_mask], y_s, p_s,
            threshold_mlp, step_seconds=step_seconds,
        )
        per_source[source] = {
            "n_test_transitions": int(source_mask.sum()),
            "test_base_rate": base_rate(y_s),
            "auprc": float(average_precision_score(y_s, p_s)) if len(np.unique(y_s)) > 1 else None,
            "median_lead_steps": lead_s["median_lead_steps"],
            "median_lead_seconds": lead_s["median_lead_seconds"],
        }

    return {
        "obs_action_dim": obs_action_dim,
        "predictor_hidden": hidden,
        "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "split_base_rates": {"train": base_rate(y_train), "val": base_rate(y_val), "test": base_rate(y_test)},
        "mlp": mlp_metrics,
        "mlp_lead_time": mlp_lead_time,
        "baselines": baselines,
        "feature_screening": screening,
        "best_feature": best_feature_name,
        "per_severity_source_breakdown": per_source,
        "model_path": str(model_save_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push-scales", dest="push_scales", type=float, nargs="+", default=list(DEFAULT_PUSH_SCALES))
    parser.add_argument("--predictor-hidden", dest="predictor_hidden", type=str, default="128,128")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    h1, h2 = (int(p) for p in args.predictor_hidden.split(","))
    hidden = (h1, h2)

    num_envs = SMOKE_NUM_ENVS if args.smoke else COLLECT_NUM_ENVS
    target_transitions = SMOKE_COLLECT_TRANSITIONS if args.smoke else TARGET_TRANSITIONS_PER_SEVERITY
    target_terminated = SMOKE_TARGET_TERMINATED_EPISODES if args.smoke else TARGET_TERMINATED_PER_SEVERITY

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[dict] = []
    labels: list[str] = []
    sidecars: list[dict] = []
    for i, scale in enumerate(args.push_scales):
        label = severity_label(i, scale)
        print(f"[multiseverity] === severity tier {i} ({label}, push_scale={scale}x) ===")
        dataset, sidecar = collect_under_wild_push(
            checkpoint_path=args.checkpoint_file,
            seed=args.seed,
            push_scale=scale,
            num_envs=num_envs,
            target_transitions=target_transitions,
            target_terminated=target_terminated,
            horizon=PREDICTOR_HORIZON,
            device=device,
        )
        print(
            f"[multiseverity] tier {i} ({label}): {sidecar['labelled_transitions']} transitions, "
            f"base_rate={sidecar['base_rate']:.3f}, terminated_episodes={sidecar['terminated_episode_count']}, "
            f"wall_time={sidecar['wall_time_s']:.1f}s"
        )
        npz_path = outputs_dir / f"go1_multiseverity_frozen_{label}_seed{args.seed}.npz"
        json_path = outputs_dir / f"go1_multiseverity_frozen_{label}_seed{args.seed}.json"
        np.savez_compressed(npz_path, **dataset)
        json_path.write_text(json.dumps(sidecar, indent=2))
        print(f"  wrote {npz_path}")
        print(f"  wrote {json_path}")

        datasets.append(dataset)
        labels.append(label)
        sidecars.append(sidecar)

    pooled = concat_datasets_with_offset(datasets, labels)
    pooled_npz_path = outputs_dir / f"go1_multiseverity_frozen_pooled_seed{args.seed}.npz"
    np.savez_compressed(pooled_npz_path, **pooled)
    print(f"[multiseverity] pooled {len(pooled['label'])} total transitions across {len(datasets)} severities")
    print(f"  wrote {pooled_npz_path}")

    step_seconds = 0.02  # Go1: decimation=4, timestep=0.005 -> 50 Hz, confirmed live this session
    metrics = train_and_evaluate_pooled(pooled, seed=args.seed, hidden=hidden, step_seconds=step_seconds)
    metrics["severity_tiers"] = [
        {"index": i, "label": labels[i], "push_scale": args.push_scales[i], "collection_sidecar": sidecars[i]}
        for i in range(len(labels))
    ]

    metrics_path = outputs_dir / f"go1_multiseverity_metrics_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(
        f"\n[multiseverity] POOLED: test_base_rate={metrics['split_base_rates']['test']:.4f} "
        f"AUPRC={metrics['mlp']['auprc']:.4f} (baseline={metrics['mlp']['auprc_baseline']:.4f}) "
        f"AUROC={metrics['mlp']['auroc']:.4f} "
        f"median_lead={metrics['mlp_lead_time']['median_lead_steps']:.1f} steps "
        f"({metrics['mlp_lead_time']['median_lead_seconds']:.2f}s)"
    )
    print("[multiseverity] per-severity breakdown (does pooling help every regime, or just dilute?):")
    for source, row in metrics["per_severity_source_breakdown"].items():
        print(
            f"  {source}: n={row['n_test_transitions']} base_rate={row['test_base_rate']:.4f} "
            f"AUPRC={row['auprc']:.4f} median_lead={row['median_lead_steps']:.1f} steps "
            f"({row['median_lead_seconds']:.2f}s)"
        )
    print(f"  wrote {metrics_path}")
    print(f"  wrote {metrics['model_path']}")


if __name__ == "__main__":
    main()
