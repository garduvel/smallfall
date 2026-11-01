"""Predictor training + evaluation on a frozen Go1 dataset (phase-2 analogue
of CLAUDE.md S8/S9/S11 gate 9 -- see DEVIATIONS.md).

Reuses every dimension-agnostic piece of scripts/eval_predictor.py directly
(classification_metrics, mcfadden_r2, expected_calibration_error,
best_threshold_by_f1, median_detection_lead_time, split_dataset_by_split_fn,
make_inputs -- none of these assume cartpole's obs width or semantics) and
cartpole_term.predictor.{model,train,split} unchanged. Two things ARE
Go1-specific and are reimplemented here rather than reused:

1. Feature screening (CLAUDE.md S9.3's analogue): cartpole's build_features()
   hardcodes named semantic features (cart_pos, cos_theta, ...) at known obs
   indices. Go1's actor obs concatenates several IMU/joint/command terms
   whose exact order wasn't confirmed by scripts/inspect_go1_task.py at the
   time this was written (see that script's "observation groups" section --
   it lists term NAMES per group but mjlab's ObservationManager decides
   their concatenation order internally). Rather than guess an order and
   silently mislabel which raw dimension is which, this screens every raw
   obs/action dimension by INDEX (obs_0, obs_1, ..., action_0, ...) plus two
   generic derived magnitudes (obs L2 norm, action L2 norm) -- still answers
   "is anything in this dataset predictive" (the actual diagnostic purpose
   of CLAUDE.md S9.3) without assuming semantics that weren't verified.
2. Per-cause breakdown: uses whatever cause names actually appear in the
   dataset's termination_cause field (written by go1_collect_frozen.py's
   dynamic cause discovery) instead of a hardcoded ("pole_angle", "cart_pos")
   tuple.

Run on the GPU machine (training itself is cheap CPU work, same as
eval_predictor.py, but this needs cartpole_term.go1 registered), after
scripts/go1_collect_frozen.py:

    python scripts/go1_eval_predictor.py --seeds 0
    python scripts/go1_eval_predictor.py --seeds 0 --operating-point target --predictor-hidden 256,256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from eval_predictor import (  # noqa: E402
    base_rate,
    best_threshold_by_f1,
    classification_metrics,
    make_inputs,
    mcfadden_r2,
    median_detection_lead_time,
    split_dataset_by_split_fn,
)

STEP_SECONDS = 0.02  # Go1 velocity task: decimation=4, timestep=0.005 -> 50 Hz; confirm against outputs/go1_task_inspection.json


def build_features_generic(obs: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    features = {f"obs_{i}": obs[:, i] for i in range(obs.shape[1])}
    features.update({f"action_{i}": action[:, i] for i in range(action.shape[1])})
    features["obs_l2norm"] = np.linalg.norm(obs, axis=1)
    features["action_l2norm"] = np.linalg.norm(action, axis=1)
    return features


def feature_screening_generic(features: dict[str, np.ndarray], y: np.ndarray) -> dict[str, dict]:
    results = {}
    has_both = len(np.unique(y)) > 1
    has_pos, has_neg = bool((y == 1).any()), bool((y == 0).any())
    for name, x in features.items():
        class_stats = {
            "positive_mean": float(x[y == 1].mean()) if has_pos else None,
            "positive_std": float(x[y == 1].std()) if has_pos else None,
            "negative_mean": float(x[y == 0].mean()) if has_neg else None,
            "negative_std": float(x[y == 0].std()) if has_neg else None,
        }
        if not has_both:
            results[name] = {"mcfadden_r2": None, "auc_roc": None, **class_stats}
            continue
        clf = LogisticRegression(max_iter=1000)
        x2 = x.reshape(-1, 1)
        clf.fit(x2, y)
        p = clf.predict_proba(x2)[:, 1]
        results[name] = {
            "mcfadden_r2": mcfadden_r2(y, p),
            "auc_roc": float(roc_auc_score(y, p)),
            **class_stats,
        }
    return results


def evaluate_one_dataset(npz_path: Path, sidecar_path: Path, seed: int, hidden: tuple[int, int], device: str = "cpu") -> dict:
    from cartpole_term.predictor.model import TerminationPredictor
    from cartpole_term.predictor.split import split_episodes
    from cartpole_term.predictor.train import train_predictor

    npz_data = dict(np.load(npz_path, allow_pickle=False))
    sidecar = json.loads(sidecar_path.read_text())

    train, val, test = split_dataset_by_split_fn(npz_data, split_episodes, seed=seed)
    for name, split in (("train", train), ("val", val), ("test", test)):
        if len(split["label"]) == 0:
            raise ValueError(f"{name} split is empty for {npz_path} -- dataset too small.")

    x_train, y_train = make_inputs(train), train["label"].astype(np.float32)
    x_val, y_val = make_inputs(val), val["label"].astype(np.float32)
    x_test, y_test = make_inputs(test), test["label"].astype(np.float32)
    obs_action_dim = x_train.shape[1]

    import torch

    torch.manual_seed(seed)
    model = TerminationPredictor(history=1, hidden=hidden, obs_action_dim=obs_action_dim)
    result = train_predictor(model, x_train, y_train, x_val, y_val, device=device, seed=seed)
    model.eval()

    model_save_path = Path("outputs") / f"go1_predictor_{sidecar['operating_point']}_seed{seed}.pt"
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
        test["episode_id"], test["steps_to_termination"], y_test, p_test_mlp, threshold_mlp, step_seconds=STEP_SECONDS
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
    features_val = build_features_generic(val["obs"], val["action"])
    features_test = build_features_generic(test["obs"], test["action"])
    screening = feature_screening_generic(features_train, y_train)

    # Single-best-feature baseline (CLAUDE.md S9.2 item 4's analogue): pick
    # the feature with the highest training-split AUC-ROC from screening,
    # rather than assuming a name like cartpole's |theta| -- we don't know
    # in advance which raw dimension will matter most for Go1.
    best_feature_name = max(
        (n for n, r in screening.items() if r["auc_roc"] is not None), key=lambda n: screening[n]["auc_roc"], default=None
    )
    if best_feature_name is not None:
        logreg_best = LogisticRegression(max_iter=1000)
        logreg_best.fit(features_train[best_feature_name].reshape(-1, 1), y_train)
        p_val_best = logreg_best.predict_proba(features_val[best_feature_name].reshape(-1, 1))[:, 1]
        p_test_best = logreg_best.predict_proba(features_test[best_feature_name].reshape(-1, 1))[:, 1]
        t_best = best_threshold_by_f1(y_val, p_val_best)
        baselines[f"logistic_regression_single_best_feature[{best_feature_name}]"] = classification_metrics(
            y_test, p_test_best, t_best
        )

    per_cause: dict[str, dict] = {}
    cause_names = sorted({c for c in test["termination_cause"].tolist() if c})
    for cause in cause_names:
        cause_mask = test["termination_cause"] == cause
        neg_mask = y_test == 0
        keep = cause_mask | neg_mask
        y_c, p_c = y_test[keep], p_test_mlp[keep]
        per_cause[cause] = {
            "n_positive": int(cause_mask.sum()),
            "auprc": float(average_precision_score(y_c, p_c)) if len(np.unique(y_c)) > 1 else None,
            "lead_time": median_detection_lead_time(
                test["episode_id"][keep], test["steps_to_termination"][keep], y_test[keep], p_test_mlp[keep],
                threshold_mlp, step_seconds=STEP_SECONDS,
            ),
        }

    return {
        "npz_path": str(npz_path),
        "collection_sidecar": sidecar,
        "obs_action_dim": obs_action_dim,
        "predictor_hidden": hidden,
        "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "split_base_rates": {"train": base_rate(y_train), "val": base_rate(y_val), "test": base_rate(y_test)},
        "mlp": mlp_metrics,
        "mlp_lead_time": mlp_lead_time,
        "baselines": baselines,
        "feature_screening": screening,
        "best_feature": best_feature_name,
        "per_cause_breakdown": per_cause,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate the Go1 termination predictor (phase-2, see DEVIATIONS.md).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--operating-point", dest="operating_point", choices=("struggling", "target", "competent", "fluent"), default="target")
    parser.add_argument("--predictor-hidden", dest="predictor_hidden", type=str, default="128,128")
    args = parser.parse_args()

    h1, h2 = (int(p) for p in args.predictor_hidden.split(","))
    hidden = (h1, h2)

    device = "cpu"
    outputs_dir = Path("outputs")

    for seed in args.seeds:
        npz_path = outputs_dir / f"go1_frozen_{args.operating_point}_seed{seed}.npz"
        sidecar_path = outputs_dir / f"go1_frozen_{args.operating_point}_seed{seed}.json"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} not found -- run scripts/go1_collect_frozen.py first.")

        print(f"[go1-eval] seed={seed} point={args.operating_point}: loading {npz_path}")
        metrics = evaluate_one_dataset(npz_path, sidecar_path, seed=seed, hidden=hidden, device=device)

        out_path = outputs_dir / f"go1_metrics_{args.operating_point}_seed{seed}.json"
        out_path.write_text(json.dumps(metrics, indent=2))

        print(
            f"[go1-eval] seed={seed} point={args.operating_point}: "
            f"obs_action_dim={metrics['obs_action_dim']} "
            f"test_base_rate={metrics['split_base_rates']['test']:.4f} "
            f"MLP AUPRC={metrics['mlp']['auprc']:.4f} (baseline={metrics['mlp']['auprc_baseline']:.4f}) "
            f"AUROC={metrics['mlp']['auroc']:.4f} best_feature={metrics['best_feature']}"
        )
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
