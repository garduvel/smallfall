"""Predictor training + evaluation on a frozen dataset (CLAUDE.md S8, S9,
S11 gates 4/9).

Loads a frozen dataset produced by scripts/collect_frozen.py, splits it by
episode_id (CLAUDE.md S7.5), trains the MLP predictor (S8) on train+val,
and reports the S9.1 metrics table, S9.2 baselines, and S9.3 feature
screening on the untouched test split.

All the metric/baseline/screening logic below is pure numpy/sklearn and
needs neither mjlab nor a GPU -- only the `cartpole_term.*` imports inside
main() do. That split keeps the highest-error-risk code (metric formulas)
locally testable.

Run on the GPU machine, from the repo root, after scripts/collect_frozen.py
has produced outputs/frozen_{point}_seed{N}.npz:

    python scripts/eval_predictor.py                              # target point, seeds 0 1 2
    python scripts/eval_predictor.py --smoke                       # S11 gate 4 tiny check
    python scripts/eval_predictor.py --seeds 0 --operating-point struggling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

STEP_SECONDS = 0.05  # CLAUDE.md S2.2: 20 Hz env step
ECE_BINS = 15  # CLAUDE.md S9.1

FEATURE_ORDER = (
    "cart_pos",
    "cos_theta",
    "sin_theta",
    "cart_vel",
    "pole_vel",
    "action",
    "abs_theta_wrapped",
    "abs_cart_pos",
)


# ---------------------------------------------------------------------------
# Pure metric functions -- no mjlab, no GPU, fully locally testable.
# ---------------------------------------------------------------------------


def base_rate(y: np.ndarray) -> float:
    return float(np.mean(y)) if len(y) else 0.0


def mcfadden_r2(y: np.ndarray, p: np.ndarray) -> float | None:
    if len(y) == 0:
        return None
    eps = 1e-12
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(y, dtype=np.float64)
    ll_model = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    pi = min(max(base_rate(y), eps), 1 - eps)
    ll_null = np.sum(y * np.log(pi) + (1 - y) * np.log(1 - pi))
    if ll_null == 0:
        return 0.0
    return float(1 - ll_model / ll_null)


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = ECE_BINS
) -> tuple[float, list[dict]]:
    """Equal-mass (quantile) binning ECE. Returns (ece, per-bin reliability data)."""
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    if n == 0:
        return 0.0, []
    order = np.argsort(p)
    y_sorted, p_sorted = y[order], p[order]
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    bins: list[dict] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        p_bin, y_bin = p_sorted[lo:hi], y_sorted[lo:hi]
        mean_p, mean_y = float(p_bin.mean()), float(y_bin.mean())
        weight = len(p_bin) / n
        ece += weight * abs(mean_p - mean_y)
        bins.append({"count": len(p_bin), "mean_predicted": mean_p, "mean_actual": mean_y})
    return ece, bins


def best_threshold_by_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    if len(np.unique(y_val)) < 2:
        return 0.5
    candidates = np.unique(np.clip(p_val, 1e-6, 1 - 1e-6))
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def classification_metrics(y_test: np.ndarray, p_test: np.ndarray, threshold: float) -> dict:
    y_test = np.asarray(y_test)
    p_test = np.asarray(p_test)
    pred = (p_test >= threshold).astype(int)
    has_both_classes = len(np.unique(y_test)) > 1

    auprc = float(average_precision_score(y_test, p_test)) if has_both_classes else None
    auroc = float(roc_auc_score(y_test, p_test)) if has_both_classes else None
    brier = float(brier_score_loss(y_test, p_test))
    ece, reliability = expected_calibration_error(y_test, p_test)
    cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()

    return {
        "base_rate": base_rate(y_test),
        "auprc": auprc,
        "auprc_baseline": base_rate(y_test),
        "auroc": auroc,
        "brier": brier,
        "ece": ece,
        "reliability_diagram": reliability,
        "threshold": threshold,
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "accuracy": float((pred == y_test).mean()) if len(y_test) else None,
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "mcfadden_r2": mcfadden_r2(y_test, p_test),
    }


def median_detection_lead_time(
    episode_id: np.ndarray,
    steps_to_termination: np.ndarray,
    label: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    step_seconds: float = STEP_SECONDS,
) -> dict:
    """For each terminated episode (identified by its positive samples),
    the earliest step in its horizon window where predicted prob exceeds
    threshold -- i.e. the LARGEST steps_to_termination among that episode's
    positives with prob > threshold (CLAUDE.md S9.1)."""
    positive_mask = label == 1
    per_episode_hits: dict[int, list[int]] = {}
    for eid, k, p in zip(
        episode_id[positive_mask], steps_to_termination[positive_mask], probs[positive_mask]
    ):
        if p > threshold:
            per_episode_hits.setdefault(int(eid), []).append(int(k))

    num_episodes_with_positives = int(len(set(episode_id[positive_mask].tolist())))
    lead_times = [max(ks) for ks in per_episode_hits.values()]

    if not lead_times:
        return {
            "median_lead_steps": None,
            "median_lead_seconds": None,
            "num_detected_episodes": 0,
            "num_episodes_with_positives": num_episodes_with_positives,
        }
    median_steps = float(np.median(lead_times))
    return {
        "median_lead_steps": median_steps,
        "median_lead_seconds": median_steps * step_seconds,
        "num_detected_episodes": len(lead_times),
        "num_episodes_with_positives": num_episodes_with_positives,
    }


def build_features(obs: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    cart_pos, cos_t, sin_t, cart_vel, pole_vel = (
        obs[:, 0],
        obs[:, 1],
        obs[:, 2],
        obs[:, 3],
        obs[:, 4],
    )
    theta_wrapped = np.arctan2(sin_t, cos_t)
    return {
        "cart_pos": cart_pos,
        "cos_theta": cos_t,
        "sin_theta": sin_t,
        "cart_vel": cart_vel,
        "pole_vel": pole_vel,
        "action": action[:, 0],
        "abs_theta_wrapped": np.abs(theta_wrapped),
        "abs_cart_pos": np.abs(cart_pos),
    }


def feature_screening(features: dict[str, np.ndarray], y: np.ndarray) -> dict[str, dict]:
    """Per-feature univariate logistic regression (CLAUDE.md S9.3), plus
    per-class (positive vs negative) mean/std for each feature. The latter
    isn't spec-mandated, but answers "why is/isn't this feature predictive"
    directly from the JSON -- added after gate 9 needed a hand-rolled
    version of exactly this to diagnose why |theta_wrapped| came out weak
    at the `target` operating point (see DEVIATIONS.md): it separates
    positives from negatives by only 1.27x there, vs |cart_pos|'s 3.13x,
    because a competent policy's failures are overwhelmingly cart-drift-
    driven, not pole-angle-driven. Having this on by default means that
    question answers itself next time instead of needing an ad hoc query.
    """
    results = {}
    has_both = len(np.unique(y)) > 1
    has_pos = bool((y == 1).any())
    has_neg = bool((y == 0).any())
    for name in FEATURE_ORDER:
        x = features[name]
        class_stats = {
            "positive_mean": float(x[y == 1].mean()) if has_pos else None,
            "positive_std": float(x[y == 1].std()) if has_pos else None,
            "negative_mean": float(x[y == 0].mean()) if has_neg else None,
            "negative_std": float(x[y == 0].std()) if has_neg else None,
        }
        x2 = x.reshape(-1, 1)
        if not has_both:
            results[name] = {"mcfadden_r2": None, "auc_roc": None, **class_stats}
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x2, y)
        p = clf.predict_proba(x2)[:, 1]
        results[name] = {
            "mcfadden_r2": mcfadden_r2(y, p),
            "auc_roc": float(roc_auc_score(y, p)),
            **class_stats,
        }
    return results


def split_dataset_by_split_fn(
    npz: dict, split_fn, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """split_fn: callable matching cartpole_term.predictor.split.split_episodes's
    signature -- injected so this stays testable without importing the
    cartpole_term package (CLAUDE.md S7.5)."""
    train_ids, val_ids, test_ids = split_fn(npz["episode_id"].tolist(), seed=seed)

    def subset(ids: set[int]) -> dict[str, np.ndarray]:
        mask = np.isin(npz["episode_id"], list(ids))
        return {k: v[mask] for k, v in npz.items()}

    return subset(train_ids), subset(val_ids), subset(test_ids)


def make_inputs(split: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([split["obs"], split["action"]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Orchestration (needs cartpole_term.predictor.{model,train} -- injected).
# ---------------------------------------------------------------------------


def evaluate_one_dataset(
    npz_path: Path,
    sidecar_path: Path,
    seed: int,
    predictor_history: int,
    model_cls,
    train_fn,
    split_fn,
    device: str = "cpu",
    model_save_path: Path | None = None,
) -> dict:
    if predictor_history != 1:
        raise NotImplementedError(
            "predictor.history > 1 is not implemented -- CLAUDE.md S8.1 lists it as "
            "optional and off by default, and it requires collect_frozen.py to store "
            "full per-episode raw trajectories (not just labelled transitions) to "
            "reconstruct history windows correctly. Not silently approximating this."
        )

    npz_data = dict(np.load(npz_path, allow_pickle=False))
    sidecar = json.loads(sidecar_path.read_text())

    train, val, test = split_dataset_by_split_fn(npz_data, split_fn, seed=seed)

    for name, split in (("train", train), ("val", val), ("test", test)):
        if len(split["label"]) == 0:
            raise ValueError(
                f"{name} split is empty for {npz_path} -- dataset too small "
                "(this can happen in --smoke mode with very few episodes)."
            )

    x_train, y_train = make_inputs(train), train["label"].astype(np.float32)
    x_val, y_val = make_inputs(val), val["label"].astype(np.float32)
    x_test, y_test = make_inputs(test), test["label"].astype(np.float32)

    # --- Phase 1 MLP predictor (CLAUDE.md S8.2, S8.3) ---
    # Seed BEFORE constructing the model: weight init happens at
    # construction time, before train_fn's own torch.manual_seed(seed)
    # call, so seeding only there left init non-reproducible (caught by a
    # Gate 4 re-run producing a different AUPRC from a byte-identical
    # dataset and the same --seeds value).
    import torch

    torch.manual_seed(seed)
    model = model_cls(history=predictor_history)
    result = train_fn(model, x_train, y_train, x_val, y_val, device=device, seed=seed)
    model.eval()

    if model_save_path is not None:
        # scripts/record_video.py's predictor-overlay video (CLAUDE.md S10
        # item 2) needs these exact trained weights -- saved here rather
        # than retraining a second time, since evaluate_one_dataset is
        # already the single place these weights exist. State dict only
        # (matches build_runner_from_checkpoint's actor-only load pattern):
        # architecture is fully fixed by CLAUDE.md S8.2, so reconstructing
        # via TerminationPredictor(history=predictor_history) before
        # load_state_dict is sufficient, no extra metadata needed.
        model_save_path = Path(model_save_path)
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
        test["episode_id"], test["steps_to_termination"], y_test, p_test_mlp, threshold_mlp
    )

    # --- S9.2 baselines ---
    baselines: dict[str, dict] = {}

    p_const_neg = np.zeros_like(y_test)
    baselines["constant_negative"] = classification_metrics(y_test, p_const_neg, threshold=0.5)

    train_base_rate = base_rate(y_train)
    p_base_rate_const = np.full_like(y_test, train_base_rate)
    baselines["base_rate_constant"] = classification_metrics(y_test, p_base_rate_const, threshold=train_base_rate)

    features_train = build_features(train["obs"], train["action"])
    features_val = build_features(val["obs"], val["action"])
    features_test = build_features(test["obs"], test["action"])

    logreg_all = LogisticRegression(max_iter=1000)
    x_train_6 = np.concatenate([train["obs"], train["action"]], axis=1)
    x_val_6 = np.concatenate([val["obs"], val["action"]], axis=1)
    x_test_6 = np.concatenate([test["obs"], test["action"]], axis=1)
    logreg_all.fit(x_train_6, y_train)
    p_val_logreg = logreg_all.predict_proba(x_val_6)[:, 1]
    p_test_logreg = logreg_all.predict_proba(x_test_6)[:, 1]
    t_logreg = best_threshold_by_f1(y_val, p_val_logreg)
    baselines["logistic_regression_6_inputs"] = classification_metrics(y_test, p_test_logreg, t_logreg)

    logreg_theta = LogisticRegression(max_iter=1000)
    logreg_theta.fit(features_train["abs_theta_wrapped"].reshape(-1, 1), y_train)
    p_val_theta = logreg_theta.predict_proba(features_val["abs_theta_wrapped"].reshape(-1, 1))[:, 1]
    p_test_theta = logreg_theta.predict_proba(features_test["abs_theta_wrapped"].reshape(-1, 1))[:, 1]
    t_theta = best_threshold_by_f1(y_val, p_val_theta)
    baselines["logistic_regression_abs_theta_only"] = classification_metrics(y_test, p_test_theta, t_theta)

    # --- S9.3 feature screening, on the training split ---
    screening = feature_screening(features_train, y_train)

    # --- Per-cause breakdown (CLAUDE.md S9.1) ---
    per_cause: dict[str, dict] = {}
    for cause in ("pole_angle", "cart_pos"):
        cause_positive_mask = test["termination_cause"] == cause
        neg_mask = y_test == 0
        keep = cause_positive_mask | neg_mask
        if not cause_positive_mask.any():
            per_cause[cause] = {"note": "no test-split positives with this cause", "n_positive": 0}
            continue
        y_c, p_c = y_test[keep], p_test_mlp[keep]
        per_cause[cause] = {
            "n_positive": int(cause_positive_mask.sum()),
            "auprc": float(average_precision_score(y_c, p_c)) if len(np.unique(y_c)) > 1 else None,
            "lead_time": median_detection_lead_time(
                test["episode_id"][cause_positive_mask | neg_mask],
                test["steps_to_termination"][cause_positive_mask | neg_mask],
                y_test[cause_positive_mask | neg_mask],
                p_test_mlp[cause_positive_mask | neg_mask],
                threshold_mlp,
            ),
        }

    return {
        "npz_path": str(npz_path),
        "collection_sidecar": sidecar,
        "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "split_base_rates": {
            "train": base_rate(y_train),
            "val": base_rate(y_val),
            "test": base_rate(y_test),
        },
        "mlp": mlp_metrics,
        "mlp_lead_time": mlp_lead_time,
        "baselines": baselines,
        "feature_screening": screening,
        "per_cause_breakdown": per_cause,
    }


def main() -> None:
    from cartpole_term.cli import build_parser
    from cartpole_term.predictor.model import TerminationPredictor
    from cartpole_term.predictor.split import split_episodes
    from cartpole_term.predictor.train import train_predictor

    parser = build_parser(
        ("eval",), description="Train + evaluate the termination predictor on a frozen dataset (CLAUDE.md S8, S9)."
    )
    args = parser.parse_args()

    if args.predictor_uncertainty != "none":
        raise NotImplementedError(
            "--predictor.uncertainty ensemble is not implemented in this pass "
            "(CLAUDE.md S8.4 phase 2, off by default)."
        )

    device = "cpu"  # metrics/training here are cheap; no need for GPU
    outputs_dir = Path("outputs")

    for seed in args.seeds:
        npz_path = outputs_dir / f"frozen_{args.operating_point}_seed{seed}.npz"
        sidecar_path = outputs_dir / f"frozen_{args.operating_point}_seed{seed}.json"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} not found -- run scripts/collect_frozen.py for this "
                "seed/operating-point first."
            )

        print(f"[eval] seed={seed} point={args.operating_point}: loading {npz_path}")
        model_save_path = outputs_dir / f"predictor_{args.operating_point}_seed{seed}.pt"
        metrics = evaluate_one_dataset(
            npz_path,
            sidecar_path,
            seed=seed,
            predictor_history=args.predictor_history,
            model_cls=TerminationPredictor,
            train_fn=train_predictor,
            split_fn=split_episodes,
            device=device,
            model_save_path=model_save_path,
        )

        out_path = outputs_dir / f"metrics_{args.operating_point}_seed{seed}.json"
        out_path.write_text(json.dumps(metrics, indent=2))
        print(f"  wrote {model_save_path}")

        print(
            f"[eval] seed={seed} point={args.operating_point}: "
            f"test_base_rate={metrics['split_base_rates']['test']:.4f} "
            f"MLP AUPRC={metrics['mlp']['auprc']:.4f} "
            f"(baseline={metrics['mlp']['auprc_baseline']:.4f}) "
            f"AUROC={metrics['mlp']['auroc']:.4f}"
        )
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
