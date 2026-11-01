"""Uncertainty and significance for the Go1 fall predictor.

Re-creates the locked predictor's test-split predictions (same dataset, same
episode-level split seed, the saved weights, the validation-chosen threshold
from the metrics file) and checks them against the recorded metrics before
doing anything else, so the statistics describe exactly the reported model.

Offline (test split):
  - 95% confidence intervals for AUPRC, AUROC, Brier, precision, recall, F1
    from a CLUSTER bootstrap that resamples whole episodes. Steps within an
    episode are strongly correlated; resampling steps would understate the
    uncertainty.
  - MLP vs logistic regression on the same 60 inputs: the paired difference in
    AUPRC and AUROC under the same episode resampling, with a bootstrap
    p-value.
  - MLP vs chance: AUPRC minus the (resampled) base rate.
  - Detection per fall: share of falling episodes flagged before the fall
    (exact binomial CI) and the median lead time (bootstrap CI over episodes).
Live (optional, from scripts/go1_handoff_eval.py's outputs): recall and
precision with exact binomial CIs, median lead time with a bootstrap CI.

What this does NOT show: variation across training seeds. One predictor was
trained; the intervals are over episodes for that fixed model.

Pure numpy / scikit-learn / torch (CPU): no GPU or mjlab needed.

Usage:
    python scripts/go1_predictor_stats.py --out-dir outputs/stats/predictor
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def _load(rel: str, name: str):
    """Load a source file by path: importing the cartpole_term package would
    register mjlab tasks and so require mjlab."""
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


stats = _load("src/cartpole_term/common/stats.py", "go1_stats")
split_mod = _load("src/cartpole_term/predictor/split.py", "go1_split")
model_mod = _load("src/cartpole_term/predictor/model.py", "go1_model")

STEP_SECONDS = 0.02
REPRO_TOLERANCE = 1e-6


def thresholded(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return precision, recall, f1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="outputs/go1_wildpush_frozen_seed0.npz")
    p.add_argument("--predictor-file", default="outputs/go1_wildpush_predictor_seed0.pt")
    p.add_argument("--metrics-file", default="outputs/go1_wildpush_metrics_seed0.json")
    p.add_argument("--live-json", default="outputs/handoff_eval/go1_handoff_eval_seed0.json")
    p.add_argument("--live-npz", default="outputs/handoff_eval/go1_handoff_eval_seed0.npz")
    p.add_argument("--seed", type=int, default=0, help="the split seed the predictor was trained with")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quiet", action="store_true", help="do not print the tables; they are still written to the .md file")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    recorded = json.loads(Path(args.metrics_file).read_text())
    threshold = float(recorded["mlp"]["threshold"])
    d = np.load(args.dataset)
    episode_id = d["episode_id"]
    train_ids, _val_ids, test_ids = split_mod.split_episodes(episode_id.tolist(), seed=args.seed)
    test_mask = np.isin(episode_id, list(test_ids))
    train_mask = np.isin(episode_id, list(train_ids))
    label = d["label"]

    def inputs(mask):
        return np.concatenate([d["obs"][mask], d["action"][mask]], axis=1).astype(np.float32)

    x_test, y_test = inputs(test_mask), label[test_mask].astype(np.int64)
    ep_test, stt_test = episode_id[test_mask], d["steps_to_termination"][test_mask]

    model = model_mod.TerminationPredictor(history=1, hidden=tuple(recorded["predictor_hidden"]),
                                           obs_action_dim=int(recorded["obs_action_dim"]))
    model.load_state_dict(torch.load(args.predictor_file, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        p_mlp = torch.sigmoid(model(torch.as_tensor(x_test)).squeeze(-1)).numpy().astype(np.float64)

    x_train, y_train = inputs(train_mask), label[train_mask].astype(np.float32)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(x_train, y_train)  # same call as scripts/go1_wild_push_predictor.py
    p_lr = logreg.predict_proba(x_test)[:, 1]
    del x_train, y_train

    # Reproduction check: these must be the reported models' test predictions.
    repro = {
        "test_size": [int(len(y_test)), int(recorded["split_sizes"]["test"])],
        "mlp_auprc": [float(average_precision_score(y_test, p_mlp)), recorded["mlp"]["auprc"]],
        "mlp_auroc": [float(roc_auc_score(y_test, p_mlp)), recorded["mlp"]["auroc"]],
        "logreg_auprc": [float(average_precision_score(y_test, p_lr)),
                         recorded["baselines"]["logistic_regression_all_inputs"]["auprc"]],
    }
    if repro["test_size"][0] != repro["test_size"][1]:
        raise RuntimeError(f"[predictor-stats] test split size differs from the recorded one: {repro['test_size']}")
    for k in ("mlp_auprc", "mlp_auroc"):
        if abs(repro[k][0] - repro[k][1]) > REPRO_TOLERANCE:
            raise RuntimeError(f"[predictor-stats] could not reproduce the recorded {k}: {repro[k]}")
    logreg_reproduced = abs(repro["logreg_auprc"][0] - repro["logreg_auprc"][1]) < 1e-3
    print(f"[predictor-stats] reproduced recorded test metrics: {repro} ({time.time() - t0:.0f}s)")

    pred_mlp = p_mlp >= threshold
    names = ["mlp_auprc", "mlp_auroc", "logreg_auprc", "logreg_auroc", "auprc_mlp_minus_logreg",
             "auroc_mlp_minus_logreg", "base_rate", "auprc_mlp_minus_base_rate", "mlp_brier",
             "mlp_precision", "mlp_recall", "mlp_f1"]

    def stat_fn(idx):
        y = y_test[idx]
        if y.min() == y.max():
            return np.full(len(names), np.nan)
        a_m, a_l = average_precision_score(y, p_mlp[idx]), average_precision_score(y, p_lr[idx])
        r_m, r_l = roc_auc_score(y, p_mlp[idx]), roc_auc_score(y, p_lr[idx])
        base = y.mean()
        prec, rec, f1 = thresholded(y, pred_mlp[idx])
        return np.array([a_m, r_m, a_l, r_l, a_m - a_l, r_m - r_l, base, a_m - base,
                         brier_score_loss(y, p_mlp[idx]), prec, rec, f1])

    cb = stats.cluster_bootstrap(ep_test, stat_fn, n_boot=args.n_boot, seed=args.seed)
    offline = {n: {"estimate": cb["point"][i], "ci95": [cb["ci95_low"][i], cb["ci95_high"][i]]}
               for i, n in enumerate(names)}
    for n in ("auprc_mlp_minus_logreg", "auroc_mlp_minus_logreg", "auprc_mlp_minus_base_rate"):
        offline[n]["bootstrap_p_two_sided"] = cb["p_vs_zero"][names.index(n)]
    print(f"[predictor-stats] cluster bootstrap done: {cb['n_clusters']} test episodes x {args.n_boot} resamples "
          f"({time.time() - t0:.0f}s)")

    # Per falling episode: was it flagged before the fall, and how early?
    pos = y_test == 1
    falling = np.unique(ep_test[pos])
    lead = {}
    for eid, k, flagged in zip(ep_test[pos], stt_test[pos], (p_mlp > threshold)[pos]):
        if flagged:
            lead[int(eid)] = max(lead.get(int(eid), 0), int(k))
    leads = np.array(list(lead.values()), dtype=np.float64)
    per_fall = {
        "falling_test_episodes": int(len(falling)), "flagged_before_fall": int(len(leads)),
        "detection_rate": float(len(leads) / len(falling)),
        "detection_rate_ci95": list(stats.clopper_pearson_ci(len(leads), len(falling))),
        "median_lead_steps": float(np.median(leads)), "median_lead_steps_ci95": list(stats.bootstrap_ci(leads, np.median)),
        "median_lead_seconds": float(np.median(leads) * STEP_SECONDS),
        "note": "lead is capped at the 16-step label horizon in this offline measure",
    }

    live = None
    if Path(args.live_json).exists() and Path(args.live_npz).exists():
        det = json.loads(Path(args.live_json).read_text())["gate6"]["nominal_detection"]
        z = np.load(args.live_npz)
        flag, fall, event = z["nominal__flag_step"], z["nominal__fall_step"], z["nominal__event"]
        live_leads = (fall - flag)[(event == 1) & (fall >= 0)].astype(np.float64)
        tp, missed, fa = det["num_true_alarms"], det["num_missed_falls"], det["num_false_alarms"]
        live = {
            "trials": det["num_trials"], "true_alarms": tp, "missed_falls": missed, "false_alarms": fa,
            "recall": tp / (tp + missed), "recall_ci95": list(stats.clopper_pearson_ci(tp, tp + missed)),
            "precision": tp / (tp + fa), "precision_ci95": list(stats.clopper_pearson_ci(tp, tp + fa)),
            "median_lead_steps": float(np.median(live_leads)),
            "median_lead_steps_ci95": list(stats.bootstrap_ci(live_leads, np.median)),
            "median_lead_seconds": float(np.median(live_leads) * STEP_SECONDS),
            "note": "live walker under the 150 N push, no horizon cap on lead; a false alarm = no fall within 3 s of the flag",
        }

    res = {"config": vars(args), "threshold": threshold, "reproduction_check": repro,
           "logreg_reproduced_within_1e-3": bool(logreg_reproduced),
           "test_transitions": int(len(y_test)), "test_episodes": cb["n_clusters"], "n_boot": args.n_boot,
           "offline": offline, "per_fall_offline": per_fall, "live": live}
    (out_dir / "predictor_stats.json").write_text(json.dumps(res, indent=2))

    def row(label, key, fmt="{:.4f}"):
        o = offline[key]
        extra = f" | {o['bootstrap_p_two_sided']:.4f}" if "bootstrap_p_two_sided" in o else " | "
        return f"| {label} | {fmt.format(o['estimate'])} | [{fmt.format(o['ci95'][0])}, {fmt.format(o['ci95'][1])}]{extra} |"

    L = ["# Fall predictor: uncertainty and significance", "",
         f"Test split: {len(y_test):,} transitions from {cb['n_clusters']:,} episodes; base rate {y_test.mean():.4f}. "
         f"Intervals: 95%, cluster bootstrap over episodes ({args.n_boot} resamples). Threshold {threshold:.4f} (chosen on validation).",
         "One trained predictor: uncertainty is over episodes, not training seeds.", "",
         "| quantity | estimate | 95% CI | bootstrap p (difference = 0) |", "|---|---|---|---|",
         row("MLP AUPRC", "mlp_auprc"), row("MLP AUROC", "mlp_auroc"),
         row("Logistic regression AUPRC", "logreg_auprc"), row("Logistic regression AUROC", "logreg_auroc"),
         row("AUPRC, MLP minus logistic regression", "auprc_mlp_minus_logreg"),
         row("AUROC, MLP minus logistic regression", "auroc_mlp_minus_logreg"),
         row("AUPRC, MLP minus base rate (chance)", "auprc_mlp_minus_base_rate"),
         row("MLP Brier score", "mlp_brier"), row("MLP precision at threshold", "mlp_precision"),
         row("MLP recall at threshold", "mlp_recall"), row("MLP F1 at threshold", "mlp_f1"), "",
         f"A bootstrap p of {1 / args.n_boot:.4f} means no resample crossed zero (p < {1 / args.n_boot:.4f}).", "",
         "## Per falling episode (offline test split)", "",
         f"- Flagged before the fall: {per_fall['flagged_before_fall']} of {per_fall['falling_test_episodes']} "
         f"({100 * per_fall['detection_rate']:.1f}%, exact 95% CI {100 * per_fall['detection_rate_ci95'][0]:.1f}-"
         f"{100 * per_fall['detection_rate_ci95'][1]:.1f}%).",
         f"- Median lead: {per_fall['median_lead_steps']:.0f} steps ({per_fall['median_lead_seconds']:.2f} s), "
         f"95% CI {per_fall['median_lead_steps_ci95'][0]:.0f}-{per_fall['median_lead_steps_ci95'][1]:.0f} steps "
         "(capped at the 16-step label horizon)."]
    if live:
        L += ["", "## Live detection (walker under the 150 N push, 1,024 trials)", "",
              f"- Recall: {live['true_alarms']} of {live['true_alarms'] + live['missed_falls']} falls flagged first "
              f"({100 * live['recall']:.2f}%, exact 95% CI {100 * live['recall_ci95'][0]:.2f}-{100 * live['recall_ci95'][1]:.2f}%).",
              f"- Precision: {live['true_alarms']} of {live['true_alarms'] + live['false_alarms']} flags followed by a fall within 3 s "
              f"({100 * live['precision']:.1f}%, exact 95% CI {100 * live['precision_ci95'][0]:.1f}-{100 * live['precision_ci95'][1]:.1f}%).",
              f"- Median lead: {live['median_lead_steps']:.0f} steps ({live['median_lead_seconds']:.2f} s), "
              f"95% CI {live['median_lead_steps_ci95'][0]:.0f}-{live['median_lead_steps_ci95'][1]:.0f} steps (no horizon cap)."]
    md = "\n".join(L) + "\n"
    (out_dir / "predictor_stats.md").write_text(md)
    if not args.quiet:
        print(md)
    print(f"[predictor-stats] wrote {out_dir}/predictor_stats.json and predictor_stats.md ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
