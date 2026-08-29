"""Phase-2 tool (outside CLAUDE.md's spec, see DEVIATIONS.md): plot a frozen
predictor's generalization metrics against a second, independent training
run's checkpoints.

Joins two JSON files that scripts/eval_predictor_generalization.py and
scripts/train.py already write:

  - outputs/predictor_generalization_<predictor>_<run_dir>.json  (per-checkpoint
    AUPRC/AUROC/base_rate/lead-time, from eval_predictor_generalization.py)
  - logs/rsl_rl/cartpole_term/<run_dir>/eval_history.json         (per-iteration
    frozen-eval mean episode length -- CLAUDE.md S6 -- i.e. the second run's
    own policy-competence measurement, independent of the predictor)

joined on `iteration`, giving predictor quality as a function of the SECOND
run's policy competence -- the interesting number here, since the predictor
was trained only on the FIRST run's `target`-point data and has never seen
this policy before.

Pure stdlib json + matplotlib. No mjlab/torch/GPU needed -- runs anywhere,
including after downloading both JSON files locally.

Usage:
    python scripts/plot_predictor_generalization.py \\
        --generalization-json outputs/predictor_generalization_predictor_target_seed0_2026-08-26_seed3.json \\
        --eval-history logs/rsl_rl/cartpole_term/2026-08-26_..._seed3/eval_history.json \\
        --out docs/figures/predictor_generalization_seed3.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_eval_history_by_iteration(path: Path) -> dict[int, dict]:
    history = json.loads(path.read_text())
    return {rec["iteration"]: rec for rec in history}


def join(generalization: dict, eval_history_by_it: dict[int, dict]) -> list[dict]:
    rows = []
    missing = []
    for r in generalization["results"]:
        it = r["iteration"]
        if it is None:
            continue
        eh = eval_history_by_it.get(it)
        if eh is None:
            missing.append(it)
            continue
        rows.append(
            {
                "iteration": it,
                "competence_mean_action_len": eh["mean_action"]["mean_episode_length"],
                "competence_sampled_action_len": eh["sampled_action"]["mean_episode_length"],
                "base_rate": r["metrics"]["base_rate"],
                "auprc": r["metrics"]["auprc"],
                "auroc": r["metrics"]["auroc"],
                "brier": r["metrics"]["brier"],
                "ece": r["metrics"]["ece"],
                "median_lead_steps": r["lead_time"]["median_lead_steps"],
            }
        )
    if missing:
        print(
            f"[plot] WARNING: {len(missing)} checkpoint iteration(s) had no matching "
            f"eval_history.json entry, dropped from the plot: {missing}. This means "
            "--eval.interval didn't evenly divide save_interval for this run -- check "
            "the training invocation if this list is large."
        )
    rows.sort(key=lambda row: row["iteration"])
    return rows


def make_figure(rows: list[dict], predictor_label: str, run_label: str, out_path: Path) -> None:
    if not rows:
        raise ValueError("no joined rows to plot -- see the WARNING above for why.")

    iterations = [r["iteration"] for r in rows]
    competence = [r["competence_mean_action_len"] for r in rows]
    base_rate = [r["base_rate"] for r in rows]
    auprc = [r["auprc"] for r in rows]
    auroc = [r["auroc"] for r in rows]
    lead = [r["median_lead_steps"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Frozen predictor ({predictor_label}) generalization vs. an independent "
        f"training run ({run_label})\ninference only -- predictor never trained on this run's data",
        fontsize=11,
    )

    ax = axes[0, 0]
    ax.plot(iterations, competence, marker="o", color="tab:gray")
    ax.set_xlabel("checkpoint iteration")
    ax.set_ylabel("frozen-eval mean episode length (policy competence)")
    ax.set_title("Second run's own competence curve")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(competence, [a if a is not None else float("nan") for a in auprc], marker="o", label="AUPRC")
    ax.plot(competence, base_rate, marker="x", linestyle="--", color="gray", label="base rate (no-skill floor)")
    ax.set_xlabel("policy competence (frozen-eval mean episode length)")
    ax.set_ylabel("AUPRC")
    ax.set_title("AUPRC vs. policy competence")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(competence, [a if a is not None else float("nan") for a in auroc], marker="o", color="tab:orange")
    ax.set_xlabel("policy competence (frozen-eval mean episode length)")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs. policy competence")
    ax.set_ylim(0.5, 1.02)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(competence, [l if l is not None else float("nan") for l in lead], marker="o", color="tab:green")
    ax.axhline(16, color="gray", linestyle=":", label="16-step horizon ceiling")
    ax.set_xlabel("policy competence (frozen-eval mean episode length)")
    ax.set_ylabel("median detection lead (steps)")
    ax.set_title("Detection lead time vs. policy competence")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[plot] wrote {out_path}")

    print("\n[plot] joined table:")
    print(f"{'iter':>6} {'competence':>11} {'base_rate':>10} {'auprc':>8} {'auroc':>8} {'lead(steps)':>12}")
    for r in rows:
        auprc_s = f"{r['auprc']:.4f}" if r["auprc"] is not None else "n/a"
        auroc_s = f"{r['auroc']:.4f}" if r["auroc"] is not None else "n/a"
        lead_s = f"{r['median_lead_steps']:.1f}" if r["median_lead_steps"] is not None else "n/a"
        print(
            f"{r['iteration']:>6} {r['competence_mean_action_len']:>11.1f} "
            f"{r['base_rate']:>10.4f} {auprc_s:>8} {auroc_s:>8} {lead_s:>12}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalization-json", type=str, required=True)
    parser.add_argument("--eval-history", type=str, required=True)
    parser.add_argument("--out", type=str, default="outputs/predictor_generalization.png")
    args = parser.parse_args()

    generalization = json.loads(Path(args.generalization_json).read_text())
    eval_history_by_it = load_eval_history_by_iteration(Path(args.eval_history))
    rows = join(generalization, eval_history_by_it)

    predictor_label = Path(generalization["predictor_weights"]).stem
    run_label = Path(args.eval_history).parent.name
    make_figure(rows, predictor_label, run_label, Path(args.out))

    table_path = Path(args.out).with_suffix(".json")
    table_path.write_text(json.dumps(rows, indent=2))
    print(f"[plot] wrote {table_path}")


if __name__ == "__main__":
    main()
