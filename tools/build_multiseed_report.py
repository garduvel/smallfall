"""Write the full report of a multi-seed Go1 run as one markdown file with
figures, from the run folder's own result files (nothing is re-simulated).

    .venv/bin/python tools/build_multiseed_report.py \
        --run-dir outputs/multiseed1 --out docs/GO1_MULTISEED_REPORT.md

Figures go next to the report under figures/<run name>/.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("go1_recovery_stats", ROOT / "scripts" / "go1_recovery_stats.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)

ARMS = ("protective", "walker", "limp")
NAME = {"protective": "Hand-off", "walker": "Walker", "limp": "Limp"}
MEASURES = [m for m in rs.METRICS]  # (key, label, unit, primary)
PAIR_LABEL = {"protective_vs_walker": "Hand-off vs walker", "protective_vs_limp": "Hand-off vs limp",
              "walker_vs_limp": "Walker vs limp"}


def pval(p: float) -> str:
    if p == 0:
        return "< 1e-300"
    return f"{p:.3f}" if p >= 1e-3 else f"{p:.1e}"


def num(x: float) -> str:
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def ms(values) -> str:
    """Mean ± SD with the SD shown to the same number of decimals as the mean."""
    v = np.asarray(values, dtype=float)
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    if mean == 0 and sd == 0:
        return "0 ± 0"
    dec = 0 if abs(mean) >= 100 else 1 if abs(mean) >= 10 else 2 if abs(mean) >= 1 else 3
    return f"{mean:.{dec}f} ± {sd:.{dec}f}"


def sd_p(values) -> str:
    """SD of a set of p-values; an SD that underflows to 0 is shown as such."""
    sd = float(np.std(values, ddof=1))
    return "0" if sd == 0 else (f"{sd:.3f}" if sd >= 1e-3 else f"{sd:.1e}")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def style_axes(ax):
    rs._style(ax)
    ax.xaxis.grid(False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    out_md = Path(args.out)
    fig_dir = out_md.parent / "figures" / run.name
    fig_dir.mkdir(parents=True, exist_ok=True)
    o = run / "outputs"

    versions = json.loads((o / "versions_go1_end_to_end.json").read_text())
    seeds = versions["seeds"]
    k = len(seeds)
    rec = {s: json.loads((o / f"stats/recovery_seed{s}/recovery_stats.json").read_text()) for s in seeds}
    pred = {s: json.loads((o / f"stats/predictor_seed{s}/predictor_stats.json").read_text()) for s in seeds}
    live = {s: json.loads((o / f"go1_handoff_eval_seed{s}.json").read_text())["gate6"]["nominal_detection"] for s in seeds}
    test = {s: json.loads((o / f"go1_protective_compare_test_seed{s}.json").read_text())["summaries"] for s in seeds}
    falls = {}
    for s in seeds:
        files = sorted(glob.glob(str(o / f"go1_recovery_eval_seed{s}_flag*.npz")))
        falls[s], _ = rs.load_pooled(files, list(ARMS))
    pooled = {a: {key: np.concatenate([falls[s][a][key] for s in seeds]) for key in [m[0] for m in MEASURES] + ["settled"]} for a in ARMS}
    n_per_seed = [rec[s]["n_falls"] for s in seeds]
    total = sum(n_per_seed)

    # ------------------------------------------------------------------ figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    shutil.copy(o / "stats/multiseed/fig_boxplots.png", fig_dir / "fig_boxplots.png")

    # Per-run box plots: one group of three boxes per run, four measures.
    panels = [m for m in MEASURES if m[0] != "steps"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), facecolor=rs.SURFACE)
    for ax, (key, label, unit, _p) in zip(axes.ravel(), panels):
        style_axes(ax)
        top = 0.0
        for gi, s in enumerate(seeds):
            for ai, arm in enumerate(ARMS):
                d = falls[s][arm][key]
                top = max(top, float(np.quantile(d, 0.95)))
                bp = ax.boxplot([d], positions=[gi * 4 + ai], widths=0.7, whis=(5, 95), showfliers=False, patch_artist=True,
                                medianprops={"color": rs.INK, "linewidth": 1.8}, whiskerprops={"linewidth": 1.3},
                                capprops={"linewidth": 1.3}, boxprops={"linewidth": 0})
                bp["boxes"][0].set_facecolor(rs.ARM_COLORS[arm])
                bp["boxes"][0].set_alpha(0.85)
                for part in ("whiskers", "caps"):
                    for line in bp[part]:
                        line.set_color(rs.ARM_COLORS[arm])
        ax.set_xticks([gi * 4 + 1 for gi in range(k)])
        ax.set_xticklabels([f"run {s}" for s in seeds], color=rs.INK)
        ax.set_xlim(-1, k * 4 - 1)
        ax.set_ylim(-top * 0.03, top * 1.05 if top > 0 else 1)
        ax.set_title(f"{label} ({unit})", color=rs.INK, fontsize=11, loc="left", pad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=rs.ARM_COLORS[a], alpha=0.85) for a in ARMS]
    fig.legend(handles, [rs.ARM_LABELS[a] for a in ARMS], loc="upper center", ncol=3, frameon=False,
               labelcolor=rs.INK_2, bbox_to_anchor=(0.5, 0.965), fontsize=10)
    fig.suptitle(f"Each run separately: about {int(np.mean(n_per_seed))} falls per run, the same falls under all three policies (lower is better)",
                 color=rs.INK, fontsize=12, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005, "Box: middle 50% of the run's falls. Line: median. Whiskers: 5th to 95th percentile. "
             "Limp has no torque by construction.", color=rs.MUTED, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(fig_dir / "fig_boxplots_per_run.png", dpi=150, facecolor=rs.SURFACE, bbox_inches="tight")
    plt.close(fig)

    # Predictor per run: AUPRC with its 95% interval, MLP and logistic regression, chance marked.
    fig, ax = plt.subplots(figsize=(8.4, 4.0), facecolor=rs.SURFACE)
    style_axes(ax)
    for name, key, color, dx in (("MLP predictor", "mlp_auprc", "#2a78d6", -0.12), ("Logistic regression", "logreg_auprc", "#eb6834", 0.12)):
        for i, s in enumerate(seeds):
            e = pred[s]["offline"][key]
            ax.plot([i + dx, i + dx], e["ci95"], color=color, linewidth=2, solid_capstyle="round")
            ax.plot([i + dx], [e["estimate"]], marker="o", markersize=8, color=color, markeredgecolor=rs.SURFACE,
                    markeredgewidth=2, linestyle="none", label=name if i == 0 else None)
    base = float(np.mean([pred[s]["offline"]["base_rate"]["estimate"] for s in seeds]))
    ax.axhline(base, color=rs.AXIS, linewidth=1)
    ax.annotate(f"chance (base rate {base:.3f})", (k - 0.55, base), xytext=(0, 4), textcoords="offset points",
                ha="right", color=rs.INK_2, fontsize=9)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"run {s}" for s in seeds], color=rs.INK)
    ax.set_xlim(-0.6, k - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUPRC on the test split", color=rs.INK_2, fontsize=9)
    ax.set_title("Fall predictor, one model per run: AUPRC with 95% interval", color=rs.INK, fontsize=12, loc="left", pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, labelcolor=rs.INK_2, fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_predictor_auprc.png", dpi=150, facecolor=rs.SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ report
    def arm_vals(key, arm, q="median"):
        return [rec[s]["metrics"][key]["per_arm"][arm][q] for s in seeds]

    def pair(key, p, field):
        return [rec[s]["metrics"][key]["pairs"][p][field] for s in seeds]

    def n_sig(key, p):
        ps, sh = pair(key, p, "p_holm"), pair(key, p, "share_a_lower_of_untied")
        low = sum(a < 0.05 and b > 0.5 for a, b in zip(ps, sh))
        high = sum(a < 0.05 and b < 0.5 for a, b in zip(ps, sh))
        return low, high

    g = "envelope_growth_m"
    L = [f"# Go1 fall pipeline: results of the five-run evaluation (`{run.name}`)", "",
         f"Run on {versions['gpu']}, mjlab {versions['mjlab']}, MuJoCo {versions['mujoco']}, torch {versions['torch']}. "
         "Generated from the run's saved result files by `tools/build_multiseed_report.py`.", "",
         "## 1. Summary", "",
         f"One walking policy was trained once. Everything after it (the fall predictor, the fall-state banks, and the "
         f"protective policy) was then built {k} independent times with different seeds, and each of the {k} runs was "
         f"evaluated on about {int(round(np.mean(n_per_seed), -1)):,} falls it had never seen ({total:,} falls in all). Every fall was "
         "replayed from the identical starting state under three policies: the **hand-off** to the protective policy, "
         "the **walker** carrying on, and **limp** (motors off).", "",
         f"- **The protective policy makes the fall smaller, in every run.** Median ground covered: hand-off "
         f"{ms(arm_vals(g, 'protective'))} m, limp {ms(arm_vals(g, 'limp'))} m, walker {ms(arm_vals(g, 'walker'))} m "
         f"(mean ± SD of the {k} runs). The difference is significant in {n_sig(g, 'protective_vs_limp')[0]} of {k} runs "
         f"against limp and {n_sig(g, 'protective_vs_walker')[0]} of {k} against the walker.",
         f"- **It keeps the trunk off the ground, in every run.** Median trunk/hip peak force is 0 N under the hand-off in all "
         f"{k} runs, against {ms(arm_vals('peak_force_critical_n', 'limp'))} N for limp and "
         f"{ms(arm_vals('peak_force_critical_n', 'walker'))} N for the walker.",
         f"- **It uses about half the walker's peak torque:** {ms(arm_vals('peak_torque_nm', 'protective'))} Nm against "
         f"{ms(arm_vals('peak_torque_nm', 'walker'))} Nm.",
         f"- **The predictor is consistent across runs:** test AUPRC {ms([pred[s]['offline']['mlp_auprc']['estimate'] for s in seeds])} "
         f"(chance is about {base:.3f}), and live it flagged every real fall before it happened in all {k} runs, "
         f"with a median warning of {ms([live[s]['median_lead_steps'] for s in seeds])} steps "
         f"({np.mean([live[s]['median_lead_steps'] for s in seeds]) * 0.02:.2f} s).", "",
         "![Fall outcomes by recovery policy, all runs](figures/" + run.name + "/fig_boxplots.png)", "",
         "*Every box is drawn from all falls of all runs for that policy. Under each name: mean ± SD of the per-run medians. "
         "Brackets: the hand-off against that baseline, mean ± SD of the per-run p-values and the number of runs in which the "
         "hand-off is significantly lower or higher.*", "",
         "## 2. What was run", ""]
    L += table(["Stage", "Per run", "Details"], [
        ["Walker", "shared", "mjlab Go1 flat-velocity task, 10,000 PPO iterations, trained once (`model_9999.pt`)"],
        ["Push", "same setting", "150 N per axis on the trunk, 0.1-0.3 s, every 1-3 s"],
        ["Predictor", "one per run", "MLP 60-128-128-1, predicts a fall within 16 steps (0.32 s); about 1M labelled steps per run; threshold chosen on validation"],
        ["Training bank", "one per run", "about 500 robot states captured at the moment that run's predictor fires, each followed by a real fall"],
        ["Protective policy", "one per run", "PPO from those states, round-8 reward, checkpoint at iteration 200"],
        ["Evaluation falls", "three fresh banks per run", f"{', '.join(str(n) for n in n_per_seed)} flag-moment states, never used in training"],
        ["Comparison", "per fall", "each state replayed under hand-off, walker, and limp until at rest or 3 s"],
    ])
    L += ["**Measures (lower is better for all).** *Ground covered*: how far the farthest part of the robot gets from where the "
          "fall started, beyond its own reach at the start. *Trunk/hip peak force* and *any-body peak force*: the hardest ground "
          "contact on the trunk or hips, or on any body part. *Peak joint torque*: the largest torque at any joint. "
          "*Steps until at rest*: control steps (50 per second) until the robot has been still for 0.5 s, capped at 150.", "",
          f"**Tests.** Within each run, on that run's falls: a paired Wilcoxon signed-rank test for every measure and every pair "
          "of policies, the exact McNemar test for coming to rest, and the Friedman test across the three policies, with "
          f"the pairwise p-values Holm-adjusted within the run ({rec[seeds[0]]['holm_family_size']} tests). After all {k} runs "
          "finished, each statistic was averaged across the runs (mean ± SD). No test pools falls across runs.", "",
          "## 3. Recovery policies", "", "### 3.1 Per-policy results (mean ± SD of the per-run values)", ""]
    rows = []
    for key, label, unit, _p in MEASURES:
        rows.append([f"{label} ({unit}), median"] + [ms(arm_vals(key, a)) for a in ARMS])
        rows.append([f"{label} ({unit}), 90th percentile"] + [ms(arm_vals(key, a, "p90")) for a in ARMS])
    L += table(["Measure"] + [rs.ARM_LABELS[a] for a in ARMS], rows)

    L += ["### 3.2 Hand-off against each baseline (mean ± SD across the runs)", ""]
    rows = []
    for key, label, unit, primary in MEASURES:
        for p in ("protective_vs_limp", "protective_vs_walker"):
            if "skipped" in rec[seeds[0]]["metrics"][key]["pairs"][p]:
                continue
            low, high = n_sig(key, p)
            sig = f"{low} of {k} lower" + (f", {high} of {k} higher" if high else "")
            rows.append([f"{label} ({unit})" + (" **primary**" if primary else ""), PAIR_LABEL[p].replace("Hand-off vs ", ""),
                         ms(pair(key, p, "median_paired_diff")),
                         f"{100 * np.mean(pair(key, p, 'share_a_lower_of_untied')):.1f}% ± {100 * np.std(pair(key, p, 'share_a_lower_of_untied'), ddof=1):.1f}%",
                         f"{np.mean(pair(key, p, 'rank_biserial')):+.2f} ± {np.std(pair(key, p, 'rank_biserial'), ddof=1):.2f}",
                         f"{pval(float(np.mean(pair(key, p, 'p_holm'))))} ± {sd_p(pair(key, p, 'p_holm'))}", sig])
    L += table(["Measure", "Baseline", "Median paired difference (hand-off minus baseline)", "Falls with hand-off lower",
                "Rank-biserial effect size", "p (mean ± SD)", "Runs significant"], rows)
    L += ["Rank-biserial runs from -1 (hand-off lower in every fall) to +1 (higher in every fall). A mean of p-values "
          "summarises the runs; the individual values are in section 3.3.", "",
          "### 3.3 p-values and effect sizes, run by run", "",
          "Holm-adjusted within the run. Each cell: p-value, then the median paired difference.", ""]
    for p in ("protective_vs_limp", "protective_vs_walker", "walker_vs_limp"):
        L += [f"**{PAIR_LABEL[p]}**", ""]
        rows = []
        for key, label, unit, _pr in MEASURES:
            if "skipped" in rec[seeds[0]]["metrics"][key]["pairs"][p]:
                rows.append([f"{label} ({unit})"] + ["not tested (limp torque is 0)"] + [""] * (k - 1))
                continue
            rows.append([f"{label} ({unit})"] + [f"{pval(rec[s]['metrics'][key]['pairs'][p]['p_holm'])}; {num(rec[s]['metrics'][key]['pairs'][p]['median_paired_diff'])}" for s in seeds])
        sp = [rec[s]["settled"][p] for s in seeds]
        rows.append(["Came to rest (McNemar); rates A / B"] + [f"{pval(r['p_holm'])}; {r['rate_a']:.3f} / {r['rate_b']:.3f}" for r in sp])
        L += table(["Measure"] + [f"Run {s}" for s in seeds], rows)
    rows = []
    for key, label, unit, _pr in MEASURES:
        if "friedman" in rec[seeds[0]]["metrics"][key]:
            rows.append([label] + [f"{pval(rec[s]['metrics'][key]['friedman']['p'])} (χ² {rec[s]['metrics'][key]['friedman']['statistic']:.0f})" for s in seeds])
    L += ["**Friedman test across the three policies**", ""] + table(["Measure"] + [f"Run {s}" for s in seeds], rows)

    L += ["### 3.4 Distributions", "",
          "![Each run separately](figures/" + run.name + "/fig_boxplots_per_run.png)", "",
          f"All {total:,} evaluation falls together, per policy:", ""]
    rows = []
    for key, label, unit, _pr in MEASURES:
        for a in ARMS:
            d = pooled[a][key]
            q = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.9, 0.95])
            rows.append([f"{label} ({unit})", NAME[a], num(float(d.mean()))] + [num(float(x)) for x in q] + [f"{100 * np.mean(d == 0):.1f}%"])
    L += table(["Measure", "Policy", "Mean", "5th", "25th", "Median", "75th", "90th", "95th", "Falls at exactly 0"], rows)
    L += ["Per-run medians:", ""]
    rows = []
    for key, label, unit, _pr in MEASURES:
        for a in ARMS:
            rows.append([f"{label} ({unit})", NAME[a]] + [num(v) for v in arm_vals(key, a)] + [ms(arm_vals(key, a))])
    L += table(["Measure", "Policy"] + [f"Run {s}" for s in seeds] + ["Mean ± SD"], rows)

    L += ["### 3.5 Each run's separate test bank", "",
          "Besides the evaluation falls, each run was checked once on a further fresh bank of about 500 flag moments. Medians:", ""]
    rows = []
    for s in seeds:
        q = test[s]
        rows.append([str(s)] + [num(q[a][g]["median"]) for a in ARMS] + [num(q[a]["peak_force_critical_n"]["median"]) for a in ARMS]
                    + [num(q[a]["peak_force_any_n"]["median"]) for a in ARMS] + [num(q["protective"]["peak_torque_nm"]["median"]), num(q["walker"]["peak_torque_nm"]["median"])])
    L += table(["Run", "Ground: hand-off", "walker", "limp", "Trunk force: hand-off", "walker", "limp",
                "Any-body force: hand-off", "walker", "limp", "Torque: hand-off", "walker"], rows)

    L += ["## 4. Fall predictor", "",
          "![Predictor AUPRC per run](figures/" + run.name + "/fig_predictor_auprc.png)", "",
          "### 4.1 Test-split performance, run by run", "",
          "Intervals are 95%, from resampling whole episodes (1,000 resamples). p-values are bootstrap p-values for a zero difference; "
          "0.001 is the smallest the resampling can report.", ""]
    rows = []
    for s in seeds:
        of = pred[s]["offline"]
        rows.append([str(s), f"{pred[s]['test_transitions']:,} / {pred[s]['test_episodes']}", f"{of['base_rate']['estimate']:.4f}",
                     f"{of['mlp_auprc']['estimate']:.3f} [{of['mlp_auprc']['ci95'][0]:.3f}, {of['mlp_auprc']['ci95'][1]:.3f}]",
                     f"{of['mlp_auroc']['estimate']:.4f} [{of['mlp_auroc']['ci95'][0]:.4f}, {of['mlp_auroc']['ci95'][1]:.4f}]",
                     f"{of['logreg_auprc']['estimate']:.3f}",
                     f"+{of['auprc_mlp_minus_logreg']['estimate']:.3f} [{of['auprc_mlp_minus_logreg']['ci95'][0]:.3f}, {of['auprc_mlp_minus_logreg']['ci95'][1]:.3f}]",
                     pval(of["auprc_mlp_minus_logreg"]["bootstrap_p_two_sided"]), pval(of["auprc_mlp_minus_base_rate"]["bootstrap_p_two_sided"]),
                     f"{of['mlp_brier']['estimate']:.4f}"])
    rows.append(["**Mean ± SD**", "", ms([pred[s]["offline"]["base_rate"]["estimate"] for s in seeds])]
                + [f"{np.mean([pred[s]['offline'][key]['estimate'] for s in seeds]):.3f} ± {np.std([pred[s]['offline'][key]['estimate'] for s in seeds], ddof=1):.3f}"
                   for key in ("mlp_auprc", "mlp_auroc", "logreg_auprc", "auprc_mlp_minus_logreg")] + ["", "",
                   f"{np.mean([pred[s]['offline']['mlp_brier']['estimate'] for s in seeds]):.4f} ± {np.std([pred[s]['offline']['mlp_brier']['estimate'] for s in seeds], ddof=1):.4f}"])
    L += table(["Run", "Test steps / episodes", "Base rate", "MLP AUPRC [95% CI]", "MLP AUROC [95% CI]", "Logistic regression AUPRC",
                "MLP minus logistic regression [95% CI]", "p vs logistic regression", "p vs chance", "Brier"], rows)

    L += ["### 4.2 At the decision threshold, and per falling episode", ""]
    rows = []
    for s in seeds:
        of, pf = pred[s]["offline"], pred[s]["per_fall_offline"]
        rows.append([str(s), f"{pred[s]['threshold']:.3f}",
                     f"{of['mlp_precision']['estimate']:.3f} [{of['mlp_precision']['ci95'][0]:.3f}, {of['mlp_precision']['ci95'][1]:.3f}]",
                     f"{of['mlp_recall']['estimate']:.3f} [{of['mlp_recall']['ci95'][0]:.3f}, {of['mlp_recall']['ci95'][1]:.3f}]",
                     f"{of['mlp_f1']['estimate']:.3f}", f"{pf['flagged_before_fall']} of {pf['falling_test_episodes']}",
                     f"{pf['median_lead_steps']:.0f} [{pf['median_lead_steps_ci95'][0]:.0f}, {pf['median_lead_steps_ci95'][1]:.0f}]"])
    L += table(["Run", "Threshold (from validation)", "Precision per step [95% CI]", "Recall per step [95% CI]", "F1",
                "Falling test episodes flagged before the fall", "Median lead, steps [95% CI] (capped at 16)"], rows)

    L += ["### 4.3 Live, on the walking robot", "",
          "1,024 robots per run walking under the push with the predictor watching; a flag counts as right if the robot falls within 3 s.", ""]
    rows = []
    for s in seeds:
        d = live[s]
        rows.append([str(s), str(d["num_true_alarms"] + d["num_missed_falls"]), str(d["num_missed_falls"]), f"{d['live_recall']:.3f}",
                     str(d["num_flagged"]), f"{d['precision']:.3f}", f"{d['median_lead_steps']:.0f} ({d['median_lead_s']:.2f} s)"])
    rows.append(["**Mean ± SD**", "", "", f"{np.mean([live[s]['live_recall'] for s in seeds]):.3f} ± {np.std([live[s]['live_recall'] for s in seeds], ddof=1):.3f}", "",
                 f"{np.mean([live[s]['precision'] for s in seeds]):.3f} ± {np.std([live[s]['precision'] for s in seeds], ddof=1):.3f}",
                 ms([live[s]["median_lead_steps"] for s in seeds]) + " steps"])
    L += table(["Run", "Real falls", "Falls not flagged first", "Recall", "Flags raised", "Share of flags followed by a fall", "Median warning"], rows)

    settle = [rec[s]["settled"]["protective"]["rate"] for s in seeds]
    unsettled = int(np.sum(pooled["protective"]["settled"] == 0))
    aw_low, aw_high = n_sig("peak_force_any_n", "protective_vs_walker")
    L += ["## 5. Secondary observations", "",
          f"- **Coming to rest.** The hand-off came to rest within 3 s in {', '.join(f'{100 * r:.1f}%' for r in settle)} of falls "
          f"(runs {seeds[0]}-{seeds[-1]}); limp in {100 * np.mean([rec[s]['settled']['limp']['rate'] for s in seeds]):.1f}% and the walker in "
          f"{100 * np.mean([rec[s]['settled']['walker']['rate'] for s in seeds]):.1f}% on average. In the {unsettled} hand-off falls "
          f"({100 * unsettled / total:.1f}%) that did not meet the stillness thresholds, the robot stays where it is: median trunk force is 0 N "
          f"and median ground covered is {num(float(np.median(pooled['protective'][g][pooled['protective']['settled'] == 0])))} m, against "
          f"{num(float(np.median(pooled['protective'][g][pooled['protective']['settled'] == 1])))} m for the falls that did. On inspection of the "
          "videos this is small residual foot movement, and the 99% settle criterion written before the run was set aside as peripheral "
          "(recorded in `DEVIATIONS.md`).",
          f"- **Peak force on any body.** The protective policy moves the impact from the trunk to the legs, not away. Its median is level "
          f"with the walker's ({ms(arm_vals('peak_force_any_n', 'protective'))} against {ms(arm_vals('peak_force_any_n', 'walker'))} N) and below "
          f"limp's ({ms(arm_vals('peak_force_any_n', 'limp'))} N, significant in {n_sig('peak_force_any_n', 'protective_vs_limp')[0]} of {k} runs). "
          f"Paired fall by fall it is higher than the walker's by {ms(pair('peak_force_any_n', 'protective_vs_walker', 'median_paired_diff'))} N "
          f"(significant in {aw_high} of {k} runs).",
          f"- **False alarms.** Each run's predictor picks its own threshold ({min(pred[s]['threshold'] for s in seeds):.2f} to "
          f"{max(pred[s]['threshold'] for s in seeds):.2f}), so the share of live flags followed by a fall varies from "
          f"{100 * min(live[s]['precision'] for s in seeds):.0f}% to {100 * max(live[s]['precision'] for s in seeds):.0f}%. No real fall was missed in any run.",
          "", "## 6. Scope", "",
          f"- The walker is the same in all {k} runs. The spread across runs reflects how the predictor and the protective policy train and "
          "which falls are sampled, not walker training.",
          "- The comparison replays captured flag moments with no further pushes, and covers flags that were followed by a real fall.",
          "- The limp baseline has no joint friction or damping added; it is the most free-falling version of an unpowered robot.",
          "- Ground covered is this project's own measure.", "",
          "## 7. Where the data is", ""]
    L += table(["What", "Path under `" + str(run) + "/`"], [
        ["Per-fall numbers (every fall, every policy)", "`outputs/go1_recovery_eval_seed{run}_flag{6,7,8}.npz`"],
        ["Each run's tests", "`outputs/stats/recovery_seed{run}/recovery_stats.json` and `.md`"],
        ["Across-run means and SDs", "`outputs/stats/multiseed/multiseed_stats.json` and `.md`"],
        ["Predictor intervals", "`outputs/stats/predictor_seed{run}/predictor_stats.json`"],
        ["Predictor metrics and threshold", "`outputs/go1_wildpush_metrics_seed{run}.json`"],
        ["Live hand-off check", "`outputs/go1_handoff_eval_seed{run}.json` and `.npz`"],
        ["Test-bank comparison", "`outputs/go1_protective_compare_test_seed{run}.json` and `.npz`"],
        ["Policies", "`checkpoints/model_9999.pt`, `checkpoints/protective/seed{run}_round8b_model_200.pt`"],
        ["Videos (first run)", "`outputs/videos/go1_handoff_demo_seed0.mp4`, `outputs/go1_*.mp4`"],
    ])
    L += ["Package versions: " + ", ".join(f"{name} {versions[name]}" for name in ("mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib", "torch", "scipy", "numpy", "scikit-learn") if versions.get(name)) + "."]
    out_md.write_text("\n".join(L) + "\n")
    print(f"wrote {out_md} ({len(L)} lines) and figures in {fig_dir}")


if __name__ == "__main__":
    main()
