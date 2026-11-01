"""Final step of the multi-seed Go1 pipeline: average the per-run results.

Each back-half run (one seed: its own predictor, fall banks, protective
policy) does its own significance tests on its own falls with
scripts/go1_recovery_stats.py and saves them (recovery_stats.json). Nothing
is combined until every run has finished. This script then reads those saved
results and reports, for every statistic, its MEAN and STANDARD DEVIATION
across the runs:

  per policy          median and 90th percentile of each measure, settle rate
  per pair of policies  median paired difference, share of falls lower,
                      rank-biserial effect size, the run's p-value
                      (Wilcoxon, Holm-adjusted within the run), and how many
                      runs were significant at 0.05
  came to rest        McNemar p-value
  predictor           test AUPRC / AUROC / precision / recall, the MLP-minus-
                      logistic-regression difference and its p-value, live
                      recall / precision / lead

No test is run on falls pooled across runs. The figure's boxes are drawn
from all falls of all runs (a picture, not a test); the numbers on it are the
across-run means and SDs.

Note on averaging p-values: the mean of several p-values is a summary of how
the runs came out, not itself a p-value for a combined test. The count of
significant runs is reported beside it.

Scope: the walker is the same in every run, so the spread across runs
reflects predictor and protective-policy training and the sampled falls, not
walker training.

Pure numpy/matplotlib: no GPU or mjlab needed.

Usage:
    python scripts/go1_multiseed_stats.py --seeds 0 1 2 3 4 \
        --recovery-stats-template "outputs/stats/recovery_seed{seed}/recovery_stats.json" \
        --compare-npz-template "outputs/go1_recovery_eval_seed{seed}_flag*.npz" \
        --predictor-stats-template "outputs/stats/predictor_seed{seed}/predictor_stats.json" \
        --handoff-json-template "outputs/go1_handoff_eval_seed{seed}.json" \
        --out-dir outputs/stats/multiseed

Writes <out-dir>/multiseed_stats.json, multiseed_stats.md, fig_boxplots.png.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rs = _load(_HERE / "go1_recovery_stats.py", "go1_recovery_stats")  # METRICS, colours, loaders, plot style

ALPHA = 0.05  # on each run's own Holm-adjusted p-values
PAIR_STATS = (  # (key in the per-run results, column label)
    ("median_paired_diff", "median paired difference (A - B)"),
    ("share_a_lower_of_untied", "share of falls with A lower"),
    ("rank_biserial", "rank-biserial"),
    ("p_holm", "p (Holm, within the run)"),
)


def mean_sd(values) -> dict:
    """Mean and sample standard deviation across runs (one value per run)."""
    v = np.asarray(values, dtype=np.float64)
    return {"values": v.tolist(), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}


def analyse(per_run: dict[int, dict], arms: list[str]) -> dict:
    """per_run: seed -> that run's recovery_stats.json contents."""
    seeds = sorted(per_run)
    first = per_run[seeds[0]]
    out = {"seeds": seeds, "arms": arms, "n_falls_per_seed": {s: per_run[s]["n_falls"] for s in seeds},
           "metrics": {}, "settled": {"per_arm": {}, "pairs": {}}}
    for key, label, unit, primary in rs.METRICS:
        m = {"label": label, "unit": unit, "primary": primary, "per_arm": {}, "pairs": {}}
        for arm in arms:
            m["per_arm"][arm] = {q: mean_sd([per_run[s]["metrics"][key]["per_arm"][arm][q] for s in seeds])
                                 for q in ("median", "p90")}
        for pair, r0 in first["metrics"][key]["pairs"].items():
            if "skipped" in r0:
                m["pairs"][pair] = {"skipped": r0["skipped"]}
                continue
            rows = [per_run[s]["metrics"][key]["pairs"][pair] for s in seeds]
            entry = {stat: mean_sd([r[stat] for r in rows]) for stat, _ in PAIR_STATS}
            sig = [r for r in rows if r["p_holm"] < ALPHA]
            entry["n_runs_significant"] = len(sig)
            entry["n_runs_significant_a_lower"] = int(sum(r["share_a_lower_of_untied"] > 0.5 for r in sig))
            entry["n_runs_significant_a_higher"] = int(sum(r["share_a_lower_of_untied"] < 0.5 for r in sig))
            m["pairs"][pair] = entry
        if "friedman" in first["metrics"][key]:
            m["friedman_p"] = mean_sd([per_run[s]["metrics"][key]["friedman"]["p"] for s in seeds])
        out["metrics"][key] = m
    for arm in arms:
        out["settled"]["per_arm"][arm] = mean_sd([per_run[s]["settled"][arm]["rate"] for s in seeds])
    for pair, r0 in first["settled"].items():
        if "_vs_" in pair:
            ps = [per_run[s]["settled"][pair]["p_holm"] for s in seeds]
            out["settled"]["pairs"][pair] = {"p_holm": mean_sd(ps), "n_runs_significant": int(sum(p < ALPHA for p in ps))}
    return out


PREDICTOR_ROWS = (
    ("mlp_auprc", "MLP AUPRC"), ("mlp_auroc", "MLP AUROC"), ("logreg_auprc", "Logistic regression AUPRC"),
    ("auprc_mlp_minus_logreg", "AUPRC, MLP minus logistic regression"),
    ("auprc_mlp_minus_base_rate", "AUPRC, MLP minus base rate"),
    ("mlp_precision", "Precision at threshold"), ("mlp_recall", "Recall at threshold"),
)
PREDICTOR_P_ROWS = (
    ("auprc_mlp_minus_logreg__p", "p, MLP vs logistic regression (AUPRC)"),
    ("auprc_mlp_minus_base_rate__p", "p, MLP vs chance (AUPRC)"),
)
LIVE_ROWS = (("live_recall", "Live recall"), ("precision", "Live precision"), ("median_lead_steps", "Live median lead (steps)"))


def predictor_summary(seeds: list[int], pred_template: str | None, live_template: str | None) -> dict:
    out = {}
    if pred_template:
        files = {s: Path(pred_template.format(seed=s)) for s in seeds}
        missing = [str(f) for f in files.values() if not f.exists()]
        if missing:
            raise FileNotFoundError(f"[multiseed] missing predictor stats: {missing}")
        data = {s: json.loads(f.read_text())["offline"] for s, f in files.items()}
        for key, _label in PREDICTOR_ROWS:
            out[key] = mean_sd([data[s][key]["estimate"] for s in seeds])
        for key, _label in PREDICTOR_P_ROWS:
            base = key[: -len("__p")]
            ps = [data[s][base]["bootstrap_p_two_sided"] for s in seeds]
            out[key] = {**mean_sd(ps), "n_runs_significant": int(sum(p < ALPHA for p in ps))}
    if live_template:
        files = {s: Path(live_template.format(seed=s)) for s in seeds}
        missing = [str(f) for f in files.values() if not f.exists()]
        if missing:
            raise FileNotFoundError(f"[multiseed] missing hand-off results: {missing}")
        det = {s: json.loads(f.read_text())["gate6"]["nominal_detection"] for s, f in files.items()}
        for key, _label in LIVE_ROWS:
            out[key] = mean_sd([float(det[s][key]) for s in seeds])
    return out


def _fmt_vals(vals, fmt=rs._n) -> str:
    return ", ".join(fmt(v) for v in vals)


def _ms(d: dict, fmt=None, sd_fmt=None) -> str:
    fmt = fmt or rs._n
    return f"{fmt(d['mean'])} \u00b1 {(sd_fmt or fmt)(d['sd'])}"


def _pfmt(x: float) -> str:
    return f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}"


def to_markdown(res: dict, pred: dict) -> str:
    seeds, arms = res["seeds"], res["arms"]
    k = len(seeds)
    L = [f"# Go1 back half: {k} runs, results averaged at the end", "",
         f"Runs (seeds) {seeds}; falls per run: {list(res['n_falls_per_seed'].values())}. Same walker in every run.",
         "Every run did its own tests on its own falls. Each cell below is the mean \u00b1 standard deviation of that "
         f"statistic across the {k} runs. Lower is better for every measure.", "",
         "## Per policy", "", "| measure | " + " | ".join(rs.ARM_LABELS.get(a, a) for a in arms) + " |",
         "|---|" + "---|" * len(arms)]
    for key, m in res["metrics"].items():
        L.append(f"| {m['label']} ({m['unit']}), median | " + " | ".join(_ms(m["per_arm"][a]["median"]) for a in arms) + " |")
        L.append(f"| {m['label']} ({m['unit']}), 90th percentile | " + " | ".join(_ms(m["per_arm"][a]["p90"]) for a in arms) + " |")
    L.append("| Came to rest (rate) | " + " | ".join(_ms(res["settled"]["per_arm"][a], lambda x: f"{x:.3f}") for a in arms) + " |")
    L += ["", "## Pairwise tests (A vs B), paired on the fall within each run", "",
          f"Test in each run: Wilcoxon signed-rank, Holm-adjusted within the run. Significant = p < {ALPHA}.", "",
          "| measure | A vs B | median paired difference A - B | share of falls with A lower | rank-biserial | p | runs significant (A lower / A higher) |",
          "|---|---|---|---|---|---|---|"]
    for key, m in res["metrics"].items():
        for pair, v in m["pairs"].items():
            a, b = pair.split("_vs_")
            if "skipped" in v:
                L.append(f"| {m['label']} ({m['unit']}) | {a} vs {b} | not tested: {v['skipped']} | | | | |")
                continue
            L.append(f"| {m['label']} ({m['unit']}){' (primary)' if m['primary'] else ''} | {a} vs {b} | "
                     f"{_ms(v['median_paired_diff'])} | {_ms(v['share_a_lower_of_untied'], lambda x: f'{100 * x:.1f}%')} | "
                     f"{_ms(v['rank_biserial'], lambda x: f'{x:+.2f}', lambda x: f'{x:.2f}')} | {_ms(v['p_holm'], _pfmt)} | "
                     f"{v['n_runs_significant']} of {k} ({v['n_runs_significant_a_lower']} / {v['n_runs_significant_a_higher']}) |")
    L += ["", "## Came to rest (exact McNemar in each run)", "", "| A vs B | p | runs significant |", "|---|---|---|"]
    for pair, v in res["settled"]["pairs"].items():
        a, b = pair.split("_vs_")
        L.append(f"| {a} vs {b} | {_ms(v['p_holm'], _pfmt)} | {v['n_runs_significant']} of {k} |")
    L += ["", "## Friedman test across the three policies (in each run)", "", "| measure | p |", "|---|---|"]
    for key, m in res["metrics"].items():
        if "friedman_p" in m:
            L.append(f"| {m['label']} | {_ms(m['friedman_p'], _pfmt)} |")
    if pred:
        L += ["", "## Predictor (one model per run)", "", "| quantity | mean \u00b1 SD | runs significant |", "|---|---|---|"]
        f4 = lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:.1f}"  # noqa: E731
        for key, label in PREDICTOR_ROWS + PREDICTOR_P_ROWS + LIVE_ROWS:
            if key in pred:
                sig = f"{pred[key]['n_runs_significant']} of {k}" if "n_runs_significant" in pred[key] else ""
                L.append(f"| {label} | {_ms(pred[key], _pfmt if key.endswith('__p') else f4)} | {sig} |")
    L += ["", "A mean of p-values summarises how the runs came out; it is not the p-value of a combined test. "
          "The per-run values are in multiseed_stats.json.",
          "The walker is the same in every run: the spread reflects predictor and protective-policy training and "
          "the sampled falls, not walker training."]
    return "\n".join(L) + "\n"


def figure(per_seed_falls: dict[int, dict], res: dict, out_path: Path) -> None:
    """Four box-and-whisker panels. Every box is drawn from all falls of all
    runs for that policy. Under each policy name: mean +/- SD of the per-run
    medians. Brackets: mean +/- SD of the per-run p-values for the hand-off
    against that baseline, and how many runs were significant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds, arms = res["seeds"], res["arms"]
    k = len(seeds)
    cand = arms[0]
    short = {"protective": "Hand-off", "walker": "Walker", "limp": "Limp"}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    # Four panels; any-body peak force is left to the tables (user, 2026-10-12).
    panels = [m for m in rs.METRICS if m[0] != "peak_force_any_n"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 5.2), facecolor=rs.SURFACE)
    for ax, (key, label, unit, _primary) in zip(np.atleast_1d(axes), panels):
        rs._style(ax)
        ax.xaxis.grid(False)
        data = [np.concatenate([per_seed_falls[s][a][key] for s in seeds]) for a in arms]
        bp = ax.boxplot(data, positions=range(len(arms)), widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops={"color": rs.INK, "linewidth": 2}, whiskerprops={"linewidth": 1.5},
                        capprops={"linewidth": 1.5}, boxprops={"linewidth": 0})
        for i, arm in enumerate(arms):
            bp["boxes"][i].set_facecolor(rs.ARM_COLORS[arm])
            bp["boxes"][i].set_alpha(0.85)
            for part in ("whiskers", "caps"):
                for line in bp[part][2 * i: 2 * i + 2]:
                    line.set_color(rs.ARM_COLORS[arm])
        top = max(float(np.quantile(d, 0.95)) for d in data)
        top = top if top > 0 else 1.0
        level = 1.06
        for base in arms[1:]:
            v = res["metrics"][key]["pairs"].get(f"{cand}_vs_{base}")
            if v is None or "skipped" in v:
                continue
            p = v["p_holm"]
            ptxt = "p < 0.001" if p["mean"] < 1e-3 else (f"p = {p['mean']:.3f}" + (f" \u00b1 {p['sd']:.3f}" if k > 1 else ""))
            # Say the direction: "significant" alone would read as "hand-off better".
            parts = []
            if v["n_runs_significant_a_lower"]:
                parts.append(f"lower in {v['n_runs_significant_a_lower']}/{k}")
            if v["n_runs_significant_a_higher"]:
                parts.append(f"HIGHER in {v['n_runs_significant_a_higher']}/{k}")
            text = f"{ptxt}; " + (", ".join(parts) if parts else f"significant in 0/{k}")
            x1, y = arms.index(base), top * level
            ax.plot([0, 0, x1, x1], [y - top * 0.015, y, y, y - top * 0.015], color=rs.MUTED, linewidth=1)
            ax.annotate(text, (x1 / 2, y), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        color=rs.INK_2, fontsize=8.5)
            level += 0.11
        ax.set_ylim(-top * 0.03, top * max(level + 0.04, 1.12))
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_xticks(range(len(arms)))
        names = []
        for arm in arms:
            md_ = res["metrics"][key]["per_arm"][arm]["median"]
            dec = 0 if abs(md_["mean"]) >= 100 else 1 if abs(md_["mean"]) >= 10 else 2
            names.append(f"{short.get(arm, arm)}\n{md_['mean']:.{dec}f}" + (f" \u00b1 {md_['sd']:.{dec}f}" if k > 1 else ""))
        ax.set_xticklabels(names, color=rs.INK, fontsize=9.5, linespacing=1.5)
        ax.set_title(label if unit == "steps" else f"{label} ({unit})", color=rs.INK, fontsize=11, loc="left", pad=8)
    total = sum(res["n_falls_per_seed"].values())
    fig.suptitle(f"Fall outcomes by recovery policy: all {total} falls from {k} runs (lower is better)",
                 color=rs.INK, fontsize=12, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005, "Box: middle 50% of all falls from all runs. Line: their median. Whiskers: 5th to 95th percentile. "
             f"Under each name: mean \u00b1 SD of the {k} per-run medians. Brackets: hand-off against that baseline; "
             f"mean \u00b1 SD of the {k} per-run p-values (paired Wilcoxon on the run's falls, Holm-adjusted) and the number of "
             f"runs in which the hand-off is significantly lower or higher (p < {ALPHA}). Limp has no torque by construction.",
             color=rs.MUTED, fontsize=8.5, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_path, dpi=160, facecolor=rs.SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--recovery-stats-template", required=True,
                   help='each run\'s saved tests, e.g. "outputs/stats/recovery_seed{seed}/recovery_stats.json"')
    p.add_argument("--compare-npz-template", required=True,
                   help='per-fall files for the box plots, a glob with {seed}, e.g. "outputs/go1_recovery_eval_seed{seed}_flag*.npz"')
    p.add_argument("--predictor-stats-template", default=None, help='e.g. "outputs/stats/predictor_seed{seed}/predictor_stats.json"')
    p.add_argument("--handoff-json-template", default=None, help='e.g. "outputs/go1_handoff_eval_seed{seed}.json"')
    p.add_argument("--arms", nargs="+", default=["protective", "walker", "limp"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run, per_seed_falls, files_used = {}, {}, {}
    for s in args.seeds:
        stats_file = Path(args.recovery_stats_template.format(seed=s))
        if not stats_file.exists():
            raise FileNotFoundError(f"[multiseed] run {s} has no saved test results: {stats_file}")
        per_run[s] = json.loads(stats_file.read_text())
        if per_run[s]["arms"] != args.arms:
            raise ValueError(f"[multiseed] run {s} was analysed with arms {per_run[s]['arms']}, expected {args.arms}")
        files = sorted(glob.glob(args.compare_npz_template.format(seed=s)))
        if not files:
            raise FileNotFoundError(f"[multiseed] no files match {args.compare_npz_template.format(seed=s)}")
        per_seed_falls[s], _ = rs.load_pooled(files, args.arms)
        if len(per_seed_falls[s][args.arms[0]]["settled"]) != per_run[s]["n_falls"]:
            raise ValueError(f"[multiseed] run {s}: the per-fall files hold {len(per_seed_falls[s][args.arms[0]]['settled'])} "
                             f"falls but its saved tests used {per_run[s]['n_falls']}")
        files_used[s] = {"stats": str(stats_file), "falls": files}
    res = analyse(per_run, args.arms)
    pred = predictor_summary(sorted(per_run), args.predictor_stats_template, args.handoff_json_template)
    res["predictor"] = pred
    res["config"] = vars(args)
    res["files"] = files_used
    (out_dir / "multiseed_stats.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res, pred)
    (out_dir / "multiseed_stats.md").write_text(md)
    figure(per_seed_falls, res, out_dir / "fig_boxplots.png")
    if not args.quiet:
        print(md)
    print(f"[multiseed] {len(per_run)} runs, {sum(res['n_falls_per_seed'].values())} falls; "
          f"wrote {out_dir}/multiseed_stats.json, multiseed_stats.md, fig_boxplots.png")


if __name__ == "__main__":
    main()
