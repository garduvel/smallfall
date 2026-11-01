"""Statistical comparison of the recovery policies (Go1): protective hand-off
vs the walker continuing vs limp (motors off).

Input: one or more per-fall result files written by
scripts/go1_protective_compare.py (`.npz`, arrays named `<arm>__<metric>`,
one row per replayed fall state). Every fall state is replayed under every
arm from an identical start, so all tests are PAIRED on the fall. Several
files (different fall banks) are pooled; a per-file table shows whether the
banks agree.

Per metric and per pair of arms:
  median paired difference with a bootstrap 95% CI, Wilcoxon signed-rank test
  (two-sided), matched-pairs rank-biserial effect size, share of falls in
  which the first arm is lower with a Wilson 95% CI, exact sign test.
Per metric across all arms: Friedman omnibus test.
Came-to-rest (binary): exact McNemar test per pair.
All pairwise p-values (Wilcoxon and McNemar) are Holm-adjusted as one family.
Peak torque is not tested against limp: limp's torque is zero by construction.

What this does NOT show: variation across training seeds. There is one
walker, one predictor, and one protective policy; the tests quantify
uncertainty over falls for those fixed policies.

Pure numpy/scipy/matplotlib: runs on any machine, no GPU or mjlab needed.

Usage:
    python scripts/go1_recovery_stats.py --compare-npz outputs/.../a.npz outputs/.../b.npz \
        --out-dir outputs/stats/recovery_confirm

Writes <out-dir>/recovery_stats.json, recovery_stats.md (all tables), and
fig_boxplots.png (box-and-whisker per policy, four panels, real units).
--quiet skips printing the tables to the terminal; they are still written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

# Loaded by path, not `import cartpole_term...`: the package __init__ registers
# mjlab tasks and so needs mjlab, which this script deliberately does not.
_STATS_PATH = Path(__file__).resolve().parent.parent / "src" / "cartpole_term" / "common" / "stats.py"
_spec = importlib.util.spec_from_file_location("go1_stats", _STATS_PATH)
stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stats)

# (key in the npz, label, unit, is the primary outcome)
METRICS = (
    ("envelope_growth_m", "Ground covered", "m", True),
    ("peak_force_critical_n", "Trunk/hip peak force", "N", False),
    ("peak_force_any_n", "Any-body peak force", "N", False),
    ("peak_torque_nm", "Peak joint torque", "Nm", False),
    ("steps", "Steps until at rest (capped at 150)", "steps", False),
)
ARM_LABELS = {"protective": "Hand-off (protective)", "walker": "Walker continues", "limp": "Limp (motors off)"}
# Categorical slots 1-3 of the reference palette (validated all-pairs for
# colour-vision deficiency); colour follows the arm in every figure.
ARM_COLORS = {"protective": "#2a78d6", "walker": "#eb6834", "limp": "#1baf7a"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load_pooled(files: list[str], arms: list[str]) -> tuple[dict, list[dict]]:
    pooled = {arm: {} for arm in arms}
    per_file = []
    keys = [m[0] for m in METRICS] + ["settled"]
    parts = {arm: {k: [] for k in keys} for arm in arms}
    for f in files:
        d = np.load(f)
        row = {"file": f, "n": None}
        for arm in arms:
            for k in keys:
                name = f"{arm}__{k}"
                if name not in d:
                    raise KeyError(f"{f} has no array {name}; arms present: "
                                   f"{sorted({n.split('__')[0] for n in d.files})}")
                parts[arm][k].append(np.asarray(d[name], dtype=np.float64))
            row["n"] = int(len(d[f"{arm}__settled"]))
            row[arm] = {k: float(np.median(d[f"{arm}__{k}"])) for k in keys[:-1]}
            row[arm]["settled"] = float(np.mean(d[f"{arm}__settled"]))
        per_file.append(row)
    for arm in arms:
        for k in keys:
            pooled[arm][k] = np.concatenate(parts[arm][k])
    return pooled, per_file


def analyse(pooled: dict, arms: list[str], n_boot: int, seed: int) -> dict:
    pairs = [(a, b) for i, a in enumerate(arms) for b in arms[i + 1:]]
    out = {"n_falls": int(len(pooled[arms[0]]["settled"])), "arms": arms, "metrics": {}, "settled": {}}
    family = {}
    for key, label, unit, primary in METRICS:
        m = {"label": label, "unit": unit, "primary": primary, "per_arm": {}, "pairs": {}}
        for arm in arms:
            v = pooled[arm][key]
            m["per_arm"][arm] = {
                "median": float(np.median(v)), "median_ci95": list(stats.bootstrap_ci(v, np.median, n_boot, seed=seed)),
                "mean": float(np.mean(v)),
                "p90": float(np.quantile(v, 0.9)),
                "p90_ci95": list(stats.bootstrap_ci(v, lambda x, axis: np.quantile(x, 0.9, axis=axis), n_boot, seed=seed)),
                "share_zero": float(np.mean(v == 0)),
            }
        tested_arms = [a for a in arms if not (key == "peak_torque_nm" and a == "limp")]
        if len(tested_arms) >= 3:
            m["friedman"] = stats.friedman(*[pooled[a][key] for a in tested_arms])
        for a, b in pairs:
            if a not in tested_arms or b not in tested_arms:
                m["pairs"][f"{a}_vs_{b}"] = {"skipped": "limp torque is zero by construction"}
                continue
            r = stats.paired_comparison(pooled[a][key], pooled[b][key], n_boot=n_boot, seed=seed)
            m["pairs"][f"{a}_vs_{b}"] = r
            family[f"{key}:{a}_vs_{b}"] = r["wilcoxon_p"]
        out["metrics"][key] = m
    for a, b in pairs:
        r = stats.mcnemar_exact(pooled[a]["settled"], pooled[b]["settled"])
        out["settled"][f"{a}_vs_{b}"] = r
        family[f"settled:{a}_vs_{b}"] = r["mcnemar_p"]
    for arm in arms:
        s = pooled[arm]["settled"]
        out["settled"][arm] = {"rate": float(s.mean()),
                               "ci95": list(stats.clopper_pearson_ci(int(s.sum()), len(s)))}
    adjusted = stats.holm(family)
    out["holm_family_size"] = len(family)
    for name, p_adj in adjusted.items():
        key, pair = name.split(":")
        target = out["settled"][pair] if key == "settled" else out["metrics"][key]["pairs"][pair]
        target["p_holm"] = p_adj
    return out


def _n(x: float) -> str:
    """Compact number: whole units once it is large, 3 significant figures otherwise."""
    return f"{x:.0f}" if abs(x) >= 100 else f"{x:.3g}"


def _p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def to_markdown(res: dict, per_file: list[dict], files: list[str]) -> str:
    arms = res["arms"]
    L = [f"# Recovery-policy comparison: {res['n_falls']} paired falls", "",
         f"Arms: {', '.join(ARM_LABELS.get(a, a) for a in arms)}. Source files: {len(files)}.",
         f"All p-values are two-sided and Holm-adjusted over {res['holm_family_size']} pairwise tests.",
         "Lower is better for every measure. One trained policy per arm: uncertainty is over falls, not training seeds.", ""]
    L += ["## Per-arm summary", "", "| measure | " + " | ".join(ARM_LABELS.get(a, a) for a in arms) + " |",
          "|---|" + "---|" * len(arms)]
    for key, m in res["metrics"].items():
        cells = []
        for a in arms:
            s = m["per_arm"][a]
            cells.append(f"{_n(s['median'])} [{_n(s['median_ci95'][0])}, {_n(s['median_ci95'][1])}]; p90 {_n(s['p90'])}")
        L.append(f"| {m['label']} ({m['unit']}): median [95% CI]; p90 | " + " | ".join(cells) + " |")
    L.append("| Came to rest: rate [95% CI] | " + " | ".join(
        f"{res['settled'][a]['rate']:.3f} [{res['settled'][a]['ci95'][0]:.3f}, {res['settled'][a]['ci95'][1]:.3f}]"
        for a in arms) + " |")
    L += ["", "## Pairwise tests (paired on the fall)", "",
          "| measure | A vs B | median A | median B | median paired diff A-B [95% CI] | A lower in (of untied) [95% CI] | ties | rank-biserial | Wilcoxon p (Holm) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for key, m in res["metrics"].items():
        for pair, r in m["pairs"].items():
            a, b = pair.split("_vs_")
            if "skipped" in r:
                L.append(f"| {m['label']} | {a} vs {b} | | | not tested: {r['skipped']} | | | | |")
                continue
            L.append(
                f"| {m['label']}{' (primary)' if m['primary'] else ''} | {a} vs {b} | {_n(r['median_a'])} | {_n(r['median_b'])} | "
                f"{_n(r['median_paired_diff'])} [{_n(r['median_paired_diff_ci95'][0])}, {_n(r['median_paired_diff_ci95'][1])}] | "
                f"{100 * r['share_a_lower_of_untied']:.1f}% [{100 * r['share_a_lower_ci95'][0]:.1f}, {100 * r['share_a_lower_ci95'][1]:.1f}] | "
                f"{r['ties']} | {r['rank_biserial']:+.2f} | {_p(r['p_holm'])} |")
    L += ["", "## Came to rest (exact McNemar, Holm-adjusted)", "",
          "| A vs B | rate A | rate B | only A settled | only B settled | p (Holm) |", "|---|---|---|---|---|---|"]
    for pair, r in res["settled"].items():
        if "_vs_" in pair:
            a, b = pair.split("_vs_")
            L.append(f"| {a} vs {b} | {r['rate_a']:.3f} | {r['rate_b']:.3f} | {r['only_a']} | {r['only_b']} | {_p(r['p_holm'])} |")
    L += ["", "## Friedman omnibus test across arms", "", "| measure | arms | chi-square | p |", "|---|---|---|---|"]
    for key, m in res["metrics"].items():
        if "friedman" in m:
            L.append(f"| {m['label']} | {m['friedman']['k']} | {m['friedman']['statistic']:.1f} | {_p(m['friedman']['p'])} |")
    if len(per_file) > 1:
        L += ["", "## Agreement between fall banks (medians per file)", "",
              "| file | n | " + " | ".join(f"{a}: ground m / trunk N / any N" for a in arms) + " |",
              "|---|---|" + "---|" * len(arms)]
        for row in per_file:
            L.append(f"| {Path(row['file']).name} | {row['n']} | " + " | ".join(
                f"{row[a]['envelope_growth_m']:.3f} / {row[a]['peak_force_critical_n']:.0f} / {row[a]['peak_force_any_n']:.0f}"
                for a in arms) + " |")
    return "\n".join(L) + "\n"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def _sig(p: float) -> str:
    return "p < 0.001" if p < 1e-3 else (f"p = {p:.3f}" if p < 0.05 else "n.s.")


def figures(pooled: dict, res: dict, out_dir: Path) -> None:
    """One figure, in real units: fig_boxplots.png, a box-and-whisker per
    policy in one panel per measure (ground covered, trunk/hip force,
    any-body force, peak torque)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = res["arms"]
    cand = arms[0]
    short = {"protective": "Hand-off", "walker": "Walker", "limp": "Limp"}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    n = res["n_falls"]

    # Box-and-whisker: box = middle 50% of falls, line = median, whiskers = 5th to 95th percentile.
    panels = [m for m in METRICS if m[0] != "steps"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.9), facecolor=SURFACE)
    for ax, (key, label, unit, _primary) in zip(np.atleast_1d(axes), panels):
        _style(ax)
        ax.xaxis.grid(False)
        data = [pooled[a][key] for a in arms]
        bp = ax.boxplot(data, positions=range(len(arms)), widths=0.5, whis=(5, 95), showfliers=False,
                        patch_artist=True, medianprops={"color": INK, "linewidth": 2},
                        whiskerprops={"linewidth": 1.5}, capprops={"linewidth": 1.5}, boxprops={"linewidth": 0})
        for k, arm in enumerate(arms):
            bp["boxes"][k].set_facecolor(ARM_COLORS[arm])
            bp["boxes"][k].set_alpha(0.85)
            for part in ("whiskers", "caps"):
                for line in bp[part][2 * k: 2 * k + 2]:
                    line.set_color(ARM_COLORS[arm])
        top = max(float(np.quantile(d, 0.95)) for d in data)
        top = top if top > 0 else 1.0
        for k, arm in enumerate(arms):  # the median, written beside its box
            med = float(np.median(data[k]))
            ax.annotate(_n(med), (k + 0.29, med), xytext=(2, 0), textcoords="offset points", va="center",
                        color=INK_2, fontsize=9)
        # Significance of the hand-off against each baseline (Holm-adjusted Wilcoxon), as brackets above the boxes.
        level = 1.06
        for base in arms[1:]:
            r = res["metrics"][key]["pairs"].get(f"{cand}_vs_{base}")
            if r is None or "skipped" in r:
                continue
            x0, x1, y = 0, arms.index(base), top * level
            ax.plot([x0, x0, x1, x1], [y - top * 0.015, y, y, y - top * 0.015], color=MUTED, linewidth=1)
            ax.annotate(_sig(r["p_holm"]), ((x0 + x1) / 2, y), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", color=INK_2, fontsize=9)
            level += 0.11
        ax.set_ylim(-top * 0.03, top * max(level + 0.04, 1.12))
        ax.set_xlim(-0.6, len(arms) - 0.05)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([short.get(a, a) for a in arms], color=INK, fontsize=10)
        ax.set_title(f"{label} ({unit})", color=INK, fontsize=11, loc="left", pad=8)
    fig.suptitle(f"Fall outcomes by recovery policy, {n} falls each replayed under all three (lower is better)",
                 color=INK, fontsize=12, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005, "Box: middle 50% of falls. Line and number: median. Whiskers: 5th to 95th percentile. "
             "Brackets: hand-off against that baseline, paired Wilcoxon test, Holm-adjusted; n.s. = not significant. "
             "Limp has no torque by construction.", color=MUTED, fontsize=8.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(out_dir / "fig_boxplots.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare-npz", nargs="+", required=True)
    p.add_argument("--arms", nargs="+", default=["protective", "walker", "limp"],
                   help="first arm is the candidate; the rest are baselines")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--quiet", action="store_true", help="do not print the tables; they are still written to the .md file")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled, per_file = load_pooled(args.compare_npz, args.arms)
    res = analyse(pooled, args.arms, args.n_boot, args.seed)
    res["config"] = vars(args)
    res["per_file"] = per_file
    (out_dir / "recovery_stats.json").write_text(json.dumps(res, indent=2))
    md = to_markdown(res, per_file, args.compare_npz)
    (out_dir / "recovery_stats.md").write_text(md)
    if not args.no_figures:
        figures(pooled, res, out_dir)
    if not args.quiet:
        print(md)
    print(f"[recovery-stats] {res['n_falls']} paired falls; wrote {out_dir}/recovery_stats.json, recovery_stats.md"
          + ("" if args.no_figures else ", fig_boxplots.png"))


if __name__ == "__main__":
    main()
