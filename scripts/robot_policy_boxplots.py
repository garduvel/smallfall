"""Box plots and paired tests for several fall policies replayed on the same
falls (G1, DEVIATIONS.md 2026-11-15). CPU only; reads compare npz files
written by scripts/go1_protective_compare.py (arrays `<arm>__<metric>`).

Each run (seed) is tested on its own falls; nothing is pooled into a test.
Per run: paired Wilcoxon signed-rank for every new policy against the
reference (the original SmallFall) and against limp, and between every two
new policies, Holm-adjusted within the run per measure. Across runs: mean
and SD of each statistic, and in how many runs it was significant. Boxes
show every fall from every run.

    python scripts/robot_policy_boxplots.py --runs "outputs/g1_policies_seed{seed}_flag*.npz" --seeds 0 1 2 \\
        --arm hullgrow="Hull growth" --arm centroid="Centroid travel" ... \\
        --new hullgrow centroid hull touched --reference sf --out-dir outputs/stats/policies
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

STEP_S = 0.02
MEASURES = (  # (npz key, label, unit, scale)
    ("envelope_growth_m", "Ground covered", "m", 1.0),
    ("peak_force_critical_n", "Trunk/pelvis peak force", "N", 1.0),
    ("peak_force_any_n", "Any-body peak force", "N", 1.0),
    ("steps", "Time until at rest", "s", STEP_S),
)
# Categorical slots in plot order, checked with the dataviz validator (light
# surface, adjacent pairs): new policies 4,7,6,5; the original SmallFall,
# walker and limp keep slots 1,2,3 as in every earlier figure.
COLORS = ["#eda100", "#6250d6", "#008300", "#e87ba4", "#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def load_runs(template: str, seeds, arms) -> dict:
    """{run: {arm: {key: array}}}; a run's files are concatenated (same falls order per arm)."""
    runs = {}
    for s in (seeds if seeds else [None]):
        files = sorted(glob.glob(template.format(seed=s) if s is not None else template))
        if not files:
            raise FileNotFoundError(template.format(seed=s) if s is not None else template)
        per = {a: {} for a in arms}
        for a in arms:
            for key in [m[0] for m in MEASURES] + ["settled"]:
                parts = []
                for f in files:
                    d = np.load(f)
                    if f"{a}__{key}" not in d.files:
                        raise KeyError(f"{f} has no {a}__{key}")
                    parts.append(d[f"{a}__{key}"])
                per[a][key] = np.concatenate(parts)
        runs[s if s is not None else 0] = per
    return runs


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    nz = d[d != 0]
    p = float(wilcoxon(a, b).pvalue) if len(nz) else 1.0
    r = 0.0
    if len(nz):
        ranks = np.argsort(np.argsort(np.abs(nz))) + 1
        r = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    return {"median_diff": float(np.median(d)), "share_lower": float(np.mean(d < 0)), "rank_biserial": r, "p": p}


def analyse(runs: dict, arms: list[str], new: list[str], reference: str) -> dict:
    pairs = [(n, reference) for n in new] + [(n, "limp") for n in new] + list(itertools.combinations(new, 2))
    out = {"pairs": [list(p) for p in pairs], "per_run": {}, "per_arm": {}}
    for run, per in runs.items():
        res = {}
        for key, *_ in MEASURES:
            tests = [paired(per[a][key], per[b][key]) for a, b in pairs]
            for t, padj in zip(tests, holm([t["p"] for t in tests])):
                t["p_holm"] = padj
            fr = friedmanchisquare(*[per[a][key] for a in arms]).pvalue
            res[key] = {"tests": tests, "friedman_p": float(fr)}
        out["per_run"][str(run)] = res
    for a in arms:
        row = {}
        for key, _, _, scale in MEASURES:
            meds = [float(np.median(runs[r][a][key]) * scale) for r in runs]
            row[key] = {"median_mean": float(np.mean(meds)), "median_sd": float(np.std(meds, ddof=1)) if len(meds) > 1 else 0.0}
        rest = [float(np.mean(runs[r][a]["settled"])) for r in runs]
        row["settled"] = {"mean": float(np.mean(rest)), "sd": float(np.std(rest, ddof=1)) if len(rest) > 1 else 0.0}
        out["per_arm"][a] = row
    summary = {}
    for key, *_ in MEASURES:
        rows = []
        for i, (a, b) in enumerate(pairs):
            ts = [out["per_run"][str(r)][key]["tests"][i] for r in runs]
            rows.append({"a": a, "b": b,
                         **{f"{k}_mean": float(np.mean([t[k] for t in ts])) for k in ("median_diff", "share_lower", "rank_biserial", "p_holm")},
                         "median_diff_sd": float(np.std([t["median_diff"] for t in ts], ddof=1)) if len(ts) > 1 else 0.0,
                         "runs_a_lower": int(sum(t["p_holm"] < 0.05 and t["median_diff"] < 0 for t in ts)),
                         "runs_a_higher": int(sum(t["p_holm"] < 0.05 and t["median_diff"] > 0 for t in ts)),
                         "runs": len(ts)})
        summary[key] = rows
    out["pairwise_summary"] = summary
    out["n_falls"] = {str(r): int(len(runs[r][arms[0]]["settled"])) for r in runs}
    return out


def plot(runs, arms, labels, new, reference, res, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    k = len(runs)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), facecolor=SURFACE)
    for ax, (key, label, unit, scale) in zip(axes.ravel(), MEASURES):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        ax.grid(True, axis="y", color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        data = [np.concatenate([runs[r][a][key] for r in runs]) * scale for a in arms]
        bp = ax.boxplot(data, positions=range(len(arms)), widths=0.6, whis=(5, 95), showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "linewidth": 2}, whiskerprops={"linewidth": 1.5},
                        capprops={"linewidth": 1.5}, boxprops={"linewidth": 0})
        for i, a in enumerate(arms):
            c = COLORS[i % len(COLORS)]
            bp["boxes"][i].set_facecolor(c)
            bp["boxes"][i].set_alpha(0.85)
            for part in ("whiskers", "caps"):
                for line in bp[part][2 * i: 2 * i + 2]:
                    line.set_color(c)
        ticks = []
        rows = {(r["a"], r["b"]): r for r in res["pairwise_summary"][key]}
        for a in arms:
            m = res["per_arm"][a][key]
            fmt = (lambda v: f"{v:.2f}") if unit in ("m", "s") else (lambda v: f"{v:.0f}")
            t = f"{labels[a].replace(' ', chr(10), 1)}\n{fmt(m['median_mean'])}" + (f" ± {fmt(m['median_sd'])}" if k > 1 else "")
            if a in new:
                r = rows[(a, reference)]
                verdict = (f"< SF {r['runs_a_lower']}/{k}" if r["runs_a_lower"] else
                           f"> SF {r['runs_a_higher']}/{k}" if r["runs_a_higher"] else "≈ SF")
                t += f"\n{verdict}"
            ticks.append(t)
        ax.set_xticks(range(len(arms)), ticks, fontsize=9, color=INK)
        ax.set_title(f"{label} ({unit})", loc="left", fontsize=12, color=INK)
    n = sum(res["n_falls"].values())
    fig.suptitle(f"{title}: {n} falls from {k} run{'s' if k > 1 else ''} (lower is better)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.01, "Box: middle 50% of all falls. Line: median. Whiskers: 5th-95th percentile. Under each name: median"
             + (" (mean ± SD of the per-run medians)" if k > 1 else "")
             + ". '< SF k/n': lower than the original SmallFall in k of n runs (paired Wilcoxon, Holm, p < 0.05).",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=3)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def write_md(res, arms, labels, path, title):
    L = [f"# {title}", "", f"Falls per run: {res['n_falls']}. Every test is paired on the fall, within a run; "
         "Holm-adjusted within the run per measure.", "", "## Per policy (median; mean ± SD across runs)", ""]
    L.append("| policy | " + " | ".join(f"{lab} ({u})" for _, lab, u, _ in MEASURES) + " | came to rest |")
    L.append("|---|" + "---|" * (len(MEASURES) + 1))
    for a in arms:
        r = res["per_arm"][a]
        L.append(f"| {labels[a]} | " + " | ".join(f"{r[k]['median_mean']:.3g} ± {r[k]['median_sd']:.2g}" for k, *_ in MEASURES)
                 + f" | {r['settled']['mean']:.3f} |")
    L += ["", "## Pairwise (A vs B)", "",
          "| measure | A vs B | median A − B | share of falls A lower | rank-biserial | p (Holm), mean | runs A lower / higher |",
          "|---|---|---|---|---|---|---|"]
    for key, lab, *_ in MEASURES:
        for r in res["pairwise_summary"][key]:
            L.append(f"| {lab} | {labels[r['a']]} vs {labels[r['b']]} | {r['median_diff_mean']:.3g} | {r['share_lower_mean']:.1%} | "
                     f"{r['rank_biserial_mean']:+.2f} | {r['p_holm_mean']:.2g} | {r['runs_a_lower']} / {r['runs_a_higher']} of {r['runs']} |")
    Path(path).write_text("\n".join(L) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", required=True, help="glob of compare npz files; {seed} is filled per seed")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--arm", action="append", required=True, metavar="KEY=LABEL", help="in plot order")
    p.add_argument("--new", nargs="+", required=True, help="the new policies (compared with each other)")
    p.add_argument("--reference", required=True, help="the original SmallFall arm")
    p.add_argument("--title", default="Fall outcomes by policy")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    labels = dict(a.split("=", 1) for a in args.arm)
    arms = list(labels)
    for need in [*args.new, args.reference, "limp"]:
        if need not in labels:
            raise ValueError(f"{need!r} must also be given with --arm")
    runs = load_runs(args.runs, args.seeds, arms)
    res = analyse(runs, arms, args.new, args.reference)
    res["config"] = vars(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy_stats.json").write_text(json.dumps(res, indent=2))
    write_md(res, arms, labels, out / "policy_stats.md", args.title)
    plot(runs, arms, labels, args.new, args.reference, res, out / "fig_policy_boxplots.png", args.title)
    print(f"[policies] wrote {out}/fig_policy_boxplots.png, policy_stats.md, policy_stats.json")


if __name__ == "__main__":
    main()
