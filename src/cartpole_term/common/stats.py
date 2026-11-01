"""Statistical tests for the Go1 evaluation (pure numpy/scipy, no mjlab).

Paired comparisons (recovery policies): every fall state is replayed under
each policy, so the unit of analysis is the fall and all tests are paired.
Clustered resampling (predictor): steps within an episode are strongly
correlated, so confidence intervals resample whole EPISODES, not steps.

What these tests do and do not cover: they quantify uncertainty over falls
(or episodes) for the ONE trained walker, predictor, and protective policy.
They say nothing about variation across training seeds.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(values: np.ndarray, stat=np.median, n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of `stat` over independent units."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = stat(values[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def clopper_pearson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if successes == 0 else float(stats.beta.ppf(alpha / 2, successes, n - successes + 1))
    hi = 1.0 if successes == n else float(stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes))
    return lo, hi


def paired_comparison(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> dict:
    """a vs b on the same units (lower is better for every metric here).

    Returns the median paired difference (a - b) with a bootstrap 95% CI, the
    Wilcoxon signed-rank test (two-sided; zero differences dropped), the
    matched-pairs rank-biserial effect size (-1 = a lower in every pair,
    +1 = a higher in every pair), win/tie/loss counts, the share of
    non-tied pairs where a is lower with a Wilson 95% CI, and the exact sign
    test on the non-tied pairs."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1
    d = a - b
    lower, higher, ties = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    nz = d[d != 0]
    if len(nz) >= 1:
        ranks = stats.rankdata(np.abs(nz))
        w_pos, w_neg = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
        rank_biserial = (w_pos - w_neg) / (w_pos + w_neg)
        p_wilcoxon = float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
        p_sign = float(stats.binomtest(lower, lower + higher, 0.5).pvalue)
    else:
        rank_biserial, p_wilcoxon, p_sign = 0.0, 1.0, 1.0
    lo, hi = bootstrap_ci(d, np.median, n_boot=n_boot, seed=seed)
    wlo, whi = wilson_ci(lower, lower + higher)
    return {
        "n": int(len(d)), "median_a": float(np.median(a)), "median_b": float(np.median(b)),
        "median_paired_diff": float(np.median(d)), "median_paired_diff_ci95": [lo, hi],
        "a_lower": lower, "a_higher": higher, "ties": ties,
        "share_a_lower_of_untied": lower / (lower + higher) if lower + higher else float("nan"),
        "share_a_lower_ci95": [wlo, whi],
        "wilcoxon_p": p_wilcoxon, "rank_biserial": float(rank_biserial), "sign_test_p": p_sign,
    }


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test for paired binary outcomes (a, b in {0, 1})."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    p = float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0
    return {"n": int(len(a)), "rate_a": float(a.mean()), "rate_b": float(b.mean()),
            "only_a": only_a, "only_b": only_b, "mcnemar_p": p}


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values, keyed like the input."""
    keys = sorted(p_values, key=lambda k: p_values[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * p_values[k]))
        out[k] = running
    return out


def friedman(*arms: np.ndarray) -> dict:
    """Friedman omnibus test: do k >= 3 paired arms differ at all?"""
    res = stats.friedmanchisquare(*[np.asarray(x, dtype=np.float64) for x in arms])
    return {"statistic": float(res.statistic), "p": float(res.pvalue), "k": len(arms), "n": int(len(arms[0]))}


def cluster_bootstrap(groups: np.ndarray, stat_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap over whole clusters (episodes).

    `stat_fn(indices)` takes row indices into the caller's arrays and returns
    a float or a 1-D array of floats (several statistics at once, e.g. two
    models' scores and their difference). Returns point estimates on the full
    data, percentile CIs, and for each statistic the two-sided bootstrap
    p-value for "statistic = 0" (meaningful for differences)."""
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="stable")
    uniq, starts = np.unique(groups[order], return_index=True)
    members = np.split(order, starts[1:])
    rng = np.random.default_rng(seed)
    point = np.atleast_1d(np.asarray(stat_fn(np.arange(len(groups))), dtype=np.float64))
    boots = np.empty((n_boot, len(point)))
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        boots[i] = np.atleast_1d(stat_fn(np.concatenate([members[j] for j in pick])))
    lo, hi = np.nanquantile(boots, alpha / 2, axis=0), np.nanquantile(boots, 1 - alpha / 2, axis=0)
    p = 2 * np.minimum((boots <= 0).mean(axis=0), (boots >= 0).mean(axis=0))
    return {"point": point.tolist(), "ci95_low": lo.tolist(), "ci95_high": hi.tolist(),
            "p_vs_zero": np.minimum(1.0, np.maximum(p, 1.0 / n_boot)).tolist(),
            "n_clusters": int(len(uniq)), "n_boot": n_boot}


def seed_level_summary(values, alpha: float = 0.05) -> dict:
    """Summarize ONE number per independent pipeline run (training seed).

    This is the across-seed analysis: each seed contributes a single value
    (e.g. its median paired difference), so n is the number of seeds, not the
    number of falls. Returns the mean, sample SD, a t-based 95% CI, the
    two-sided one-sample t-test against 0, the range, and how many seeds are
    below / above zero. With n < 2 the interval and test are undefined (nan).
    """
    v = np.asarray(values, dtype=np.float64)
    n = int(len(v))
    out = {"n_seeds": n, "values": v.tolist(), "mean": float(v.mean()) if n else float("nan"),
           "min": float(v.min()) if n else float("nan"), "max": float(v.max()) if n else float("nan"),
           "n_below_zero": int((v < 0).sum()), "n_above_zero": int((v > 0).sum())}
    if n < 2:
        return {**out, "sd": float("nan"), "ci95": [float("nan"), float("nan")], "t_test_p": float("nan")}
    sd = float(v.std(ddof=1))
    if sd == 0.0:
        return {**out, "sd": 0.0, "ci95": [out["mean"], out["mean"]], "t_test_p": 0.0 if out["mean"] != 0 else 1.0}
    half = float(stats.t.ppf(1 - alpha / 2, n - 1)) * sd / np.sqrt(n)
    p = float(stats.ttest_1samp(v, 0.0).pvalue)
    return {**out, "sd": sd, "ci95": [out["mean"] - half, out["mean"] + half], "t_test_p": p}
