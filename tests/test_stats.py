"""Tests for src/cartpole_term/common/stats.py (pure numpy/scipy)."""

from __future__ import annotations

import numpy as np

from _helpers import load_module

stats = load_module("common/stats.py", "_stats_under_test")


def test_paired_comparison_detects_a_shift_and_counts_pairs():
    rng = np.random.default_rng(1)
    b = rng.normal(1.0, 0.2, 300)
    a = b - 0.1 + rng.normal(0, 0.05, 300)
    r = stats.paired_comparison(a, b, n_boot=2000)
    assert r["n"] == 300 and r["a_lower"] + r["a_higher"] + r["ties"] == 300
    assert r["wilcoxon_p"] < 1e-10 and r["rank_biserial"] < -0.8
    lo, hi = r["median_paired_diff_ci95"]
    assert lo < r["median_paired_diff"] < hi < 0 and abs(r["median_paired_diff"] + 0.1) < 0.02


def test_paired_comparison_no_effect_and_all_ties():
    rng = np.random.default_rng(2)
    b = rng.normal(0, 1, 400)
    r = stats.paired_comparison(b + rng.normal(0, 1, 400), b, n_boot=1000)
    assert r["wilcoxon_p"] > 0.01 and r["median_paired_diff_ci95"][0] < 0 < r["median_paired_diff_ci95"][1]
    z = stats.paired_comparison(np.zeros(10), np.zeros(10), n_boot=100)
    assert z["ties"] == 10 and z["wilcoxon_p"] == 1.0 and z["sign_test_p"] == 1.0


def test_zero_inflated_pairs_drop_ties():
    a = np.array([0.0] * 50 + [0.0] * 40)
    b = np.array([0.0] * 50 + [300.0] * 40)
    r = stats.paired_comparison(a, b, n_boot=500)
    assert r["ties"] == 50 and r["a_lower"] == 40 and r["a_higher"] == 0
    assert r["rank_biserial"] == -1.0 and r["sign_test_p"] < 1e-9 and r["share_a_lower_of_untied"] == 1.0


def test_mcnemar_and_holm_and_intervals():
    a = np.array([1] * 90 + [1] * 10 + [0] * 0)
    b = np.array([1] * 90 + [0] * 10)
    m = stats.mcnemar_exact(a, b)
    assert m["only_a"] == 10 and m["only_b"] == 0 and abs(m["mcnemar_p"] - 2 * 0.5 ** 10) < 1e-12
    h = stats.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert abs(h["x"] - 0.03) < 1e-12 and abs(h["z"] - 0.06) < 1e-12 and abs(h["y"] - 0.06) < 1e-12
    lo, hi = stats.clopper_pearson_ci(763, 764)
    assert 0.99 < lo < 763 / 764 < hi <= 1.0
    assert stats.clopper_pearson_ci(10, 10)[1] == 1.0 and stats.clopper_pearson_ci(0, 10)[0] == 0.0
    wlo, whi = stats.wilson_ci(50, 100)
    assert abs((wlo + whi) / 2 - 0.5) < 1e-9 and 0.39 < wlo < 0.41


def test_friedman_separates_three_arms():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, 200)
    f = stats.friedman(base, base + 0.5, base + 1.0)
    assert f["p"] < 1e-20 and f["k"] == 3


def test_cluster_bootstrap_is_wider_than_naive_when_clusters_are_correlated():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(40), 50)                 # 40 episodes x 50 steps
    values = np.repeat(rng.normal(0, 1, 40), 50) + rng.normal(0, 0.05, 2000)
    cb = stats.cluster_bootstrap(groups, lambda idx: values[idx].mean(), n_boot=600)
    naive = stats.bootstrap_ci(values, np.mean, n_boot=600)
    assert cb["n_clusters"] == 40
    assert (cb["ci95_high"][0] - cb["ci95_low"][0]) > 3 * (naive[1] - naive[0])
    two = stats.cluster_bootstrap(groups, lambda idx: np.array([values[idx].mean(), 5.0]), n_boot=200)
    assert len(two["point"]) == 2 and two["p_vs_zero"][1] <= 2 / 200


def test_seed_level_summary():
    r = stats.seed_level_summary([-0.10, -0.12, -0.09, -0.11, -0.13])
    assert r["n_seeds"] == 5 and r["n_below_zero"] == 5 and r["n_above_zero"] == 0
    assert abs(r["mean"] + 0.11) < 1e-12 and r["ci95"][0] < r["mean"] < r["ci95"][1] < 0
    assert r["t_test_p"] < 0.001
    # n = 5 with a wide spread: the interval must cross zero and the test must not be significant
    w = stats.seed_level_summary([-0.5, 0.4, -0.1, 0.3, -0.2])
    assert w["ci95"][0] < 0 < w["ci95"][1] and w["t_test_p"] > 0.3
    # t-interval, not a normal one: half-width = t(0.975, 4) * sd / sqrt(5)
    half = (r["ci95"][1] - r["ci95"][0]) / 2
    assert abs(half - 2.7764451 * r["sd"] / np.sqrt(5)) < 1e-6
    one = stats.seed_level_summary([-0.2])
    assert one["n_seeds"] == 1 and np.isnan(one["t_test_p"]) and np.isnan(one["sd"])
