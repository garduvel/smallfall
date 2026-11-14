"""Which policy falls smallest overall, across many ways of measuring fall
size (metric ablation, DEVIATIONS.md 2026-11-13). Numpy (+ scipy for the
paired tests); no torch or mjlab, so it runs on the CPU from a compare npz.

Input: per_fall[policy][metric] -> array over the SAME falls in the same
order for every policy (the compare script replays one bank under each
policy). Lower = smaller for every metric.

Three ways of combining the measures, all fixed before any data:
  A. dominance  -- policy A dominates B if A's median is <= B's on every
                   measure and < on at least one. The answer is the set of
                   policies nothing dominates. No weighting at all.
  B. mean rank  -- on each fall and each measure, rank the policies (1 =
                   smallest, ties averaged); average over falls and measures.
  C. ratio to the reference (limp) -- per fall and measure, policy / limp;
                   geometric mean over falls and measures. 0.6 = 40% smaller
                   than doing nothing. A small floor (1% of the measure's
                   pooled median) keeps zeros from dividing.
Overall answer: if B's winner, C's winner, and A's set all name the same
single policy (or B and C agree and that policy is in A's set), that policy;
otherwise the methods disagree and A's set is reported as the answer.
Robustness: the falls are resampled with replacement (paired across
policies) and B, C, and A are redone; how often each policy wins is reported.
"""

from __future__ import annotations

import itertools

import numpy as np

RATIO_FLOOR_FRACTION = 0.01
BOOTSTRAP_RESAMPLES = 1000


def _stack(per_fall: dict[str, dict[str, np.ndarray]], policies, metrics) -> np.ndarray:
    """[P, M, F] array; checks every policy has the same number of falls."""
    n = {len(per_fall[p][m]) for p in policies for m in metrics}
    if len(n) != 1:
        raise ValueError(f"policies/metrics have different numbers of falls: {sorted(n)}")
    return np.stack([np.stack([np.asarray(per_fall[p][m], dtype=float) for m in metrics]) for p in policies])


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """Ranks along axis 0 (1 = smallest), ties averaged. x [P, ...]."""
    from scipy.stats import rankdata

    return rankdata(x, method="average", axis=0)


def dominance(medians: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """medians [P, M] -> (dom [P, P] bool, dom[i, j] = i dominates j;
    undominated [P] bool)."""
    le = (medians[:, None, :] <= medians[None, :, :]).all(-1)
    lt = (medians[:, None, :] < medians[None, :, :]).any(-1)
    dom = le & lt
    return dom, ~dom.any(0)


def mean_rank(x: np.ndarray) -> np.ndarray:
    """x [P, M, F] -> mean rank per policy [P]."""
    return _rank_rows(x).mean(axis=(1, 2))


def ratio_to_reference(x: np.ndarray, ref: int, floors: np.ndarray) -> np.ndarray:
    """x [P, M, F], floors [M] -> geometric-mean ratio per policy [P]."""
    f = floors[None, :, None]
    lr = np.log((x + f) / (x[ref][None] + f))
    return np.exp(lr.mean(axis=(1, 2)))


def _floors(x: np.ndarray) -> np.ndarray:
    med = np.median(np.abs(x), axis=(0, 2))
    return np.maximum(RATIO_FLOOR_FRACTION * med, 1e-9)


def kendall_tau(a, b) -> float:
    """Kendall's tau-a between two score vectors (no scipy needed)."""
    n, s = len(a), 0
    for i, j in itertools.combinations(range(n), 2):
        s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(s / (n * (n - 1) / 2)) if n > 1 else 1.0


def _combine(x, policies, ref):
    med = np.median(x, axis=2)
    _, undom = dominance(med)
    mr = mean_rank(x)
    rr = ratio_to_reference(x, ref, _floors(x))
    return med, undom, mr, rr


def overall(winner_b: str, winner_c: str, undominated: list[str]) -> dict:
    agree = winner_b == winner_c and winner_b in undominated
    return {
        "methods_agree": agree,
        "answer": [winner_b] if agree else list(undominated),
        "note": ("mean rank, ratio to limp, and dominance agree" if agree else
                 "the methods disagree: the answer is the set of policies no other policy dominates"),
    }


def analyse(per_fall: dict[str, dict[str, np.ndarray]], metrics, reference: str = "limp",
            policies=None, bootstrap: int = BOOTSTRAP_RESAMPLES, seed: int = 0,
            scale: dict[str, float] | None = None) -> dict:
    """Full ablation report (JSON-serialisable). `scale` optionally divides
    each metric (e.g. lengths by standing height) for the reported medians;
    rankings are unaffected by it."""
    policies = list(policies or per_fall)
    metrics = list(metrics)
    if reference not in policies:
        raise ValueError(f"reference policy {reference!r} not among {policies}")
    ref = policies.index(reference)
    x = _stack(per_fall, policies, metrics)
    P, M, F = x.shape
    med, undom, mr, rr = _combine(x, policies, ref)

    # Per-measure rankings and how much the measures agree.
    rank_by_metric = {m: [policies[i] for i in np.argsort(med[:, k], kind="stable")] for k, m in enumerate(metrics)}
    first_by_metric = {m: r[0] for m, r in rank_by_metric.items()}
    ranking_tau = {f"{a}|{b}": kendall_tau(med[:, i], med[:, j])
                   for (i, a), (j, b) in itertools.combinations(enumerate(metrics), 2)}
    # Do the measures agree fall by fall? Spearman over every (policy, fall).
    ranks = _rank_rows(x.transpose(1, 0, 2).reshape(M, -1).T).T  # rank each measure over all (policy, fall)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_fall_spearman = np.corrcoef(ranks) if F * P > 1 else np.ones((M, M))

    # Paired per-fall comparisons with each policy against every other.
    pairwise = {}
    try:
        from scipy.stats import wilcoxon
    except ImportError:  # pragma: no cover
        wilcoxon = None
    dom_matrix = dominance(med)[0]
    for k, m in enumerate(metrics):
        rows = {}
        for i, j in itertools.permutations(range(P), 2):
            d = x[i, k] - x[j, k]
            p = None
            if wilcoxon is not None and np.any(d != 0):
                p = float(wilcoxon(x[i, k], x[j, k], zero_method="wilcox").pvalue)
            rows[f"{policies[i]}|{policies[j]}"] = {"smaller_share": float((d < 0).mean()),
                                                    "tie_share": float((d == 0).mean()), "wilcoxon_p": p}
        pairwise[m] = rows

    winner_b = policies[int(np.argmin(mr))]
    winner_c = policies[int(np.argmin(rr))]
    undominated = [p for p, u in zip(policies, undom) if u]

    # Bootstrap over falls.
    rng = np.random.default_rng(seed)
    wins_b, wins_c, in_a, agree = np.zeros(P), np.zeros(P), np.zeros(P), 0
    for _ in range(bootstrap):
        idx = rng.integers(0, F, F)
        _, u_s, mr_s, rr_s = _combine(x[:, :, idx], policies, ref)
        b, c = int(np.argmin(mr_s)), int(np.argmin(rr_s))
        wins_b[b] += 1
        wins_c[c] += 1
        in_a += u_s
        agree += int(b == c and u_s[b])

    s = np.array([(scale or {}).get(m, 1.0) for m in metrics])
    return {
        "policies": policies,
        "metrics": metrics,
        "reference": reference,
        "num_falls": F,
        "medians": {p: {m: float(med[i, k] / s[k]) for k, m in enumerate(metrics)} for i, p in enumerate(policies)},
        "scale": {m: float(v) for m, v in zip(metrics, s)},
        "rank_by_metric": rank_by_metric,
        "smallest_by_metric": first_by_metric,
        "ranking_kendall_tau": ranking_tau,
        "mean_ranking_kendall_tau": float(np.mean(list(ranking_tau.values()))) if ranking_tau else 1.0,
        "per_fall_spearman": {f"{a}|{b}": float(per_fall_spearman[i, j])
                              for (i, a), (j, b) in itertools.combinations(enumerate(metrics), 2)},
        "pairwise": pairwise,
        "A_dominance": {"undominated": undominated,
                        "dominates": {p: [policies[j] for j in range(P) if dom_matrix[i, j]]
                                      for i, p in enumerate(policies)}},
        "B_mean_rank": {"scores": dict(zip(policies, map(float, mr))), "winner": winner_b},
        "C_ratio_to_reference": {"scores": dict(zip(policies, map(float, rr))), "winner": winner_c},
        "overall": overall(winner_b, winner_c, undominated),
        "bootstrap": {"resamples": bootstrap,
                      "B_win_share": dict(zip(policies, map(float, wins_b / max(bootstrap, 1)))),
                      "C_win_share": dict(zip(policies, map(float, wins_c / max(bootstrap, 1)))),
                      "A_undominated_share": dict(zip(policies, map(float, in_a / max(bootstrap, 1)))),
                      "methods_agree_share": agree / max(bootstrap, 1)},
    }


def exceedance_curve(values: np.ndarray, radii: np.ndarray) -> list[float]:
    """Share of falls whose value is above each radius (e.g. max_reach_m:
    the share of falls that would hit an object at that distance)."""
    v = np.asarray(values, dtype=float)
    return [float((v > r).mean()) for r in radii]


# --- Reward ablation: one policy trained per measure ------------------------------

# Short policy labels: the policy trained with measure m is "r_<tag>".
REWARD_TAGS = {
    "envelope_growth_m": "ground", "max_reach_m": "reach", "root_travel_end_m": "rootend",
    "root_travel_max_m": "rootmax", "root_path_m": "rootpath", "centroid_travel_end_m": "centroid",
    "swept_hull_area_m2": "hull", "hull_growth_m2": "hullgrow", "contact_reach_m": "contact",
    "touched_area_m2": "touched",
}
TAG_TO_MEASURE = {v: k for k, v in REWARD_TAGS.items()}
MIN_SETTLED_RATE = 0.99  # as in compare.select_checkpoint


def select_by_measure(per_arm: dict[str, dict[str, np.ndarray]], candidates: list[str], measure: str,
                      min_settled: float = MIN_SETTLED_RATE) -> dict:
    """Pick the checkpoint for a policy trained with `measure`, judged by
    that same measure on a selection bank: among candidates that come to
    rest in >= min_settled of falls, the lowest median of the measure (ties
    -> first candidate). If none comes to rest that often, the lowest median
    overall, flagged eligible=False. per_arm[label] holds 'settled' and
    'size__<measure>' arrays."""
    rows = []
    for c in candidates:
        rows.append({"label": c, "settled": float(np.mean(per_arm[c]["settled"])),
                     "median": float(np.median(per_arm[c][f"size__{measure}"]))})
    for r in rows:
        r["eligible"] = r["settled"] >= min_settled
    pool = [r for r in rows if r["eligible"]] or rows
    best = min(pool, key=lambda r: r["median"])
    return {"measure": measure, "selected": best["label"], "selected_is_eligible": best["eligible"], "candidates": rows}


def trained_on_report(rep: dict) -> dict | None:
    """For policies labelled r_<tag>: the ratio-to-reference medians matrix
    (rows = reward trained on, columns = measure) and whether each policy
    is the smallest on its own measure. None if there are no r_ policies."""
    own = {p: TAG_TO_MEASURE[p[2:]] for p in rep["policies"] if p.startswith("r_") and p[2:] in TAG_TO_MEASURE}
    if not own:
        return None
    ref = rep["reference"]
    med = rep["medians"]
    ratio = {p: {m: (med[p][m] / med[ref][m] if med[ref][m] else float("nan")) for m in rep["metrics"]} for p in own}
    trained = list(own)
    own_rank = {}
    for p, m in own.items():
        among = sorted(trained, key=lambda q: med[q][m])
        own_rank[p] = {"measure": m, "rank_among_trained": among.index(p) + 1, "of": len(among),
                       "rank_among_all": rep["rank_by_metric"][m].index(p) + 1,
                       "smallest_trained_on_it": among[0]}
    return {"ratio_to_reference": ratio, "own_measure": own_rank,
            "wins_own_measure": sum(r["rank_among_trained"] == 1 for r in own_rank.values()),
            "num_trained": len(own)}


# --- Round 2 (G1): smallest fall that is still safe --------------------------------
# DEVIATIONS.md 2026-11-14. "Size score": one measure per family the reward
# ablation found (they trade off against each other), each relative to limp
# on the same fall, geometric mean over the four and over falls. 0.8 = 20%
# smaller than limp, averaged over the four kinds of "small".
FAMILY_MEASURES = ("envelope_growth_m", "centroid_travel_end_m", "hull_growth_m2", "touched_area_m2")
SAFE_MIN_SETTLED = 0.97  # the ablation showed 0.99 is out of reach for good policies (r_ground 0.97-0.98)


def size_floors(per_arm: dict[str, dict[str, np.ndarray]], measures=FAMILY_MEASURES) -> dict[str, float]:
    """1% of each measure's pooled median over every arm (as in method C)."""
    return {m: max(RATIO_FLOOR_FRACTION * float(np.median(np.abs(np.concatenate(
        [per_arm[a][f"size__{m}"] for a in per_arm])))), 1e-9) for m in measures}


def per_fall_size_score(arm: dict[str, np.ndarray], ref: dict[str, np.ndarray], floors: dict[str, float],
                        measures=FAMILY_MEASURES) -> np.ndarray:
    """Per fall: geometric mean over `measures` of (arm + floor) / (ref + floor)."""
    logs = [np.log((arm[f"size__{m}"] + floors[m]) / (ref[f"size__{m}"] + floors[m])) for m in measures]
    return np.exp(np.mean(logs, axis=0))


def safety_row(arm: dict[str, np.ndarray]) -> dict[str, float]:
    return {"settled": float(np.mean(arm["settled"])),
            "trunk_force_median": float(np.median(arm["peak_force_critical_n"])),
            "any_force_median": float(np.median(arm["peak_force_any_n"])),
            "ground_covered_median": float(np.median(arm["size__envelope_growth_m"]))}


def select_safe_small(per_arm: dict[str, dict[str, np.ndarray]], candidates: list[str], reference: str = "limp",
                      walker: str = "walker", min_settled: float = SAFE_MIN_SETTLED) -> dict:
    """Round-2 checkpoint rule (fixed before the data):
      safe     = median trunk/pelvis peak force <= the walker's AND median
                 any-body peak force < limp's;
      eligible = safe AND comes to rest in >= min_settled of falls;
      choose the eligible candidate with the lowest size score (vs limp).
    Fallbacks, flagged: lowest score among those that settle often enough,
    then among all."""
    floors = size_floors(per_arm)
    w, ref = safety_row(per_arm[walker]), safety_row(per_arm[reference])
    rows = []
    for c in candidates:
        r = {"label": c, **safety_row(per_arm[c])}
        s = per_fall_size_score(per_arm[c], per_arm[reference], floors)
        r["size_score"] = float(np.exp(np.mean(np.log(s))))
        r["safe"] = r["trunk_force_median"] <= w["trunk_force_median"] and r["any_force_median"] < ref["any_force_median"]
        r["eligible"] = r["safe"] and r["settled"] >= min_settled
        rows.append(r)
    pool = ([r for r in rows if r["eligible"]] or [r for r in rows if r["settled"] >= min_settled] or rows)
    best = min(pool, key=lambda r: (r["size_score"], r["label"]))
    return {"selected": best["label"], "selected_is_eligible": best["eligible"], "size_score": best["size_score"],
            "floors": floors, "walker": w, "reference": ref, "candidates": rows}
