"""Re-sizing the protective reward weights for a new robot by the Go1's rule
(G1 port, DEVIATIONS.md 2026-11-13).

On the Go1, four terms had weights sized from the Go1's own measured
magnitudes so that each contributed a comparable amount to the reward
(env_cfg.py: joint torques, joint limits, ground covered, peak force). The
other terms use arXiv:2512.01336's weights and are kept as they are.

For a new robot: train briefly with the Go1 weights, read each sized term's
logged contribution (mjlab's Episode_Reward/<term>, weight x mean raw value
per second) over iterations 0-10, and scale its weight so that contribution
matches the Go1's over the same iterations of its final training run
(round 8b). Early iterations are used on both robots because the policy is
then near its random initialisation, so both measurements describe the
same condition.
"""

from __future__ import annotations

CALIBRATED_TERMS = ("joint_torques_l2", "joint_pos_limits", "fall_envelope_growth", "peak_contact_force_growth")

# Go1 round 8b (the reported policy's training run), mean Episode_Reward/<term>
# over iterations 0-10, read from
# logs/rsl_rl/go1_protective_fall/round8b_2026-10-03 (TensorBoard events).
GO1_REFERENCE_CONTRIBUTIONS = {
    "joint_torques_l2": -0.0580,
    "joint_pos_limits": -0.0016,
    "fall_envelope_growth": -0.1190,
    "peak_contact_force_growth": -0.1787,
}
CALIBRATION_ITERATIONS = (0, 10)  # inclusive window, on both robots
FACTOR_LIMITS = (0.02, 50.0)  # guard against a near-zero measurement blowing a weight up


def calibrate(measured: dict[str, float], base_weights: dict[str, float],
              reference: dict[str, float] = GO1_REFERENCE_CONTRIBUTIONS,
              limits: tuple[float, float] = FACTOR_LIMITS) -> dict:
    """New weights for CALIBRATED_TERMS: base weight x reference / measured,
    with the factor clamped to `limits`. A term whose measured contribution is
    zero, missing or of the opposite sign keeps its base weight (flagged)."""
    lo, hi = limits
    weights, rows = {}, {}
    for term in CALIBRATED_TERMS:
        base, ref, got = base_weights[term], reference[term], measured.get(term)
        row = {"base_weight": base, "go1_contribution": ref, "measured_contribution": got}
        if got is None or got == 0.0 or (got > 0) != (ref > 0):
            row.update(factor=1.0, clamped=False, kept_base=True)
        else:
            raw = ref / got
            factor = min(max(raw, lo), hi)
            row.update(factor=factor, raw_factor=raw, clamped=factor != raw, kept_base=False)
        row["weight"] = base * row["factor"]
        weights[term] = row["weight"]
        rows[term] = row
    return {"weights": weights, "terms": rows, "reference": dict(reference), "limits": list(limits)}
