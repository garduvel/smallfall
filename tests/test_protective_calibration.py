"""Reward re-sizing for a new robot (src/cartpole_term/protective/calibration.py)."""

from __future__ import annotations

import pytest

from _helpers import load_module

cal = load_module("protective/calibration.py", "_calibration_under_test")

BASE = {"joint_torques_l2": -5e-4, "joint_pos_limits": -10.0, "fall_envelope_growth": -30.0,
        "peak_contact_force_growth": -0.02}


def test_matching_contribution_keeps_the_weight():
    out = cal.calibrate(dict(cal.GO1_REFERENCE_CONTRIBUTIONS), BASE)
    assert out["weights"] == pytest.approx(BASE)


def test_term_twice_as_large_gets_half_the_weight():
    measured = dict(cal.GO1_REFERENCE_CONTRIBUTIONS)
    measured["joint_torques_l2"] *= 2
    out = cal.calibrate(measured, BASE)
    assert out["weights"]["joint_torques_l2"] == pytest.approx(-2.5e-4)
    assert out["weights"]["fall_envelope_growth"] == pytest.approx(-30.0)


def test_factor_is_clamped_and_flagged():
    measured = dict(cal.GO1_REFERENCE_CONTRIBUTIONS)
    measured["joint_pos_limits"] = -0.0016 / 1000  # 1000x smaller -> factor 1000, clamped to 50
    out = cal.calibrate(measured, BASE)
    assert out["terms"]["joint_pos_limits"]["clamped"]
    assert out["weights"]["joint_pos_limits"] == pytest.approx(-10.0 * 50)


@pytest.mark.parametrize("bad", [None, 0.0, +0.1])
def test_missing_zero_or_wrong_sign_keeps_base(bad):
    measured = dict(cal.GO1_REFERENCE_CONTRIBUTIONS)
    if bad is None:
        del measured["fall_envelope_growth"]
    else:
        measured["fall_envelope_growth"] = bad
    out = cal.calibrate(measured, BASE)
    assert out["terms"]["fall_envelope_growth"]["kept_base"]
    assert out["weights"]["fall_envelope_growth"] == -30.0


def test_weights_stay_penalties():
    measured = {k: v * 7 for k, v in cal.GO1_REFERENCE_CONTRIBUTIONS.items()}
    out = cal.calibrate(measured, BASE)
    assert all(w < 0 for w in out["weights"].values())


def test_go1_protective_settings_are_unchanged_by_the_port():
    env_cfg = load_module("protective/env_cfg.py", "_protective_env_cfg_under_test")
    assert env_cfg.TASK_IDS["go1"] == env_cfg.TASK_ID == "Mjlab-Go1-Protective-Fall"
    assert env_cfg.PROTECTIVE_EPISODE_LENGTH_S_BY_ROBOT == {"go1": 3.0, "g1": 5.0}
    assert env_cfg.GO1_REWARD_WEIGHTS == {
        "root_linear_velocity": -1.03125, "action_rate_l2": -0.00667, "joint_pos_limits": -10.0,
        "joint_torques_l2": -5e-4, "fall_envelope_growth": -30.0, "peak_contact_force_growth": -0.02,
    }
    assert set(cal.CALIBRATED_TERMS) <= set(env_cfg.GO1_REWARD_WEIGHTS)
