"""Tests for scripts/eval_predictor.py's pure metric functions and the full
evaluate_one_dataset pipeline (CLAUDE.md S8, S9).

No mjlab or GPU required (CLAUDE.md S0.2.5): eval_predictor.py only
consumes already-collected .npz datasets, and its metric/orchestration
logic takes model/train/split as injected callables specifically so it can
be tested this way.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from _helpers import load_module, load_script

eval_predictor = load_script("scripts/eval_predictor.py", "_eval_predictor_under_test")
model_mod = load_module("predictor/model.py", "_model_under_test")
train_mod = load_module("predictor/train.py", "_train_under_test")
split_mod = load_module("predictor/split.py", "_split_under_test")


def test_mcfadden_r2_perfect_and_null():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p_perfect = np.array([0.001, 0.001, 0.001, 0.001, 0.999, 0.999, 0.999, 0.999])
    p_null = np.full(8, y.mean())

    assert eval_predictor.mcfadden_r2(y, p_perfect) > 0.9
    assert abs(eval_predictor.mcfadden_r2(y, p_null)) < 1e-9


def test_ece_low_for_well_calibrated_predictions():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    ece, bins = eval_predictor.expected_calibration_error(y, p, n_bins=15)
    assert ece < 0.08
    assert len(bins) == 15


def test_ece_high_for_badly_miscalibrated_predictions():
    # Predictor always says ~0.9 regardless of true label -- badly miscalibrated.
    y = np.array([0] * 900 + [1] * 100)
    p = np.full(1000, 0.9)
    ece, _ = eval_predictor.expected_calibration_error(y, p, n_bins=15)
    assert ece > 0.5


def test_median_detection_lead_time():
    episode_id = np.array([0, 0, 0, 1, 1, 1])
    steps_to_term = np.array([3, 2, 1, 10, 5, 1])
    label = np.array([1, 1, 1, 1, 1, 1])
    # episode 0: all three (k=3,2,1) exceed threshold -> earliest alarm at k=3
    # episode 1: only k=5 and k=1 exceed threshold (k=10 does not) -> earliest alarm at k=5
    probs = np.array([0.9, 0.9, 0.9, 0.1, 0.9, 0.9])

    result = eval_predictor.median_detection_lead_time(episode_id, steps_to_term, label, probs, threshold=0.5)
    assert result["num_detected_episodes"] == 2
    assert result["median_lead_steps"] == 4.0  # median of [3, 5]
    assert result["median_lead_seconds"] == 4.0 * eval_predictor.STEP_SECONDS


def test_median_detection_lead_time_handles_no_detections():
    episode_id = np.array([0, 0])
    steps_to_term = np.array([2, 1])
    label = np.array([1, 1])
    probs = np.array([0.1, 0.1])  # never crosses threshold

    result = eval_predictor.median_detection_lead_time(episode_id, steps_to_term, label, probs, threshold=0.5)
    assert result["num_detected_episodes"] == 0
    assert result["median_lead_steps"] is None
    assert result["num_episodes_with_positives"] == 1


def _synthetic_frozen_npz(n_episodes: int = 200, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    episode_id = 0
    for ep_i in range(n_episodes):
        ep_len = int(rng.integers(20, 40))
        cause = "pole_angle" if ep_i % 3 != 0 else "cart_pos"
        n_pos = min(16, ep_len - 1)
        n_neg = ep_len - 1 - n_pos
        for step in range(n_neg):
            theta = rng.normal(0, 0.05)
            x = rng.normal(0, 0.1)
            rows.append(
                dict(
                    obs=[x, np.cos(theta), np.sin(theta), rng.normal(0, 0.1), rng.normal(0, 0.1)],
                    action=[rng.uniform(-1, 1)],
                    label=0,
                    env_id=ep_i % 8,
                    episode_id=episode_id,
                    step_in_episode=step,
                    steps_to_termination=-1,
                    termination_cause="",
                )
            )
        for k in range(n_pos, 0, -1):
            severity = (n_pos - k + 1) / n_pos
            if cause == "pole_angle":
                theta = 0.15 + 0.15 * severity + rng.normal(0, 0.02)
                x = rng.normal(0, 0.1)
            else:
                theta = rng.normal(0, 0.05)
                x = 0.6 + 0.5 * severity + rng.normal(0, 0.05)
            rows.append(
                dict(
                    obs=[x, np.cos(theta), np.sin(theta), rng.normal(0, 0.1), rng.normal(0, 0.1)],
                    action=[rng.uniform(-1, 1)],
                    label=1,
                    env_id=ep_i % 8,
                    episode_id=episode_id,
                    step_in_episode=n_neg + (n_pos - k),
                    steps_to_termination=k,
                    termination_cause=cause,
                )
            )
        episode_id += 1

    return {
        "obs": np.array([r["obs"] for r in rows], dtype=np.float32),
        "action": np.array([r["action"] for r in rows], dtype=np.float32),
        "label": np.array([r["label"] for r in rows], dtype=np.int64),
        "env_id": np.array([r["env_id"] for r in rows], dtype=np.int64),
        "episode_id": np.array([r["episode_id"] for r in rows], dtype=np.int64),
        "step_in_episode": np.array([r["step_in_episode"] for r in rows], dtype=np.int64),
        "steps_to_termination": np.array([r["steps_to_termination"] for r in rows], dtype=np.int64),
        "termination_cause": np.array([r["termination_cause"] for r in rows], dtype="U16"),
    }


def test_evaluate_one_dataset_end_to_end_on_learnable_synthetic_data():
    npz_data = _synthetic_frozen_npz(n_episodes=150, seed=0)

    with tempfile.TemporaryDirectory() as d:
        npz_path = Path(d) / "frozen_target_seed0.npz"
        sidecar_path = Path(d) / "frozen_target_seed0.json"
        np.savez_compressed(npz_path, **npz_data)
        sidecar_path.write_text(json.dumps({"seed": 0, "operating_point": "target"}))

        result = eval_predictor.evaluate_one_dataset(
            npz_path,
            sidecar_path,
            seed=0,
            predictor_history=1,
            model_cls=model_mod.TerminationPredictor,
            train_fn=train_mod.train_predictor,
            split_fn=split_mod.split_episodes,
            device="cpu",
        )

    # MLP should clearly beat the base-rate baseline on learnable data.
    assert result["mlp"]["auprc"] > result["mlp"]["auprc_baseline"]
    # abs_theta_wrapped must come out strongly predictive -- CLAUDE.md S9.3:
    # "If it doesn't, the labels are wrong; stop and debug."
    assert result["feature_screening"]["abs_theta_wrapped"]["auc_roc"] > 0.7
    # Baselines must be well-formed and not accidentally beat the real model
    # by a wide margin on data explicitly constructed to be learnable.
    assert result["baselines"]["constant_negative"]["auprc"] is None or (
        result["baselines"]["constant_negative"]["auprc"] <= result["mlp"]["auprc"]
    )
    # Both causes present in this synthetic construction should get a breakdown.
    assert result["per_cause_breakdown"]["pole_angle"]["n_positive"] > 0
    assert result["per_cause_breakdown"]["cart_pos"]["n_positive"] > 0


def test_evaluate_one_dataset_rejects_history_greater_than_one():
    npz_data = _synthetic_frozen_npz(n_episodes=10, seed=0)
    with tempfile.TemporaryDirectory() as d:
        npz_path = Path(d) / "frozen_target_seed0.npz"
        sidecar_path = Path(d) / "frozen_target_seed0.json"
        np.savez_compressed(npz_path, **npz_data)
        sidecar_path.write_text(json.dumps({}))

        try:
            eval_predictor.evaluate_one_dataset(
                npz_path,
                sidecar_path,
                seed=0,
                predictor_history=2,
                model_cls=model_mod.TerminationPredictor,
                train_fn=train_mod.train_predictor,
                split_fn=split_mod.split_episodes,
                device="cpu",
            )
            raised = False
        except NotImplementedError:
            raised = True
        assert raised


def test_evaluate_one_dataset_is_deterministic_given_same_seed():
    # Regression test: model_cls(...) used to be constructed (randomly
    # initializing weights) BEFORE train_fn's internal torch.manual_seed(seed)
    # call ever ran, so the seed silently never controlled weight init.
    # Caught in practice by a Gate 4 re-run producing a different AUPRC from
    # a byte-identical dataset and the same seed (CLAUDE.md S8.3: "Fixed
    # torch seed per run; report it").
    npz_data = _synthetic_frozen_npz(n_episodes=150, seed=0)

    auprcs = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            npz_path = Path(d) / "frozen_target_seed0.npz"
            sidecar_path = Path(d) / "frozen_target_seed0.json"
            np.savez_compressed(npz_path, **npz_data)
            sidecar_path.write_text(json.dumps({}))

            result = eval_predictor.evaluate_one_dataset(
                npz_path,
                sidecar_path,
                seed=0,
                predictor_history=1,
                model_cls=model_mod.TerminationPredictor,
                train_fn=train_mod.train_predictor,
                split_fn=split_mod.split_episodes,
                device="cpu",
            )
            auprcs.append(result["mlp"]["auprc"])

    assert auprcs[0] == auprcs[1], auprcs


def test_evaluate_one_dataset_saves_model_that_reproduces_test_auprc():
    # scripts/record_video.py's predictor-overlay video loads these saved
    # weights rather than retraining -- this checks the round trip actually
    # reproduces the same predictions the reported metrics were computed
    # from, not just that torch.save/load don't raise.
    import torch

    npz_data = _synthetic_frozen_npz(n_episodes=150, seed=0)

    with tempfile.TemporaryDirectory() as d:
        npz_path = Path(d) / "frozen_target_seed0.npz"
        sidecar_path = Path(d) / "frozen_target_seed0.json"
        model_save_path = Path(d) / "predictor_target_seed0.pt"
        np.savez_compressed(npz_path, **npz_data)
        sidecar_path.write_text(json.dumps({}))

        result = eval_predictor.evaluate_one_dataset(
            npz_path,
            sidecar_path,
            seed=0,
            predictor_history=1,
            model_cls=model_mod.TerminationPredictor,
            train_fn=train_mod.train_predictor,
            split_fn=split_mod.split_episodes,
            device="cpu",
            model_save_path=model_save_path,
        )

        assert model_save_path.exists()

        npz_full = dict(np.load(npz_path, allow_pickle=False))
        _, _, test = eval_predictor.split_dataset_by_split_fn(
            npz_full, split_mod.split_episodes, seed=0
        )
        x_test = eval_predictor.make_inputs(test)
        y_test = test["label"].astype(np.float32)

        reloaded = model_mod.TerminationPredictor(history=1)
        reloaded.load_state_dict(torch.load(model_save_path, weights_only=True))
        reloaded.eval()
        with torch.no_grad():
            p_test = torch.sigmoid(reloaded(torch.as_tensor(x_test)).squeeze(-1)).numpy()

        reloaded_auprc = eval_predictor.average_precision_score(y_test, p_test)
        assert reloaded_auprc == result["mlp"]["auprc"], (reloaded_auprc, result["mlp"]["auprc"])
