"""Tests for cartpole_term.predictor.history (CLAUDE.md S8.1's
`--predictor.history H`, implemented for the Go1 wild-push predictor --
see DEVIATIONS.md). Pure Python/numpy, no GPU/mjlab needed -- run in-sandbox
per this project's standing rule that anything runnable without a GPU must
actually be run, same as tests/test_labeling.py and tests/test_split.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from _helpers import load_module

history = load_module("predictor/history.py", "_history_under_test")
build_history_stacked_dataset = history.build_history_stacked_dataset


def _make_dataset(episode_id, step_in_episode, obs_dim=2, action_dim=1):
    n = len(episode_id)
    # obs/action encode (episode_id, step_in_episode) directly so stacked
    # windows can be checked by value, not just by shape.
    if n == 0:
        obs = np.zeros((0, obs_dim), dtype=np.float32)
        action = np.zeros((0, action_dim), dtype=np.float32)
    else:
        obs = np.array(
            [[eid * 1000.0 + s, -(eid * 1000.0 + s)] for eid, s in zip(episode_id, step_in_episode)],
            dtype=np.float32,
        )[:, :obs_dim]
        action = np.array(
            [[eid * 1000.0 + s + 0.5] for eid, s in zip(episode_id, step_in_episode)], dtype=np.float32
        )[:, :action_dim]
    return {
        "obs": obs,
        "action": action,
        "episode_id": np.array(episode_id, dtype=np.int64),
        "step_in_episode": np.array(step_in_episode, dtype=np.int64),
        "label": np.zeros(n, dtype=np.int64),
        "steps_to_termination": np.full(n, -1, dtype=np.int64),
        "termination_cause": np.array([""] * n, dtype="U32"),
    }


def test_history_1_is_noop():
    ds = _make_dataset([0, 0, 0], [0, 1, 2])
    out = build_history_stacked_dataset(ds, history=1)
    assert out is ds


def test_single_episode_stacking_and_boundary_drop():
    ds = _make_dataset(episode_id=[0] * 5, step_in_episode=[0, 1, 2, 3, 4])
    out = build_history_stacked_dataset(ds, history=3)

    # Rows needing a frame before step 0 (steps 0, 1) are dropped.
    assert sorted(out["step_in_episode"].tolist()) == [2, 3, 4]
    assert out["obs"].shape == (3, 3 * 2)  # history * obs_dim

    # Row for step 4 (episode_id=0, so eid*1000=0): oldest-first ->
    # [obs(2), obs(3), obs(4)], last slice is current frame.
    row = out["obs"][out["step_in_episode"] == 4][0]
    expected = np.concatenate([[2.0, -2.0], [3.0, -3.0], [4.0, -4.0]])
    np.testing.assert_array_equal(row, expected)
    # Last obs_dim slice is exactly the unstacked current-frame obs.
    np.testing.assert_array_equal(row[-2:], [4.0, -4.0])


def test_multiple_episodes_no_cross_contamination_and_input_order_irrelevant():
    # Two episodes, interleaved in the raw array (not pre-sorted) -- the
    # function must sort internally and never let episode 1's early steps
    # see episode 0's frames.
    episode_id = [1, 0, 1, 0, 0, 1]
    step_in_episode = [1, 1, 0, 0, 2, 2]
    ds = _make_dataset(episode_id, step_in_episode)
    out = build_history_stacked_dataset(ds, history=2)

    # Episode 0 has steps {0,1,2}; step 0 dropped (needs history-1=1 prior
    # frame it doesn't have). Episode 1 has steps {0,1,2}; same rule.
    kept = {(int(e), int(s)) for e, s in zip(out["episode_id"], out["step_in_episode"])}
    assert kept == {(0, 1), (0, 2), (1, 1), (1, 2)}

    row = out["obs"][(out["episode_id"] == 1) & (out["step_in_episode"] == 2)][0]
    expected = np.concatenate([[1001.0, -1001.0], [1002.0, -1002.0]])
    np.testing.assert_array_equal(row, expected)


def test_other_fields_preserved_and_aligned():
    ds = _make_dataset(episode_id=[0] * 4, step_in_episode=[0, 1, 2, 3])
    ds["label"] = np.array([0, 0, 1, 1], dtype=np.int64)
    out = build_history_stacked_dataset(ds, history=2)
    for eid, step, label in zip(out["episode_id"], out["step_in_episode"], out["label"]):
        assert label == (1 if step >= 2 else 0)


def test_raises_on_mid_episode_gap():
    # Step 1 is missing entirely for episode 0 -- violates the contiguous-
    # prefix assumption this module depends on (module docstring).
    ds = _make_dataset(episode_id=[0, 0, 0], step_in_episode=[0, 2, 3])
    with pytest.raises(AssertionError):
        build_history_stacked_dataset(ds, history=2)


def test_empty_dataset():
    ds = _make_dataset(episode_id=[], step_in_episode=[])
    out = build_history_stacked_dataset(ds, history=4)
    assert len(out["episode_id"]) == 0
