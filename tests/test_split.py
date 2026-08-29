"""Tests for src/cartpole_term/predictor/split.py (CLAUDE.md S7.5).

No mjlab or GPU required (CLAUDE.md S0.2.5).
"""

from __future__ import annotations

from _helpers import load_module

split_mod = load_module("predictor/split.py", "_split_under_test")
split_episodes = split_mod.split_episodes


def test_splits_are_disjoint_and_cover_all_episodes():
    episode_ids = list(range(1000))
    train, val, test = split_episodes(episode_ids, seed=0)

    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert train | val | test == set(episode_ids)


def test_split_proportions_are_roughly_70_15_15():
    episode_ids = list(range(1000))
    train, val, test = split_episodes(episode_ids, seed=0)

    n = len(episode_ids)
    assert abs(len(train) / n - 0.70) < 0.02
    assert abs(len(val) / n - 0.15) < 0.02
    assert abs(len(test) / n - 0.15) < 0.02


def test_split_is_deterministic_given_seed():
    episode_ids = list(range(500))
    a = split_episodes(episode_ids, seed=42)
    b = split_episodes(episode_ids, seed=42)
    assert a == b


def test_different_seeds_give_different_splits():
    episode_ids = list(range(500))
    a = split_episodes(episode_ids, seed=0)
    b = split_episodes(episode_ids, seed=1)
    assert a != b


def test_duplicate_episode_ids_in_input_are_deduplicated():
    # Simulates the real use case: many transitions share an episode_id, so
    # the input list has heavy duplication -- only unique ids are split.
    episode_ids = [i for i in range(100) for _ in range(20)]  # 100 unique, 2000 total
    train, val, test = split_episodes(episode_ids, seed=0)
    assert train | val | test == set(range(100))
    assert len(train) + len(val) + len(test) == 100


def test_transition_level_membership_is_single_split():
    # Every transition (episode_id, step) must map into exactly one split,
    # via its episode_id -- this is the core anti-leakage property S7.5
    # requires (a transition never appears in two splits).
    transitions = [(episode_id, step) for episode_id in range(200) for step in range(15)]
    episode_ids = [t[0] for t in transitions]
    train, val, test = split_episodes(episode_ids, seed=7)

    for episode_id, _ in transitions:
        memberships = sum(episode_id in s for s in (train, val, test))
        assert memberships == 1, episode_id


def test_rejects_invalid_fractions():
    import pytest

    with pytest.raises(ValueError):
        split_episodes([1, 2, 3], seed=0, train_frac=0.9, val_frac=0.2)
