"""Episode-level train/val/test split (CLAUDE.md S7.5).

Splitting by transition would leak strongly-correlated consecutive steps
(and near-duplicate steps within one horizon window) across splits and
inflate AUROC to a meaningless number. Split by episode_id instead: every
step of a given episode lands in exactly one split.
"""

from __future__ import annotations

import random


def split_episodes(
    episode_ids: list[int],
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[set[int], set[int], set[int]]:
    """Deterministically partition unique episode ids into train/val/test.

    Returns (train_ids, val_ids, test_ids): disjoint sets whose union is the
    set of unique input episode_ids. The remainder after train_frac and
    val_frac (nominally 1 - train_frac - val_frac, e.g. 0.15 for the default
    70/15/15 split) goes to test.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac={train_frac}, val_frac={val_frac} must each be in "
            "(0, 1) and sum to less than 1"
        )
    unique_ids = sorted(set(episode_ids))
    rng = random.Random(seed)
    shuffled = unique_ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = min(round(n * train_frac), n)
    n_val = min(round(n * val_frac), n - n_train)

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])
    test_ids = set(shuffled[n_train + n_val :])
    return train_ids, val_ids, test_ids
