"""History-frame stacking (CLAUDE.md S8.1's `--predictor.history H` option --
listed in the spec as optional/off-by-default, never actually implemented:
scripts/eval_predictor.py raises NotImplementedError for history != 1 to this
day, precisely to avoid "silently approximating" a history-window
reconstruction it hadn't verified. This module is that reconstruction, built
generically over any flat per-step dataset (works for cartpole's 6-wide
obs+action or Go1's 60-wide one unchanged).

Built for the Go1 wild-push predictor's short-lead-time tail (see
DEVIATIONS.md): a single-frame snapshot can't distinguish "about to fall
fast" from "about to recover" until very late, because it has no access to a
trend. Stacking the last H frames gives the model that trend directly
(matches STATE-NAV's own gait-cycle aggregation, CLAUDE.md's own analogy for
why history exists at all).

**Why "drop", not literally "zero-pad and mask" as CLAUDE.md's prose says:**
zero-padding a step near episode start and then masking it out of the loss
produces the exact same trained model as just never including that row in
the first place -- the row never contributes gradient either way. Dropping
is simpler and, more importantly, avoids ever handing a real (but disabled)
input to `train_predictor`'s existing generic code path, which has no
concept of a per-sample loss mask -- adding one would be new surface area
for a result that's already achieved by exclusion. Also avoids any risk of
the model latching onto "an all-zero older frame" as an artifact correlated
with "near episode start" -- which is itself close to the CLAUDE.md S8.1
forbidden `step_in_episode` leak.

**Why simple index-shifting (not a per-row dict lookup) is correct here:**
CLAUDE.md S7.4's labelling ring buffer emits, per episode, EITHER the full
contiguous prefix {0, ..., T-1} (terminated episodes: steps age out to
negatives one at a time up to T-17, then the terminal burst emits T-16..T-1
as positives all at once) OR the contiguous prefix {0, ..., T-17} (truncated
episodes: the last 16 steps are censored and never emitted at all). Either
way, every step_in_episode actually present in the saved flat dataset for a
given episode_id is preceded by every smaller step_in_episode for that same
episode -- there are no mid-episode gaps, only a possible truncation at the
tail. That means "the row H-1 positions earlier in the (episode_id,
step_in_episode)-sorted array, same episode_id" is exactly "H-1 steps
earlier in wall-clock time" -- no need to look anything up by value.
`_assert_contiguous_prefix` below verifies this holds for the actual data
before trusting it, rather than assuming it silently.
"""

from __future__ import annotations

import numpy as np


def _assert_contiguous_prefix(episode_id: np.ndarray, step_in_episode: np.ndarray, order: np.ndarray) -> None:
    sorted_eid = episode_id[order]
    sorted_step = step_in_episode[order]
    n = len(order)
    is_episode_start = np.empty(n, dtype=bool)
    is_episode_start[0] = True
    is_episode_start[1:] = sorted_eid[1:] != sorted_eid[:-1]
    group_id = np.cumsum(is_episode_start) - 1
    episode_start_positions = np.nonzero(is_episode_start)[0][group_id]
    expected_step = np.arange(n) - episode_start_positions
    if not np.array_equal(expected_step, sorted_step):
        bad = np.nonzero(expected_step != sorted_step)[0][0]
        raise AssertionError(
            "History-stacking's contiguous-prefix assumption (module docstring) is "
            f"violated at sorted position {bad}: episode_id={sorted_eid[bad]}, "
            f"step_in_episode={sorted_step[bad]}, expected {expected_step[bad]}. "
            "The dataset has a mid-episode gap this module doesn't know how to handle "
            "-- do not silently build history windows across it."
        )


def build_history_stacked_dataset(dataset: dict[str, np.ndarray], history: int) -> dict[str, np.ndarray]:
    """Returns a NEW dataset dict with 'obs'/'action' replaced by their
    H-frame history-stacked versions (oldest frame first, so the LAST
    obs_dim/action_dim slice of the stacked row is always the current
    frame -- unstacking is just `stacked[..., -obs_dim:]`), and every row
    that would need a frame from before its own episode started DROPPED.
    Every other field (label, episode_id, step_in_episode,
    steps_to_termination, termination_cause, ...) is subset to match the
    kept rows, in the same relative order convention (not required to
    match the INPUT order -- callers that need a specific order, e.g. for
    determinism, should not rely on it, same as any other dict-of-arrays
    dataset in this project).

    history=1 returns `dataset` unchanged (a true no-op) -- every existing
    call site can pass history=1 with zero behavior change.
    """
    if history < 1:
        raise ValueError(f"history must be >= 1, got {history}")
    if history == 1:
        return dataset

    episode_id = dataset["episode_id"]
    step_in_episode = dataset["step_in_episode"]
    obs = dataset["obs"]
    action = dataset["action"]
    n = len(episode_id)
    if n == 0:
        return dict(dataset)

    order = np.lexsort((step_in_episode, episode_id))
    _assert_contiguous_prefix(episode_id, step_in_episode, order)

    sorted_eid = episode_id[order]
    sorted_obs = obs[order]
    sorted_action = action[order]

    valid = np.ones(n, dtype=bool)
    obs_frames = [sorted_obs]  # index 0 = current frame (k=0)
    action_frames = [sorted_action]
    for k in range(1, history):
        shifted_eid = np.full(n, -1, dtype=sorted_eid.dtype)
        shifted_obs = np.zeros_like(sorted_obs)
        shifted_action = np.zeros_like(sorted_action)
        shifted_eid[k:] = sorted_eid[:-k]
        shifted_obs[k:] = sorted_obs[:-k]
        shifted_action[k:] = sorted_action[:-k]
        valid &= shifted_eid == sorted_eid
        obs_frames.append(shifted_obs)
        action_frames.append(shifted_action)

    # oldest-first: reverse so frame[0]=t-(H-1), ..., frame[-1]=t (current)
    stacked_obs = np.concatenate(list(reversed(obs_frames)), axis=1)
    stacked_action = np.concatenate(list(reversed(action_frames)), axis=1)

    kept = valid
    result: dict[str, np.ndarray] = {}
    for key, values in dataset.items():
        sorted_values = values[order]
        if key == "obs":
            result[key] = stacked_obs[kept]
        elif key == "action":
            result[key] = stacked_action[kept]
        else:
            result[key] = sorted_values[kept]
    return result
