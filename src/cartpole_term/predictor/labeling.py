"""Termination-horizon labelling for the predictor dataset (CLAUDE.md S7.4).

y_t^i = 1  iff  the env terminates (failure) at step t+k for some k in
                {1, ..., horizon}
y_t^i = 0  iff  the env survives all of steps t+1 .. t+horizon
y_t^i = discarded (censored) otherwise -- in particular, every pending
                observation is discarded, uncounted, when an episode ends by
                *truncation* (time-out): its true future is unknown, since
                the episode would have continued but for the time limit.

Implementation: a per-environment ring buffer of pending observations.
Terminated/truncated are checked with terminated taking priority if somehow
both fire on the same step (failure wins).

One subtlety this implementation gets right on purpose: an entry that ages
past the horizon (k = horizon + 1) is a NEGATIVE by the formal definition
above, and this must hold *even if* termination also fires that same step
for a different (younger) entry. The eviction of such entries therefore runs
unconditionally every step, before the terminated/truncated branch -- not
folded into an "else" branch keyed on this step's outcome. Folding it into
the branch would silently drop that one entry (neither positive nor
negative) on the rare step where both conditions coincide, which would be a
quiet, hard-to-notice bug in exactly the boundary case the horizon is
supposed to get right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class LabeledSample(Generic[T]):
    env_id: int
    payload: T
    label: int  # 0 or 1
    steps_to_termination: int | None = None
    """For positives, the number of steps k (1..horizon) from this
    observation to the actual termination -- exactly the entry's age at
    emission. None (sentinel) for negatives: CLAUDE.md S7.3 calls for a
    sentinel here since a negative's real future termination step, if any,
    is unknown/irrelevant -- all we've established is survival past the
    horizon."""


@dataclass
class _PendingEntry(Generic[T]):
    payload: T
    age: int = 0


class TerminationLabeler(Generic[T]):
    """Call `.step(...)` once per env step, with one payload/flag per env."""

    def __init__(self, num_envs: int, horizon: int = 16):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.num_envs = num_envs
        self.horizon = horizon
        self._buffers: list[list[_PendingEntry[T]]] = [[] for _ in range(num_envs)]

    def step(
        self,
        payloads: list[T],
        terminated: list[bool],
        truncated: list[bool],
    ) -> list[LabeledSample[T]]:
        if not (len(payloads) == len(terminated) == len(truncated) == self.num_envs):
            raise ValueError(
                f"Expected length {self.num_envs} for payloads/terminated/"
                f"truncated, got {len(payloads)}/{len(terminated)}/{len(truncated)}"
            )

        emitted: list[LabeledSample[T]] = []

        for env_id in range(self.num_envs):
            buf = self._buffers[env_id]

            # 1. Age every entry pushed on a prior step.
            for entry in buf:
                entry.age += 1

            # 2. Unconditional eviction: an entry that just aged past the
            # horizon is a negative, regardless of this step's outcome.
            still_pending: list[_PendingEntry[T]] = []
            for entry in buf:
                if entry.age == self.horizon + 1:
                    emitted.append(LabeledSample(env_id, entry.payload, 0, steps_to_termination=None))
                else:
                    still_pending.append(entry)
            buf = still_pending

            # 3. Push this step's observation at age 0.
            buf.append(_PendingEntry(payloads[env_id], age=0))

            # 4. Branch on this step's outcome. Terminated takes priority
            # over truncated if both somehow fire the same step.
            if terminated[env_id]:
                for entry in buf:
                    if 1 <= entry.age <= self.horizon:
                        emitted.append(
                            LabeledSample(env_id, entry.payload, 1, steps_to_termination=entry.age)
                        )
                buf = []
            elif truncated[env_id]:
                buf = []  # censored: discard every pending entry, emit nothing
            # else: buf stays as-is, pending for future steps.

            self._buffers[env_id] = buf

        return emitted

    def pending_counts(self) -> list[int]:
        """Number of still-pending (unresolved) entries per env."""
        return [len(b) for b in self._buffers]
