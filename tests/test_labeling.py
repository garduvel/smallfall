"""Mandatory tests for src/cartpole_term/predictor/labeling.py (CLAUDE.md
S7.4). Do not run any dataset collection until every test here passes --
this is the highest-risk file in the project.

No mjlab or GPU required (CLAUDE.md S0.2.5): the labeler is pure Python
against opaque payloads.

Note on the worked-example numbers: CLAUDE.md S7.4's terminated-episode
example text reads "16 positives (steps 24-39)" / "23 negatives (steps
0-22)" / "Step 23 is the last positive" -- these don't add up internally
(24-39 overlaps the stated "terminal step 39 emits nothing", and 0-22 plus
24-39 skips step 23 with no accounting for it). The formal definition
(y=1 iff terminated at t+k, k in {1..16}) and the section's OWN truncated-
episode example ("steps 23-39 are discarded", 17 steps) both independently
pin down the self-consistent numbers used below: negatives are steps 0-22
(23 of them), positives are steps 23-38 (16 of them), step 39 (terminal)
emits nothing. See the chat log / DEVIATIONS.md for the full derivation.
"""

from __future__ import annotations

from _helpers import load_module

labeling = load_module("predictor/labeling.py", "_labeling_under_test")

TerminationLabeler = labeling.TerminationLabeler
HORIZON = 16


def _run_solo_episode(length: int, end_kind: str) -> list:
    """Run one env through a `length`-step episode, ending (at the last
    step) via `end_kind` in {"terminated", "truncated"}. Returns the list of
    emitted LabeledSample.
    """
    assert end_kind in ("terminated", "truncated")
    labeler = TerminationLabeler(num_envs=1, horizon=HORIZON)
    emitted = []
    for step in range(length):
        is_last = step == length - 1
        terminated = [is_last and end_kind == "terminated"]
        truncated = [is_last and end_kind == "truncated"]
        emitted += labeler.step([step], terminated, truncated)
    return emitted


def test_length_40_terminated_emits_16_positive_23_negative():
    emitted = _run_solo_episode(40, "terminated")

    positives = sorted(s.payload for s in emitted if s.label == 1)
    negatives = sorted(s.payload for s in emitted if s.label == 0)

    assert len(positives) == 16, positives
    assert len(negatives) == 23, negatives
    assert negatives == list(range(0, 23))
    assert positives == list(range(23, 39))
    assert 39 not in positives and 39 not in negatives  # terminal step: nothing emitted
    assert len(emitted) == 39  # every non-terminal step accounted for exactly once


def test_length_40_terminated_steps_to_termination_is_correct():
    # CLAUDE.md S7.3: steps_to_termination must be recoverable per positive
    # (k, the exact number of steps from the observation to termination),
    # with a sentinel (None) for negatives.
    emitted = _run_solo_episode(40, "terminated")
    by_payload = {s.payload: s for s in emitted}

    for step in range(23, 39):  # positives, terminal step 39
        sample = by_payload[step]
        assert sample.label == 1
        assert sample.steps_to_termination == 39 - step, (step, sample.steps_to_termination)

    for step in range(0, 23):  # negatives
        sample = by_payload[step]
        assert sample.label == 0
        assert sample.steps_to_termination is None


def test_length_40_truncated_emits_23_negative_0_positive():
    emitted = _run_solo_episode(40, "truncated")

    positives = [s.payload for s in emitted if s.label == 1]
    negatives = sorted(s.payload for s in emitted if s.label == 0)

    assert len(positives) == 0
    assert len(negatives) == 23, negatives
    assert negatives == list(range(0, 23))
    assert all(s.steps_to_termination is None for s in emitted)
    # steps 23..39 (17 steps) are censored: never emitted at all.
    emitted_steps = {s.payload for s in emitted}
    for step in range(23, 40):
        assert step not in emitted_steps, step


def test_length_10_terminated_emits_9_positive_0_negative():
    emitted = _run_solo_episode(10, "terminated")

    positives = sorted(s.payload for s in emitted if s.label == 1)
    negatives = [s.payload for s in emitted if s.label == 0]

    assert len(positives) == 9, positives
    assert len(negatives) == 0
    assert positives == list(range(0, 9))
    assert 9 not in positives  # terminal step emits nothing


def test_interleaved_envs_do_not_cross_contaminate():
    # env 0: one continuous 40-step terminated episode.
    # env 1: a 10-step terminated episode, immediately followed by a second,
    # still-open (never-terminating within this test) episode, running
    # concurrently with env 0's single long episode in the SAME labeler.
    labeler = TerminationLabeler(num_envs=2, horizon=HORIZON)
    emitted = []
    for step in range(40):
        terminated0 = step == 39
        if step < 10:
            payload1 = ("ep1", step)
            terminated1 = step == 9
        else:
            payload1 = ("ep2", step - 10)
            terminated1 = False
        emitted += labeler.step(
            payloads=[("env0", step), payload1],
            terminated=[terminated0, terminated1],
            truncated=[False, False],
        )

    env0_emitted = [(s.payload[1], s.label) for s in emitted if s.env_id == 0]
    env1_ep1_emitted = [
        (s.payload[1], s.label) for s in emitted if s.env_id == 1 and s.payload[0] == "ep1"
    ]
    env1_ep2_emitted = [
        (s.payload[1], s.label) for s in emitted if s.env_id == 1 and s.payload[0] == "ep2"
    ]

    solo0 = _run_solo_episode(40, "terminated")
    solo1 = _run_solo_episode(10, "terminated")

    assert sorted(env0_emitted) == sorted((s.payload, s.label) for s in solo0)
    assert sorted(env1_ep1_emitted) == sorted((s.payload, s.label) for s in solo1)
    # env1's second episode never terminates/truncates within this test's 30
    # remaining steps (local steps 0..29), but entries that individually age
    # past the horizon are still correctly negatives regardless -- local
    # steps 0..12 each reach age 17 before the window ends (12 + 17 = 29),
    # so exactly those 13 are emitted as negatives; nothing else is (no
    # positives, since ep2 never terminates here; steps 13..29 stay pending).
    assert env1_ep2_emitted == [(i, 0) for i in range(13)]
    # And no positives leaked in from env0's termination at step 39 -- ep2's
    # local step 29 (global step 39) is unaffected by env0 terminating.
    assert all(label == 0 for _, label in env1_ep2_emitted)


def test_total_emitted_count_invariant_to_num_envs():
    solo = _run_solo_episode(40, "terminated")
    solo_count = len(solo)

    for num_envs in (2, 5, 8):
        labeler = TerminationLabeler(num_envs=num_envs, horizon=HORIZON)
        emitted = []
        for step in range(40):
            payloads = [("target", step)] + [("filler", i, step) for i in range(num_envs - 1)]
            terminated = [step == 39] + [False] * (num_envs - 1)
            truncated = [False] * num_envs
            emitted += labeler.step(payloads, terminated, truncated)

        target_emitted = [s for s in emitted if s.payload[0] == "target"]
        assert len(target_emitted) == solo_count, (num_envs, len(target_emitted))
