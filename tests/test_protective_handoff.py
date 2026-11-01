"""Tests for the Gate 6/7 hand-off bookkeeping in protective/compare.py
(FirstFallTracker.begin, HandoffTrials). Pure torch."""

from __future__ import annotations

import numpy as np
import torch

from _helpers import load_module

compare = load_module("protective/compare.py", "_protective_compare_handoff_under_test")

CRIT = torch.tensor([True, False])


def _xy(xs):
    root = torch.tensor([[x, 0.0] for x in xs])
    return root, torch.stack([root, root + torch.tensor([0.3, 0.0])], dim=1)


def _trials(n, handoff=True, walk_steps=100, window_steps=5, hold_steps=2):
    root, bodies = _xy([0.0] * n)
    return compare.HandoffTrials(
        handoff=handoff, walk_steps=walk_steps, window_steps=window_steps, hold_steps=hold_steps,
        lin_threshold=0.2, ang_threshold=0.5, root_xy=root, body_xy=bodies,
    )


def _pre(tr, t, probs, xs):
    root, bodies = _xy(xs)
    pose = torch.cat([torch.cat([root, torch.zeros(len(xs), 1)], -1), torch.zeros(len(xs), 4)], -1)
    return tr.pre_step(t, torch.tensor(probs), 0.5, root, bodies, pose)


def _post(tr, t, xs, *, fell=None, speed=None, force=10.0, torque=1.0, time_out=None):
    n = len(xs)
    root, bodies = _xy(xs)
    pose = torch.cat([torch.cat([root, torch.zeros(n, 1)], -1), torch.zeros(n, 4)], -1)
    speed = torch.tensor(speed if speed is not None else [1.0] * n)
    tr.post_step(
        t, fell=torch.tensor(fell or [False] * n), time_out=torch.tensor(time_out or [False] * n),
        force_mag=torch.full((n, 2), float(force)), critical_mask=CRIT, torque=torch.full((n, 3), float(torque)),
        root_xy=root, body_xy=bodies, reward=torch.full((n,), -1.0),
        lin_speed=speed, ang_speed=torch.zeros(n), root_pose=pose,
    )


def test_tracker_begin_starts_windows_independently():
    root, bodies = _xy([0.0, 0.0])
    tr = compare.FirstFallTracker(root, bodies, start_active=False)
    assert not tr.active.any()
    args = dict(critical_mask=CRIT, torque=torch.ones(2, 3), reward=torch.zeros(2),
                terminated=torch.zeros(2, dtype=torch.bool), truncated=torch.zeros(2, dtype=torch.bool))
    # nothing recorded before begin()
    tr.update(force_mag=torch.full((2, 2), 999.0), root_xy=root, body_xy=bodies, **args)
    r1, b1 = _xy([1.0, 2.0])
    tr.begin(torch.tensor([True, False]), r1, b1)
    r2, b2 = _xy([1.5, 2.0])
    tr.update(force_mag=torch.full((2, 2), 7.0), root_xy=r2, body_xy=b2, **args)
    pf = tr.per_fall()
    assert pf["peak_force_any_n"].tolist() == [7.0, 0.0]
    assert pf["steps"].tolist() == [1.0, 0.0]
    assert np.allclose(pf["root_travel_end_m"], [0.5, 0.0])  # origin = position at begin, not reset
    assert np.allclose(pf["envelope_growth_m"], [0.5, 0.0])


def test_flag_latches_and_window_ends_on_settle():
    tr = _trials(2)
    latched = _pre(tr, 0, [0.9, 0.1], [0.0, 0.0])
    assert latched.tolist() == [True, False]
    _post(tr, 0, [0.1, 0.0], speed=[0.0, 1.0])
    latched = _pre(tr, 1, [0.0, 0.1], [0.1, 0.0])  # low prob later: latch must hold
    assert latched.tolist() == [True, False]
    _post(tr, 1, [0.1, 0.0], speed=[0.0, 1.0])  # 2 still steps -> settled
    r = tr.results()
    assert r["event"].tolist() == [compare.EVENT_FLAG, compare.EVENT_NONE]
    assert r["flag_step"].tolist() == [0, -1]
    assert r["settled"].tolist() == [1.0, 0.0]
    assert r["steps"].tolist() == [2.0, 0.0]
    assert tr.phase.tolist() == [compare.PHASE_DONE, compare.PHASE_WALKING]


def test_nominal_arm_never_latches_but_records_the_same_window():
    tr = _trials(1, handoff=False)
    assert _pre(tr, 3, [0.9], [0.0]).tolist() == [False]
    _post(tr, 3, [0.0], fell=[True])
    r = tr.results()
    assert r["event"].tolist() == [compare.EVENT_FLAG]
    assert r["fall_step"].tolist() == [3] and r["flag_step"].tolist() == [3]


def test_missed_fall_opens_window_next_step_and_window_caps():
    tr = _trials(1, window_steps=3)
    _pre(tr, 0, [0.0], [0.0])
    _post(tr, 0, [0.0], fell=[True])
    assert tr.event.tolist() == [compare.EVENT_MISSED_FALL]
    assert tr.window_start.tolist() == [1] and not tr.latched.any()
    for t in (1, 2, 3):
        _pre(tr, t, [0.99], [0.0])  # already in a window: no flag, no latch
        _post(tr, t, [0.0], fell=[True])
    r = tr.results()
    assert r["flag_step"].tolist() == [-1] and not r["latched"].any()
    assert r["steps"].tolist() == [3.0] and r["settled"].tolist() == [0.0]
    assert tr.all_done


def test_no_event_by_walk_limit_and_no_new_flags_after_it():
    tr = _trials(2, walk_steps=2)
    for t in (0, 1):
        _pre(tr, t, [0.0, 0.0], [0.0, 0.0])
        _post(tr, t, [0.0, 0.0])
    assert tr.all_done
    assert _pre(tr, 2, [0.9, 0.9], [0.0, 0.0]).tolist() == [False, False]
    assert tr.results()["event"].tolist() == [compare.EVENT_NONE] * 2


def test_time_out_closes_open_window():
    tr = _trials(1, window_steps=50)
    _pre(tr, 0, [0.9], [0.0])
    _post(tr, 0, [0.0], time_out=[True])
    assert tr.all_done and tr.results()["settled"].tolist() == [0.0]


def test_inference_mode_safe():
    with torch.inference_mode():
        tr = _trials(2)
    _pre(tr, 0, [0.9, 0.1], [0.0, 0.0])
    _post(tr, 0, [0.0, 0.0])


def test_down_latches_on_fall_or_low_root_and_stays():
    root, bodies = _xy([0.0, 0.0, 0.0])
    tr = compare.HandoffTrials(
        handoff=True, walk_steps=100, window_steps=50, hold_steps=25, lin_threshold=0.2, ang_threshold=0.5,
        root_xy=root, body_xy=bodies, down_height=torch.tensor([0.15, 0.15, 0.15]),
    )
    pose = torch.zeros(3, 7)

    def post(t, fell, z):
        tr.post_step(
            t, fell=torch.tensor(fell), time_out=torch.zeros(3, dtype=torch.bool),
            force_mag=torch.zeros(3, 2), critical_mask=CRIT, torque=torch.zeros(3, 3), root_xy=root,
            body_xy=bodies, reward=torch.zeros(3), lin_speed=torch.ones(3), ang_speed=torch.zeros(3),
            root_pose=pose, root_z=torch.tensor(z),
        )

    post(0, [False, False, False], [0.30, 0.30, 0.30])
    assert tr.down.tolist() == [False, False, False]
    post(1, [True, False, False], [0.30, 0.10, 0.30])   # env 0 falls over, env 1 sinks to the ground
    assert tr.down.tolist() == [True, True, False]
    post(2, [False, False, False], [0.30, 0.30, 0.30])  # back up / fall flag cleared: still down
    assert tr.down.tolist() == [True, True, False]
    assert tr.results()["down"].tolist() == [True, True, False]


def test_down_without_height_uses_fall_only():
    tr = _trials(1)
    _post(tr, 0, [0.0], fell=[True])
    assert tr.down.tolist() == [True]
