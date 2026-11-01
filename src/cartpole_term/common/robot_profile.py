"""Per-robot configuration for the multi-robot phase-2 pipeline (Go1 + G1
humanoid -- see DEVIATIONS.md's "multi-robot scaffolding" entry, and the
plan this was built from, for the full audit that motivated this file).

A RobotProfile holds everything that varies between robots so scripts and
modules can be written ONCE, parameterized by --robot, instead of
hand-duplicated per robot -- the pattern every prior go1_*.py file used,
which new (robot_*.py) scripts replace going forward. The Go1-only
scripts already gated/working (scripts/go1_*.py,
cartpole_term/go1/*) are left exactly as they are; only NEW work uses this.

expected_obs_dim/expected_action_dim/step_hz/max_episode_steps start as
None for a robot until scripts/inspect_robot_task.py has confirmed the
real, live values on the GPU machine -- never pre-guessed (CLAUDE.md
S0.2's rule: never invent an API/value that hasn't been confirmed against
the installed version). Fill them in via a follow-up edit to this file
once that script's live output is available, not by mutating this frozen
dataclass at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotProfile:
    name: str  # "go1" | "g1" -- also every new output file's tag
    registered_task_id: str  # our own registered task id, e.g. "Mjlab-Go1-Velocity-Term"
    task_id_candidates: tuple[str, ...] = ()  # exact mjlab-shipped ids to try first
    task_id_must: tuple[str, ...] = ()  # regex fragments, ANDed, word-bounded (see discovery.py)
    task_id_exclude: tuple[str, ...] = ()  # regex fragments, word-bounded, any-match excludes
    reward_overrides: dict[str, float] = field(default_factory=dict)
    expected_obs_dim: int | None = None
    expected_action_dim: int | None = None
    obs_layout: dict[str, slice] = field(default_factory=dict)
    vulnerability_regions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    step_hz: float | None = None
    max_episode_steps: int | None = None


GO1 = RobotProfile(
    name="go1",
    registered_task_id="Mjlab-Go1-Velocity-Term",
    task_id_candidates=("Mjlab-Velocity-Flat-Unitree-Go1",),
    task_id_must=("go1", "velocity", "flat"),
    task_id_exclude=("getup",),
    # IsaacLab-sourced, see cartpole_term.go1.env_cfg.GO1_AIR_TIME_WEIGHT's
    # comment and DEVIATIONS.md for the full citation (mjlab Discussion #220).
    reward_overrides={"air_time": 0.25},
    # Live-confirmed via scripts/inspect_robot_task.py --robot go1
    # (outputs/go1_task_inspection.json), 2026-09-13: actor obs (4, 48),
    # action_dim 12, terminations {time_out: True, fell_over: False},
    # push_robot params match exactly (see G1's below -- byte-identical).
    expected_obs_dim=48,
    expected_action_dim=12,
    step_hz=50.0,  # decimation=4, timestep=0.005 -> 0.02s/step
    max_episode_steps=1000,  # episode_length_s=20.0 / 0.02s
    # Live-confirmed actor obs term order (inspect_robot_task.py's
    # ObservationManager dump): base_lin_vel(3) + base_ang_vel(3) +
    # projected_gravity(3) + joint_pos(12) + joint_vel(12) + actions(12) +
    # command(3) = 48. Real slices, not re-derived per script (fixes the
    # audit's flagged gap: go1_watch_episodes.py hardcoded these as
    # standalone module constants instead of a shared, robot-keyed source).
    obs_layout={
        "base_lin_vel": slice(0, 3),
        "base_ang_vel": slice(3, 6),
        "projected_gravity": slice(6, 9),
        "joint_pos": slice(9, 21),
        "joint_vel": slice(21, 33),
        "actions": slice(33, 45),
        "command": slice(45, 48),
    },
    # Real body names (13, live-confirmed): trunk + 4 legs x {hip,thigh,calf}
    # -- no separate foot/ankle body exists. "calf" is the link nearest
    # ground contact in normal operation, so it's the low-vulnerability tier
    # here, not a literal foot.
    vulnerability_regions={
        "high": ("trunk",),
        "medium": ("FR_hip", "FL_hip", "RR_hip", "RL_hip", "FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh"),
        "low": ("FR_calf", "FL_calf", "RR_calf", "RL_calf"),
    },
)

G1 = RobotProfile(
    name="g1",
    registered_task_id="Mjlab-G1-Velocity-Term",
    task_id_candidates=("Mjlab-Velocity-Flat-Unitree-G1",),
    task_id_must=("g1", "velocity", "flat"),
    task_id_exclude=("getup",),
    # G1 ships the same air_time term at weight 0.0 (confirmed via GitHub
    # source review, then re-confirmed live by
    # scripts/inspect_robot_task.py --robot g1) -- same override, same
    # rationale as Go1's. Re-confirmed again at apply time by
    # g1/env_cfg.py's loud RuntimeError guard.
    reward_overrides={"air_time": 0.25},
    # Live-confirmed via scripts/inspect_robot_task.py --robot g1
    # (outputs/g1_task_inspection.json), 2026-09-13: actor obs (4, 99) --
    # NOT the ~286 estimated from static source review (that estimate
    # wrongly assumed the flat variant carries a 187-ray height_scan term;
    # it doesn't -- that's rough-terrain only). action_dim 29. Terminations
    # {time_out: True, fell_over: False} -- SAME structure as Go1, no
    # collision-named termination exists (self_collisions is reward-only,
    # weight -1.0). push_robot params byte-identical to Go1's. mjlab's
    # shipped rl_cfg.max_iterations is 30,000 for G1 (vs. Go1's 10,000) --
    # a real difference, not carried over from Go1's default.
    expected_obs_dim=99,
    expected_action_dim=29,
    step_hz=50.0,
    max_episode_steps=1000,
    # Live-confirmed actor obs term order, same method as Go1's above:
    # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) +
    # joint_pos(29) + joint_vel(29) + actions(29) + command(3) = 99.
    obs_layout={
        "base_lin_vel": slice(0, 3),
        "base_ang_vel": slice(3, 6),
        "projected_gravity": slice(6, 9),
        "joint_pos": slice(9, 38),
        "joint_vel": slice(38, 67),
        "actions": slice(67, 96),
        "command": slice(96, 99),
    },
    # Real body names (30, live-confirmed): pelvis, legs (hip/knee/ankle
    # x2), waist+torso, arms (shoulder/elbow/wrist x2) -- NO head or hand
    # link exists on this model, so the SafeFall/arXiv:2512.01336
    # head/torso/pelvis vulnerability template is adapted, not copied
    # verbatim: pelvis/torso/waist stand in for the "core" high-vulnerability
    # tier since there's no head; ankles (nearest ground contact) are the
    # low tier; everything else (hips/knees/arms) is medium.
    vulnerability_regions={
        "high": ("pelvis", "torso_link", "waist_yaw_link", "waist_roll_link"),
        "medium": (
            "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link",
            "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link",
            "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
            "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
            "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
        ),
        "low": ("left_ankle_pitch_link", "left_ankle_roll_link", "right_ankle_pitch_link", "right_ankle_roll_link"),
    },
)

PROFILES: dict[str, RobotProfile] = {"go1": GO1, "g1": G1}


def get_profile(name: str) -> RobotProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown robot {name!r} -- must be one of {sorted(PROFILES)}.")
    return PROFILES[name]
