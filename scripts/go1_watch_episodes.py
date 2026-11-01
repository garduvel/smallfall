"""Phase-2 tool (outside CLAUDE.md's cartpole-only spec, see DEVIATIONS.md):
plain rollout video of a Go1 checkpoint -- N episodes back to back, no
predictor overlay, no dependency on go1_select_checkpoints.py or
go1_eval_predictor.py having run at all.

go1_record_video.py (the predictor-overlay analogue of the gated
predictor_overlay.mp4) requires a trained predictor's .pt/.json to exist for
the given operating point. For a plain "does this checkpoint actually walk"
check right after a training-only run -- no dataset collection, no predictor
training -- those files don't exist yet and shouldn't need to. This script
takes a checkpoint path directly and just renders it.

Reuses record_video.py's already-tested _sanity_check_renderer (checks the
EGL renderer produces a real frame before committing to a full rollout) via
import, same as go1_record_video.py does; does NOT need
_composite_overlay_frame since there is no predictor output to draw.
_ensure_egl_configured is copied verbatim for the same reason every other
phase-2 video script copies it (must run before mujoco/mjlab's first import
anywhere in the process).

Episode boundaries are detected the same way go1_record_video.py detects
falls: keep stepping and watch `terminated`/`truncated` go True on a given
step -- mjlab's ManagerBasedRlEnv auto-resets internally on that same step,
so no manual env.reset() between episodes is needed (confirmed by
go1_record_video.py's identical pattern, which never resets between its
counted failure episodes).

Usage, from the repo root, on the GPU machine, right after go1_train.py --
no other script needs to have run:

    python scripts/go1_watch_episodes.py --checkpoint-file logs/rsl_rl/go1_velocity_term/<run>/model_999.pt
    python scripts/go1_watch_episodes.py --checkpoint-file <path> --episodes 3 --stochastic
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_egl_configured() -> None:
    """Verbatim copy of scripts/record_video.py's own helper -- see that
    file's docstring for why this must run before mujoco's first import."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    icd_path = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(icd_path):
        try:
            os.makedirs(os.path.dirname(icd_path), exist_ok=True)
            with open(icd_path, "w") as f:
                f.write(
                    '{"file_format_version": "1.0.0", '
                    '"ICD": {"library_path": "libEGL_nvidia.so.0"}}'
                )
            print(f"[go1-watch] wrote missing EGL vendor ICD file: {icd_path}")
        except OSError as e:
            print(f"[go1-watch] could not write {icd_path} ({e}); continuing")


_ensure_egl_configured()

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term.go1  # noqa: F401,E402
from cartpole_term.go1.env_cfg import go1_velocity_term_env_cfg  # noqa: E402
from cartpole_term.go1.runner import build_go1_runner_from_checkpoint  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from record_video import _sanity_check_renderer  # noqa: E402

STEP_SECONDS = 0.02  # Go1 velocity task: decimation=4, timestep=0.005 -> 50 Hz
MAX_STEPS_PER_EPISODE = 1000  # matches the task's own episode cap (safety bound only)
# Actor obs layout confirmed via scripts/inspect_go1_task.py (outputs/go1_task_inspection.json):
# base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_pos(12)
# + joint_vel(12) + actions(12) + command(3) = 48. command is [vx, vy, wz],
# base_lin_vel is the robot's OWN body-frame linear velocity (confirmed by
# reading mjlab's mdp.track_linear_velocity source on GitHub -- it compares
# against asset.data.root_link_lin_vel_b, a body-frame quantity).
BASE_LIN_VEL_SLICE = slice(0, 3)
COMMAND_SLICE = slice(45, 48)
# mjlab's shipped reset_base event randomizes spawn yaw uniformly over the
# full circle every episode (pose_range["yaw"] = (-3.14, 3.14) in
# src/mjlab/tasks/velocity/velocity_env_cfg.py). Correct and intentional for
# TRAINING (it's what makes the policy heading-agnostic), but combined with
# track_linear_velocity being defined in the robot's own body frame, a
# policy that's genuinely walking straight "forward" relative to itself can
# point in an arbitrary WORLD-frame direction each episode -- which, viewed
# through a world-fixed camera, can look like it's wandering or walking
# sideways even when body-frame tracking is perfect. Pinning yaw here (demo
# env only -- never for training/collection/eval, which must match the real
# distribution) removes that confound so the video is actually a fair test
# of gait quality instead of a per-episode coin flip on camera framing.
WATCH_SPAWN_YAW = 0.0
# Total-step safety cap so a labeling/termination-detection bug can't hang
# the process forever instead of just producing a wrong-looking video.
MAX_TOTAL_STEPS_PER_EPISODE_BUDGET = 2 * MAX_STEPS_PER_EPISODE


def _build_watch_env_cfg(args: argparse.Namespace):
    cfg = go1_velocity_term_env_cfg(play=False)  # play=True not implemented, see env_cfg.py
    cfg.scene.num_envs = 1
    cfg.viewer.azimuth = args.camera_azimuth
    cfg.viewer.elevation = args.camera_elevation
    if args.camera_distance is not None:
        cfg.viewer.distance = args.camera_distance
    if args.lin_vel_x is not None or args.lin_vel_y is not None or args.ang_vel_z is not None:
        from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

        # Same range-pinning trick mjlab's own shipped play-mode config uses
        # (unitree_go1_flat_env_cfg's `twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)`
        # etc.) -- sampling from a degenerate [v, v] range always returns v,
        # with no need to guess at an internal CommandManager override API.
        # Any axis left as None keeps the shipped training-distribution range.
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg), (
            f"expected UniformVelocityCommandCfg for 'twist', got {type(twist_cmd)}"
        )
        if args.lin_vel_x is not None:
            twist_cmd.ranges.lin_vel_x = (args.lin_vel_x, args.lin_vel_x)
        if args.lin_vel_y is not None:
            twist_cmd.ranges.lin_vel_y = (args.lin_vel_y, args.lin_vel_y)
        if args.ang_vel_z is not None:
            twist_cmd.ranges.ang_vel_z = (args.ang_vel_z, args.ang_vel_z)

        # Go1's shipped `command_vel` curriculum term (mjlab/tasks/velocity/
        # mdp/curriculums.py's commands_vel) OVERWRITES cfg.ranges.{lin_vel_x,
        # lin_vel_y,ang_vel_z} directly, every step, from a staged schedule
        # keyed off env.common_step_counter -- which starts at 0 for any
        # freshly-built env regardless of which checkpoint is loaded. Left
        # in place, it silently clobbers the pin above back down to the
        # curriculum's easiest/smallest stage almost immediately (confirmed:
        # this is exactly what produced small, still-randomized-looking
        # velocities despite the "pinned" print line above). Must be removed
        # whenever an override is requested, or the pin is a no-op in practice.
        removed = cfg.curriculum.pop("command_vel", None)
        print(
            f"[go1-watch] pinned command ranges: lin_vel_x={twist_cmd.ranges.lin_vel_x}, "
            f"lin_vel_y={twist_cmd.ranges.lin_vel_y}, ang_vel_z={twist_cmd.ranges.ang_vel_z} "
            "(likely outside the training distribution if far from the shipped default -- "
            "that's the point of overriding it). "
            f"Removed the 'command_vel' curriculum term ({'was present' if removed else 'was absent'}) "
            "so it can't overwrite this pin mid-rollout."
        )
    if not args.random_spawn_yaw:
        cfg.events["reset_base"].params["pose_range"]["yaw"] = (WATCH_SPAWN_YAW, WATCH_SPAWN_YAW)
        print(
            f"[go1-watch] pinned spawn yaw to {WATCH_SPAWN_YAW} for this demo "
            "(training/collection/eval still randomize it -- see module docstring). "
            "Pass --random-spawn-yaw to see the real per-episode distribution instead."
        )
    print(
        f"[go1-watch] viewer config: origin_type={cfg.viewer.origin_type}, "
        f"azimuth={cfg.viewer.azimuth}, elevation={cfg.viewer.elevation}, "
        f"distance={cfg.viewer.distance}, body_name={cfg.viewer.body_name}"
    )
    return cfg


def _print_velocity_diagnostics(obs, prefix: str) -> None:
    cmd = obs["actor"][0, COMMAND_SLICE].detach().cpu().numpy()
    actual = obs["actor"][0, BASE_LIN_VEL_SLICE].detach().cpu().numpy()
    print(
        f"[go1-watch] {prefix} command (vx, vy, wz) = "
        f"({cmd[0]:+.3f}, {cmd[1]:+.3f}, {cmd[2]:+.3f}), "
        f"actual body-frame lin_vel (x, y, z) = "
        f"({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f})"
    )


def record_go1_episodes(args: argparse.Namespace, device: str) -> None:
    print(f"[go1-watch] loading checkpoint {args.checkpoint_file}")
    runner = build_go1_runner_from_checkpoint(args.checkpoint_file, seed=args.seed, device=device)
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    env_cfg = _build_watch_env_cfg(args)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    _sanity_check_renderer(raw_env, args.seed)

    obs, _ = raw_env.reset(seed=args.seed)
    frames: list[np.ndarray] = []

    episode_count = 0
    episode_step = 0
    step_idx = 0
    max_total_steps = args.episodes * MAX_TOTAL_STEPS_PER_EPISODE_BUDGET
    _print_velocity_diagnostics(obs, f"episode {episode_count + 1}/{args.episodes} start,")
    with torch.inference_mode():
        while episode_count < args.episodes and step_idx < max_total_steps:
            frames.append(raw_env.render())

            actions = runner.alg.actor(obs, stochastic_output=args.stochastic)
            obs, _reward, terminated, truncated, _info = raw_env.step(actions.to(raw_env.device))

            step_idx += 1
            episode_step += 1
            done = bool(terminated.item()) or bool(truncated.item())
            if done or episode_step >= MAX_STEPS_PER_EPISODE:
                episode_count += 1
                cause = "fell_over" if bool(terminated.item()) else "time_out"
                print(f"[go1-watch] episode {episode_count}/{args.episodes} ended after {episode_step} steps ({cause})")
                episode_step = 0
                # obs here is already the auto-reset observation for the NEXT
                # episode (mjlab's ManagerBasedRlEnv resets internally on the
                # same step a done fires) -- print its command now, before
                # the loop condition can end things.
                if episode_count < args.episodes:
                    _print_velocity_diagnostics(obs, f"episode {episode_count + 1}/{args.episodes} start,")

    raw_env.close()

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during recording"

    if episode_count < args.episodes:
        print(
            f"[go1-watch] WARNING only captured {episode_count}/{args.episodes} "
            f"episodes within the {max_total_steps}-step safety cap. Writing what was captured anyway."
        )

    import imageio

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    final_path = outputs_dir / f"go1_watch_seed{args.seed}_ep{args.episodes}.mp4"
    fps = round(1.0 / STEP_SECONDS)
    writer = imageio.get_writer(str(final_path), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    size_mb = final_path.stat().st_size / 1e6
    print(f"[go1-watch] wrote {final_path} ({size_mb:.1f} MB, {step_idx} frames = {step_idx * STEP_SECONDS:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed and actor-load seed (not a training seed).")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use sampled (exploration-noise) actions instead of the deterministic mean action. "
        "Default is deterministic, matching CLAUDE.md S10.1's policy_final.mp4 convention for a clean demo.",
    )
    parser.add_argument("--camera-azimuth", dest="camera_azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", dest="camera_elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance", dest="camera_distance", type=float, default=None,
                         help="Override mjlab's default viewer distance for this task if the robot reads too small/large on camera.")
    parser.add_argument(
        "--random-spawn-yaw", dest="random_spawn_yaw", action="store_true",
        help="Use the real training-distribution random spawn yaw instead of pinning it to "
        f"{WATCH_SPAWN_YAW} -- see module docstring for why the default pins it for this demo.",
    )
    parser.add_argument(
        "--lin-vel-x", dest="lin_vel_x", type=float, default=None,
        help="Pin the commanded forward velocity (vx) to this exact value instead of the "
        "CommandManager's normal random per-episode sampling. Default: unpinned (real distribution).",
    )
    parser.add_argument(
        "--lin-vel-y", dest="lin_vel_y", type=float, default=None,
        help="Pin the commanded lateral velocity (vy) to this exact value. Default: unpinned.",
    )
    parser.add_argument(
        "--ang-vel-z", dest="ang_vel_z", type=float, default=None,
        help="Pin the commanded yaw rate (wz, 'z velocity') to this exact value -- e.g. 3.0 for a "
        "hard, likely out-of-training-distribution spin test. Default: unpinned.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = "cuda"

    record_go1_episodes(args, device)


if __name__ == "__main__":
    main()
