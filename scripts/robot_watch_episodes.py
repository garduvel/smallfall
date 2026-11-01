"""Phase-2 tool (outside CLAUDE.md's cartpole-only spec, see DEVIATIONS.md):
plain rollout video of a robot checkpoint -- N episodes back to back, no
predictor overlay. Generalizes scripts/go1_watch_episodes.py via
cartpole_term.common (RobotProfile/backend) instead of Go1-only literals
-- fixes the audit's flagged gap (go1_watch_episodes.py hardcoded Go1's
exact 48-dim obs-slice layout as standalone module constants).

Serves Gate 1's own pass criterion #3 (a quick visual check that the
final trained checkpoint actually walks) and sets up the same
forward-only-for-demos mechanism scripts/robot_record_demo_video.py
(a later gate) will also use.

Forward-only by default (cartpole_term.common.commands.
force_forward_only_commands) -- pass --natural-commands to see the real,
unrestricted command distribution instead (useful for confirming the
earlier "smooth sideways walking" finding is really just the shipped
command distribution, not a policy defect). This override, like
go1_watch_episodes.py's spawn-yaw pin, is applied ONLY in this demo env --
never for training/collection/eval.

Reuses record_video.py's already-tested _sanity_check_renderer via
import, same as every other phase-2 video script; copies
_ensure_egl_configured verbatim for the same reason (must run before
mujoco/mjlab's first import anywhere in the process).

Usage, from the repo root, on the GPU machine:

    python scripts/robot_watch_episodes.py --robot go1 --checkpoint-file logs/rsl_rl/go1_velocity_term/<run>/model_9999.pt
    python scripts/robot_watch_episodes.py --robot go1 --checkpoint-file <path> --episodes 3 --natural-commands
    python scripts/robot_watch_episodes.py --robot go1 --checkpoint-file <path> --commanded-speed 3.0
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
            print(f"[robot-watch] wrote missing EGL vendor ICD file: {icd_path}")
        except OSError as e:
            print(f"[robot-watch] could not write {icd_path} ({e}); continuing")


_ensure_egl_configured()

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cartpole_term.common.backend import get_robot_backend  # noqa: E402
from cartpole_term.common.commands import force_forward_only_commands  # noqa: E402
from cartpole_term.common.robot_profile import get_profile  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from record_video import _sanity_check_renderer  # noqa: E402

MAX_TOTAL_STEPS_PER_EPISODE_BUDGET_MULTIPLIER = 2  # safety cap, see module docstring's precedent


def _build_watch_env_cfg(args: argparse.Namespace, backend, profile) -> object:
    cfg = backend.env_cfg_factory(play=False)  # play=True not implemented for either robot yet
    cfg.scene.num_envs = 1
    cfg.viewer.azimuth = args.camera_azimuth
    cfg.viewer.elevation = args.camera_elevation
    if args.camera_distance is not None:
        cfg.viewer.distance = args.camera_distance
    if not args.natural_commands:
        force_forward_only_commands(cfg, fixed_speed=args.commanded_speed)
        speed_note = f"pinned at {args.commanded_speed} m/s" if args.commanded_speed is not None else "natural forward-only range"
        print(
            f"[robot-watch] forced forward-only commands for this demo ({speed_note}) "
            "(training/collection/eval still use the real command distribution -- see module docstring). "
            "Pass --natural-commands to see the real per-episode distribution instead."
        )
    print(
        f"[robot-watch] viewer config: origin_type={cfg.viewer.origin_type}, "
        f"azimuth={cfg.viewer.azimuth}, elevation={cfg.viewer.elevation}, "
        f"distance={cfg.viewer.distance}"
    )
    return cfg


def _print_velocity_diagnostics(obs, profile, prefix: str) -> None:
    base_lin_vel_slice = profile.obs_layout["base_lin_vel"]
    command_slice = profile.obs_layout["command"]
    cmd = obs["actor"][0, command_slice].detach().cpu().numpy()
    actual = obs["actor"][0, base_lin_vel_slice].detach().cpu().numpy()
    print(
        f"[robot-watch] {prefix} command (vx, vy, wz) = "
        f"({cmd[0]:+.3f}, {cmd[1]:+.3f}, {cmd[2]:+.3f}), "
        f"actual body-frame lin_vel (x, y, z) = "
        f"({actual[0]:+.3f}, {actual[1]:+.3f}, {actual[2]:+.3f})"
    )


def record_robot_episodes(args: argparse.Namespace, device: str) -> None:
    profile = get_profile(args.robot)
    backend = get_robot_backend(args.robot)

    print(f"[robot-watch] robot={args.robot} loading checkpoint {args.checkpoint_file}")
    from cartpole_term.runner import build_runner_from_checkpoint

    runner = build_runner_from_checkpoint(
        args.checkpoint_file,
        seed=args.seed,
        device=device,
        runner_cls=backend.runner_cls,
        env_cfg_factory=backend.env_cfg_factory,
        task_id=backend.task_id,
        extra_scalar_metric_fn=None,
    )
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    env_cfg = _build_watch_env_cfg(args, backend, profile)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    _sanity_check_renderer(raw_env, args.seed)

    max_steps_per_episode = profile.max_episode_steps
    step_seconds = 1.0 / profile.step_hz
    max_total_steps_budget = args.episodes * max_steps_per_episode * MAX_TOTAL_STEPS_PER_EPISODE_BUDGET_MULTIPLIER

    obs, _ = raw_env.reset(seed=args.seed)
    frames: list[np.ndarray] = []

    episode_count = 0
    episode_step = 0
    step_idx = 0
    _print_velocity_diagnostics(obs, profile, f"episode {episode_count + 1}/{args.episodes} start,")
    with torch.inference_mode():
        while episode_count < args.episodes and step_idx < max_total_steps_budget:
            frames.append(raw_env.render())

            actions = runner.alg.actor(obs, stochastic_output=args.stochastic)
            obs, _reward, terminated, truncated, _info = raw_env.step(actions.to(raw_env.device))

            step_idx += 1
            episode_step += 1
            done = bool(terminated.item()) or bool(truncated.item())
            if done or episode_step >= max_steps_per_episode:
                episode_count += 1
                cause = "fell_over" if bool(terminated.item()) else "time_out"
                print(f"[robot-watch] episode {episode_count}/{args.episodes} ended after {episode_step} steps ({cause})")
                episode_step = 0
                if episode_count < args.episodes:
                    _print_velocity_diagnostics(obs, profile, f"episode {episode_count + 1}/{args.episodes} start,")

    raw_env.close()

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during recording"

    if episode_count < args.episodes:
        print(
            f"[robot-watch] WARNING only captured {episode_count}/{args.episodes} "
            f"episodes within the {max_total_steps_budget}-step safety cap. Writing what was captured anyway."
        )

    import imageio

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    final_path = outputs_dir / f"{profile.name}_watch_seed{args.seed}_ep{args.episodes}.mp4"
    fps = round(1.0 / step_seconds)
    writer = imageio.get_writer(str(final_path), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    size_mb = final_path.stat().st_size / 1e6
    print(f"[robot-watch] wrote {final_path} ({size_mb:.1f} MB, {step_idx} frames = {step_idx * step_seconds:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", required=True, choices=("go1", "g1"))
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
    parser.add_argument("--camera-distance", dest="camera_distance", type=float, default=None)
    parser.add_argument(
        "--natural-commands", dest="natural_commands", action="store_true",
        help="Use the real, unrestricted command distribution instead of forcing forward-only -- "
        "see module docstring for why the default forces forward-only for this demo specifically.",
    )
    parser.add_argument(
        "--commanded-speed", dest="commanded_speed", type=float, default=None,
        help="Pin the forward command to exactly this speed (m/s) every episode, instead of the "
        "natural forward-only sampled range. Ignored if --natural-commands is set. "
        "cartpole_term.common.commands.MAX_FORWARD_COMMAND_LIN_VEL_X (3.0) is the absolute highest "
        "this task's shipped command_vel curriculum ever asks for.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = "cuda"

    record_robot_episodes(args, device)


if __name__ == "__main__":
    main()
