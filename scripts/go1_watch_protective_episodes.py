"""Phase-2 tool (Gate 5, see DEVIATIONS.md): plain rollout video of a
trained protective-fall checkpoint -- N variable-length episodes back to
back, each starting from a different real injected fall state, each
running until the robot settles (or the safety-cap budget expires).
Modeled directly on scripts/go1_watch_episodes.py's already-tested
structure (same EGL setup, same renderer sanity check, same episode-
boundary detection), generalized to the protective task's env cfg and
checkpoint-loading path instead of the velocity-walking one.

Purpose stated plainly: after three rounds of reward-weight fixes chasing
numbers in a training log, the actual go/no-go question ("does the fall
look controlled, or does it look crazy") can only be answered by watching
it, not by reading Metrics/* further. This is that check.

**Episode length, revised (see env_cfg.py / terminations.py /
DEVIATIONS.md):** episodes used to be a fixed 10 steps (0.2s), which this
same video-watching check surfaced as ending mid-fall before the robot had
settled. Episodes now end via the `settled` termination (once the robot
has genuinely stopped moving, held for a sustained duration) or, failing
that, the enlarged PROTECTIVE_EPISODE_LENGTH_S safety cap -- so each
printed "episode ended after N steps" line below can legitimately vary a
lot run to run; that variability is expected, not a bug. `terminated`
(from `settled`) vs `truncated` (from `time_out`, now a safety cap only)
are reported per episode below specifically so a run that keeps hitting
the safety cap (never settling) is visible at a glance.

Episode-boundary detection watches `terminated | truncated` go True, same
mechanism go1_watch_episodes.py already uses -- mjlab auto-resets
internally on that same step (this script's env is built with the DEFAULT
auto_reset=True, unlike go1_collect_flagged_states.py's DeferredResetEnv
use elsewhere in this project; there is no reason to hold a protective
episode open past its own termination).

Usage, from the repo root, on the GPU machine, after
scripts/go1_train_protective.py has produced a checkpoint:

    python scripts/go1_watch_protective_episodes.py \
        --checkpoint-file logs/rsl_rl/go1_protective_fall/<run>/model_499.pt
    python scripts/go1_watch_protective_episodes.py --checkpoint-file <path> --episodes 8
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_egl_configured() -> None:
    """Verbatim copy of scripts/record_video.py's own helper -- see that
    file's docstring for why this must run before mujoco's first import."""
    # Must precede `import torch`: mjlab's @torch.jit.script math utils crash
    # on this Colab image (missing libnvrtc-builtins.so.13.0). See DEVIATIONS.md.
    os.environ.setdefault("PYTORCH_JIT", "0")
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
            print(f"[protective-watch] wrote missing EGL vendor ICD file: {icd_path}")
        except OSError as e:
            print(f"[protective-watch] could not write {icd_path} ({e}); continuing")


_ensure_egl_configured()

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term.protective  # noqa: F401,E402 -- side effect: registers the task
from cartpole_term.protective.env_cfg import (  # noqa: E402
    PROTECTIVE_EPISODE_LENGTH_S_BY_ROBOT,
    TASK_IDS,
    protective_fall_env_cfg,
)
from cartpole_term.protective.runner import ProtectiveFrozenEvalRunner  # noqa: E402
from cartpole_term.runner import build_runner_from_checkpoint  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from record_video import _sanity_check_renderer  # noqa: E402

STEP_SECONDS = 0.02  # Go1: decimation=4, timestep=0.005 -> 50 Hz
# Episodes are now variable length (end via the `settled` termination, see
# env_cfg.py/terminations.py) -- this is a SAFETY bound only, derived from
# the env's own safety-cap budget so it never fights the env's own
# time_out. +1 step of slack for the render-then-step frame ordering below.
# Longest episode budget of any robot (the G1's 5 s), so one cap fits both.
MAX_STEPS_PER_EPISODE = round(max(PROTECTIVE_EPISODE_LENGTH_S_BY_ROBOT.values()) / STEP_SECONDS) + 1
MAX_TOTAL_STEPS_PER_EPISODE_BUDGET = MAX_STEPS_PER_EPISODE
# Log-only space metrics (src/cartpole_term/protective/space.py), reported
# by mjlab in the step's extras["log"] on the step an episode resets.
SPACE_METRICS = ("fall_envelope_radius", "root_travel_at_end", "body_spread_at_end", "body_spread_mean")


def _build_watch_env_cfg(args: argparse.Namespace):
    cfg = protective_fall_env_cfg(args.robot, play=False, state_bank_file=args.state_bank_file)
    cfg.scene.num_envs = 1
    cfg.viewer.azimuth = args.camera_azimuth
    cfg.viewer.elevation = args.camera_elevation
    if args.camera_distance is not None:
        cfg.viewer.distance = args.camera_distance
    print(
        f"[protective-watch] viewer config: origin_type={cfg.viewer.origin_type}, "
        f"azimuth={cfg.viewer.azimuth}, elevation={cfg.viewer.elevation}, "
        f"distance={cfg.viewer.distance}, body_name={cfg.viewer.body_name}"
    )
    return cfg


def record_protective_episodes(args: argparse.Namespace, device: str) -> None:
    print(f"[protective-watch] loading checkpoint {args.checkpoint_file}")

    def env_cfg_factory(play: bool = False):
        return protective_fall_env_cfg(args.robot, play=play, state_bank_file=args.state_bank_file)

    runner = build_runner_from_checkpoint(
        args.checkpoint_file,
        seed=args.seed,
        eval_num_envs=1,
        device=device,
        runner_cls=ProtectiveFrozenEvalRunner,
        env_cfg_factory=env_cfg_factory,
        task_id=TASK_IDS[args.robot],
        extra_scalar_metric_fn=None,
    )
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
    episode_rewards: list[float] = []
    space_values: dict[str, list[float]] = {}
    current_episode_reward = 0.0
    max_total_steps = args.episodes * MAX_TOTAL_STEPS_PER_EPISODE_BUDGET
    with torch.inference_mode():
        while episode_count < args.episodes and step_idx < max_total_steps:
            frames.append(raw_env.render())

            actions = runner.alg.actor(obs, stochastic_output=args.stochastic)
            obs, reward, terminated, truncated, _info = raw_env.step(actions.to(raw_env.device))

            current_episode_reward += float(reward.item())
            step_idx += 1
            episode_step += 1
            done = bool(terminated.item()) or bool(truncated.item())
            if done or episode_step >= MAX_STEPS_PER_EPISODE:
                episode_count += 1
                if bool(terminated.item()):
                    cause = "settled"
                elif bool(truncated.item()):
                    cause = "time_out (never settled -- check this run)"
                else:
                    cause = "script safety cutoff (env's own termination never fired -- bug?)"
                episode_rewards.append(current_episode_reward)
                space = {
                    k.removeprefix("Episode_Metrics/"): float(v)
                    for k, v in _info.get("log", {}).items()
                    if k.removeprefix("Episode_Metrics/") in SPACE_METRICS
                }
                for k, v in space.items():
                    space_values.setdefault(k, []).append(v)
                space_str = ", ".join(f"{k}={v:.2f}m" for k, v in space.items())
                print(
                    f"[protective-watch] episode {episode_count}/{args.episodes} ended after "
                    f"{episode_step} steps ({cause}), cumulative reward {current_episode_reward:+.3f}"
                    + (f" | {space_str}" if space_str else "")
                )
                episode_step = 0
                current_episode_reward = 0.0

    raw_env.close()

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during recording"

    if episode_count < args.episodes:
        print(
            f"[protective-watch] WARNING only captured {episode_count}/{args.episodes} "
            f"episodes within the {max_total_steps}-step safety cap. Writing what was captured anyway."
        )
    if episode_rewards:
        print(
            f"[protective-watch] cumulative reward across {len(episode_rewards)} episodes: "
            f"mean={np.mean(episode_rewards):+.3f}, min={min(episode_rewards):+.3f}, "
            f"max={max(episode_rewards):+.3f}"
        )

    for k, vals in space_values.items():
        print(f"[protective-watch] {k}: mean={np.mean(vals):.2f}m, max={max(vals):.2f}m")

    import imageio

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    final_path = outputs_dir / f"{args.robot}_protective_watch_seed{args.seed}_ep{args.episodes}.mp4"
    fps = round(1.0 / STEP_SECONDS)
    writer = imageio.get_writer(str(final_path), fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    size_mb = final_path.stat().st_size / 1e6
    print(f"[protective-watch] wrote {final_path} ({size_mb:.1f} MB, {step_idx} frames = {step_idx * STEP_SECONDS:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=("go1", "g1"), default="go1")
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, required=True)
    parser.add_argument(
        "--state-bank-file", dest="state_bank_file", type=str, default="outputs/go1_fall_state_bank_seed0.npz",
        help="Must match (or be compatible with) whatever bank the checkpoint was trained against.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Rollout seed and actor-load seed (not a training seed).")
    parser.add_argument("--episodes", type=int, default=8, help="More than the walking demo's default of 5 -- episodes are shorter on average (they end once settled), even though the safety-cap budget per episode is now longer.")
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use sampled (exploration-noise) actions instead of the deterministic mean action. "
        "Default is deterministic, matching CLAUDE.md S10.1's policy_final.mp4 convention for a clean demo.",
    )
    parser.add_argument("--camera-azimuth", dest="camera_azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", dest="camera_elevation", type=float, default=-20.0)
    parser.add_argument(
        "--camera-distance", dest="camera_distance", type=float, default=None,
        help="Override mjlab's default viewer distance if the robot reads too small/large on camera.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = "cuda"

    record_protective_episodes(args, device)


if __name__ == "__main__":
    main()
