"""Phase-2 tool (outside CLAUDE.md's cartpole-only spec, see DEVIATIONS.md):
predictor_overlay.mp4 analogue for the WILD-PUSH pipeline -- the trained
Go1 policy walking, getting shoved by the wild push, with the
termination predictor's probability overlaid per frame (text, bar,
threshold marker, ground-truth label once known), stopping once
--min-failures real falls are captured.

The video shows ONLY the falls: one clip per fall, the last --clip-seconds
of that episode up to the fall, captioned "Fall k of N" with a countdown.
Episodes that end on the time limit without falling are recorded (the
rollout is continuous) but not shown. --full-rollout writes everything.

Adapted from scripts/go1_record_video.py (the natural-fall predictor
video for the original operating-point sweep), with three real
differences for this pipeline:

1. Supports BOTH perturbation mechanisms go1_wild_push_predictor.py now
   supports: `--mechanism impulse` (default, recommended -- a real
   force/torque wrench via mjlab's apply_body_impulse, respecting the
   robot's actual mass/inertia) and `--mechanism velocity` (the original
   instantaneous velocity-teleport scaling, kept for comparison only --
   direct visual inspection this session confirmed it produces
   physically-implausible results, "goes flying," at the scales needed
   to induce a reasonable failure rate). Whichever mechanism is chosen
   must be baked into the env cfg BEFORE ManagerBasedRlEnv construction,
   not applied to an already-built env -- mjlab's EventManager deep-copies
   its cfg at __init__ time (confirmed via mjlab/managers/event_manager.py's
   real source this session), so a post-construction mutation silently
   does nothing. Same fix already applied in
   scripts/go1_wild_push_predictor.py's `collect_under_wild_push`/
   `collect_under_impulse_push`.
2. Forces forward-only WALKING commands (cartpole_term.common.commands.
   force_forward_only_commands, natural sampled speed -- not pinned to
   the max) so the "before the push" baseline reads as clean forward
   walking on camera, same demo-only reasoning as
   scripts/robot_watch_episodes.py. The wild push itself still introduces
   real, uncontrolled motion once it fires -- that's the point; this only
   removes ambiguity about the BASELINE gait.
3. Loads the checkpoint/predictor/metrics by explicit file paths
   (--checkpoint-file/--predictor-weights/--predictor-metrics) instead of
   the operating-point/selected-checkpoints.json convention
   go1_record_video.py uses -- this pipeline's files
   (go1_wild_push_predictor.py / go1_multiseverity_predictor.py) are
   named by push-severity label, not by competence operating point.

Reuses scripts/record_video.py's already-tested, task-agnostic helpers
(_sanity_check_renderer, _composite_overlay_frame) via import, same as
go1_record_video.py does; that file is a completed, gated CLAUDE.md
deliverable and is not modified here. `_ensure_egl_configured` is copied
verbatim for the same reason every other phase-2 video script copies it.

Usage, from the repo root, on the GPU machine, after
scripts/go1_wild_push_predictor.py (or go1_multiseverity_predictor.py)
has produced a predictor + metrics file:

    python scripts/go1_wildpush_video.py \\
        --checkpoint-file model_9999.pt \\
        --predictor-weights outputs/go1_wildpush_predictor_seed0.pt \\
        --predictor-metrics outputs/go1_wildpush_metrics_seed0.json
    python scripts/go1_wildpush_video.py \\
        --checkpoint-file model_9999.pt \\
        --predictor-weights outputs/go1_wildpush_predictor_seed0.pt \\
        --predictor-metrics outputs/go1_wildpush_metrics_seed0.json \\
        --mechanism velocity --push-scale 4.0
"""

from __future__ import annotations

import argparse
import json
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
            print(f"[wildpush-video] wrote missing EGL vendor ICD file: {icd_path}")
        except OSError as e:
            print(f"[wildpush-video] could not write {icd_path} ({e}); continuing")


_ensure_egl_configured()

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cartpole_term.common.commands import force_forward_only_commands  # noqa: E402
from cartpole_term.go1.env_cfg import TASK_ID, go1_velocity_term_env_cfg  # noqa: E402
from cartpole_term.go1.runner import Go1FrozenEvalRunner  # noqa: E402
from cartpole_term.runner import build_runner_from_checkpoint  # noqa: E402
from go1_wild_push_predictor import (  # noqa: E402
    DEFAULT_IMPULSE_COOLDOWN_S,
    DEFAULT_IMPULSE_DURATION_S,
    DEFAULT_IMPULSE_FORCE_N,
    DEFAULT_IMPULSE_TORQUE_N,
    MASS_SCALED_FORCE_RANGE_N,
    scaled_velocity_range,
)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from record_video import _composite_overlay_frame, _sanity_check_renderer  # noqa: E402

STEP_SECONDS = 0.02  # Go1: decimation=4, timestep=0.005 -> 50 Hz, confirmed live this session
PREDICTION_HORIZON = 16  # matches go1_collect_frozen.py's --predictor.horizon default
MIN_FAILURE_EPISODES_FLOOR = 3
MAX_OVERLAY_STEPS = 20_000
FALL_HOLD_SECONDS = 0.75  # freeze on the last frame before each fall
TAKE_SEED_OFFSET = 400  # distinct from every other seed offset already in use (100/200/300)


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
    """A one-line caption on a dark strip along the bottom edge of the frame."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([0, h - 16, w, h], fill=(20, 20, 20))
    d.text((6, h - 14), text, fill=(255, 255, 255))
    return np.asarray(img)


def record_wildpush_predictor_overlay(args: argparse.Namespace, device: str) -> None:
    from cartpole_term.predictor.labeling import TerminationLabeler
    from cartpole_term.predictor.model import TerminationPredictor

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.predictor_metrics)
    predictor_path = Path(args.predictor_weights)
    if not predictor_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"{predictor_path} and/or {metrics_path} not found.")
    metrics = json.loads(metrics_path.read_text())
    threshold = metrics["mlp"]["threshold"]
    obs_action_dim = metrics["obs_action_dim"]
    hidden = tuple(metrics["predictor_hidden"])
    print(f"[wildpush-video] loaded threshold={threshold:.4f}, obs_action_dim={obs_action_dim}, hidden={hidden}")

    predictor = TerminationPredictor(history=1, hidden=hidden, obs_action_dim=obs_action_dim)
    predictor.load_state_dict(torch.load(predictor_path, weights_only=True))
    predictor.eval()

    # Same construction-order-safe pattern as
    # go1_wild_push_predictor.py's collect_under_wild_push/
    # collect_under_impulse_push: whichever mechanism's override is
    # baked into the cfg returned by this factory, BEFORE
    # ManagerBasedRlEnv (and therefore EventManager) is ever constructed.
    shipped_velocity_range: dict = {}

    def wild_video_env_cfg_factory(play: bool = False):
        cfg = go1_velocity_term_env_cfg(play=play)
        cfg.scene.num_envs = 1
        cfg.viewer.azimuth = args.camera_azimuth
        cfg.viewer.elevation = args.camera_elevation
        if args.camera_distance is not None:
            cfg.viewer.distance = args.camera_distance

        if args.mechanism == "impulse":
            from mjlab.envs.mdp.events import apply_body_impulse
            from mjlab.managers.event_manager import EventTermCfg
            from mjlab.managers.scene_entity_config import SceneEntityCfg

            cfg.events["wild_impulse"] = EventTermCfg(
                func=apply_body_impulse,
                mode="step",
                params={
                    "force_range": (-args.impulse_force, args.impulse_force),
                    "torque_range": (-args.impulse_torque, args.impulse_torque),
                    "duration_s": DEFAULT_IMPULSE_DURATION_S,
                    "cooldown_s": DEFAULT_IMPULSE_COOLDOWN_S,
                    "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
                },
            )
        else:
            push_cfg = cfg.events["push_robot"]
            base_range = dict(push_cfg.params["velocity_range"])
            shipped_velocity_range.clear()
            shipped_velocity_range.update(base_range)
            push_cfg.params["velocity_range"] = scaled_velocity_range(base_range, args.push_scale)

        if not args.natural_commands:
            force_forward_only_commands(cfg)  # natural sampled forward speed -- NOT pinned to max
        return cfg

    runner = build_runner_from_checkpoint(
        args.checkpoint_file,
        seed=args.seed,
        device=device,
        runner_cls=Go1FrozenEvalRunner,
        env_cfg_factory=wild_video_env_cfg_factory,
        task_id=TASK_ID,
        extra_scalar_metric_fn=None,
    )
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    rollout_seed = args.seed + args.take * TAKE_SEED_OFFSET
    video_env_cfg = wild_video_env_cfg_factory(play=False)
    if args.mechanism == "impulse":
        print(
            f"[wildpush-video] mechanism=impulse: force=+-{args.impulse_force}N, "
            f"torque=+-{args.impulse_torque}Nm, body=trunk "
            f"(mass-scaled reference range {MASS_SCALED_FORCE_RANGE_N} N)"
        )
    else:
        wild_velocity_range = dict(video_env_cfg.events["push_robot"].params["velocity_range"])
        print(f"[wildpush-video] mechanism=velocity, shipped push_robot.velocity_range: {shipped_velocity_range}")
        print(f"[wildpush-video] mechanism=velocity, wild push_robot.velocity_range (scale={args.push_scale}x): {wild_velocity_range}")

    raw_env = ManagerBasedRlEnv(cfg=video_env_cfg, device=device, render_mode="rgb_array")
    _sanity_check_renderer(raw_env, rollout_seed)

    obs, _ = raw_env.reset(seed=rollout_seed)
    labeler = TerminationLabeler(num_envs=1, horizon=PREDICTION_HORIZON)
    frames: list[np.ndarray] = []
    probs: list[float] = []
    labels: dict[int, int] = {}

    failure_count = 0
    step_idx = 0
    episode_start = 0
    fall_clips: list[tuple[int, int]] = []  # (first frame shown, last frame before the fall), one per fall
    clip_steps = round(args.clip_seconds / STEP_SECONDS)
    with torch.inference_mode():
        while failure_count < args.min_failures and step_idx < MAX_OVERLAY_STEPS:
            frame = raw_env.render()
            frames.append(frame)

            obs_np = obs["actor"].detach().cpu().numpy()
            # Sampled, not mean action -- the predictor was trained exclusively
            # on stochastic-behaviour-policy transitions, same rule as every
            # other frozen collection in this project.
            actions = runner.alg.actor(obs, stochastic_output=True)
            action_np = actions.detach().cpu().numpy()
            x = np.concatenate([obs_np, action_np], axis=1).astype(np.float32)
            prob = float(torch.sigmoid(predictor(torch.as_tensor(x))).item())
            probs.append(prob)

            obs, _reward, terminated, truncated, _info = raw_env.step(actions.to(raw_env.device))
            emitted = labeler.step([{"idx": step_idx}], terminated.tolist(), truncated.tolist())
            for sample in emitted:
                labels[sample.payload["idx"]] = sample.label

            if bool(terminated.item()):
                failure_count += 1
                print(f"[wildpush-video] fall #{failure_count} at step {step_idx} (p just before fall: {prob:.3f})")
                # Never reach back into the previous episode: a clip starts at this episode's own first frame at the earliest.
                fall_clips.append((max(episode_start, step_idx - clip_steps + 1), step_idx))
            if bool(terminated.item()) or bool(truncated.item()):
                episode_start = step_idx + 1
            step_idx += 1

    raw_env.close()

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during recording"

    if failure_count < args.min_failures:
        print(
            f"[wildpush-video] WARNING only captured {failure_count}/{args.min_failures} "
            f"failures within the {MAX_OVERLAY_STEPS}-step safety cap. Writing what was captured anyway."
        )
    print(
        f"[wildpush-video] rollout done -- {step_idx} steps, {failure_count} failure episode(s), "
        f"{len(labels)}/{step_idx} frames have a resolved ground-truth label."
    )

    import imageio

    tag = f"impulse{args.impulse_force}N" if args.mechanism == "impulse" else f"scale{args.push_scale}"
    final_path = outputs_dir / f"go1_wildpush_overlay_{tag}_seed{args.seed}_take{args.take}.mp4"
    fps = round(1.0 / STEP_SECONDS)
    writer = imageio.get_writer(str(final_path), fps=fps)
    written = 0
    if args.full_rollout:
        # The whole continuous recording, including episodes that never fall.
        for idx, frame in enumerate(frames):
            writer.append_data(_composite_overlay_frame(frame, probs[idx], threshold, labels.get(idx)))
            written += 1
    else:
        # Default: only the falls. One clip per fall -- the last --clip-seconds of
        # that episode up to the fall -- so no episode that survives is shown.
        hold = round(FALL_HOLD_SECONDS / STEP_SECONDS)
        for k, (first, last) in enumerate(fall_clips, start=1):
            for idx in range(first, last + 1):
                composite = _composite_overlay_frame(frames[idx], probs[idx], threshold, labels.get(idx))
                caption = f"Fall {k} of {len(fall_clips)}   falls in {(last + 1 - idx) * STEP_SECONDS:4.2f} s"
                writer.append_data(_caption(composite, caption))
                written += 1
            final = _caption(_composite_overlay_frame(frames[last], probs[last], threshold, labels.get(last)),
                             f"Fall {k} of {len(fall_clips)}   FELL")
            for _ in range(hold):
                writer.append_data(final)
                written += 1
    writer.close()

    size_mb = final_path.stat().st_size / 1e6
    what = "full rollout" if args.full_rollout else f"{len(fall_clips)} fall clips of up to {args.clip_seconds:.0f} s"
    print(f"[wildpush-video] wrote {final_path} ({size_mb:.1f} MB, {what}, {written} frames = {written * STEP_SECONDS:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, required=True)
    parser.add_argument("--predictor-weights", dest="predictor_weights", type=str, required=True)
    parser.add_argument("--predictor-metrics", dest="predictor_metrics", type=str, required=True)
    parser.add_argument(
        "--mechanism", choices=("impulse", "velocity"), default="impulse",
        help="'impulse' (default, recommended): real force/torque wrench, respects mass/inertia. "
        "'velocity': the original instantaneous teleport -- confirmed to look physically implausible "
        "('goes flying') at scales strong enough to induce a reasonable failure rate.",
    )
    parser.add_argument("--push-scale", dest="push_scale", type=float, default=None, help="velocity mechanism only.")
    parser.add_argument("--impulse-force", dest="impulse_force", type=float, default=DEFAULT_IMPULSE_FORCE_N, help="impulse mechanism only, Newtons.")
    parser.add_argument("--impulse-torque", dest="impulse_torque", type=float, default=DEFAULT_IMPULSE_TORQUE_N, help="impulse mechanism only, Nm.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-azimuth", dest="camera_azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", dest="camera_elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance", dest="camera_distance", type=float, default=None)
    parser.add_argument("--min-failures", dest="min_failures", type=int, default=5)
    parser.add_argument(
        "--natural-commands", dest="natural_commands", action="store_true",
        help="Use the real, unrestricted command distribution instead of forcing forward-only walking.",
    )
    parser.add_argument(
        "--clip-seconds", dest="clip_seconds", type=float, default=4.0,
        help="Seconds shown before each fall. The video contains only the falls, one clip each.",
    )
    parser.add_argument(
        "--full-rollout", dest="full_rollout", action="store_true",
        help="Write the whole continuous recording instead (includes episodes that never fall).",
    )
    parser.add_argument(
        "--take", type=int, default=0,
        help="Which rollout take (0, 1, ...) -- same checkpoint, different rollout seed and output "
        "filename, so multiple independent takes don't overwrite each other.",
    )
    args = parser.parse_args()
    if args.min_failures < MIN_FAILURE_EPISODES_FLOOR:
        raise ValueError(f"--min-failures={args.min_failures} is below the required minimum of {MIN_FAILURE_EPISODES_FLOOR}.")
    if args.mechanism == "velocity" and args.push_scale is None:
        raise ValueError("--mechanism velocity requires --push-scale.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = "cuda"

    record_wildpush_predictor_overlay(args, device)


if __name__ == "__main__":
    main()
