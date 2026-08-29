"""policy_final.mp4 and predictor_overlay.mp4 (CLAUDE.md S10, S11 gate 11).

policy_final.mp4: the fully trained (competent-point) policy balancing,
>=20s, one environment, single camera.

predictor_overlay.mp4: the target-point policy with the predictor's
probability overlaid per frame (text, bar, threshold marker, ground-truth
label once known), running until >=3 failure episodes are captured
(CLAUDE.md S10 item 2).

Built directly against the real installed mjlab API -- verified via
scripts/inspect_video_and_termmgr.py (CLAUDE.md S0.2: never guess an mjlab
API), not assumed from the spec text alone. Deviations from the literal
S10 text, all logged in DEVIATIONS.md:

1. `env.render()` already returns a plain frame via mjlab's own internal
   OffscreenRenderer (render_mode="rgb_array") -- no manual
   `mujoco.Renderer(model, ...)` setup is needed for the overlay video
   either, contrary to S10 item 2's literal text. Both videos here call
   `env.render()` directly.
2. `play=True` disables randomization AND sets episode length to
   (near-)infinite (S6.2.2's own reason frozen eval must not use it).
   `policy_final.mp4` bounds its recording with an explicit VideoRecorder
   `video_length` instead of relying on an episode boundary that may not
   exist; `predictor_overlay.mp4` stops on failure-episode count instead
   (it needs real episode ends to work at all, so play=True's near-infinite
   single episode does not block it -- `terminated` still fires normally
   even though `time_out` effectively cannot).
3. The raw (unwrapped) ManagerBasedRlEnv.step() used by both videos
   returns a 5-tuple `(obs, reward, terminated, truncated, info)`, not the
   4-tuple RslRlVecEnvWrapper.step() returns (CLAUDE.md S2.9, used
   everywhere else in this project).

Run on the GPU machine, from the repo root, after
scripts/select_checkpoints.py (both videos) and scripts/eval_predictor.py
--operating-point target (predictor_overlay.mp4 only, needs the saved
outputs/predictor_target_seed{N}.pt) have produced their outputs:

    python scripts/record_video.py                       # both videos, seed 0
    python scripts/record_video.py --which policy_final
    python scripts/record_video.py --which predictor_overlay --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_egl_configured() -> None:
    """Colab-specific EGL setup (reference/old_notebook_setup.md S2-S3, in
    CLAUDE.md S4.1's permitted scope: "MUJOCO_GL / EGL configuration...
    known platform quirks"). Must run before mujoco is imported anywhere in
    this process -- MUJOCO_GL only takes effect on mujoco's first import,
    and setting it after is silently ignored. Two failure modes this
    guards against, both silent (no exception, no traceback):
    1. MUJOCO_GL/PYOPENGL_PLATFORM unset or set too late -> falls back to
       a software/no-op renderer.
    2. Colab sometimes lacks the NVIDIA EGL vendor ICD file entirely ->
       a non-functional GL context, observed here as the process dying
       with zero output partway through OffscreenRenderer.initialize()
       (CLAUDE.md S10: "EGL failures are silent").
    """
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
            print(f"[video] wrote missing EGL vendor ICD file: {icd_path}")
        except OSError as e:
            # Not fatal here -- some environments (non-Colab, non-NVIDIA)
            # don't need this file at all. If rendering still fails, the
            # renderer sanity check below will say so explicitly.
            print(f"[video] could not write {icd_path} ({e}); continuing")


_ensure_egl_configured()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import cartpole_term  # noqa: F401,E402
from cartpole_term.cli import build_parser  # noqa: E402
from cartpole_term.env_cfg import PREDICTION_HORIZON, cartpole_balance_term_env_cfg  # noqa: E402
from cartpole_term.runner import build_runner_from_checkpoint  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.utils.wrappers import VideoRecorder  # noqa: E402

STEP_SECONDS = 0.05  # CLAUDE.md S2.2: 20 Hz env step

MIN_POLICY_VIDEO_SECONDS = 20.0
# A few seconds of margin over the S10 minimum so a slightly-early episode
# end doesn't leave the video right at the boundary.
POLICY_VIDEO_FRAMES = int((MIN_POLICY_VIDEO_SECONDS + 5.0) / STEP_SECONDS)

MIN_FAILURE_EPISODES_FLOOR = 3  # CLAUDE.md S10 item 2: "at least three failure episodes"
# Engineering safety cap, not a spec requirement -- mirrors
# collect_frozen.py's MAX_STEPS_SAFETY_CAP so a pathological policy can't
# loop forever chasing failures that never happen.
MAX_OVERLAY_STEPS = 20_000


def _build_video_env_cfg(args: argparse.Namespace):
    cfg = cartpole_balance_term_env_cfg(play=True)
    cfg.scene.num_envs = 1
    # Shipped cartpole config overrides the viewer's azimuth to 0.0, which
    # looks straight down the rail (slider joint axis="1 0 0" -- CLAUDE.md
    # S2.1): motion along the rail then only changes apparent depth, not
    # left-right position, so back-and-forth movement barely reads on
    # camera. mjlab's own ViewerConfig class default is azimuth=90.0 (a
    # 90-degree rotation around the vertical axis from the shipped
    # override) -- confirmed via inspecting ViewerConfig directly, not
    # guessed. --camera-azimuth defaults to that side-on view; still
    # overridable if 90 isn't quite right once seen.
    cfg.viewer.azimuth = args.camera_azimuth
    cfg.viewer.elevation = args.camera_elevation
    print(
        f"[video] viewer config: origin_type={cfg.viewer.origin_type}, "
        f"azimuth={cfg.viewer.azimuth}, elevation={cfg.viewer.elevation}, "
        f"distance={cfg.viewer.distance}, body_name={cfg.viewer.body_name}"
    )
    return cfg


def resolve_checkpoint(seed: int, operating_point: str, checkpoint_file: str | None) -> str:
    if checkpoint_file is not None:
        return checkpoint_file
    selected_path = Path("outputs/selected_checkpoints.json")
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{selected_path} not found and --checkpoint-file not given -- run "
            "scripts/select_checkpoints.py first, or pass --checkpoint-file explicitly."
        )
    selected = json.loads(selected_path.read_text())
    seed_key = str(seed)
    if seed_key not in selected:
        raise KeyError(f"seed {seed} not found in {selected_path}.")
    if operating_point not in selected[seed_key]:
        raise KeyError(f"operating point '{operating_point}' not found for seed {seed}.")
    return selected[seed_key][operating_point]["checkpoint_path"]


def _sanity_check_renderer(raw_env: ManagerBasedRlEnv, seed: int) -> None:
    raw_env.reset(seed=seed)
    frame = raw_env.render()
    if frame is None:
        raise AssertionError(
            "raw_env.render() returned None -- EGL/rendering setup failed. "
            "Confirm MUJOCO_GL=egl is set before this script runs (CLAUDE.md S10)."
        )
    print(f"[video] renderer sanity check OK: frame shape {frame.shape}, dtype {frame.dtype}")


def record_policy_final(args: argparse.Namespace, device: str) -> None:
    # CLAUDE.md S10 item 1 wants "the fully trained policy" -- not
    # user-selectable per S5.5 (video is not in --operating-point's scoped
    # script list), so this is a hardcoded internal choice, not a CLI flag.
    checkpoint_path = resolve_checkpoint(args.seed, "competent", args.checkpoint_file)
    print(f"[video] policy_final: loading checkpoint {checkpoint_path}")

    runner = build_runner_from_checkpoint(checkpoint_path, seed=args.seed, device=device)
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    video_env_cfg = _build_video_env_cfg(args)
    raw_env = ManagerBasedRlEnv(cfg=video_env_cfg, device=device, render_mode="rgb_array")
    _sanity_check_renderer(raw_env, args.seed)

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    video = VideoRecorder(
        raw_env,
        video_folder=str(outputs_dir),
        episode_trigger=lambda ep: ep == 0,
        video_length=POLICY_VIDEO_FRAMES,
        name_prefix="policy_final_raw",
    )

    obs, _ = video.reset(seed=args.seed)
    with torch.inference_mode():
        for step in range(POLICY_VIDEO_FRAMES + 1):
            actions = runner.alg.actor(obs, stochastic_output=False)  # deterministic action for a clean demo
            obs, _reward, _terminated, _truncated, _info = video.step(actions.to(raw_env.device))
            if video.video_count >= 1:
                print(f"[video] policy_final: recording finished at step {step + 1}")
                break
    video.close()

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during policy_final recording"

    raw_clips = sorted(outputs_dir.glob("policy_final_raw-episode-*.mp4"))
    if not raw_clips:
        raise RuntimeError(f"VideoRecorder produced no clip in {outputs_dir} -- check its logging above.")
    final_path = outputs_dir / "policy_final.mp4"
    raw_clips[-1].replace(final_path)
    for leftover in raw_clips[:-1]:
        leftover.unlink()

    size_mb = final_path.stat().st_size / 1e6
    print(f"[video] wrote {final_path} ({size_mb:.1f} MB, {POLICY_VIDEO_FRAMES} frame budget = "
          f"{POLICY_VIDEO_FRAMES * STEP_SECONDS:.1f}s max)")


def _composite_overlay_frame(
    frame: np.ndarray, prob: float, threshold: float, label: int | None
) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, _h = img.size

    bar_x0, bar_y0 = 10, 10
    bar_w, bar_h = w - 20, 14
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline=(255, 255, 255))
    p_clamped = max(0.0, min(1.0, prob))
    fill_w = int(bar_w * p_clamped)
    bar_color = (220, 50, 50) if prob >= threshold else (60, 160, 60)
    if fill_w > 0:
        draw.rectangle([bar_x0, bar_y0, bar_x0 + fill_w, bar_y0 + bar_h], fill=bar_color)
    thresh_x = bar_x0 + int(bar_w * max(0.0, min(1.0, threshold)))
    draw.line([(thresh_x, bar_y0 - 2), (thresh_x, bar_y0 + bar_h + 2)], fill=(255, 255, 0), width=2)

    draw.text((bar_x0, bar_y0 + bar_h + 4), f"p={prob:.2f}  thresh={threshold:.2f}", fill=(255, 255, 255))

    if label is None:
        label_text, label_color = "label: pending", (200, 200, 200)
    elif label == 1:
        label_text, label_color = "label: FAIL <=16 steps", (255, 80, 80)
    else:
        label_text, label_color = "label: safe", (120, 220, 120)
    draw.text((bar_x0, bar_y0 + bar_h + 18), label_text, fill=label_color)

    return np.asarray(img)


def record_predictor_overlay(args: argparse.Namespace, device: str) -> None:
    from cartpole_term.predictor.labeling import TerminationLabeler
    from cartpole_term.predictor.model import TerminationPredictor

    outputs_dir = Path("outputs")
    # CLAUDE.md S10 item 2 hardcodes `target` for this video, unlike item 1.
    checkpoint_path = resolve_checkpoint(args.seed, "target", args.checkpoint_file)
    print(f"[video] predictor_overlay: loading policy checkpoint {checkpoint_path}")

    predictor_path = outputs_dir / f"predictor_target_seed{args.seed}.pt"
    metrics_path = outputs_dir / f"metrics_target_seed{args.seed}.json"
    if not predictor_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"{predictor_path} and/or {metrics_path} not found -- run "
            f"scripts/eval_predictor.py --operating-point target --seeds {args.seed} first "
            "(it saves predictor weights alongside the metrics JSON, see DEVIATIONS.md)."
        )
    threshold = json.loads(metrics_path.read_text())["mlp"]["threshold"]
    print(f"[video] predictor_overlay: loaded threshold={threshold:.4f} from {metrics_path}")

    predictor = TerminationPredictor(history=1)
    predictor.load_state_dict(torch.load(predictor_path, weights_only=True))
    predictor.eval()

    runner = build_runner_from_checkpoint(checkpoint_path, seed=args.seed, device=device)
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    video_env_cfg = _build_video_env_cfg(args)
    raw_env = ManagerBasedRlEnv(cfg=video_env_cfg, device=device, render_mode="rgb_array")
    _sanity_check_renderer(raw_env, args.seed)

    obs, _ = raw_env.reset(seed=args.seed)
    labeler = TerminationLabeler(num_envs=1, horizon=PREDICTION_HORIZON)
    frames: list[np.ndarray] = []
    probs: list[float] = []
    labels: dict[int, int] = {}

    failure_count = 0
    step_idx = 0
    with torch.inference_mode():
        while failure_count < args.min_failures and step_idx < MAX_OVERLAY_STEPS:
            frame = raw_env.render()
            frames.append(frame)

            obs_np = obs["actor"].detach().cpu().numpy()
            # S7.3: sampled, not the mean action -- the predictor was
            # trained exclusively on stochastic-behaviour-policy
            # transitions, and the mean action rarely fails at all (see
            # DEVIATIONS.md), so this is necessary, not just consistent.
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
            step_idx += 1

    raw_env.close()

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during predictor_overlay recording"

    if failure_count < args.min_failures:
        print(
            f"[video] predictor_overlay: WARNING only captured "
            f"{failure_count}/{args.min_failures} failures within the "
            f"{MAX_OVERLAY_STEPS}-step safety cap (CLAUDE.md S10 item 2 wants >= 3). "
            "Writing what was captured anyway."
        )
    print(
        f"[video] predictor_overlay: rollout done -- {step_idx} steps, "
        f"{failure_count} failure episode(s), {len(labels)}/{step_idx} frames have a "
        "resolved ground-truth label (the rest are still-pending horizon windows, "
        "correctly left unlabelled rather than guessed -- CLAUDE.md S7.4)."
    )

    import imageio

    final_path = outputs_dir / "predictor_overlay.mp4"
    fps = round(1.0 / STEP_SECONDS)
    writer = imageio.get_writer(str(final_path), fps=fps)
    for idx, frame in enumerate(frames):
        composite = _composite_overlay_frame(frame, probs[idx], threshold, labels.get(idx))
        writer.append_data(composite)
    writer.close()

    size_mb = final_path.stat().st_size / 1e6
    print(f"[video] wrote {final_path} ({size_mb:.1f} MB, {step_idx} frames = "
          f"{step_idx * STEP_SECONDS:.1f}s)")


def main() -> None:
    parser = build_parser(
        ("video",), description="Record CLAUDE.md S10 video deliverables (S11 gate 11)."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--which", choices=("policy_final", "predictor_overlay", "both"), default="both"
    )
    parser.add_argument(
        "--camera-azimuth",
        dest="camera_azimuth",
        type=float,
        default=90.0,
        help="Viewer azimuth in degrees (mjlab ViewerConfig). Default 90.0 -- a "
        "side-on view of the rail; the shipped cartpole config's own default "
        "(0.0) looks straight down the rail instead. See DEVIATIONS.md.",
    )
    parser.add_argument(
        "--camera-elevation",
        dest="camera_elevation",
        type=float,
        default=-25.0,
        help="Viewer elevation in degrees (mjlab ViewerConfig). Default -25.0, "
        "nudged from the shipped cartpole config's -15.0 for a slightly higher "
        "camera so the pole reads more centered in frame -- not yet confirmed "
        "against a real render, tune if it goes the wrong way.",
    )
    parser.add_argument(
        "--min-failures",
        dest="min_failures",
        type=int,
        default=10,
        help="predictor_overlay.mp4 stops once this many failure episodes are "
        "captured (default 10, for a longer/more demonstrative clip than the "
        "CLAUDE.md S10 item 2 floor). Must be >= 3 -- that floor is a spec "
        "requirement, not a default.",
    )
    args = parser.parse_args()
    if args.min_failures < MIN_FAILURE_EPISODES_FLOOR:
        raise ValueError(
            f"--min-failures={args.min_failures} is below CLAUDE.md S10 item 2's "
            f"required minimum of {MIN_FAILURE_EPISODES_FLOOR} failure episodes."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    if args.which in ("policy_final", "both"):
        record_policy_final(args, device)
    if args.which in ("predictor_overlay", "both"):
        record_predictor_overlay(args, device)


if __name__ == "__main__":
    main()
