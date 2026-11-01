"""Phase-2 tool (outside CLAUDE.md's spec, see DEVIATIONS.md): one video
showing the STRATIFIED predictor (scripts/train_stratified_predictor.py's
output, trained across all 4 competence stacks at once) running its
overlay against all 4 of its own source checkpoints in turn -- struggling,
target, competent, fluent -- back to back in one mp4, so it's visible at a
glance whether the same frozen predictor reads sensibly across the whole
competence range it was trained on, not just at one operating point.

This is a straightforward extension of scripts/record_video.py's
record_predictor_overlay() (CLAUDE.md S10 item 2) to 4 checkpoints and one
already-trained predictor instead of 1 checkpoint and its own
operating-point-matched predictor. scripts/record_video.py is a completed,
gated CLAUDE.md S11 gate-11 deliverable and is not modified here -- this is
a new file that imports its already-tested helpers
(_build_video_env_cfg, _sanity_check_renderer, _composite_overlay_frame)
rather than duplicating or editing them. `_ensure_egl_configured` is the
one exception: copied verbatim rather than imported, because it MUST run
before mujoco/mjlab is imported anywhere in the process, and relying on
"importing record_video.py happens to run it first" is a fragile ordering
assumption to build on.

Each checkpoint segment runs until EITHER --min-failures-per-checkpoint
failure episodes are captured OR --max-seconds-per-checkpoint elapses,
whichever comes first -- not "keep going until N failures" alone. The
`fluent` checkpoint's whole point is a base rate near zero (CLAUDE.md
S9's cross-operating-point story, extended to a 4th point by this
project's stratified predictor); waiting for it to fail organically could
run a very long time or never happen at all, so it is bounded by the
per-checkpoint time cap and simply shows a long stable segment with the
predictor's probability staying low throughout -- itself the honest,
informative result for that regime, not a bug to work around.

Checkpoint paths and the decision threshold are read directly from
outputs/metrics_stratified_seed{seed}.json (written by
train_stratified_predictor.py) rather than re-derived, so the video is
guaranteed to show the exact same 4 checkpoints the predictor was actually
trained on.

Usage, from the repo root, on the GPU machine, after
scripts/train_stratified_predictor.py --seed 0 has produced
outputs/predictor_stratified_seed0.pt and outputs/metrics_stratified_seed0.json:

    python scripts/record_predictor_stratified_video.py --seed 0
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
            print(f"[video] wrote missing EGL vendor ICD file: {icd_path}")
        except OSError as e:
            print(f"[video] could not write {icd_path} ({e}); continuing")


_ensure_egl_configured()

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import cartpole_term  # noqa: F401,E402
from cartpole_term.env_cfg import PREDICTION_HORIZON  # noqa: E402
from cartpole_term.runner import build_runner_from_checkpoint  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from record_video import (  # noqa: E402
    _build_video_env_cfg,
    _composite_overlay_frame,
    _sanity_check_renderer,
)

STEP_SECONDS = 0.05  # CLAUDE.md S2.2: 20 Hz env step
SEGMENT_ORDER = ("struggling", "target", "competent", "fluent")
SEGMENT_LABELS = {
    "struggling": "STRUGGLING",
    "target": "TARGET",
    "competent": "COMPETENT",
    "fluent": "FLUENT (~1000-step ceiling)",
}


def _composite_with_title(frame: np.ndarray, prob: float, threshold: float, label: int | None, title: str) -> np.ndarray:
    composed = _composite_overlay_frame(frame, prob, threshold, label)
    img = Image.fromarray(composed)
    draw = ImageDraw.Draw(img)
    _w, h = img.size
    draw.text((10, h - 20), title, fill=(255, 255, 0))
    return np.asarray(img)


def record_segment(
    label: str,
    checkpoint_path: str,
    predictor,
    threshold: float,
    seed: int,
    device: str,
    max_steps: int,
    min_failures: int,
    camera_args: argparse.Namespace,
) -> list[np.ndarray]:
    from cartpole_term.predictor.labeling import TerminationLabeler

    print(f"[video] [{label}] loading checkpoint {checkpoint_path}")
    runner = build_runner_from_checkpoint(checkpoint_path, seed=seed, device=device)
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    video_env_cfg = _build_video_env_cfg(camera_args)
    raw_env = ManagerBasedRlEnv(cfg=video_env_cfg, device=device, render_mode="rgb_array")
    _sanity_check_renderer(raw_env, seed)

    obs, _ = raw_env.reset(seed=seed)
    labeler = TerminationLabeler(num_envs=1, horizon=PREDICTION_HORIZON)
    frames: list[np.ndarray] = []
    probs: list[float] = []
    labels: dict[int, int] = {}

    failure_count = 0
    step_idx = 0
    title = SEGMENT_LABELS[label]
    with torch.inference_mode():
        while failure_count < min_failures and step_idx < max_steps:
            frame = raw_env.render()

            obs_np = obs["actor"].detach().cpu().numpy()
            # S7.3: sampled, not the mean action -- matches how the predictor was trained.
            actions = runner.alg.actor(obs, stochastic_output=True)
            action_np = actions.detach().cpu().numpy()
            x = np.concatenate([obs_np, action_np], axis=1).astype(np.float32)
            prob = float(torch.sigmoid(predictor(torch.as_tensor(x))).item())

            obs, _reward, terminated, truncated, _info = raw_env.step(actions.to(raw_env.device))
            emitted = labeler.step([{"idx": step_idx}], terminated.tolist(), truncated.tolist())
            for sample in emitted:
                labels[sample.payload["idx"]] = sample.label

            frames.append(_composite_with_title(frame, prob, threshold, labels.get(step_idx), title))
            probs.append(prob)

            if bool(terminated.item()):
                failure_count += 1
            step_idx += 1

    raw_env.close()

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"[{label}] actor.{name} changed during recording"

    print(
        f"[video] [{label}] {step_idx} steps ({step_idx * STEP_SECONDS:.1f}s), "
        f"{failure_count} failure episode(s), mean p={np.mean(probs):.3f}"
    )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--predictor-weights", type=str, default=None)
    parser.add_argument("--predictor-hidden", type=str, default="128,128")
    parser.add_argument("--metrics-file", type=str, default=None)
    parser.add_argument("--max-seconds-per-checkpoint", type=float, default=20.0)
    parser.add_argument("--min-failures-per-checkpoint", type=int, default=2)
    parser.add_argument("--camera-azimuth", dest="camera_azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", dest="camera_elevation", type=float, default=-25.0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2).")
    device = "cuda"

    outputs_dir = Path("outputs")
    metrics_path = Path(args.metrics_file) if args.metrics_file else outputs_dir / f"metrics_stratified_seed{args.seed}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found -- run scripts/train_stratified_predictor.py --seed {args.seed} first."
        )
    metrics = json.loads(metrics_path.read_text())
    checkpoints = metrics["checkpoints"]
    threshold = metrics["threshold"]
    print(f"[video] loaded checkpoints + threshold={threshold} from {metrics_path}")

    weights_path = Path(args.predictor_weights) if args.predictor_weights else outputs_dir / f"predictor_stratified_seed{args.seed}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"{weights_path} not found.")
    hidden_parts = args.predictor_hidden.split(",")
    hidden = (int(hidden_parts[0]), int(hidden_parts[1]))

    from cartpole_term.predictor.model import TerminationPredictor

    predictor = TerminationPredictor(history=1, hidden=hidden)
    predictor.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    predictor.eval()
    print(f"[video] loaded predictor {weights_path} (hidden={hidden})")

    max_steps = round(args.max_seconds_per_checkpoint / STEP_SECONDS)

    all_frames: list[np.ndarray] = []
    for label in SEGMENT_ORDER:
        segment_frames = record_segment(
            label=label,
            checkpoint_path=checkpoints[label],
            predictor=predictor,
            threshold=threshold,
            seed=args.seed,
            device=device,
            max_steps=max_steps,
            min_failures=args.min_failures_per_checkpoint,
            camera_args=args,
        )
        all_frames.extend(segment_frames)

    import imageio

    out_path = Path(args.out) if args.out else outputs_dir / f"predictor_stratified_overlay_seed{args.seed}.mp4"
    fps = round(1.0 / STEP_SECONDS)
    writer = imageio.get_writer(str(out_path), fps=fps)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    size_mb = out_path.stat().st_size / 1e6
    print(
        f"[video] wrote {out_path} ({size_mb:.1f} MB, {len(all_frames)} frames = "
        f"{len(all_frames) * STEP_SECONDS:.1f}s total across {len(SEGMENT_ORDER)} segments)"
    )


if __name__ == "__main__":
    main()
