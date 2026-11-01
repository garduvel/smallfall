"""Phase-2 tool (outside CLAUDE.md's cartpole-only spec, see DEVIATIONS.md):
predictor_overlay.mp4 analogue for Go1 -- the policy at a chosen operating
point, running with the predictor's probability overlaid per frame (text,
bar, threshold marker, ground-truth label once known), stopping once
--min-failures real fall episodes are captured.

Written specifically in response to the AUPRC~0.98 result across all four
operating points looking suspiciously high for a visual sanity check, not
just to take the numbers on faith -- this renders the actual rollout so a
human can see whether the probability genuinely rises before a real fall,
the same way scripts/record_video.py's predictor_overlay.mp4 let a human
eyeball cartpole's result instead of trusting the metrics file alone.

Reuses scripts/record_video.py's already-tested, task-agnostic helpers
(_sanity_check_renderer, _composite_overlay_frame) via import rather than
duplicating them -- that file is a completed, gated CLAUDE.md deliverable
and is not modified here. `_ensure_egl_configured` is the one exception,
copied verbatim rather than imported, for the same reason
record_predictor_stratified_video.py copied it: it must run before
mujoco/mjlab's first import anywhere in the process, and relying on import
order is fragile.

One real gap from record_video.py's version: `play=True` isn't implemented
for Go1 yet (cartpole_term/go1/env_cfg.py's go1_velocity_term_env_cfg
raises NotImplementedError for play=True -- mjlab's play-cfg loader API for
a non-cartpole task wasn't confirmed, see DEVIATIONS.md). This script uses
play=False (the training-shaped cfg, same one frozen-eval/collection use),
which means domain randomization and random pushes are visually present in
the video -- not a bug, arguably a more honest demonstration (the policy
handling real perturbations on camera), but worth knowing why the ground
looks less "clean" than a typical demo reel.

Usage, from the repo root, on the GPU machine, after
scripts/go1_select_checkpoints.py and scripts/go1_eval_predictor.py
--operating-point <point> have produced their outputs:

    python scripts/go1_record_video.py --operating-point target
    python scripts/go1_record_video.py --operating-point fluent --min-failures 3
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
            print(f"[go1-video] wrote missing EGL vendor ICD file: {icd_path}")
        except OSError as e:
            print(f"[go1-video] could not write {icd_path} ({e}); continuing")


_ensure_egl_configured()

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term.go1  # noqa: F401,E402
from cartpole_term.go1.env_cfg import go1_velocity_term_env_cfg  # noqa: E402
from cartpole_term.go1.runner import build_go1_runner_from_checkpoint  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from record_video import _composite_overlay_frame, _sanity_check_renderer  # noqa: E402

STEP_SECONDS = 0.02  # Go1 velocity task: decimation=4, timestep=0.005 -> 50 Hz (confirmed via inspect_go1_task.py)
PREDICTION_HORIZON = 16  # matches go1_collect_frozen.py's --predictor.horizon default
MIN_FAILURE_EPISODES_FLOOR = 3
MAX_OVERLAY_STEPS = 20_000
# Offset for --take's rollout seed, distinct from every other seed offset
# already in use across this project (100/200/300) so a "take" never
# accidentally reuses e.g. the S6.2.3 eval seed or S7.3 collection seed.
TAKE_SEED_OFFSET = 400


def _build_go1_video_env_cfg(args: argparse.Namespace):
    cfg = go1_velocity_term_env_cfg(play=False)  # play=True not implemented yet, see module docstring
    cfg.scene.num_envs = 1
    cfg.viewer.azimuth = args.camera_azimuth
    cfg.viewer.elevation = args.camera_elevation
    if args.camera_distance is not None:
        cfg.viewer.distance = args.camera_distance
    print(
        f"[go1-video] viewer config: origin_type={cfg.viewer.origin_type}, "
        f"azimuth={cfg.viewer.azimuth}, elevation={cfg.viewer.elevation}, "
        f"distance={cfg.viewer.distance}, body_name={cfg.viewer.body_name}"
    )
    return cfg


def resolve_go1_checkpoint(seed: int, operating_point: str, checkpoint_file: str | None) -> str:
    if checkpoint_file is not None:
        return checkpoint_file
    selected_path = Path("outputs/go1_selected_checkpoints.json")
    if not selected_path.exists():
        raise FileNotFoundError(
            f"{selected_path} not found and --checkpoint-file not given -- run "
            "scripts/go1_select_checkpoints.py first, or pass --checkpoint-file explicitly."
        )
    selected = json.loads(selected_path.read_text())
    seed_key = str(seed)
    if seed_key not in selected:
        raise KeyError(f"seed {seed} not found in {selected_path}.")
    if operating_point not in selected[seed_key]:
        raise KeyError(f"operating point '{operating_point}' not found for seed {seed}.")
    return selected[seed_key][operating_point]["checkpoint_path"]


def record_go1_predictor_overlay(args: argparse.Namespace, device: str) -> None:
    from cartpole_term.predictor.labeling import TerminationLabeler
    from cartpole_term.predictor.model import TerminationPredictor

    outputs_dir = Path("outputs")
    checkpoint_path = resolve_go1_checkpoint(args.seed, args.operating_point, args.checkpoint_file)
    print(f"[go1-video] loading policy checkpoint {checkpoint_path}")

    predictor_path = outputs_dir / f"go1_predictor_{args.operating_point}_seed{args.seed}.pt"
    metrics_path = outputs_dir / f"go1_metrics_{args.operating_point}_seed{args.seed}.json"
    if not predictor_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"{predictor_path} and/or {metrics_path} not found -- run "
            f"scripts/go1_eval_predictor.py --operating-point {args.operating_point} "
            f"--seeds {args.seed} first."
        )
    metrics = json.loads(metrics_path.read_text())
    threshold = metrics["mlp"]["threshold"]
    obs_action_dim = metrics["obs_action_dim"]
    hidden = tuple(metrics["predictor_hidden"])
    print(f"[go1-video] loaded threshold={threshold:.4f}, obs_action_dim={obs_action_dim}, hidden={hidden}")

    predictor = TerminationPredictor(history=1, hidden=hidden, obs_action_dim=obs_action_dim)
    predictor.load_state_dict(torch.load(predictor_path, weights_only=True))
    predictor.eval()

    runner = build_go1_runner_from_checkpoint(checkpoint_path, seed=args.seed, device=device)
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    rollout_seed = args.seed + args.take * TAKE_SEED_OFFSET
    video_env_cfg = _build_go1_video_env_cfg(args)
    raw_env = ManagerBasedRlEnv(cfg=video_env_cfg, device=device, render_mode="rgb_array")
    _sanity_check_renderer(raw_env, rollout_seed)

    obs, _ = raw_env.reset(seed=rollout_seed)
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
            # Sampled, not mean action -- the predictor was trained exclusively
            # on stochastic-behaviour-policy transitions (CLAUDE.md S7.3's rule,
            # ported unchanged for Go1).
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
                print(f"[go1-video] fall #{failure_count} at step {step_idx} (p just before fall: {prob:.3f})")
            step_idx += 1

    raw_env.close()

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during recording"

    if failure_count < args.min_failures:
        print(
            f"[go1-video] WARNING only captured {failure_count}/{args.min_failures} "
            f"failures within the {MAX_OVERLAY_STEPS}-step safety cap. Writing what was captured anyway."
        )
    print(
        f"[go1-video] rollout done -- {step_idx} steps, {failure_count} failure episode(s), "
        f"{len(labels)}/{step_idx} frames have a resolved ground-truth label."
    )

    import imageio

    final_path = outputs_dir / f"go1_predictor_overlay_{args.operating_point}_seed{args.seed}_take{args.take}.mp4"
    fps = round(1.0 / STEP_SECONDS)
    writer = imageio.get_writer(str(final_path), fps=fps)
    for idx, frame in enumerate(frames):
        composite = _composite_overlay_frame(frame, probs[idx], threshold, labels.get(idx))
        writer.append_data(composite)
    writer.close()

    size_mb = final_path.stat().st_size / 1e6
    print(f"[go1-video] wrote {final_path} ({size_mb:.1f} MB, {step_idx} frames = {step_idx * STEP_SECONDS:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operating-point", dest="operating_point", choices=("struggling", "target", "competent", "fluent"), default="target")
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, default=None)
    parser.add_argument("--camera-azimuth", dest="camera_azimuth", type=float, default=90.0)
    parser.add_argument("--camera-elevation", dest="camera_elevation", type=float, default=-20.0)
    parser.add_argument("--camera-distance", dest="camera_distance", type=float, default=None,
                         help="Override mjlab's default viewer distance for this task if the robot reads too small/large on camera.")
    parser.add_argument("--min-failures", dest="min_failures", type=int, default=5)
    parser.add_argument("--take", type=int, default=0,
                         help="Which rollout take (0, 1, ...) for this operating point -- same "
                         "checkpoint, different rollout seed and output filename, so multiple "
                         "independent takes don't overwrite each other.")
    args = parser.parse_args()
    if args.min_failures < MIN_FAILURE_EPISODES_FLOOR:
        raise ValueError(f"--min-failures={args.min_failures} is below the required minimum of {MIN_FAILURE_EPISODES_FLOOR}.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = "cuda"

    record_go1_predictor_overlay(args, device)


if __name__ == "__main__":
    main()
