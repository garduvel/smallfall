"""Gate 7 demo video (Go1): the same fall, three ways, side by side.

Left: the walker alone (predictor watching, not wired). Middle: limp (when
the predictor fires, the motors are switched off: zero torque at every joint,
via mjlab's dr.pd_gains scaled to 0 at that step; asserted from the measured
actuator force). Right: the combined controller (predictor fires -> protective
policy takes over for good). All three panels are the same rollout seed
through the same live system as scripts/go1_handoff_eval.py (same env cfg
factory, same HandoffTrials bookkeeping, same models), one env each. Each take
runs until the predictor fires, then until the robot settles (or the 3 s
window runs out). Only takes where the walker really falls after the flag (a
true alarm, per the left panel) are kept, so every take shows an actual fall.
Pushes stop for a robot once it is on the ground.

Overlay per panel: the predictor probability bar with the threshold marker
(while the walker is in control), who is in control, the moment of the flag,
and the end state (settled / still moving, trunk peak force, ground covered).

The GPU sim is not bit-deterministic, so the panels can diverge before the
flag; each is then still a correct recording of its own arm. Takes whose
panels all flag on the same step are preferred, and the JSON says which did.

Usage (repo root, GPU machine):
    python scripts/go1_handoff_demo_video.py \
        --protective-checkpoint checkpoints/protective/round8b_model_200.pt

Writes outputs/videos/go1_handoff_demo_seed{N}.mp4 and a .json sidecar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace


def _ensure_egl_configured() -> None:
    """Same helper as scripts/record_video.py; must run before mujoco's first import."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    icd_path = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    if not os.path.exists(icd_path):
        try:
            os.makedirs(os.path.dirname(icd_path), exist_ok=True)
            with open(icd_path, "w") as f:
                f.write('{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}')
            print(f"[demo] wrote missing EGL vendor ICD file: {icd_path}")
        except OSError as e:
            print(f"[demo] could not write {icd_path} ({e}); continuing")


_ensure_egl_configured()
# Must precede `import torch`: see DEVIATIONS.md (PYTORCH_JIT on Colab).
os.environ.setdefault("PYTORCH_JIT", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from cartpole_term.common.commands import force_forward_only_commands  # noqa: E402
from cartpole_term.common.deferred_reset import DeferredResetEnv  # noqa: E402
from cartpole_term.common.robot_profile import GO1  # noqa: E402
from cartpole_term.protective.compare import EVENT_FLAG, HandoffTrials  # noqa: E402
from cartpole_term.protective.env_cfg import (  # noqa: E402
    ANGULAR_SETTLED_THRESHOLD_RAD_S,
    LINEAR_SETTLED_THRESHOLD_MPS,
)
from go1_collect_flagged_states import STEP_SECONDS, _load_predictor  # noqa: E402
from go1_handoff_eval import DOWN_HEIGHT_FRACTION, HOLD_STEPS, WINDOW_STEPS, make_env_cfg_factory  # noqa: E402
from go1_protective_compare import CONTACT_SENSOR, _load_actor  # noqa: E402
from go1_wild_push_predictor import DEFAULT_IMPULSE_FORCE_N, DEFAULT_IMPULSE_TORQUE_N  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

ARMS = ("nominal", "limp", "handoff")  # left, middle, right
TITLES = {
    "nominal": "Walker alone (predictor not wired)",
    "limp": "Limp: motors off when predictor fires",
    "handoff": "Hand-off to protective policy",
}
# After the predictor fires: (who-is-in-control text, its colour, banner text)
AFTER_FLAG = {
    "nominal": ("IN CONTROL: walker", (90, 200, 90), "PREDICTOR FIRED (not wired)"),
    "limp": ("MOTORS OFF (limp)", (190, 190, 190), "PREDICTOR FIRED -> MOTORS OFF"),
    "handoff": ("IN CONTROL: protective policy", (255, 150, 40), "PREDICTOR FIRED -> HAND-OFF"),
}
LIMP_TORQUE_TOLERANCE_NM = 1e-4
VIDEO_SEED_OFFSET = 700  # distinct from every other offset in use
LEAD_IN_S = 2.0  # walking shown before the flag
END_HOLD_S = 1.0  # freeze on the final frame
BANNER_S = 1.0  # how long the "predictor fired" banner stays up
# Frames kept from before the flag: the lead-in shown, plus slack in case the
# panels flag a few steps apart. Older walking frames are dropped as we go, so
# memory stays flat however long the robot walks before the push lands.
PRE_FLAG_KEEP = round(LEAD_IN_S / 0.02) + 50


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


FONT, FONT_BIG = _font(16), _font(22)


def _draw_fitted(d, xy, text, fill, size, max_width):
    """Draw `text` at the largest font size <= `size` that fits in
    `max_width` pixels, so no line is ever cut off at the frame edge."""
    for s in range(size, 7, -1):
        font = _font(s)
        if d.textlength(text, font=font) <= max_width:
            break
    d.text(xy, text, fill=fill, font=font)


def set_motor_gains(raw, scale: float) -> None:
    """Scale every joint motor's PD gains (kp and kd) by `scale` for all envs:
    1.0 = the robot's normal motors, 0.0 = motors off (limp). mjlab's own
    dr.pd_gains; needs the identity startup event video_factory registers so
    the gain fields exist per env."""
    from mjlab.envs.mdp import dr
    from mjlab.managers.scene_entity_config import SceneEntityCfg

    dr.pd_gains(raw, None, kp_range=(scale, scale), kd_range=(scale, scale),
                asset_cfg=SceneEntityCfg("robot"), operation="scale")


def record_take(arm, raw, deferred, walker_actor, protective_actor, predictor, threshold, seed, args, device):
    set_motor_gains(raw, 1.0)  # a previous limp take leaves them off
    motors_off, limp_peak_torque = False, 0.0
    robot = raw.scene["robot"]
    sensor = raw.scene[CONTACT_SENSOR]
    critical = set(GO1.vulnerability_regions["high"])
    critical_mask = torch.tensor([name in critical for name in sensor.primary_names], device=device)

    def root_pose_local():
        return torch.cat([robot.data.root_link_pos_w - raw.scene.env_origins, robot.data.root_link_quat_w], dim=-1)

    torch.manual_seed(seed)
    obs_dict, _ = deferred.reset(seed=seed)
    obs = TensorDict(obs_dict, batch_size=[1])
    raw.perturbations_off = torch.zeros(1, dtype=torch.bool, device=device)
    trials = HandoffTrials(
        handoff=(arm != "nominal"), walk_steps=args.walk_steps, window_steps=WINDOW_STEPS,
        hold_steps=HOLD_STEPS, lin_threshold=LINEAR_SETTLED_THRESHOLD_MPS,
        ang_threshold=ANGULAR_SETTLED_THRESHOLD_RAD_S,
        root_xy=robot.data.root_link_pos_w[:, :2], body_xy=robot.data.body_link_pos_w[..., :2],
        down_height=DOWN_HEIGHT_FRACTION * root_pose_local()[:, 2],
    )
    frames, info = deque(maxlen=PRE_FLAG_KEEP), deque(maxlen=PRE_FLAG_KEEP)
    with torch.inference_mode():
        for t in range(args.walk_steps + WINDOW_STEPS + 5):
            walker_actions = walker_actor(obs, stochastic_output=True)
            prob = torch.sigmoid(predictor(torch.cat([obs["actor"], walker_actions], dim=-1)).squeeze(-1))
            latched = trials.pre_step(t, prob, threshold, robot.data.root_link_pos_w[:, :2],
                                      robot.data.body_link_pos_w[..., :2], root_pose_local())
            actions = walker_actions
            if arm == "handoff" and latched.any():
                actions = torch.where(latched.unsqueeze(-1), protective_actor(obs, stochastic_output=False), walker_actions)
            elif arm == "limp" and latched.any() and not motors_off:
                set_motor_gains(raw, 0.0)  # from this step on the action is irrelevant: zero torque
                motors_off = True
            if int(trials.phase[0]) == 1 and frames.maxlen is not None:
                frames, info = deque(frames), deque(info)  # window open: keep every frame from here on
            frames.append(raw.render())
            info.append({"t": t, "prob": float(prob[0]), "latched": bool(latched[0]),
                         "in_window": int(trials.phase[0]) == 1})

            next_obs, reward, terminated, truncated, _ = deferred.step(actions)
            deferred.hold((terminated | truncated).nonzero(as_tuple=False).squeeze(-1))
            trials.post_step(
                t, fell=raw.termination_manager.get_term("fell_over").bool(), time_out=truncated.bool(),
                force_mag=torch.norm(sensor.data.force, dim=-1), critical_mask=critical_mask,
                torque=robot.data.actuator_force, root_xy=robot.data.root_link_pos_w[:, :2],
                body_xy=robot.data.body_link_pos_w[..., :2], reward=reward,
                lin_speed=torch.norm(robot.data.root_link_lin_vel_w, dim=-1),
                ang_speed=torch.norm(robot.data.root_link_ang_vel_w, dim=-1), root_pose=root_pose_local(),
                root_z=root_pose_local()[:, 2],
            )
            raw.perturbations_off = trials.down  # no more pushes once the robot is on the ground
            if motors_off:
                limp_peak_torque = max(limp_peak_torque, float(robot.data.actuator_force.abs().max()))
            obs = TensorDict(next_obs, batch_size=[1])
            if trials.all_done:
                frames.append(raw.render())
                info.append({**info[-1], "t": t + 1})
                break
    set_motor_gains(raw, 1.0)
    if limp_peak_torque > LIMP_TORQUE_TOLERANCE_NM:
        raise RuntimeError(f"[demo] limp panel is not limp: actuator torque {limp_peak_torque:.4f} Nm after motors-off")
    r = {k: v[0] for k, v in trials.results().items()}  # one env: per-env arrays -> scalars
    return list(frames), list(info), r


def overlay(frame, arm, i, info, r, threshold):
    img = Image.fromarray(frame).convert("RGB")
    if img.width < 640:  # renderer ignored the requested size: upscale so the text still fits
        img = img.resize((640, round(img.height * 640 / img.width)), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([0, 0, w, 30], fill=(0, 0, 0))
    _draw_fitted(d, (8, 4), TITLES[arm], (255, 255, 255), 22, w - 16)

    flag_step = r["flag_step"]
    handed_off = arm != "nominal" and info["latched"]  # walker no longer in control
    if handed_off:
        ctrl, color = AFTER_FLAG[arm][:2]
    else:
        ctrl, color = "IN CONTROL: walker", (90, 200, 90)
    clock = f"t={info['t'] * STEP_SECONDS:5.2f}s"
    clock_w = d.textlength(clock, font=FONT)
    d.text((w - 8 - clock_w, 36), clock, fill=(255, 255, 255), font=FONT)
    _draw_fitted(d, (8, 36), ctrl, color, 16, w - 32 - clock_w)

    # Predictor bar: live while the walker is choosing actions.
    if not handed_off:
        x0, y0, bw, bh = 8, 60, w - 16, 12
        d.rectangle([x0, y0, x0 + bw, y0 + bh], outline=(255, 255, 255))
        p = max(0.0, min(1.0, info["prob"]))
        d.rectangle([x0, y0, x0 + int(bw * p), y0 + bh], fill=(220, 50, 50) if p >= threshold else (60, 160, 60))
        tx = x0 + int(bw * threshold)
        d.line([(tx, y0 - 3), (tx, y0 + bh + 3)], fill=(255, 255, 0), width=2)
        _draw_fitted(d, (x0, y0 + bh + 3), f"fall predictor p={info['prob']:.2f} (threshold {threshold:.2f})",
                     (255, 255, 255), 16, w - 16)

    banner_steps = round(BANNER_S / STEP_SECONDS)
    if i == -1:  # final frozen frame: outcome of this panel
        end = "came to rest" if r["settled"] else "still moving at the 3 s cap"
        d.rectangle([0, h - 44, w, h], fill=(30, 30, 30))
        _draw_fitted(d, (10, h - 36), f"{end}; trunk peak force {r['peak_force_critical_n']:.0f} N, "
                     f"ground covered {r['envelope_growth_m']:.2f} m", (255, 255, 255), 16, w - 20)
    elif flag_step >= 0 and 0 <= info["t"] - flag_step < banner_steps:
        text = AFTER_FLAG[arm][2]
        d.rectangle([0, h - 44, w, h], fill=(150, 20, 20))
        _draw_fitted(d, (10, h - 36), text, (255, 255, 255), 22, w - 20)
    return np.asarray(img)


def compose_take(takes, threshold):
    """takes: {arm: (frames, info, r)} -> yields side-by-side frames, one at a
    time (nothing composed is held in memory). frames/info hold only the tail
    of the walk (see PRE_FLAG_KEEP); info["t"] is the absolute step, used to
    start every panel at the same moment. A panel that ends early freezes on
    its final frame, which carries its outcome line."""
    lead_in = round(LEAD_IN_S / STEP_SECONDS)
    start_t = max(0, min(int(takes[a][2]["window_start"]) for a in ARMS) - lead_in)
    shown = {arm: [k for k in range(len(takes[arm][0])) if takes[arm][1][k]["t"] >= start_t] for arm in ARMS}
    final = {arm: overlay(takes[arm][0][-1], arm, -1, takes[arm][1][-1], takes[arm][2], threshold) for arm in ARMS}
    sep = np.full((final[ARMS[0]].shape[0], 6, 3), 255, dtype=np.uint8)
    total = max(len(v) for v in shown.values()) + round(END_HOLD_S / STEP_SECONDS)
    for j in range(total):
        row = []
        for arm in ARMS:
            frames, info, r = takes[arm]
            if j < len(shown[arm]):
                k = shown[arm][j]
                panel = overlay(frames[k], arm, k, info[k], r, threshold)
            else:
                panel = final[arm]
            row += [panel] if not row else [sep, panel]
        yield np.concatenate(row, axis=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protective-checkpoint", required=True)
    p.add_argument("--walker-checkpoint", default="checkpoints/model_9999.pt")
    p.add_argument("--predictor-file", default="outputs/go1_wildpush_predictor_seed0.pt")
    p.add_argument("--metrics-file", default="outputs/go1_wildpush_metrics_seed0.json")
    p.add_argument("--state-bank-file", default="outputs/go1_fall_state_bank_seed0.npz",
                   help="only used to construct the protective runner's 1-env shape env")
    p.add_argument("--takes", type=int, default=3, help="falls to show")
    p.add_argument("--max-tries", type=int, default=30, help="rollout seeds to try before giving up")
    p.add_argument("--walk-steps", type=int, default=1000)
    p.add_argument("--impulse-force", type=float, default=DEFAULT_IMPULSE_FORCE_N)
    p.add_argument("--impulse-torque", type=float, default=DEFAULT_IMPULSE_TORQUE_N)
    p.add_argument("--camera-azimuth", type=float, default=90.0)
    p.add_argument("--camera-elevation", type=float, default=-20.0)
    p.add_argument("--camera-distance", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = "cuda"
    for f in (args.protective_checkpoint, args.walker_checkpoint, args.predictor_file,
              args.metrics_file, args.state_bank_file):
        if not Path(f).exists():
            raise FileNotFoundError(f"[demo] missing input: {f}")

    metrics = json.loads(Path(args.metrics_file).read_text())
    threshold = float(metrics["mlp"]["threshold"])
    predictor = _load_predictor(args.predictor_file, int(metrics["obs_action_dim"]),
                                tuple(metrics["predictor_hidden"]), device)

    base_factory = make_env_cfg_factory(args)

    def video_factory(play: bool = False):
        cfg = base_factory(play=play)
        cfg.scene.num_envs = 1
        # mjlab's default is 320x240 (ViewerConfig); the overlay text is laid out for 640x480.
        cfg.viewer.width, cfg.viewer.height = 640, 480
        cfg.viewer.azimuth = args.camera_azimuth
        cfg.viewer.elevation = args.camera_elevation
        if args.camera_distance is not None:
            cfg.viewer.distance = args.camera_distance
        force_forward_only_commands(cfg)  # clean forward walking before the push, as in go1_wildpush_video.py
        # Identity gain "randomization" at startup: changes nothing (scale 1),
        # but registers dr.pd_gains so mjlab allocates the motor-gain fields
        # per env, which the limp panel needs to switch the motors off mid-episode.
        from mjlab.envs.mdp import dr
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        cfg.events["motor_gains_per_env"] = EventTermCfg(
            func=dr.pd_gains, mode="startup",
            params={"kp_range": (1.0, 1.0), "kd_range": (1.0, 1.0), "operation": "scale",
                    "asset_cfg": SceneEntityCfg("robot")},
        )
        return cfg

    from cartpole_term.go1.env_cfg import TASK_ID as GO1_TASK_ID
    from cartpole_term.go1.runner import Go1FrozenEvalRunner
    from cartpole_term.runner import build_runner_from_checkpoint
    from record_video import _sanity_check_renderer

    walker_runner = build_runner_from_checkpoint(
        args.walker_checkpoint, seed=args.seed, eval_num_envs=1, device=device,
        runner_cls=Go1FrozenEvalRunner, env_cfg_factory=video_factory,
        task_id=GO1_TASK_ID, extra_scalar_metric_fn=None,
    )
    walker_runner.alg.eval_mode()
    walker_actor = walker_runner.alg.actor
    protective_actor = _load_actor(args.protective_checkpoint,
                                   SimpleNamespace(state_bank_file=args.state_bank_file, seed=args.seed), device)
    nets = {"walker": walker_actor, "protective": protective_actor, "predictor": predictor}
    snapshots = {k: {n: v.clone() for n, v in m.state_dict().items()} for k, m in nets.items()}

    raw = ManagerBasedRlEnv(cfg=video_factory(), device=device, render_mode="rgb_array")
    deferred = DeferredResetEnv(raw)
    _sanity_check_renderer(raw, args.seed)

    # MuJoCo Warp is not bit-deterministic (mjlab's own seed_rng note, mujoco_warp#562),
    # so the panels can drift apart before the flag. Takes whose panels all flag
    # on the same step are preferred; others fill in only if too few match.
    import imageio

    out_dir = Path("outputs/videos")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"go1_handoff_demo_seed{args.seed}.mp4"
    writer = imageio.get_writer(str(out), fps=round(1.0 / STEP_SECONDS))
    num_frames = 0

    def write_take(takes):
        nonlocal num_frames
        for fr in compose_take(takes, threshold):
            writer.append_data(fr)
            num_frames += 1

    kept, spare = [], []
    for k in range(args.max_tries):
        if len(kept) >= args.takes:
            break
        seed = args.seed * 1000 + VIDEO_SEED_OFFSET + k
        # Whole take inside inference mode, including its reset and the motor-gain
        # writes: once the env has been stepped in inference mode, a reset outside
        # it can fail while clearing a push in progress (see go1_handoff_eval.py).
        with torch.inference_mode():
            takes = {arm: record_take(arm, raw, deferred, walker_actor, protective_actor, predictor,
                                      threshold, seed, args, device) for arm in ARMS}
        res = {arm: takes[arm][2] for arm in ARMS}
        rn = res["nominal"]
        true_alarm = (rn["event"] == EVENT_FLAG and rn["fall_step"] >= 0
                      and all(res[arm]["event"] == EVENT_FLAG for arm in ARMS))
        flag_steps = {arm: int(res[arm]["flag_step"]) for arm in ARMS}
        summary = {"rollout_seed": seed, "kept": bool(true_alarm), "flag_step": flag_steps,
                   "flag_steps_match": len(set(flag_steps.values())) == 1}
        for arm in ARMS:
            summary[arm] = {m: float(res[arm][m]) for m in ("peak_force_critical_n", "peak_force_any_n", "peak_torque_nm",
                                                            "envelope_growth_m", "settled", "steps")}
        matched = true_alarm and summary["flag_steps_match"]
        print(f"[demo] seed {seed}: {'KEPT' if matched else 'spare' if true_alarm else 'skipped (no true alarm)'} "
              f"{json.dumps(summary)}")
        if matched:
            kept.append(summary)
            write_take(takes)  # straight to the file: nothing composed is held in memory
        elif true_alarm and len(spare) < args.takes:
            spare.append((summary, takes))
        del takes
    for summary, takes in spare[: max(0, args.takes - len(kept))]:
        print(f"[demo] using unmatched take {summary['rollout_seed']} (panels flagged at different steps)")
        kept.append(summary)
        write_take(takes)
    writer.close()
    raw.close()

    for name, m in nets.items():
        for n_, v in m.state_dict().items():
            assert torch.equal(snapshots[name][n_], v), f"[demo] {name}.{n_} changed during recording"
    if not kept:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"[demo] no true-alarm fall in {args.max_tries} rollout seeds; raise --max-tries")
    if len(kept) < args.takes:
        print(f"[demo] WARNING only {len(kept)}/{args.takes} takes found in {args.max_tries} seeds; writing those")

    out.with_suffix(".json").write_text(json.dumps({"config": vars(args), "threshold": threshold, "takes": kept}, indent=2))
    print(f"[demo] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {num_frames} frames, {len(kept)} takes)")


if __name__ == "__main__":
    main()
