"""Gate 5 pass test: replay the SAME injected fall states under three policies
and compare how hard and how far the robot falls.

Policies ("arms"):
  protective   -- the trained protective checkpoint, deterministic actions
  walker       -- the nominal walking policy (checkpoints/model_9999.pt)
                  simply continuing, i.e. what happens with no hand-off
  limp         -- motors off: zero torque at every joint from the first
                  step (both PD gains scaled to 0 with mjlab's own
                  dr.pd_gains at startup, so the actuators cannot produce
                  force whatever the action). The robot falls as a passive
                  linkage; joint limits and the motors' reflected inertia
                  (armature) still act. Asserted: measured torque is 0.

(An earlier `zero_action` arm, action 0 = a stiff PD hold of the default pose,
was removed on 2026-10-03: limp was better than it on ground covered, trunk
force, and any-body force, with zero torque. Result files written before that
date still contain it.)

Every bank state is replayed once per arm (env i <- bank state i, via
inject_fall_state's `sequential=True`). The env runs with auto_reset=False so
readings are each fall's true final state; only each env's first episode is
recorded (src/cartpole_term/protective/compare.py). Per fall: peak ground-
contact force (critical trunk / any body), peak joint torque, envelope
growth (ground covered beyond the robot's own extent), root travel, steps to
settle, settled or not, return under the protective reward.

Printed verdicts: the four-metric verdict (protective median below the
walker's on peak critical force, peak any-body force, peak torque, and
envelope growth), and the same medians against limp (torque excepted). The
project's pass rule is space-first and is applied by the notebooks: ground
covered below both the walker's and limp's, and the fall comes to rest.

Usage (repo root, GPU machine):
    python scripts/go1_protective_compare.py \
        --protective-checkpoint logs/rsl_rl/go1_protective_fall/<run>/model_299.pt \
        --walker-checkpoint checkpoints/model_9999.pt

Writes outputs/go1_protective_compare_seed{N}.json (+ .npz per-fall arrays).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Must precede `import torch`: mjlab's @torch.jit.script math utils crash on
# this Colab image (missing libnvrtc-builtins.so.13.0). See DEVIATIONS.md.
os.environ.setdefault("PYTORCH_JIT", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term.protective  # noqa: F401,E402 -- registers the task
from cartpole_term.common.robot_profile import get_profile  # noqa: E402
from cartpole_term.protective.compare import PASS_METRICS, FirstFallTracker, summarize, verdict  # noqa: E402
from cartpole_term.protective.fall_size import FallSizeTracker  # noqa: E402
from cartpole_term.protective.env_cfg import TASK_IDS, protective_fall_env_cfg  # noqa: E402
from cartpole_term.protective.runner import ProtectiveFrozenEvalRunner  # noqa: E402
from cartpole_term.runner import build_runner_from_checkpoint  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402

CONTACT_SENSOR = "protective_body_ground_contact"
ARMS = ("protective", "walker", "limp")
LIMP_TORQUE_TOLERANCE_NM = 1e-4
CONTACT_N = 1.0  # a body counts as touching the ground above this net contact force


def _env_cfg(state_bank_file: str, num_envs: int, limp: bool = False, robot: str = "go1"):
    # Rewards never affect a replay, so the Go1 weights are fine for any robot.
    cfg = protective_fall_env_cfg(robot, play=False, state_bank_file=state_bank_file)
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    cfg.events["reset_fall_state"].params["sequential"] = True
    if limp:
        # Go1's actuators are BuiltinPositionActuator (MuJoCo <position>:
        # force = kp*(ctrl - q) - kd*qd). mjlab's dr.pd_gains with
        # operation="scale" multiplies kp and kd per env; (0, 0) makes the
        # force identically zero. Registered as an event so mjlab expands
        # actuator_gainprm/biasprm per world (requires_model_fields).
        from mjlab.envs.mdp import dr
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        cfg.events["limp_motors_off"] = EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={"kp_range": (0.0, 0.0), "kd_range": (0.0, 0.0), "operation": "scale",
                    "asset_cfg": SceneEntityCfg("robot")},
        )
    return cfg


def _load_actor(checkpoint: str, args, device: str):
    def factory(play: bool = False):
        return protective_fall_env_cfg(args.robot, play=play, state_bank_file=args.state_bank_file)

    runner = build_runner_from_checkpoint(
        checkpoint, seed=args.seed, eval_num_envs=1, device=device,
        runner_cls=ProtectiveFrozenEvalRunner, env_cfg_factory=factory,
        task_id=TASK_IDS[args.robot], extra_scalar_metric_fn=None,
    )
    runner.alg.eval_mode()
    return runner.alg.actor


def run_arm(arm: str, actor, args, num_envs: int, device: str):
    env = ManagerBasedRlEnv(cfg=_env_cfg(args.state_bank_file, num_envs, limp=(arm == "limp"), robot=args.robot),
                            device=device)
    robot = env.scene["robot"]
    sensor = env.scene[CONTACT_SENSOR]
    critical = set(get_profile(args.robot).vulnerability_regions["high"])
    critical_mask = torch.tensor([n in critical for n in sensor.primary_names], device=device)
    assert critical_mask.any(), f"no critical bodies among sensor primaries {sensor.primary_names}"

    obs, _ = env.reset(seed=args.seed)
    start_pose = robot.data.root_link_pose_w.clone() - torch.cat(
        [env.scene.env_origins, torch.zeros(num_envs, 4, device=device)], dim=-1
    )
    tracker = FirstFallTracker(robot.data.root_link_pos_w[:, :2], robot.data.body_link_pos_w[..., :2])
    # Fall-size ablation measures (protective/fall_size.py), stored as size__<metric>.
    body_names = list(robot.body_names)
    assert len(body_names) == robot.data.body_link_pos_w.shape[1], "body_names / body_link_pos_w mismatch"
    contact_idx = torch.tensor([body_names.index(n) for n in sensor.primary_names], device=device)
    sizer = FallSizeTracker(robot.data.root_link_pos_w[:, :2], robot.data.body_link_pos_w[..., :2])
    snapshot = {k: v.clone() for k, v in actor.state_dict().items()} if actor is not None else None

    max_steps = int(env.max_episode_length) + 5
    with torch.inference_mode():
        for _ in range(max_steps):
            if actor is None:
                actions = torch.zeros(num_envs, env.action_manager.total_action_dim, device=device)
            else:
                actions = actor(obs, stochastic_output=False)
            obs, reward, terminated, truncated, _ = env.step(actions)
            ended = tracker.update(
                force_mag=torch.norm(sensor.data.force, dim=-1),
                critical_mask=critical_mask,
                torque=robot.data.actuator_force,
                root_xy=robot.data.root_link_pos_w[:, :2],
                body_xy=robot.data.body_link_pos_w[..., :2],
                reward=reward,
                terminated=terminated.bool(),
                truncated=truncated.bool(),
            )
            body_xy = robot.data.body_link_pos_w[..., :2]
            sizer.update(root_xy=robot.data.root_link_pos_w[:, :2], body_xy=body_xy,
                         contact_xy=body_xy[:, contact_idx], in_contact=torch.norm(sensor.data.force, dim=-1) > CONTACT_N,
                         ended=(terminated | truncated).bool())
            if tracker.all_done:
                break
            done_ids = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                obs, _ = env.reset(env_ids=done_ids)
            del ended
    env.close()

    if not tracker.all_done:
        raise RuntimeError(f"[compare] arm {arm}: {int(tracker.active.sum())} falls never ended within {max_steps} steps")
    if snapshot is not None:
        for k, v in actor.state_dict().items():
            assert torch.equal(snapshot[k], v), f"[compare] {arm}: actor.{k} changed during rollout"
    per_fall = tracker.per_fall()
    assert sizer.all_done, "fall-size tracker still recording after every first fall ended"
    sizes = sizer.per_fall()
    assert np.allclose(sizes["envelope_growth_m"], per_fall["envelope_growth_m"], atol=1e-5), \
        "the two trackers disagree on ground covered"
    per_fall.update({f"size__{k}": v for k, v in sizes.items()})
    if arm == "limp":
        worst = float(per_fall["peak_torque_nm"].max())
        if worst > LIMP_TORQUE_TOLERANCE_NM:
            raise RuntimeError(f"[compare] limp arm is not limp: peak actuator torque {worst:.4f} Nm (expected 0)")
        print(f"[compare] limp: peak actuator torque over all falls = {worst:.2e} Nm (motors off confirmed)")
    return per_fall, start_pose


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", choices=("go1", "g1"), default="go1")
    p.add_argument("--protective-checkpoint", required=True)
    p.add_argument("--walker-checkpoint", default="checkpoints/model_9999.pt")
    p.add_argument("--state-bank-file", default="outputs/go1_fall_state_bank_seed0.npz")
    p.add_argument("--num-states", type=int, default=None, help="default: every state in the bank, once each")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    p.add_argument("--extra-policy", action="append", default=[], metavar="LABEL=CHECKPOINT",
                   help="also replay this checkpoint on the same falls (per-fall arrays only; "
                        "the verdicts stay protective vs walker/limp). Repeatable.")
    args = p.parse_args()
    extras = {}
    for item in args.extra_policy:
        label, _, ckpt = item.partition("=")
        if not label or not ckpt or label in ARMS or label in extras or "__" in label:
            raise ValueError(f"[compare] bad --extra-policy {item!r} (want LABEL=CHECKPOINT, unique label)")
        extras[label] = ckpt

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = "cuda"
    for f in (args.protective_checkpoint, args.walker_checkpoint, args.state_bank_file, *extras.values()):
        if not Path(f).exists():
            raise FileNotFoundError(f"[compare] missing input: {f}")

    bank_size = len(np.load(args.state_bank_file)["episode_id"])
    num_envs = min(args.num_states or bank_size, bank_size)
    out_path = Path(args.output or f"outputs/{args.robot}_protective_compare_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arms = ARMS + tuple(extras)
    print(f"[compare] {num_envs} fall states x {len(arms)} arms")

    actors = {
        "protective": _load_actor(args.protective_checkpoint, args, device),
        "walker": _load_actor(args.walker_checkpoint, args, device),
        "limp": None,
    } | {label: _load_actor(ckpt, args, device) for label, ckpt in extras.items()}

    per_fall, start_poses, summaries = {}, {}, {}
    for arm in arms:
        t0 = time.time()
        per_fall[arm], start_poses[arm] = run_arm(arm, actors[arm], args, num_envs, device)
        summaries[arm] = summarize(per_fall[arm])
        print(f"[compare] {arm}: done in {time.time() - t0:.1f}s")

    for arm in arms[1:]:
        if not torch.allclose(start_poses[arm], start_poses[ARMS[0]], atol=1e-5):
            raise RuntimeError(f"[compare] arm {arm} did not start from the same fall states as {ARMS[0]}")

    # Four-metric verdict against the walker continuing. Limp is reported
    # beside it rather than inside it: a motors-off arm has zero torque by
    # construction, so it cannot be "beaten" on torque.
    v = verdict({arm: summaries[arm] for arm in ("protective", "walker")})
    vs_limp = {
        m: {"candidate_median": summaries["protective"][m]["median"], "limp_median": summaries["limp"][m]["median"],
            "candidate_lower": summaries["protective"][m]["median"] < summaries["limp"][m]["median"]}
        for m in PASS_METRICS
    }
    report = {
        "config": vars(args) | {"num_states": num_envs, "arms": list(arms), "extra_policies": extras},
        "summaries": summaries,
        "verdict": v,
        "vs_limp": vs_limp,
    }
    out_path.write_text(json.dumps(report, indent=2))
    np.savez(out_path.with_suffix(".npz"), **{f"{arm}__{k}": a for arm, d in per_fall.items() for k, a in d.items()})

    cols = ["peak_force_critical_n", "peak_force_any_n", "peak_torque_nm", "envelope_growth_m",
            "root_travel_end_m", "steps", "settled", "return"]
    print("\n[compare] medians (settled = mean rate)")
    print(f"{'arm':12s}" + "".join(f"{c:>23s}" for c in cols))
    for arm in ARMS:
        s = summaries[arm]
        vals = [s[c]["mean"] if c == "settled" else s[c]["median"] for c in cols]
        print(f"{arm:12s}" + "".join(f"{x:23.3f}" for x in vals))
    print("\n[compare] verdict:")
    for m, r in v.items():
        if isinstance(r, dict):
            print(f"  {m}: protective {r['candidate_median']:.3f} vs {r['baseline_medians']} -> "
                  f"{'BEATS' if r['beats_all_baselines'] else 'does NOT beat'} all baselines")
    print(f"  FOUR-METRIC VERDICT (vs the walker): {v['pass']}")
    print("\n[compare] protective vs limp (motors off), medians:")
    for m, r in vs_limp.items():
        print(f"  {m}: protective {r['candidate_median']:.3f} vs limp {r['limp_median']:.3f} -> "
              f"{'lower' if r['candidate_lower'] else 'NOT lower'}")
    print(f"[compare] wrote {out_path} and {out_path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
