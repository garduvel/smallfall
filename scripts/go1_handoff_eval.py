"""Gates 6 + 7 (Go1): the live hand-off controller, and its ablation.

The whole pipeline running live, as it would on a robot: the nominal walking
policy (checkpoints/model_9999.pt) walks under the locked 150 N collection
push; every step the frozen wild-push predictor scores the walker's own
(obs, sampled action), exactly as in scripts/go1_collect_flagged_states.py;
on the first crossing of its val-selected threshold, control switches
IRREVERSIBLY (a per-env latch) to the frozen protective policy (Gate 5:
round 7, iteration 120, deterministic actions) for the rest of the episode.

Two arms, each a fresh env built and seeded identically (same env seed, same
torch seed, same RNG consumption: the walker is sampled for every env every
step in both arms, even where its action is then overridden):
  handoff  -- the combined controller above
  nominal  -- identical, but the latch is never wired: the walker keeps
              control (the predictor still scores; Gate 7's "without handoff")

One trial per env (its first episode; nothing is ever reset, fallen envs keep
simulating via DeferredResetEnv). A trial's measurement window opens at the
flag (or at an unflagged real fall) and closes when the robot has been still
for 0.5 s (the protective task's `settled` thresholds), after 150 steps (3 s,
the protective episode budget), or at the time limit. Per window: peak ground-
contact force (critical trunk / any body), peak torque, envelope growth
(ground covered, origin at the window start), settled. See
src/cartpole_term/protective/compare.py (HandoffTrials, unit-tested).

Pairing: if the simulation is deterministic, each env is identical in both
arms up to its flag, so the nominal arm says whether that flag was followed
by a real fall (true alarm) or not (false alarm). The script measures how
many envs actually pair (same event, same flag step, same root pose at the
window start) and reports it; paired analyses are only reported if at least
95% of event trials pair.

Pre-registered criteria (DEVIATIONS.md, "Gate 6/7"):
  Gate 6: actor + predictor weights unchanged (asserted); no env ever
          switches back (asserted); the handoff arm switches exactly the
          envs it flags (asserted); live recall >= 0.95 in the nominal arm.
  Gate 7: on the paired FALL trials (events whose nominal-arm counterpart
          really fell in the window), the handoff median beats the nominal
          median on peak critical force, peak any-body force, peak torque,
          and envelope growth. False alarms are reported separately.

Usage (repo root, GPU machine):
    python scripts/go1_handoff_eval.py \
        --protective-checkpoint checkpoints/protective/round7_model_120.pt

Writes outputs/go1_handoff_eval_seed{N}.json (+ .npz per-trial arrays).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Must precede `import torch`: mjlab's @torch.jit.script math utils crash on
# this Colab image (missing libnvrtc-builtins.so.13.0). See DEVIATIONS.md.
os.environ.setdefault("PYTORCH_JIT", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from cartpole_term.common.deferred_reset import DeferredResetEnv  # noqa: E402
from cartpole_term.common.perturbation import stop_perturbations_when_down  # noqa: E402
from cartpole_term.common.robot_profile import GO1  # noqa: E402
from cartpole_term.protective.compare import (  # noqa: E402
    EVENT_FLAG,
    EVENT_MISSED_FALL,
    EVENT_NONE,
    PASS_METRICS,
    HandoffTrials,
    summarize,
    verdict,
)
from cartpole_term.protective.env_cfg import (  # noqa: E402
    ANGULAR_SETTLED_THRESHOLD_RAD_S,
    LINEAR_SETTLED_THRESHOLD_MPS,
    PROTECTIVE_EPISODE_LENGTH_S,
    SETTLED_HOLD_S,
    protective_contact_sensor_cfg,
)
from go1_collect_flagged_states import STEP_SECONDS, _load_predictor  # noqa: E402
from go1_protective_compare import CONTACT_SENSOR, _load_actor  # noqa: E402
from go1_wild_push_predictor import (  # noqa: E402
    DEFAULT_IMPULSE_COOLDOWN_S,
    DEFAULT_IMPULSE_DURATION_S,
    DEFAULT_IMPULSE_FORCE_N,
    DEFAULT_IMPULSE_TORQUE_N,
    build_impulse_env_cfg_factory,
)

ARMS = ("handoff", "nominal")
SEED_OFFSET = 500  # distinct from the collectors' offsets (300, 400)
WINDOW_STEPS = round(PROTECTIVE_EPISODE_LENGTH_S / STEP_SECONDS)  # 150
HOLD_STEPS = round(SETTLED_HOLD_S / STEP_SECONDS)  # 25
# A robot counts as down (pushes stop) once the fall termination fires or its
# root drops below this fraction of its own height at reset. Our definition.
DOWN_HEIGHT_FRACTION = 0.5
MIN_PAIRED_FRACTION = 0.95
MIN_LIVE_RECALL = 0.95
REPORT_METRICS = PASS_METRICS + ("root_travel_end_m", "steps", "settled", "mean_torque_sq")


def make_env_cfg_factory(args):
    base = build_impulse_env_cfg_factory(
        args.impulse_force, args.impulse_torque, DEFAULT_IMPULSE_DURATION_S, DEFAULT_IMPULSE_COOLDOWN_S
    )

    def factory(play: bool = False):
        cfg = base(play=play)
        cfg.auto_reset = False  # nothing is ever reset; DeferredResetEnv holds every done env
        # Room for a window opened on the last walking step to run its full 150 steps.
        cfg.episode_length_s = (args.walk_steps + WINDOW_STEPS) * STEP_SECONDS
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (protective_contact_sensor_cfg(),)
        if not getattr(args, "keep_pushing_when_down", False):
            # Pushes stop for a robot once it is on the ground (user requirement,
            # 2026-10-03); the loop sets env.perturbations_off from HandoffTrials.down.
            stop_perturbations_when_down(cfg)
        return cfg

    return factory


def run_arm(arm, raw, walker_actor, protective_actor, predictor, threshold, args, device):
    """Both arms run in the SAME env instance: mjlab's startup events
    (per-env physics randomization) run once at construction, before any
    seeding here, so separately built envs simulate different robots
    (the first run's 0% pairing, DEVIATIONS.md). Reseed + full reset per arm."""
    deferred = DeferredResetEnv(raw)
    n = raw.num_envs
    robot = raw.scene["robot"]
    sensor = raw.scene[CONTACT_SENSOR]
    critical = set(GO1.vulnerability_regions["high"])
    critical_mask = torch.tensor([name in critical for name in sensor.primary_names], device=device)
    assert critical_mask.any(), f"no critical bodies among sensor primaries {sensor.primary_names}"
    assert "fell_over" in raw.cfg.terminations, f"no fell_over term: {list(raw.cfg.terminations)}"

    def root_pose_local():
        return torch.cat([robot.data.root_link_pos_w - raw.scene.env_origins, robot.data.root_link_quat_w], dim=-1)

    torch.manual_seed(args.seed)
    raw.seed(args.seed + SEED_OFFSET)
    raw.common_step_counter = 0  # drives step-keyed curricula; never reset by env.reset()
    obs_dict, _ = deferred.reset()
    obs = TensorDict(obs_dict, batch_size=[n])  # raw env returns a dict; see go1_collect_flagged_states.py
    initial_pose = root_pose_local().clone()
    raw.perturbations_off = torch.zeros(n, dtype=torch.bool, device=device)
    initial_obs = obs["actor"].clone()

    trials = HandoffTrials(
        handoff=(arm == "handoff"), walk_steps=args.walk_steps, window_steps=WINDOW_STEPS,
        hold_steps=HOLD_STEPS, lin_threshold=LINEAR_SETTLED_THRESHOLD_MPS,
        ang_threshold=ANGULAR_SETTLED_THRESHOLD_RAD_S,
        root_xy=robot.data.root_link_pos_w[:, :2], body_xy=robot.data.body_link_pos_w[..., :2],
        down_height=DOWN_HEIGHT_FRACTION * initial_pose[:, 2],
    )
    protective_env_steps = 0
    max_steps = args.walk_steps + WINDOW_STEPS + 5
    t0 = time.time()
    with torch.inference_mode():
        for t in range(max_steps):
            walker_actions = walker_actor(obs, stochastic_output=True)
            prob = torch.sigmoid(predictor(torch.cat([obs["actor"], walker_actions], dim=-1)).squeeze(-1))
            latched = trials.pre_step(
                t, prob, threshold, robot.data.root_link_pos_w[:, :2],
                robot.data.body_link_pos_w[..., :2], root_pose_local(),
            )
            actions = walker_actions
            if latched.any():
                protective_actions = protective_actor(obs, stochastic_output=False)
                actions = torch.where(latched.unsqueeze(-1), protective_actions, walker_actions)
                protective_env_steps += int(latched.sum())

            next_obs, reward, terminated, truncated, _ = deferred.step(actions)
            deferred.hold((terminated | truncated).nonzero(as_tuple=False).squeeze(-1))
            trials.post_step(
                t,
                fell=raw.termination_manager.get_term("fell_over").bool(),
                time_out=truncated.bool(),
                force_mag=torch.norm(sensor.data.force, dim=-1),
                critical_mask=critical_mask,
                torque=robot.data.actuator_force,
                root_xy=robot.data.root_link_pos_w[:, :2],
                body_xy=robot.data.body_link_pos_w[..., :2],
                reward=reward,
                lin_speed=torch.norm(robot.data.root_link_lin_vel_w, dim=-1),
                ang_speed=torch.norm(robot.data.root_link_ang_vel_w, dim=-1),
                root_pose=root_pose_local(),
                root_z=root_pose_local()[:, 2],
            )
            raw.perturbations_off = trials.down
            obs = TensorDict(next_obs, batch_size=[n])
            if trials.all_done:
                break
            if t % 200 == 0:
                done = int((trials.phase == 2).sum())
                print(f"[handoff] {arm}: step {t}, trials done {done}/{n}, latched {int(trials.latched.sum())}")
    if not trials.all_done:
        raise RuntimeError(f"[handoff] {arm}: {int((trials.phase != 2).sum())} trials unfinished after {max_steps} steps")
    print(f"[handoff] {arm}: done in {time.time() - t0:.1f}s ({t + 1} steps)")
    result = trials.results()
    if arm == "handoff":
        flagged = result["event"] == EVENT_FLAG
        assert np.array_equal(result["latched"], flagged), "handoff arm: latched set != flagged set"
    else:
        assert not result["latched"].any(), "nominal arm latched an env"
    result["protective_env_steps"] = np.array(protective_env_steps)
    result["initial_root_pose"] = initial_pose.cpu().numpy()
    result["initial_actor_obs"] = initial_obs.cpu().numpy()
    return result


def detection_stats(r):
    """Gate 6 detection numbers. Meaningful on the nominal arm, where a
    flag does not change what happens next."""
    flag = r["event"] == EVENT_FLAG
    missed = r["event"] == EVENT_MISSED_FALL
    fell_after_flag = flag & (r["fall_step"] >= 0)
    lead = (r["fall_step"] - r["flag_step"])[fell_after_flag]
    n_falls = int(fell_after_flag.sum() + missed.sum())
    return {
        "num_trials": int(len(r["event"])),
        "num_flagged": int(flag.sum()),
        "num_missed_falls": int(missed.sum()),
        "num_no_event": int((r["event"] == EVENT_NONE).sum()),
        "num_true_alarms": int(fell_after_flag.sum()),
        "num_false_alarms": int((flag & (r["fall_step"] < 0)).sum()),
        "live_recall": float(fell_after_flag.sum() / n_falls) if n_falls else None,
        "precision": float(fell_after_flag.sum() / flag.sum()) if flag.any() else None,
        "median_lead_steps": float(np.median(lead)) if len(lead) else None,
        "median_lead_s": float(np.median(lead) * STEP_SECONDS) if len(lead) else None,
    }


def summarize_subset(r, mask):
    return summarize({k: r[k][mask] for k in REPORT_METRICS}) if mask.any() else None


def paired_fractions(h, nm, mask):
    """Share of paired trials where handoff is lower / higher than nominal."""
    out = {}
    for m in PASS_METRICS:
        a, b = h[m][mask], nm[m][mask]
        out[m] = {"handoff_lower": float(np.mean(a < b)), "handoff_higher": float(np.mean(a > b))}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protective-checkpoint", required=True)
    p.add_argument("--walker-checkpoint", default="checkpoints/model_9999.pt")
    p.add_argument("--predictor-file", default="outputs/go1_wildpush_predictor_seed0.pt")
    p.add_argument("--metrics-file", default="outputs/go1_wildpush_metrics_seed0.json")
    p.add_argument("--state-bank-file", default="outputs/go1_fall_state_bank_seed0.npz",
                   help="only used to construct the protective runner's 1-env shape env; never sampled here")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--walk-steps", type=int, default=1000, help="walking budget per trial (1000 = the walker's 20 s episode)")
    p.add_argument("--impulse-force", type=float, default=DEFAULT_IMPULSE_FORCE_N)
    p.add_argument("--impulse-torque", type=float, default=DEFAULT_IMPULSE_TORQUE_N)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-pushing-when-down", action="store_true",
                   help="old behaviour: pushes keep firing on robots that are already on the ground")
    p.add_argument("--smoke", action="store_true", help="64 envs, 200 walking steps")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if args.smoke:
        args.num_envs, args.walk_steps = 64, 200

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")
    device = "cuda"
    for f in (args.protective_checkpoint, args.walker_checkpoint, args.predictor_file,
              args.metrics_file, args.state_bank_file):
        if not Path(f).exists():
            raise FileNotFoundError(f"[handoff] missing input: {f}")

    metrics = json.loads(Path(args.metrics_file).read_text())
    threshold = float(metrics["mlp"]["threshold"])
    predictor = _load_predictor(args.predictor_file, int(metrics["obs_action_dim"]),
                                tuple(metrics["predictor_hidden"]), device)

    from cartpole_term.go1.env_cfg import TASK_ID as GO1_TASK_ID
    from cartpole_term.go1.runner import Go1FrozenEvalRunner
    from cartpole_term.runner import build_runner_from_checkpoint

    walker_runner = build_runner_from_checkpoint(
        args.walker_checkpoint, seed=args.seed, eval_num_envs=args.num_envs, device=device,
        runner_cls=Go1FrozenEvalRunner, env_cfg_factory=make_env_cfg_factory(args),
        task_id=GO1_TASK_ID, extra_scalar_metric_fn=None,
    )
    walker_runner.alg.eval_mode()
    protective_actor = _load_actor(args.protective_checkpoint,
                                   SimpleNamespace(state_bank_file=args.state_bank_file, seed=args.seed), device)

    nets = {"walker": walker_runner.alg.actor, "protective": protective_actor, "predictor": predictor}
    snapshots = {k: {n: v.clone() for n, v in m.state_dict().items()} for k, m in nets.items()}

    tag = "_smoke" if args.smoke else ""
    out_path = Path(args.output or f"outputs/go1_handoff_eval{tag}_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[handoff] {args.num_envs} envs x {len(ARMS)} arms, threshold {threshold:.4f}, "
          f"walk {args.walk_steps} steps, window {WINDOW_STEPS}, settle hold {HOLD_STEPS}")

    env = walker_runner._build_eval_env()
    # Everything that touches the env after it is built stays inside inference
    # mode, INCLUDING the reset between arms. Stepping in inference mode leaves
    # sim buffers (e.g. xfrc_applied) as inference tensors; a reset outside
    # inference mode then fails when it clears a push still in progress
    # ("Inplace update to inference tensor outside InferenceMode", hit on the
    # first end-to-end run, DEVIATIONS.md). The collector and the compare script
    # already reset mid-run inside inference mode.
    with torch.inference_mode():
        res = {arm: run_arm(arm, env.unwrapped, walker_runner.alg.actor, protective_actor, predictor, threshold, args, device)
               for arm in ARMS}
    env.close()

    for k, m in nets.items():
        for name, v in m.state_dict().items():
            assert torch.equal(snapshots[k][name], v), f"[handoff] {k}.{name} changed during the rollouts"

    h, nm = res["handoff"], res["nominal"]
    initial_identical = bool(np.array_equal(h["initial_root_pose"], nm["initial_root_pose"])
                             and np.array_equal(h["initial_actor_obs"], nm["initial_actor_obs"]))
    event_any = (h["event"] != EVENT_NONE) | (nm["event"] != EVENT_NONE)
    same = (
        (h["event"] == nm["event"])
        & (h["flag_step"] == nm["flag_step"])
        & np.all(np.isclose(h["root_pose_at_start"], nm["root_pose_at_start"], atol=1e-4, equal_nan=True), axis=-1)
    )
    paired_fraction = float(same[event_any].mean()) if event_any.any() else None
    pairing_ok = paired_fraction is not None and paired_fraction >= MIN_PAIRED_FRACTION

    det = detection_stats(nm)
    gate6 = {
        "weights_unchanged": True,  # asserted above
        "no_switch_back": True,  # asserted every step in HandoffTrials.pre_step
        "switched_exactly_flagged": True,  # asserted in run_arm
        "handoff_arm_switches": int(h["latched"].sum()),
        "nominal_detection": det,
        "handoff_arm_detection_pre_switch": {k: detection_stats(h)[k] for k in ("num_flagged", "num_missed_falls", "num_no_event")},
        "live_recall_ok": det["live_recall"] is not None and det["live_recall"] >= MIN_LIVE_RECALL,
    }

    nominal_fell = nm["fall_step"] >= 0
    fall_trials = same & (nm["event"] != EVENT_NONE) & nominal_fell  # true alarms + missed falls, paired
    false_alarms = same & (nm["event"] == EVENT_FLAG) & ~nominal_fell
    all_events = {arm: res[arm]["event"] != EVENT_NONE for arm in ARMS}

    ablation = {
        "initial_states_identical": initial_identical,
        "paired_fraction_of_event_trials": paired_fraction,
        "pairing_ok": pairing_ok,
        "all_event_trials_unpaired": {arm: summarize_subset(res[arm], all_events[arm]) for arm in ARMS},
    }
    gate7_pass = None
    if pairing_ok:
        fall_summ = {arm: summarize_subset(res[arm], fall_trials) for arm in ARMS}
        v = verdict(fall_summ, candidate="handoff") if fall_trials.any() else None
        gate7_pass = v["pass"] if v else None
        ablation.update({
            "num_paired_fall_trials": int(fall_trials.sum()),
            "paired_fall_trials": fall_summ,
            "paired_fall_trials_per_trial": paired_fractions(h, nm, fall_trials) if fall_trials.any() else None,
            "verdict": v,
            "num_paired_false_alarms": int(false_alarms.sum()),
            "paired_false_alarms": {arm: summarize_subset(res[arm], false_alarms) for arm in ARMS},
        })

    report = {
        "config": vars(args) | {"threshold": threshold, "window_steps": WINDOW_STEPS, "hold_steps": HOLD_STEPS,
                                "seed_offset": SEED_OFFSET, "arms": list(ARMS)},
        "gate6": gate6,
        "gate7": ablation,
        "GATE6_PASS": gate6["live_recall_ok"],
        "GATE7_PASS": gate7_pass,
    }
    out_path.write_text(json.dumps(report, indent=2))
    np.savez(out_path.with_suffix(".npz"), **{f"{arm}__{k}": a for arm, d in res.items() for k, a in d.items()})

    print("\n[handoff] GATE 6 (nominal arm = what the predictor sees without interference)")
    for k, val in det.items():
        print(f"  {k}: {val}")
    print(f"  handoff arm switched {gate6['handoff_arm_switches']} envs (= its flagged envs); none switched back")
    print(f"  GATE 6 PASS (weights frozen, latch holds, live recall >= {MIN_LIVE_RECALL}): {report['GATE6_PASS']}")
    print(f"\n[handoff] initial states bit-identical across arms: {initial_identical}")
    print(f"[handoff] pairing: {paired_fraction if paired_fraction is None else round(paired_fraction, 4)} "
          f"of event trials identical across arms up to the window start (need >= {MIN_PAIRED_FRACTION})")
    if pairing_ok:
        cols = ("peak_force_critical_n", "peak_force_any_n", "peak_torque_nm", "envelope_growth_m", "steps")
        print(f"\n[handoff] GATE 7, paired fall trials (n={int(fall_trials.sum())}), medians; settled = rate")
        print(f"{'arm':10s}" + "".join(f"{c:>23s}" for c in cols) + f"{'settled':>9s}")
        for arm in ARMS:
            s = ablation["paired_fall_trials"][arm]
            print(f"{arm:10s}" + "".join(f"{s[c]['median']:23.3f}" for c in cols) + f"{s['settled']['mean']:9.3f}")
        if ablation["paired_fall_trials_per_trial"]:
            for m, fr in ablation["paired_fall_trials_per_trial"].items():
                print(f"  {m}: handoff lower in {100 * fr['handoff_lower']:.1f}% of falls, higher in {100 * fr['handoff_higher']:.1f}%")
        print(f"  false alarms (flag, but the nominal walker did not fall): {int(false_alarms.sum())}")
        print(f"  GATE 7 PASS: {gate7_pass}")
    else:
        print("[handoff] pairing below threshold: paired Gate 7 not evaluable; unpaired all-event summaries are in the JSON.")
    print(f"[handoff] wrote {out_path} and {out_path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
