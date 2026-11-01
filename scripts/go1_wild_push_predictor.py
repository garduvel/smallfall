"""New, standalone script (phase-2, outside CLAUDE.md's cartpole-only
spec, see DEVIATIONS.md): take the fully-trained Go1 checkpoint (the
10,000-iteration run), inflict a MUCH stronger-than-shipped push on it,
collect a labelled frozen dataset under that wild perturbation, and train
+ evaluate the termination predictor on it -- all in one invocation.

This intentionally combines what the multi-robot plan's Gate 2 (push
tuning) / Gate 3 (collection) / Gate 4 (predictor) would otherwise do as
three separate steps, at the user's request for one script rather than a
sweep. It stays "measured, not guessed" anyway: the wild push's actual
resulting failure rate is measured from real rollout data as collection
happens (printed), not assumed, and the push scale itself is grounded
(see WILD_PUSH_SCALE below), not invented from nothing.

Reuses, unmodified:
  - cartpole_term.go1.runner.build_go1_runner_from_checkpoint (loads the
    frozen checkpoint, builds a same-dynamics env).
  - cartpole_term.predictor.labeling.TerminationLabeler (the ring-buffer
    horizon-censoring labeller, same one every dataset in this project
    uses).
  - cartpole_term.predictor.{model.TerminationPredictor, train.train_predictor,
    split.split_episodes} (same architecture/loss/optimizer/split as
    every other predictor in this project -- CLAUDE.md S8.2/S8.3).
  - scripts/eval_predictor.py's already-generic metric functions
    (classification_metrics, best_threshold_by_f1, median_detection_lead_time,
    split_dataset_by_split_fn, make_inputs, mcfadden_r2, base_rate) --
    the same ones scripts/go1_eval_predictor.py already imports, proven
    dimension-agnostic.

NEW in this file (not reused from elsewhere, because nothing else needs
to inject a push override BETWEEN building the collection env and
starting rollout): the collection loop itself, adapted from
scripts/go1_collect_frozen.py's collect_one, with the wild push override
applied to the built env's `push_robot` event right before rollout
starts. Training (scripts/robot_train.py) is never touched -- the
checkpoint being loaded here was trained under mjlab's shipped, mild
push, same as every checkpoint in this project; only THIS script's
collection env sees the wild version.

WILD_PUSH_SCALE / verification: an earlier version of this script skipped
straight to the full 1M-transition/1000-episode collection at a fixed
push_scale (8.0, the same upper end of
scripts/measure_push_failure_rate.py's own escalation ladder) with no
check that 8x actually produces a usable failure rate for THIS
checkpoint first. Real run: 8x gave only 13 falls in 7 minutes --
nowhere near enough to reach 1000 in any reasonable time. Fixed: this
script now runs a QUICK verification pass first (imports
measure_push_failure_rate.run_one_config directly, same mechanism that
script already uses for its own sweep), escalating through
`PUSH_SCALE_ESCALATION_MULTIPLIERS` applied to `--push-scale` until
finding a scale whose measured failure rate clears
MIN_ACCEPTABLE_FAILURE_RATE, THEN runs the full collection at that
verified (possibly escalated) scale -- not the originally requested one
if it had to escalate. Raises loud (does not proceed) if every
candidate in the ladder stays below SAFETY_MIN_FAILURE_RATE, since that
would mean the big collection run might never reach its terminated-
episode target at all.

Run on the GPU machine, from the repo root, after scripts/robot_train.py
(or scripts/go1_train_full.ipynb) has produced the 10,000-iteration
checkpoint:

    python scripts/go1_wild_push_predictor.py --checkpoint-file checkpoints/model_9999.pt
    python scripts/go1_wild_push_predictor.py --checkpoint-file checkpoints/model_9999.pt --push-scale 12.0
    python scripts/go1_wild_push_predictor.py --checkpoint-file checkpoints/model_9999.pt --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from cartpole_term.cli import SMOKE_COLLECT_TRANSITIONS, SMOKE_NUM_ENVS  # noqa: E402
from cartpole_term.predictor.labeling import TerminationLabeler  # noqa: E402
from eval_predictor import (  # noqa: E402
    base_rate,
    best_threshold_by_f1,
    classification_metrics,
    make_inputs,
    mcfadden_r2,
    median_detection_lead_time,
    split_dataset_by_split_fn,
)
from measure_push_failure_rate import run_one_config  # noqa: E402

# Real run showed 8x gave only 13 falls in 7 minutes -- nowhere near
# enough. Doubled the starting point; VERIFY_ESCALATION_MULTIPLIERS below
# tries this and beyond BEFORE committing to the full collection.
WILD_PUSH_SCALE = 16.0
# Escalation ceiling, deliberately bounded: at 16x, push_robot's shipped
# velocity_range (+-0.5 m/s etc.) already becomes +-8 m/s linear /
# +-12.5 rad/s yaw -- an instantaneous, mass-independent velocity
# teleport (push_by_setting_velocity's own real mechanism, confirmed via
# mjlab/envs/mdp/events.py). Scaling this further stops resembling any
# real disturbance: every perturbation-based failure-prediction paper
# checked this session (SafeFall arXiv:2511.18509; the self-protective
# falling policy, arXiv:2512.01336; GYF arXiv:2310.04828) induces falls
# with FORCE (Newtons, sustained over a duration) or a momentum-transfer
# collision, never a raw velocity-scale multiplier pushed into the tens
# of m/s. If escalating up to 3x this starting point still can't clear
# MIN_ACCEPTABLE_FAILURE_RATE, the literature-grounded next step is
# switching mechanisms (apply_body_impulse, force_range up to 850N --
# the self-protective-falling paper's own real induced-fall ceiling),
# not scaling velocity further -- see the RuntimeError this raises below.
VERIFY_ESCALATION_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)
VERIFY_EPISODES = 300  # quick, bounded probe -- not the full collection targets
MIN_ACCEPTABLE_FAILURE_RATE = 0.10  # this project's own established target band (measure_push_failure_rate.py), not literature-sourced
SAFETY_MIN_FAILURE_RATE = 0.02  # below this, the full collection would likely never reach its terminated-episode target at all
TARGET_TRANSITIONS = 1_000_000
TARGET_TERMINATED_EPISODES = 1_000
COLLECT_NUM_ENVS = 4096
MAX_STEPS_SAFETY_CAP = 2_000_000
COLLECT_SEED_OFFSET = 300
PREDICTOR_HORIZON = 16  # matches CLAUDE.md S7.4's cartpole default, reused unchanged for Go1
PREDICTOR_HIDDEN = (128, 128)
SMOKE_TARGET_TERMINATED_EPISODES = 20


def scaled_velocity_range(base_velocity_range: dict[str, tuple[float, float]], scale: float) -> dict[str, tuple[float, float]]:
    return {axis: (lo * scale, hi * scale) for axis, (lo, hi) in base_velocity_range.items()}


def verify_push_scale(checkpoint_path: str, seed: int, device: str, starting_scale: float) -> tuple[float, float]:
    """Quick, bounded probe (VERIFY_EPISODES episodes per candidate, not
    the full collection targets) that measures the REAL failure rate at
    each escalation candidate before committing to the expensive full
    collection run -- the check this script skipped the first time,
    which is why 8x silently produced only 13 falls in 7 minutes instead
    of failing loud or escalating. Reuses
    measure_push_failure_rate.run_one_config directly (same mechanism
    that script's own sweep uses), so this is the same measurement, not a
    reimplementation. Returns (scale, measured_failure_rate) for
    whichever candidate cleared MIN_ACCEPTABLE_FAILURE_RATE first; raises
    if none did even at the escalation ceiling."""
    from cartpole_term.common.backend import get_robot_backend
    from cartpole_term.common.robot_profile import get_profile

    profile = get_profile("go1")
    backend = get_robot_backend("go1")
    base_env_cfg = backend.env_cfg_factory(play=False)
    base_velocity_range = dict(base_env_cfg.events["push_robot"].params["velocity_range"])

    print(
        f"[wild-push][verify] measuring failure rate at {VERIFY_EPISODES} episodes/candidate "
        f"before committing to the full collection run..."
    )
    tried: list[tuple[float, float]] = []
    for mult in VERIFY_ESCALATION_MULTIPLIERS:
        scale = starting_scale * mult
        config_params = {"velocity_range": scaled_velocity_range(base_velocity_range, scale)}
        row = run_one_config(checkpoint_path, profile, backend, seed, VERIFY_EPISODES, device, "velocity", config_params)
        rate = row["failure_rate"]
        tried.append((scale, rate))
        print(f"[wild-push][verify] velocity_scale={scale:.1f}x -> failure_rate={rate:.3f}")
        if rate >= MIN_ACCEPTABLE_FAILURE_RATE:
            return scale, rate

    best_scale, best_rate = max(tried, key=lambda t: t[1])
    if best_rate < SAFETY_MIN_FAILURE_RATE:
        raise RuntimeError(
            f"Every escalation candidate stayed below the safety floor ({SAFETY_MIN_FAILURE_RATE:.0%}) -- "
            f"tried: {tried}. The full collection run would likely never reach its "
            f"{TARGET_TERMINATED_EPISODES}-terminated-episode target in any reasonable time. Not proceeding. "
            "Per this session's literature check (SafeFall arXiv:2511.18509, the self-protective falling "
            "policy arXiv:2512.01336, GYF arXiv:2310.04828), the next step is NOT scaling velocity further -- "
            "every one of those papers induces falls with a sustained FORCE (Newtons) or a momentum-transfer "
            "collision, not a raw velocity multiplier. Wire up apply_body_impulse "
            "(mjlab/envs/mdp/events.py, force_range up to ~850N per the self-protective-falling paper's own "
            "real induced-fall ceiling) instead of escalating this mechanism past its physically-meaningful range."
        )
    print(
        f"[wild-push][verify] WARNING: no candidate cleared the {MIN_ACCEPTABLE_FAILURE_RATE:.0%} target; "
        f"proceeding with the best found (scale={best_scale:.1f}x, rate={best_rate:.3f}) -- "
        "collection may take a while."
    )
    return best_scale, best_rate


GO1_MASS_KG = 12.7434  # confirmed by summing every <geom mass=...> in mjlab's real go1.xml this session
# Reference humanoid mass used to mass-scale the self-protective-falling
# paper's (arXiv:2512.01336) own induced-fall force range (50-850N) down
# to something appropriate for Go1's much smaller body -- SafeFall
# (arXiv:2511.18509, the closest architectural match to this whole
# project's design) validates on "full-scale Unitree G1", whose publicly
# published mass is ~35kg; the self-protective-falling paper's own test
# subject's exact mass wasn't independently confirmed this session, so
# this is a reasoned, transparent estimate (same mass-class of robot),
# not a precise citation match. F=ma: the same force produces far more
# acceleration on a 12.7kg robot than a 35kg one, so copying 50-850N
# verbatim would be proportionally MORE violent than what that paper
# actually validated, not equivalent.
REFERENCE_HUMANOID_MASS_KG = 35.0
LITERATURE_FORCE_RANGE_N = (50.0, 850.0)
MASS_SCALE_RATIO = GO1_MASS_KG / REFERENCE_HUMANOID_MASS_KG
MASS_SCALED_FORCE_RANGE_N = tuple(f * MASS_SCALE_RATIO for f in LITERATURE_FORCE_RANGE_N)  # ~(18, 310) N
# Starting point within that scaled range, biased toward the gentler end
# -- the multi-severity experiment already showed gentler is consistently
# better for both predictor quality AND visual plausibility, so there's
# no reason to start near the top of the range.
DEFAULT_IMPULSE_FORCE_N = 150.0
DEFAULT_IMPULSE_TORQUE_N = 0.0  # zero on purpose: pure linear force at the trunk's own CoM, no added spin -- see module docstring
DEFAULT_IMPULSE_DURATION_S = (0.1, 0.3)
DEFAULT_IMPULSE_COOLDOWN_S = (1.0, 3.0)  # matches push_robot's own shipped interval_range_s


def _run_collection_loop(
    runner,
    env,
    raw_env,
    checkpoint_path: str,
    seed: int,
    num_envs: int,
    target_transitions: int,
    target_terminated: int,
    horizon: int,
    device: str,
) -> tuple[dict, dict]:
    """Shared rollout/labelling/dataset-assembly loop, used by BOTH the
    velocity-teleport mechanism (collect_under_wild_push) and the
    force-impulse mechanism (collect_under_impulse_push) -- identical
    either way once the env is built; only the mechanism-specific fields
    added to the returned sidecar differ between the two callers."""
    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    runner.alg.eval_mode()

    cfg_terminations = raw_env.cfg.terminations
    failure_cause_names = [
        name for name, term_cfg in cfg_terminations.items() if not getattr(term_cfg, "time_out", False)
    ]
    if not failure_cause_names:
        raise RuntimeError(
            "No non-time_out termination terms found on this env cfg -- nothing can ever "
            "emit label=1. Check cartpole_term/go1/env_cfg.py wiring."
        )
    print(f"[wild-push] failure-cause terms: {failure_cause_names}")

    labeler = TerminationLabeler(num_envs=num_envs, horizon=horizon)

    env.seed(seed + COLLECT_SEED_OFFSET)
    obs, _ = env.reset()
    obs = obs.to(device)

    next_episode_id = num_envs
    episode_id = np.arange(num_envs, dtype=np.int64)
    step_in_episode = np.zeros(num_envs, dtype=np.int64)
    cur_episode_len = np.zeros(num_envs, dtype=np.int64)
    episode_lengths: list[int] = []

    records: list[dict] = []
    terminated_episode_count = 0
    total_steps = 0
    total_pushed = 0

    t0 = time.time()
    with torch.inference_mode():
        while len(records) < target_transitions or terminated_episode_count < target_terminated:
            if total_steps > MAX_STEPS_SAFETY_CAP:
                print(
                    f"[wild-push] SAFETY CAP hit ({MAX_STEPS_SAFETY_CAP} steps) before reaching "
                    f"targets (transitions={len(records)}/{target_transitions}, "
                    f"terminated_episodes={terminated_episode_count}/{target_terminated}). Stopping."
                )
                break

            actions = runner.alg.actor(obs, stochastic_output=True)
            obs_actor_np = obs["actor"].detach().cpu().numpy()
            action_np = actions.detach().cpu().numpy()

            next_obs, _rewards, dones, extras = env.step(actions)
            dones_cpu = dones.bool().cpu()
            time_outs = extras.get("time_outs")
            truncated_cpu = time_outs.bool().cpu() if time_outs is not None else torch.zeros_like(dones_cpu)
            terminated_cpu = dones_cpu & ~truncated_cpu

            cause_hits = {name: raw_env.termination_manager.get_term(name).cpu() for name in failure_cause_names}

            payloads = [
                {
                    "obs": obs_actor_np[i],
                    "action": action_np[i],
                    "env_id": i,
                    "episode_id": int(episode_id[i]),
                    "step_in_episode": int(step_in_episode[i]),
                }
                for i in range(num_envs)
            ]
            total_pushed += num_envs
            emitted = labeler.step(payloads, terminated_cpu.tolist(), truncated_cpu.tolist())
            for sample in emitted:
                cause = ""
                if sample.label == 1:
                    env_id = sample.env_id
                    hit_names = sorted(name for name, hits in cause_hits.items() if bool(hits[env_id]))
                    cause = "+".join(hit_names) if hit_names else "unknown"
                records.append(
                    {
                        **sample.payload,
                        "label": sample.label,
                        "steps_to_termination": (
                            sample.steps_to_termination if sample.steps_to_termination is not None else -1
                        ),
                        "termination_cause": cause,
                    }
                )

            terminated_episode_count += int(terminated_cpu.sum().item())
            total_steps += 1

            cur_episode_len += 1
            step_in_episode += 1
            done_np = dones_cpu.numpy()
            if done_np.any():
                idx = np.nonzero(done_np)[0]
                episode_lengths.extend(cur_episode_len[idx].tolist())
                cur_episode_len[idx] = 0
                step_in_episode[idx] = 0
                episode_id[idx] = next_episode_id + np.arange(len(idx))
                next_episode_id += len(idx)

            obs = next_obs.to(device)

            if total_steps % 200 == 0:
                running_base_rate = (
                    sum(1 for r in records if r["label"] == 1) / len(records) if records else 0.0
                )
                print(
                    f"[wild-push] progress: steps={total_steps} transitions={len(records)}/{target_transitions} "
                    f"terminated_episodes={terminated_episode_count}/{target_terminated} "
                    f"running_base_rate={running_base_rate:.3f}"
                )

    wall_time = time.time() - t0

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during frozen-policy collection"

    pending_at_end = sum(labeler.pending_counts())
    terminal_step_discard_count = terminated_episode_count
    censored_count = total_pushed - len(records) - pending_at_end - terminal_step_discard_count

    obs_arr = np.stack([r["obs"] for r in records]).astype(np.float32)
    action_arr = np.stack([r["action"] for r in records]).astype(np.float32)
    label_arr = np.array([r["label"] for r in records], dtype=np.int64)
    env_id_arr = np.array([r["env_id"] for r in records], dtype=np.int64)
    episode_id_arr = np.array([r["episode_id"] for r in records], dtype=np.int64)
    step_in_episode_arr = np.array([r["step_in_episode"] for r in records], dtype=np.int64)
    steps_to_term_arr = np.array([r["steps_to_termination"] for r in records], dtype=np.int64)
    cause_arr = np.array([r["termination_cause"] for r in records], dtype="U32")

    dataset = {
        "obs": obs_arr,
        "action": action_arr,
        "label": label_arr,
        "env_id": env_id_arr,
        "episode_id": episode_id_arr,
        "step_in_episode": step_in_episode_arr,
        "steps_to_termination": steps_to_term_arr,
        "termination_cause": cause_arr,
    }

    base_rate_val = float(label_arr.mean()) if len(label_arr) > 0 else 0.0
    cause_counts = {c: int((cause_arr == c).sum()) for c in sorted(set(cause_arr.tolist())) if c}
    lengths_arr = np.array(episode_lengths, dtype=np.int64)
    sidecar = {
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "collect_seed": seed + COLLECT_SEED_OFFSET,
        "num_envs": num_envs,
        "horizon": horizon,
        "obs_dim": int(obs_arr.shape[1]) if len(obs_arr) else None,
        "action_dim": int(action_arr.shape[1]) if len(action_arr) else None,
        "failure_cause_terms": failure_cause_names,
        "wall_time_s": wall_time,
        "total_env_steps": total_steps,
        "total_transitions_pushed": total_pushed,
        "labelled_transitions": len(records),
        "base_rate": base_rate_val,
        "censored_step_count": int(censored_count),
        "terminal_step_discard_count": int(terminal_step_discard_count),
        "pending_at_end_discarded": int(pending_at_end),
        "episode_count": len(episode_lengths),
        "terminated_episode_count": terminated_episode_count,
        "termination_cause_counts": cause_counts,
        "episode_length_mean": float(lengths_arr.mean()) if len(lengths_arr) else None,
        "episode_length_median": float(np.median(lengths_arr)) if len(lengths_arr) else None,
        "hit_transitions_target": len(records) >= target_transitions,
        "hit_terminated_episodes_target": terminated_episode_count >= target_terminated,
        "hit_safety_cap": total_steps > MAX_STEPS_SAFETY_CAP,
    }

    return dataset, sidecar


def collect_under_wild_push(
    checkpoint_path: str,
    seed: int,
    push_scale: float,
    num_envs: int,
    target_transitions: int,
    target_terminated: int,
    horizon: int,
    device: str,
) -> tuple[dict, dict]:
    """Velocity-teleport mechanism (mjlab's shipped push_robot event,
    scaled). NOT RECOMMENDED past small scales -- confirmed by direct
    visual inspection this session that scaling this mechanism produces
    physically-implausible results ("goes flying") well before it stops
    being effective numerically. See collect_under_impulse_push for the
    literature-grounded alternative (a real, mass-respecting force)."""
    from cartpole_term.go1.env_cfg import TASK_ID, go1_velocity_term_env_cfg
    from cartpole_term.go1.runner import Go1FrozenEvalRunner
    from cartpole_term.runner import build_runner_from_checkpoint

    # CRITICAL: mjlab's EventManager does `self.cfg = deepcopy(cfg)` at
    # __init__ time (confirmed via mjlab/managers/event_manager.py's real
    # source) -- the push override must be applied to the env cfg BEFORE
    # ManagerBasedRlEnv construction, inside the factory itself. Mutating
    # an already-built env's `.cfg.events[...]` afterward has NO EFFECT
    # on the live simulation -- confirmed the hard way in an earlier
    # version of this function.
    shipped_velocity_range: dict = {}

    def wild_env_cfg_factory(play: bool = False):
        cfg = go1_velocity_term_env_cfg(play=play)
        push_cfg = cfg.events["push_robot"]
        base_range = dict(push_cfg.params["velocity_range"])
        shipped_velocity_range.clear()
        shipped_velocity_range.update(base_range)
        push_cfg.params["velocity_range"] = scaled_velocity_range(base_range, push_scale)
        return cfg

    runner = build_runner_from_checkpoint(
        checkpoint_path,
        seed=seed,
        eval_num_envs=num_envs,
        device=device,
        runner_cls=Go1FrozenEvalRunner,
        env_cfg_factory=wild_env_cfg_factory,
        task_id=TASK_ID,
        extra_scalar_metric_fn=None,
    )
    env = runner._build_eval_env()
    raw_env = env.unwrapped

    wild_velocity_range = dict(raw_env.cfg.events["push_robot"].params["velocity_range"])
    print(f"[wild-push] mechanism=velocity, shipped push_robot.velocity_range: {shipped_velocity_range}")
    print(f"[wild-push] mechanism=velocity, wild push_robot.velocity_range (scale={push_scale}x): {wild_velocity_range}")

    dataset, sidecar = _run_collection_loop(
        runner, env, raw_env, checkpoint_path, seed, num_envs, target_transitions, target_terminated, horizon, device
    )
    sidecar["mechanism"] = "velocity"
    sidecar["push_scale"] = push_scale
    sidecar["shipped_velocity_range"] = shipped_velocity_range
    sidecar["wild_velocity_range"] = wild_velocity_range
    return dataset, sidecar


def build_impulse_env_cfg_factory(
    force_n: float,
    torque_n: float,
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
):
    """Factory-builder for the force-impulse mechanism (mjlab's
    apply_body_impulse, confirmed real signature via
    mjlab/envs/mdp/events.py this session) -- a sustained force/torque
    wrench applied to the trunk for a sampled duration, then a cooldown,
    repeating. Respects the robot's actual mass/inertia (F=ma) instead of
    teleporting its velocity, which is WHY every real perturbation-based
    failure paper checked this session (SafeFall, arXiv:2511.18509; the
    self-protective falling policy, arXiv:2512.01336; GYF,
    arXiv:2310.04828) uses this kind of mechanism rather than a raw
    velocity-scale multiplier.

    mjlab's shipped push_robot event (velocity-teleport, mild, unscaled)
    is left completely untouched here, on purpose -- it was never flagged
    as visually problematic at its shipped magnitude, so this impulse is
    layered ON TOP of it as an ADDITIONAL stressor (matching the
    literature's own "combine multiple factors" approach) rather than
    replacing it.

    Extracted to module level (was a nested closure inside
    collect_under_impulse_push) so scripts/go1_collect_flagged_states.py's
    live-rollout lead-time measurement can reuse the exact same push
    mechanism/parameters without duplicating this construction-order-
    sensitive logic."""

    def impulse_env_cfg_factory(play: bool = False):
        from mjlab.envs.mdp.events import apply_body_impulse
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        from cartpole_term.go1.env_cfg import go1_velocity_term_env_cfg

        cfg = go1_velocity_term_env_cfg(play=play)
        # Same construction-order requirement as the velocity mechanism
        # above -- this event term must exist in cfg.events BEFORE
        # ManagerBasedRlEnv construction; apply_body_impulse is a
        # STATEFUL CLASS, instantiated once at env-build time (confirmed
        # via mjlab/managers/event_manager.py's _prepare_terms), so it
        # cannot be added to an already-built env at all, not even via
        # the "mutate params on an existing term" trick the velocity
        # mechanism uses.
        cfg.events["wild_impulse"] = EventTermCfg(
            func=apply_body_impulse,
            mode="step",
            params={
                "force_range": (-force_n, force_n),
                "torque_range": (-torque_n, torque_n),
                "duration_s": duration_s,
                "cooldown_s": cooldown_s,
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
            },
        )
        return cfg

    return impulse_env_cfg_factory


def collect_under_impulse_push(
    checkpoint_path: str,
    seed: int,
    force_n: float,
    torque_n: float,
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    num_envs: int,
    target_transitions: int,
    target_terminated: int,
    horizon: int,
    device: str,
) -> tuple[dict, dict]:
    """Force-impulse mechanism -- see build_impulse_env_cfg_factory for the
    mechanism itself. This function wires it into a runner + the shared
    offline collection/labelling loop (_run_collection_loop)."""
    from cartpole_term.go1.env_cfg import TASK_ID
    from cartpole_term.go1.runner import Go1FrozenEvalRunner
    from cartpole_term.runner import build_runner_from_checkpoint

    impulse_env_cfg_factory = build_impulse_env_cfg_factory(force_n, torque_n, duration_s, cooldown_s)

    runner = build_runner_from_checkpoint(
        checkpoint_path,
        seed=seed,
        eval_num_envs=num_envs,
        device=device,
        runner_cls=Go1FrozenEvalRunner,
        env_cfg_factory=impulse_env_cfg_factory,
        task_id=TASK_ID,
        extra_scalar_metric_fn=None,
    )
    env = runner._build_eval_env()
    raw_env = env.unwrapped

    print(
        f"[wild-push] mechanism=impulse, force_range=({-force_n},{force_n})N, "
        f"torque_range=({-torque_n},{torque_n})Nm, duration_s={duration_s}, cooldown_s={cooldown_s}, "
        f"body=trunk (mass-scaled from arXiv:2512.01336's 50-850N via Go1/G1 mass ratio "
        f"{MASS_SCALE_RATIO:.3f} -> {MASS_SCALED_FORCE_RANGE_N} N reference range)"
    )

    dataset, sidecar = _run_collection_loop(
        runner, env, raw_env, checkpoint_path, seed, num_envs, target_transitions, target_terminated, horizon, device
    )
    sidecar["mechanism"] = "impulse"
    sidecar["force_range_n"] = (-force_n, force_n)
    sidecar["torque_range_nm"] = (-torque_n, torque_n)
    sidecar["duration_s"] = duration_s
    sidecar["cooldown_s"] = cooldown_s
    sidecar["mass_scaled_reference_range_n"] = MASS_SCALED_FORCE_RANGE_N
    return dataset, sidecar


def build_features_generic(obs: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
    features = {f"obs_{i}": obs[:, i] for i in range(obs.shape[1])}
    features.update({f"action_{i}": action[:, i] for i in range(action.shape[1])})
    features["obs_l2norm"] = np.linalg.norm(obs, axis=1)
    features["action_l2norm"] = np.linalg.norm(action, axis=1)
    return features


def feature_screening_generic(features: dict[str, np.ndarray], y: np.ndarray) -> dict[str, dict]:
    results = {}
    has_both = len(np.unique(y)) > 1
    has_pos, has_neg = bool((y == 1).any()), bool((y == 0).any())
    for name, x in features.items():
        class_stats = {
            "positive_mean": float(x[y == 1].mean()) if has_pos else None,
            "negative_mean": float(x[y == 0].mean()) if has_neg else None,
        }
        if not has_both:
            results[name] = {"mcfadden_r2": None, "auc_roc": None, **class_stats}
            continue
        clf = LogisticRegression(max_iter=1000)
        x2 = x.reshape(-1, 1)
        clf.fit(x2, y)
        p = clf.predict_proba(x2)[:, 1]
        results[name] = {"mcfadden_r2": mcfadden_r2(y, p), "auc_roc": float(roc_auc_score(y, p)), **class_stats}
    return results


def train_and_evaluate_predictor(
    dataset: dict, sidecar: dict, seed: int, hidden: tuple[int, int], step_seconds: float, tag: str = ""
) -> dict:
    """`tag` (default "", zero behavior change for this file's own call
    site below) lets a caller retrain on a DERIVED dataset -- e.g.
    scripts/go1_retrain_predictor_history.py's history-stacked version of
    this same collection -- without overwriting the locked H=1 predictor's
    output files."""
    from cartpole_term.predictor.model import TerminationPredictor
    from cartpole_term.predictor.split import split_episodes
    from cartpole_term.predictor.train import train_predictor

    train, val, test = split_dataset_by_split_fn(dataset, split_episodes, seed=seed)
    for name, split in (("train", train), ("val", val), ("test", test)):
        if len(split["label"]) == 0:
            raise ValueError(f"{name} split is empty -- dataset too small.")

    x_train, y_train = make_inputs(train), train["label"].astype(np.float32)
    x_val, y_val = make_inputs(val), val["label"].astype(np.float32)
    x_test, y_test = make_inputs(test), test["label"].astype(np.float32)
    obs_action_dim = x_train.shape[1]

    torch.manual_seed(seed)
    model = TerminationPredictor(history=1, hidden=hidden, obs_action_dim=obs_action_dim)
    result = train_predictor(model, x_train, y_train, x_val, y_val, device="cpu", seed=seed)
    model.eval()

    model_save_path = Path("outputs") / f"go1_wildpush_predictor{tag}_seed{seed}.pt"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    with torch.no_grad():
        p_val_mlp = torch.sigmoid(model(torch.as_tensor(x_val)).squeeze(-1)).numpy()
        p_test_mlp = torch.sigmoid(model(torch.as_tensor(x_test)).squeeze(-1)).numpy()
    threshold_mlp = best_threshold_by_f1(y_val, p_val_mlp)
    mlp_metrics = classification_metrics(y_test, p_test_mlp, threshold_mlp)
    mlp_metrics["best_val_epoch"] = result.best_epoch
    mlp_metrics["best_val_auprc"] = result.best_val_auprc
    mlp_lead_time = median_detection_lead_time(
        test["episode_id"], test["steps_to_termination"], y_test, p_test_mlp, threshold_mlp, step_seconds=step_seconds
    )

    baselines: dict[str, dict] = {}
    baselines["constant_negative"] = classification_metrics(y_test, np.zeros_like(y_test), threshold=0.5)
    train_base_rate = base_rate(y_train)
    baselines["base_rate_constant"] = classification_metrics(
        y_test, np.full_like(y_test, train_base_rate), threshold=train_base_rate
    )

    logreg_all = LogisticRegression(max_iter=1000)
    logreg_all.fit(x_train, y_train)
    p_val_logreg = logreg_all.predict_proba(x_val)[:, 1]
    p_test_logreg = logreg_all.predict_proba(x_test)[:, 1]
    t_logreg = best_threshold_by_f1(y_val, p_val_logreg)
    baselines["logistic_regression_all_inputs"] = classification_metrics(y_test, p_test_logreg, t_logreg)

    features_train = build_features_generic(train["obs"], train["action"])
    screening = feature_screening_generic(features_train, y_train)
    best_feature_name = max(
        (n for n, r in screening.items() if r["auc_roc"] is not None), key=lambda n: screening[n]["auc_roc"], default=None
    )

    per_cause: dict[str, dict] = {}
    cause_names = sorted({c for c in test["termination_cause"].tolist() if c})
    for cause in cause_names:
        cause_mask = test["termination_cause"] == cause
        neg_mask = y_test == 0
        keep = cause_mask | neg_mask
        y_c, p_c = y_test[keep], p_test_mlp[keep]
        per_cause[cause] = {
            "n_positive": int(cause_mask.sum()),
            "auprc": float(average_precision_score(y_c, p_c)) if len(np.unique(y_c)) > 1 else None,
        }

    return {
        "collection_sidecar": sidecar,
        "obs_action_dim": obs_action_dim,
        "predictor_hidden": hidden,
        "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "split_base_rates": {"train": base_rate(y_train), "val": base_rate(y_val), "test": base_rate(y_test)},
        "mlp": mlp_metrics,
        "mlp_lead_time": mlp_lead_time,
        "baselines": baselines,
        "feature_screening": screening,
        "best_feature": best_feature_name,
        "per_cause_breakdown": per_cause,
        "model_path": str(model_save_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mechanism", choices=("impulse", "velocity"), default="impulse",
        help="'impulse' (default, recommended): a real force/torque wrench (mjlab's apply_body_impulse), "
        "respects the robot's mass/inertia, matches every perturbation-based failure paper checked this "
        "session. 'velocity': the original instantaneous velocity-teleport scaling -- confirmed by direct "
        "visual inspection to produce physically-implausible results ('goes flying') at the scales needed "
        "to induce a reasonable failure rate. Kept only for comparison/debugging.",
    )
    parser.add_argument("--push-scale", dest="push_scale", type=float, default=WILD_PUSH_SCALE, help="velocity mechanism only.")
    parser.add_argument(
        "--impulse-force", dest="impulse_force", type=float, default=DEFAULT_IMPULSE_FORCE_N,
        help=f"impulse mechanism only, Newtons. Mass-scaled reference range from arXiv:2512.01336's own "
        f"induced-fall force (50-850N) via Go1/G1 mass ratio: {MASS_SCALED_FORCE_RANGE_N} N.",
    )
    parser.add_argument("--impulse-torque", dest="impulse_torque", type=float, default=DEFAULT_IMPULSE_TORQUE_N, help="impulse mechanism only, Nm. 0.0 default = pure linear force, no added spin.")
    parser.add_argument("--predictor-hidden", dest="predictor_hidden", type=str, default="128,128")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-verify", dest="skip_verify", action="store_true",
        help="velocity mechanism only: skip the pre-collection failure-rate verification and use "
        "--push-scale exactly as given. Not recommended for that mechanism.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    h1, h2 = (int(p) for p in args.predictor_hidden.split(","))
    hidden = (h1, h2)

    num_envs = SMOKE_NUM_ENVS if args.smoke else COLLECT_NUM_ENVS
    target_transitions = SMOKE_COLLECT_TRANSITIONS if args.smoke else TARGET_TRANSITIONS
    target_terminated = SMOKE_TARGET_TERMINATED_EPISODES if args.smoke else TARGET_TERMINATED_EPISODES

    if args.mechanism == "impulse":
        print(
            f"[wild-push] mechanism=impulse: force=+-{args.impulse_force}N, torque=+-{args.impulse_torque}Nm "
            f"(mass-scaled reference range {MASS_SCALED_FORCE_RANGE_N} N -- no automated verification ladder "
            "for this mechanism yet, watch the printed running_base_rate as collection starts)"
        )
        dataset, sidecar = collect_under_impulse_push(
            checkpoint_path=args.checkpoint_file,
            seed=args.seed,
            force_n=args.impulse_force,
            torque_n=args.impulse_torque,
            duration_s=DEFAULT_IMPULSE_DURATION_S,
            cooldown_s=DEFAULT_IMPULSE_COOLDOWN_S,
            num_envs=num_envs,
            target_transitions=target_transitions,
            target_terminated=target_terminated,
            horizon=PREDICTOR_HORIZON,
            device=device,
        )
    else:
        if args.skip_verify:
            verified_scale = args.push_scale
            print(f"[wild-push] --skip-verify set: using push_scale={verified_scale}x with no pre-check")
        else:
            verified_scale, verified_rate = verify_push_scale(
                checkpoint_path=args.checkpoint_file, seed=args.seed, device=device, starting_scale=args.push_scale
            )
            print(f"[wild-push] verified push_scale={verified_scale:.1f}x (measured failure_rate={verified_rate:.3f}) -- proceeding to full collection")

        print(f"[wild-push] loading checkpoint {args.checkpoint_file}, push_scale={verified_scale}x")
        dataset, sidecar = collect_under_wild_push(
            checkpoint_path=args.checkpoint_file,
            seed=args.seed,
            push_scale=verified_scale,
            num_envs=num_envs,
            target_transitions=target_transitions,
            target_terminated=target_terminated,
            horizon=PREDICTOR_HORIZON,
            device=device,
        )

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    npz_path = outputs_dir / f"go1_wildpush_frozen_seed{args.seed}.npz"
    json_path = outputs_dir / f"go1_wildpush_frozen_seed{args.seed}.json"
    np.savez_compressed(npz_path, **dataset)
    json_path.write_text(json.dumps(sidecar, indent=2))
    print(
        f"[wild-push] collected {sidecar['labelled_transitions']} labelled transitions, "
        f"base_rate={sidecar['base_rate']:.3f}, terminated_episodes={sidecar['terminated_episode_count']}, "
        f"wall_time={sidecar['wall_time_s']:.1f}s"
    )
    print(f"  wrote {npz_path}")
    print(f"  wrote {json_path}")

    step_seconds = 0.02  # Go1: decimation=4, timestep=0.005 -> 50 Hz, confirmed live this session
    metrics = train_and_evaluate_predictor(dataset, sidecar, seed=args.seed, hidden=hidden, step_seconds=step_seconds)

    metrics_path = outputs_dir / f"go1_wildpush_metrics_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(
        f"[wild-push] predictor: test_base_rate={metrics['split_base_rates']['test']:.4f} "
        f"AUPRC={metrics['mlp']['auprc']:.4f} (baseline={metrics['mlp']['auprc_baseline']:.4f}) "
        f"AUROC={metrics['mlp']['auroc']:.4f} "
        f"median_lead={metrics['mlp_lead_time']['median_lead_steps']:.1f} steps "
        f"({metrics['mlp_lead_time']['median_lead_seconds']:.2f}s) "
        f"best_feature={metrics['best_feature']}"
    )
    print(f"  wrote {metrics_path}")
    print(f"  wrote {metrics['model_path']}")


if __name__ == "__main__":
    main()
