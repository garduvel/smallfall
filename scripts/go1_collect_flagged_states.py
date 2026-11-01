"""New, standalone script (phase-2, outside CLAUDE.md's cartpole-only
spec, see DEVIATIONS.md). Feeds Gate 5 of the multi-robot plan
(~/.claude/plans/snug-booping-gosling.md) -- the protective "small-fall"
policy's rollout-then-handoff initial-state mechanism needs "the real
predictor-positive window measured in Gate 4", and this script is what
measures it under a REAL live rollout rather than reading it off the
already-collected offline frozen dataset.

Important distinction from the existing `mlp_lead_time` figure already in
outputs/go1_wildpush_metrics_seed0.json (median=12 steps): that number is
computed POST HOC from a frozen, 16-step-horizon-censored dataset
(scripts/eval_predictor.py's median_detection_lead_time) -- it can only
ever report a lead time inside the training horizon, because any step
more than 16 steps before termination was never labelled positive in the
first place, whether or not the predictor was actually confident there.

This script instead runs the SAME locked checkpoint + SAME locked push
mechanism (impulse, 150N, 0Nm -- reused verbatim from
scripts/go1_wild_push_predictor.py's build_impulse_env_cfg_factory, not
duplicated) live, step by step, with the trained predictor scoring every
step online. For each episode it records the FIRST step at which the
predictor's probability crosses the val-selected decision threshold
(from outputs/go1_wildpush_metrics_seed0.json) -- that's the "frozen
state" the user asked for (obs, action, probability, step, episode id) --
then keeps rolling that same episode forward, unmodified, until it
actually ends. If it ends in a real fall (`terminated`), the actual lead
time is `termination_step - detection_step`, with NO 16-step ceiling.
This is a strictly online measurement: nothing here was labelled by
looking into the future beyond "did the episode terminate."

Episodes that truncate (time_out) after having been flagged are censored
(discarded from the lead-time stat) for the same reason CLAUDE.md S7.4
discards truncated tails elsewhere in this project: the counterfactual
"would it have fallen anyway" is unknown. Episodes that terminate WITHOUT
ever crossing the threshold are counted separately as missed detections
(a live recall diagnostic) -- not silently dropped.

**`--capture-state-bank` (Gate 5 feed):** when set, for every resolved
lead-time trial, also captures the robot's REAL raw physics state (root
pose/velocity + joint pos/velocity -- confirmed real mjlab Entity APIs
this session: write_root_link_pose_to_sim/write_root_link_velocity_to_sim/
write_joint_state_to_sim, src/mjlab/entity/entity.py) at a step drawn
UNIFORMLY AT RANDOM from [flag_step, termination_step) via reservoir
sampling (standard algorithm: at step k since flagging, replace the held
candidate with probability 1/k -- no need to know termination_step in
advance). This directly supersedes the multi-robot plan's Gate 5 "rollout-
then-handoff" design, which was chosen specifically because literal state
injection was believed to need "an unconfirmed mjlab 'set full state
directly' API" -- that API is now confirmed real and used here. See
DEVIATIONS.md. Output: outputs/go1_fall_state_bank_seed{N}.npz.

Run on the GPU machine, from the repo root, after
scripts/go1_wild_push_predictor.py has produced both the frozen checkpoint
and the trained predictor:

    python scripts/go1_collect_flagged_states.py \
        --checkpoint-file checkpoints/model_9999.pt \
        --predictor-file outputs/go1_wildpush_predictor_seed0.pt \
        --metrics-file outputs/go1_wildpush_metrics_seed0.json
    python scripts/go1_collect_flagged_states.py --checkpoint-file checkpoints/model_9999.pt --smoke
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from cartpole_term.cli import SMOKE_NUM_ENVS  # noqa: E402
from cartpole_term.common.deferred_reset import DeferredResetEnv  # noqa: E402
from go1_wild_push_predictor import (  # noqa: E402
    DEFAULT_IMPULSE_COOLDOWN_S,
    DEFAULT_IMPULSE_DURATION_S,
    DEFAULT_IMPULSE_FORCE_N,
    DEFAULT_IMPULSE_TORQUE_N,
    build_impulse_env_cfg_factory,
)

STEP_SECONDS = 0.02  # Go1: decimation=4, timestep=0.005 -> 50 Hz, confirmed live this session (go1_wild_push_predictor.py)
COLLECT_SEED_OFFSET = 400  # distinct from go1_wild_push_predictor.py's COLLECT_SEED_OFFSET (300) -- different rollout, must not reuse identical trajectories
DEFAULT_NUM_ENVS = 4096
DEFAULT_TARGET_TRIALS = 500  # episodes with a confirmed flag-then-real-fall pair
MAX_STEPS_SAFETY_CAP = 2_000_000
SMOKE_TARGET_TRIALS = 20
# Post-termination hold window, in env steps (50 Hz -> 0.6s) -- see
# src/cartpole_term/common/deferred_reset.py. Grounded by analogy to
# SafeFall's (arXiv:2511.18509) ~40-step recovery-training episode length,
# not an exact citation match (that paper doesn't use this exact mechanic).
# 0 disables the whole hold/DeferredResetEnv pathway; the loop then behaves
# identically to the pre-hold version except that resets happen one wrapper-
# step later (see DeferredResetEnv's documented timing tradeoff) -- this
# does not affect lead-time correctness since lead time is computed at the
# moment of termination detection, before any reset (deferred or not).
DEFAULT_HOLD_POST_FALL_STEPS = 30


def _load_predictor(predictor_file: str, obs_action_dim: int, hidden: tuple[int, int], device: str):
    from cartpole_term.predictor.model import TerminationPredictor

    model = TerminationPredictor(history=1, hidden=hidden, obs_action_dim=obs_action_dim)
    state_dict = torch.load(predictor_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_flagged_states(
    checkpoint_path: str,
    predictor_file: str,
    obs_action_dim: int,
    predictor_hidden: tuple[int, int],
    threshold: float,
    seed: int,
    force_n: float,
    torque_n: float,
    num_envs: int,
    target_trials: int,
    hold_post_fall_steps: int,
    capture_state_bank: bool,
    device: str,
    capture_at_flag: bool = False,
) -> tuple[dict, dict, dict | None]:
    from cartpole_term.go1.env_cfg import TASK_ID
    from cartpole_term.go1.runner import Go1FrozenEvalRunner
    from cartpole_term.runner import build_runner_from_checkpoint

    base_impulse_factory = build_impulse_env_cfg_factory(
        force_n, torque_n, DEFAULT_IMPULSE_DURATION_S, DEFAULT_IMPULSE_COOLDOWN_S
    )

    def impulse_env_cfg_factory(play: bool = False):
        # auto_reset=False always, so DeferredResetEnv can hold a fallen env's
        # physics running past termination instead of mjlab snapping it back
        # to a fresh reset before this script ever observes the fall. See
        # src/cartpole_term/common/deferred_reset.py.
        cfg = base_impulse_factory(play=play)
        cfg.auto_reset = False
        return cfg

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
    deferred = DeferredResetEnv(raw_env)

    predictor = _load_predictor(predictor_file, obs_action_dim, predictor_hidden, device)

    print(
        f"[flagged-states] mechanism=impulse, force=+-{force_n}N, torque=+-{torque_n}Nm, "
        f"threshold={threshold:.4f}, target_trials={target_trials}, "
        f"hold_post_fall_steps={hold_post_fall_steps}"
    )

    actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    predictor_snapshot = {k: v.clone() for k, v in predictor.state_dict().items()}
    runner.alg.eval_mode()

    cfg_terminations = raw_env.cfg.terminations
    failure_cause_names = [
        name for name, term_cfg in cfg_terminations.items() if not getattr(term_cfg, "time_out", False)
    ]
    if not failure_cause_names:
        raise RuntimeError(
            "No non-time_out termination terms found on this env cfg -- nothing can ever be a real fall."
        )
    print(f"[flagged-states] failure-cause terms: {failure_cause_names}")

    raw_env.seed(seed + COLLECT_SEED_OFFSET)
    obs_dict, _ = deferred.reset()
    # raw ManagerBasedRlEnv.reset()/step() return a plain dict, not the
    # TensorDict runner.alg.actor() expects -- confirmed via mjlab's real
    # RslRlVecEnvWrapper source (src/mjlab/rl/vecenv_wrapper.py), which does
    # exactly this wrapping itself: TensorDict(obs_dict, batch_size=[num_envs]).
    # Reproduced here instead of bypassing it, so actor() sees the identical
    # input type it always does -- tensors are already on raw_env's own
    # device (confirmed "cuda" in this env's printed table), no .to() needed.
    obs = TensorDict(obs_dict, batch_size=[num_envs])

    next_episode_id = num_envs
    episode_id = np.arange(num_envs, dtype=np.int64)
    step_in_episode = np.zeros(num_envs, dtype=np.int64)
    flagged = np.zeros(num_envs, dtype=bool)
    flag_step = np.full(num_envs, -1, dtype=np.int64)
    held = np.zeros(num_envs, dtype=bool)  # currently in a post-fall hold window (DeferredResetEnv)
    post_fall_remaining = np.zeros(num_envs, dtype=np.int64)

    flagged_records: list[dict] = []  # one per episode that ever crossed threshold; kept until resolved
    lead_time_trials: list[dict] = []  # resolved: flagged episode that then really terminated
    missed_terminations = 0  # terminated but never crossed threshold
    censored_after_flag = 0  # truncated after flagging -- discard, counterfactual unknown

    held_trajectories: dict[int, dict] = {}  # episode_id -> {"obs": [...], "action": [...]}
    pending_trial_index_by_eid: dict[int, int] = {}  # episode_id -> index into lead_time_trials, while held

    if capture_state_bank:
        raw_asset = raw_env.scene["robot"]
        num_joints = raw_asset.data.joint_pos.shape[1]
        steps_since_flag = np.zeros(num_envs, dtype=np.int64)
        # Reservoir-sampled candidate state per env, live physics-frame values
        # (root position is env-LOCAL, i.e. with this env's own origin already
        # subtracted -- see the save-time re-basing note below and
        # mjlab/envs/mdp/events.py's reset_root_state_uniform, which does the
        # equivalent add-back at injection time).
        candidate_root_pos_local = torch.zeros(num_envs, 3, device=device)
        candidate_root_quat = torch.zeros(num_envs, 4, device=device)
        candidate_root_lin_vel = torch.zeros(num_envs, 3, device=device)
        candidate_root_ang_vel = torch.zeros(num_envs, 3, device=device)
        candidate_joint_pos = torch.zeros(num_envs, num_joints, device=device)
        candidate_joint_vel = torch.zeros(num_envs, num_joints, device=device)
        candidate_step_in_episode = np.zeros(num_envs, dtype=np.int64)
        state_bank_records: list[dict] = []

    total_steps = 0
    t0 = time.time()
    with torch.inference_mode():
        while len(lead_time_trials) < target_trials:
            if total_steps > MAX_STEPS_SAFETY_CAP:
                print(
                    f"[flagged-states] SAFETY CAP hit ({MAX_STEPS_SAFETY_CAP} steps) before reaching "
                    f"target_trials={target_trials} (got {len(lead_time_trials)}). Stopping."
                )
                break

            actions = runner.alg.actor(obs, stochastic_output=True)
            obs_actor_np = obs["actor"].detach().cpu().numpy()
            action_np = actions.detach().cpu().numpy()

            x = torch.as_tensor(
                np.concatenate([obs_actor_np, action_np], axis=1).astype(np.float32), device=device
            )
            prob_np = torch.sigmoid(predictor(x).squeeze(-1)).detach().cpu().numpy()

            # held envs are already-resolved falls (predictor still scores them,
            # harmlessly, but must never be treated as a fresh detection).
            newly_flagged = (~flagged) & (~held) & (prob_np > threshold)
            if newly_flagged.any():
                idx = np.nonzero(newly_flagged)[0]
                flagged[idx] = True
                flag_step[idx] = step_in_episode[idx]
                for i in idx:
                    flagged_records.append(
                        {
                            "episode_id": int(episode_id[i]),
                            "flag_step": int(step_in_episode[i]),
                            "obs": obs_actor_np[i].copy(),
                            "action": action_np[i].copy(),
                            "probability_at_flag": float(prob_np[i]),
                        }
                    )

            # Record this step's (obs, action) for envs currently in their
            # post-fall hold window, BEFORE stepping -- this is the actual
            # post-termination trajectory the wrapper exists to expose.
            if held.any():
                for i in np.nonzero(held)[0]:
                    traj = held_trajectories[int(episode_id[i])]
                    traj["obs"].append(obs_actor_np[i].copy())
                    traj["action"].append(action_np[i].copy())

            # Reservoir-sample one raw-physics-state candidate per still-open
            # (flagged, not yet resolved/held) episode -- BEFORE stepping, so
            # this is the physics state that PRODUCED this step's obs/action,
            # not the state after this step's action has been applied.
            if capture_state_bank:
                active_flagged = flagged & (~held)
                if active_flagged.any():
                    idx = np.nonzero(active_flagged)[0]
                    steps_since_flag[idx] += 1
                    replace_prob = 1.0 / steps_since_flag[idx].astype(np.float64)
                    if capture_at_flag:
                        # Gate 7: the state at the flag step itself (the hand-off
                        # moment), not a uniform draw from [flag, fall).
                        replace_prob = (steps_since_flag[idx] == 1).astype(np.float64)
                    replace_mask_np = np.random.random_sample(len(idx)) < replace_prob
                    if replace_mask_np.any():
                        replace_idx = idx[replace_mask_np]
                        replace_idx_t = torch.as_tensor(replace_idx, device=device, dtype=torch.long)
                        env_origins = raw_env.scene.env_origins[replace_idx_t]
                        candidate_root_pos_local[replace_idx_t] = (
                            raw_asset.data.root_link_pos_w[replace_idx_t] - env_origins
                        )
                        candidate_root_quat[replace_idx_t] = raw_asset.data.root_link_quat_w[replace_idx_t]
                        candidate_root_lin_vel[replace_idx_t] = raw_asset.data.root_link_lin_vel_w[replace_idx_t]
                        candidate_root_ang_vel[replace_idx_t] = raw_asset.data.root_link_ang_vel_w[replace_idx_t]
                        candidate_joint_pos[replace_idx_t] = raw_asset.data.joint_pos[replace_idx_t]
                        candidate_joint_vel[replace_idx_t] = raw_asset.data.joint_vel[replace_idx_t]
                        candidate_step_in_episode[replace_idx] = step_in_episode[replace_idx]

            next_obs, _rewards, terminated, truncated, extras = deferred.step(actions)
            terminated_cpu = terminated.bool().cpu()
            truncated_cpu = truncated.bool().cpu()
            done_cpu = terminated_cpu | truncated_cpu

            cause_hits = {name: raw_env.termination_manager.get_term(name).cpu() for name in failure_cause_names}

            done_np = done_cpu.numpy()
            terminated_np = terminated_cpu.numpy()
            truncated_np = truncated_cpu.numpy()

            newly_held_ids: list[int] = []
            if done_np.any():
                idx = np.nonzero(done_np)[0]
                pending_by_episode = {r["episode_id"]: r for r in flagged_records}
                for i in idx:
                    if held[i]:
                        # Fall already resolved and currently held past termination --
                        # this is just the same terminated/truncated condition
                        # continuing to read True. Not a new event; countdown below
                        # handles it.
                        continue

                    eid = int(episode_id[i])
                    start_hold = False
                    if terminated_np[i]:
                        if flagged[i]:
                            term_step = int(step_in_episode[i])
                            lead = term_step - int(flag_step[i])
                            hit_names = sorted(name for name, hits in cause_hits.items() if bool(hits[i]))
                            cause = "+".join(hit_names) if hit_names else "unknown"
                            lead_time_trials.append(
                                {
                                    "episode_id": eid,
                                    "flag_step": int(flag_step[i]),
                                    "termination_step": term_step,
                                    "lead_steps": lead,
                                    "termination_cause": cause,
                                    "probability_at_flag": pending_by_episode[eid]["probability_at_flag"],
                                    "obs": pending_by_episode[eid]["obs"],
                                    "action": pending_by_episode[eid]["action"],
                                }
                            )
                            start_hold = hold_post_fall_steps > 0
                            if capture_state_bank:
                                # steps_since_flag[i] >= 1 is guaranteed: the flag
                                # step itself always takes one reservoir sample
                                # (module note above) before this env can resolve.
                                state_bank_records.append(
                                    {
                                        "episode_id": eid,
                                        "flag_step": int(flag_step[i]),
                                        "termination_step": term_step,
                                        "captured_step": int(candidate_step_in_episode[i]),
                                        "termination_cause": cause,
                                        "root_pos_local": candidate_root_pos_local[i].detach().cpu().numpy().copy(),
                                        "root_quat": candidate_root_quat[i].detach().cpu().numpy().copy(),
                                        "root_lin_vel": candidate_root_lin_vel[i].detach().cpu().numpy().copy(),
                                        "root_ang_vel": candidate_root_ang_vel[i].detach().cpu().numpy().copy(),
                                        "joint_pos": candidate_joint_pos[i].detach().cpu().numpy().copy(),
                                        "joint_vel": candidate_joint_vel[i].detach().cpu().numpy().copy(),
                                    }
                                )
                        else:
                            missed_terminations += 1
                    elif truncated_np[i] and flagged[i]:
                        censored_after_flag += 1

                    flagged[i] = False
                    flag_step[i] = -1
                    if capture_state_bank:
                        steps_since_flag[i] = 0

                    if start_hold:
                        # Defer the episode-id/step-counter reset until the hold
                        # window actually ends -- this env's physics keeps running.
                        held[i] = True
                        post_fall_remaining[i] = hold_post_fall_steps
                        held_trajectories[eid] = {"obs": [], "action": []}
                        pending_trial_index_by_eid[eid] = len(lead_time_trials) - 1
                        newly_held_ids.append(i)
                    else:
                        step_in_episode[i] = -1  # incremented back to 0 below
                        episode_id[i] = next_episode_id
                        next_episode_id += 1

                still_pending_eids = {int(episode_id[i]) for i in range(num_envs) if flagged[i]}
                flagged_records = [r for r in flagged_records if r["episode_id"] in still_pending_eids]

            if newly_held_ids:
                deferred.hold(torch.as_tensor(newly_held_ids, device=deferred.held.device, dtype=torch.long))

            if held.any():
                post_fall_remaining[held] -= 1
                finished = held & (post_fall_remaining <= 0)
                if finished.any():
                    finished_idx = np.nonzero(finished)[0]
                    for i in finished_idx:
                        eid = int(episode_id[i])
                        traj = held_trajectories.pop(eid)
                        trial_idx = pending_trial_index_by_eid.pop(eid)
                        lead_time_trials[trial_idx]["post_fall_obs"] = np.stack(traj["obs"]).astype(np.float32)
                        lead_time_trials[trial_idx]["post_fall_action"] = np.stack(traj["action"]).astype(np.float32)
                        held[i] = False
                        post_fall_remaining[i] = 0
                        step_in_episode[i] = -1
                        episode_id[i] = next_episode_id
                        next_episode_id += 1
                    deferred.release(torch.as_tensor(finished_idx, device=deferred.held.device, dtype=torch.long))

            step_in_episode += 1
            total_steps += 1
            obs = TensorDict(next_obs, batch_size=[num_envs])  # see reset() above

            if total_steps % 200 == 0:
                print(
                    f"[flagged-states] progress: steps={total_steps} "
                    f"lead_time_trials={len(lead_time_trials)}/{target_trials} "
                    f"missed={missed_terminations} censored_after_flag={censored_after_flag} "
                    f"currently_held={int(held.sum())}"
                )

    wall_time = time.time() - t0

    for name, before in actor_snapshot.items():
        after = runner.alg.actor.state_dict()[name]
        assert torch.equal(before, after), f"actor.{name} changed during frozen-policy rollout"
    for name, before in predictor_snapshot.items():
        after = predictor.state_dict()[name]
        assert torch.equal(before, after), f"predictor.{name} changed during frozen-policy rollout"

    lead_steps_arr = np.array([t["lead_steps"] for t in lead_time_trials], dtype=np.int64)
    lead_seconds_arr = lead_steps_arr.astype(np.float64) * STEP_SECONDS

    per_cause: dict[str, dict] = {}
    for cause in sorted({t["termination_cause"] for t in lead_time_trials}):
        cause_leads = np.array([t["lead_steps"] for t in lead_time_trials if t["termination_cause"] == cause])
        per_cause[cause] = {
            "n": int(len(cause_leads)),
            "mean_lead_steps": float(cause_leads.mean()) if len(cause_leads) else None,
            "median_lead_steps": float(np.median(cause_leads)) if len(cause_leads) else None,
        }

    action_dim = action_np.shape[1]
    obs_dim = obs_action_dim - action_dim

    # Post-fall trajectories (see DeferredResetEnv): only present for trials whose
    # hold window actually finished before the loop stopped. A trial whose hold
    # was still in progress when target_trials/the safety cap was hit is kept in
    # the lead-time stats above (unaffected) but has no post-fall data here.
    post_fall_trial_indices = [idx for idx, t in enumerate(lead_time_trials) if "post_fall_obs" in t]
    if hold_post_fall_steps > 0 and post_fall_trial_indices:
        post_fall_obs_arr = np.stack([lead_time_trials[idx]["post_fall_obs"] for idx in post_fall_trial_indices])
        post_fall_action_arr = np.stack([lead_time_trials[idx]["post_fall_action"] for idx in post_fall_trial_indices])
    else:
        post_fall_obs_arr = np.zeros((0, max(hold_post_fall_steps, 1), obs_dim), dtype=np.float32)
        post_fall_action_arr = np.zeros((0, max(hold_post_fall_steps, 1), action_dim), dtype=np.float32)
    post_fall_trial_index_arr = np.array(post_fall_trial_indices, dtype=np.int64)

    dataset = {
        "obs": np.stack([t["obs"] for t in lead_time_trials]).astype(np.float32) if lead_time_trials else np.zeros((0, obs_dim), dtype=np.float32),
        "action": np.stack([t["action"] for t in lead_time_trials]).astype(np.float32) if lead_time_trials else np.zeros((0, action_dim), dtype=np.float32),
        "episode_id": np.array([t["episode_id"] for t in lead_time_trials], dtype=np.int64),
        "flag_step": np.array([t["flag_step"] for t in lead_time_trials], dtype=np.int64),
        "termination_step": np.array([t["termination_step"] for t in lead_time_trials], dtype=np.int64),
        "lead_steps": lead_steps_arr,
        "probability_at_flag": np.array([t["probability_at_flag"] for t in lead_time_trials], dtype=np.float32),
        "termination_cause": np.array([t["termination_cause"] for t in lead_time_trials], dtype="U32"),
        # post_fall_obs/action[k] is the (hold_post_fall_steps, dim) continuation
        # trajectory for trial row post_fall_trial_index[k] -- i.e. what actually
        # happened physically AFTER termination fired, under the (still nominal,
        # not-yet-protective) policy. This is real Gate-5 seed data once a
        # protective policy exists to compare against.
        "post_fall_obs": post_fall_obs_arr,
        "post_fall_action": post_fall_action_arr,
        "post_fall_trial_index": post_fall_trial_index_arr,
    }

    total_resolved = len(lead_time_trials) + missed_terminations
    sidecar = {
        "checkpoint_path": checkpoint_path,
        "predictor_file": predictor_file,
        "obs_action_dim": obs_action_dim,
        "predictor_hidden": list(predictor_hidden),
        "threshold": threshold,
        "seed": seed,
        "collect_seed": seed + COLLECT_SEED_OFFSET,
        "num_envs": num_envs,
        "mechanism": "impulse",
        "force_range_n": (-force_n, force_n),
        "torque_range_nm": (-torque_n, torque_n),
        "target_trials": target_trials,
        "wall_time_s": wall_time,
        "total_env_steps": total_steps,
        "num_lead_time_trials": len(lead_time_trials),
        "num_missed_terminations": missed_terminations,
        "num_censored_after_flag": censored_after_flag,
        "live_recall": (
            float(len(lead_time_trials) / total_resolved) if total_resolved > 0 else None
        ),  # fraction of REAL falls that were ever flagged beforehand, no 16-step ceiling
        "mean_lead_steps": float(lead_steps_arr.mean()) if len(lead_steps_arr) else None,
        "median_lead_steps": float(np.median(lead_steps_arr)) if len(lead_steps_arr) else None,
        "mean_lead_seconds": float(lead_seconds_arr.mean()) if len(lead_seconds_arr) else None,
        "median_lead_seconds": float(np.median(lead_seconds_arr)) if len(lead_seconds_arr) else None,
        "std_lead_steps": float(lead_steps_arr.std()) if len(lead_steps_arr) else None,
        "min_lead_steps": int(lead_steps_arr.min()) if len(lead_steps_arr) else None,
        "max_lead_steps": int(lead_steps_arr.max()) if len(lead_steps_arr) else None,
        "per_cause_breakdown": per_cause,
        "step_seconds": STEP_SECONDS,
        "hit_target_trials": len(lead_time_trials) >= target_trials,
        "hit_safety_cap": total_steps > MAX_STEPS_SAFETY_CAP,
        "hold_post_fall_steps": hold_post_fall_steps,
        "num_post_fall_trajectories_captured": len(post_fall_trial_indices),
        "num_hold_windows_incomplete_at_stop": len(pending_trial_index_by_eid),
        "capture_state_bank": capture_state_bank,
        "capture_at_flag": capture_at_flag,
    }

    state_bank = None
    if capture_state_bank:
        state_bank = {
            "episode_id": np.array([r["episode_id"] for r in state_bank_records], dtype=np.int64),
            "flag_step": np.array([r["flag_step"] for r in state_bank_records], dtype=np.int64),
            "termination_step": np.array([r["termination_step"] for r in state_bank_records], dtype=np.int64),
            "captured_step": np.array([r["captured_step"] for r in state_bank_records], dtype=np.int64),
            "termination_cause": np.array([r["termination_cause"] for r in state_bank_records], dtype="U32"),
            # root_pos_local is relative to THIS collection's own env origin --
            # re-add the INJECTING env's own env_origins at use time (see
            # mjlab/envs/mdp/events.py's reset_root_state_uniform for the
            # identical pattern), never reuse this collection's origins directly.
            "root_pos_local": np.stack([r["root_pos_local"] for r in state_bank_records]).astype(np.float32)
            if state_bank_records
            else np.zeros((0, 3), dtype=np.float32),
            "root_quat": np.stack([r["root_quat"] for r in state_bank_records]).astype(np.float32)
            if state_bank_records
            else np.zeros((0, 4), dtype=np.float32),
            "root_lin_vel": np.stack([r["root_lin_vel"] for r in state_bank_records]).astype(np.float32)
            if state_bank_records
            else np.zeros((0, 3), dtype=np.float32),
            "root_ang_vel": np.stack([r["root_ang_vel"] for r in state_bank_records]).astype(np.float32)
            if state_bank_records
            else np.zeros((0, 3), dtype=np.float32),
            "joint_pos": np.stack([r["joint_pos"] for r in state_bank_records]).astype(np.float32)
            if state_bank_records
            else np.zeros((0, num_joints), dtype=np.float32),
            "joint_vel": np.stack([r["joint_vel"] for r in state_bank_records]).astype(np.float32)
            if state_bank_records
            else np.zeros((0, num_joints), dtype=np.float32),
        }

    return dataset, sidecar, state_bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, required=True)
    parser.add_argument("--predictor-file", dest="predictor_file", type=str, default="outputs/go1_wildpush_predictor_seed0.pt")
    parser.add_argument("--metrics-file", dest="metrics_file", type=str, default="outputs/go1_wildpush_metrics_seed0.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--impulse-force", dest="impulse_force", type=float, default=DEFAULT_IMPULSE_FORCE_N)
    parser.add_argument("--impulse-torque", dest="impulse_torque", type=float, default=DEFAULT_IMPULSE_TORQUE_N)
    parser.add_argument("--num-envs", dest="num_envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--target-trials", dest="target_trials", type=int, default=DEFAULT_TARGET_TRIALS)
    parser.add_argument("--threshold", type=float, default=None, help="override the metrics-file's val-selected threshold")
    parser.add_argument(
        "--hold-post-fall-steps", dest="hold_post_fall_steps", type=int, default=DEFAULT_HOLD_POST_FALL_STEPS,
        help="env steps to keep simulating a flagged episode's env PAST real termination "
        "(DeferredResetEnv, cartpole_term/common/deferred_reset.py) instead of letting mjlab "
        "reset it immediately -- so a future Gate-5 protective policy has real post-fall states "
        "to act on / be evaluated against. 0 disables this (no hold, no post_fall_* arrays).",
    )
    parser.add_argument(
        "--capture-state-bank", dest="capture_state_bank", action="store_true",
        help="For every resolved lead-time trial, also reservoir-sample and save the robot's real "
        "raw physics state (root pose/velocity + joint pos/velocity) at a uniformly-random step in "
        "[flag_step, termination_step) -- Gate 5's protective-policy init-state bank. See module "
        "docstring. Output: outputs/go1_fall_state_bank_seed{N}.npz.",
    )
    parser.add_argument(
        "--capture-at-flag", dest="capture_at_flag", action="store_true",
        help="with --capture-state-bank: save the state AT the flag step (the hand-off moment, Gate 7) "
        "instead of a uniform draw from [flag, fall). Output: outputs/go1_flag_state_bank_seed{N}.npz.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.capture_at_flag and not args.capture_state_bank:
        raise ValueError("--capture-at-flag requires --capture-state-bank")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required (CLAUDE.md S0.2 / S5.4).")
    device = "cuda"

    metrics = json.loads(Path(args.metrics_file).read_text())
    obs_action_dim = int(metrics["obs_action_dim"])
    predictor_hidden = tuple(metrics["predictor_hidden"])
    threshold = args.threshold if args.threshold is not None else float(metrics["mlp"]["threshold"])

    num_envs = SMOKE_NUM_ENVS if args.smoke else args.num_envs
    target_trials = SMOKE_TARGET_TRIALS if args.smoke else args.target_trials

    dataset, sidecar, state_bank = collect_flagged_states(
        checkpoint_path=args.checkpoint_file,
        predictor_file=args.predictor_file,
        obs_action_dim=obs_action_dim,
        predictor_hidden=predictor_hidden,
        threshold=threshold,
        seed=args.seed,
        force_n=args.impulse_force,
        torque_n=args.impulse_torque,
        num_envs=num_envs,
        capture_state_bank=args.capture_state_bank,
        target_trials=target_trials,
        hold_post_fall_steps=args.hold_post_fall_steps,
        device=device,
        capture_at_flag=args.capture_at_flag,
    )

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    npz_path = outputs_dir / f"go1_flagged_states_seed{args.seed}.npz"
    json_path = outputs_dir / f"go1_flagged_states_seed{args.seed}.json"
    np.savez_compressed(npz_path, **dataset)
    json_path.write_text(json.dumps(sidecar, indent=2))

    print(
        f"[flagged-states] resolved trials={sidecar['num_lead_time_trials']} "
        f"missed={sidecar['num_missed_terminations']} censored_after_flag={sidecar['num_censored_after_flag']} "
        f"live_recall={sidecar['live_recall']}"
    )
    print(
        f"[flagged-states] LEAD TIME: mean={sidecar['mean_lead_steps']:.2f} steps "
        f"({sidecar['mean_lead_seconds']:.3f}s), median={sidecar['median_lead_steps']:.2f} steps "
        f"({sidecar['median_lead_seconds']:.3f}s), std={sidecar['std_lead_steps']:.2f}, "
        f"range=[{sidecar['min_lead_steps']}, {sidecar['max_lead_steps']}]"
        if sidecar["num_lead_time_trials"] > 0
        else "[flagged-states] no resolved lead-time trials -- nothing to report."
    )
    if sidecar["hold_post_fall_steps"] > 0:
        print(
            f"[flagged-states] post-fall trajectories captured: "
            f"{sidecar['num_post_fall_trajectories_captured']} "
            f"(hold_post_fall_steps={sidecar['hold_post_fall_steps']}, "
            f"{sidecar['num_hold_windows_incomplete_at_stop']} still in progress when the run stopped)"
        )
    print(f"  wrote {npz_path}")
    print(f"  wrote {json_path}")

    if state_bank is not None:
        bank_name = "go1_flag_state_bank" if args.capture_at_flag else "go1_fall_state_bank"
        bank_path = outputs_dir / f"{bank_name}_seed{args.seed}.npz"
        np.savez_compressed(bank_path, **state_bank)
        print(
            f"[flagged-states] state bank: {len(state_bank['episode_id'])} raw physics states captured "
            f"(one per resolved lead-time trial, " + ("at the flag step" if args.capture_at_flag else "reservoir-sampled from [flag_step, termination_step)") + ")"
        )
        print(f"  wrote {bank_path}")


if __name__ == "__main__":
    main()
