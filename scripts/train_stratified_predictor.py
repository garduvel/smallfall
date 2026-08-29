"""Phase-2 tool (outside CLAUDE.md's spec, see DEVIATIONS.md): ground-up
rewrite of the multi-competence predictor experiment, replacing both
train_unified_predictor.py and train_multi_checkpoint_predictor.py.

Why this replaces the previous two attempts: train_multi_checkpoint_predictor.py's
own per-source test breakdown showed the near-ceiling source (`it16`) ended
up with 767,534 test transitions -- 56% of the entire pooled test set, ~5x
every other source -- because collect_one's stopping rule ("both a
transitions target AND a terminated-episode target") keeps running far
longer for a low-base-rate policy, so it accumulates far MORE raw data, not
less. And it still failed (AUPRC 0.0056 against a 0.0004 base rate). That
result falsifies the "just needs more representation" hypothesis outright:
the near-ceiling regime was never under-represented by volume, it was
over-represented, and still didn't learn. Pooling everything into one
array and letting a single DataLoader shuffle it (both previous scripts)
also means whichever source has the most rows dominates every gradient
step in proportion to its row count -- an accident of how much data
collection happened to produce, not a deliberate choice.

This version fixes both problems structurally instead of by re-tuning
constants:

1. Collection stops on a SINGLE, simple, per-policy TRANSITIONS threshold
   (no terminated-episode target at all) -- this makes collection time
   roughly base-rate-independent (transitions accumulate at ~envs_per_policy
   per env-step for ANY policy, competent or not), and every one of the 4
   competence "stacks" ends up the same size by construction, not by luck.
2. All 4 frozen policies are collected SIMULTANEOUSLY: one env instance per
   checkpoint, stepped in a round-robin loop each iteration, instead of
   collect_one's run-to-completion-then-move-to-the-next. Wall time is
   bounded by whichever policy is slowest, not the sum of all four.
3. Training draws each minibatch by sampling an equal number of rows from
   EACH stack, every step -- so gradient signal is never dominated by
   whichever stack happens to be larger. Early stopping tracks the MEAN of
   the 4 stacks' own validation AUPRC (not one pooled number), so a strong
   score on the easy stacks can't paper over a collapse on the hard one --
   exactly the failure mode a single pooled metric hid in the last run.

Architecture, loss, and optimizer are otherwise unchanged from CLAUDE.md
S8.2/S8.3 (same TerminationPredictor MLP class, same BCEWithLogitsLoss, no
reweighting, same Adam lr=1e-3). Network width reverts to the S8.2 spec
default (128, 128) this time: the whole point of the equal-size-stacks
redesign is a much smaller total dataset (4 x --transitions-per-policy,
default 1,000,000 total) than the ~9M transitions the last attempt pooled,
so there's no longer a strong argument for a wider network (override with
--hidden if wanted).

The 4 operating points: `struggling`/`target`/`competent` checkpoints are
read directly from outputs/selected_checkpoints.json (the same,
already-trusted CLAUDE.md S7.2 mechanical selection the gated pipeline
itself uses -- not re-derived here). `fluent` (mean episode length >=
--fluent-threshold, default 990 out of the 1000-step cap) is selected the
same "earliest checkpoint that qualifies" way as S7.2, since CLAUDE.md's
three named operating points don't include a near-ceiling point and this
script must not edit scripts/select_checkpoints.py to add one.

Usage, from the repo root, on the GPU machine, after
scripts/select_checkpoints.py has already produced
outputs/selected_checkpoints.json for --seed:

    python scripts/train_stratified_predictor.py --seed 0

Generalization against an independent run is unchanged -- point
scripts/eval_predictor_generalization.py (with --predictor-hidden matching
whatever --hidden this was trained with) at the saved weights and a second
run's --log-dir, same as every other phase-2 predictor here.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from torch import nn  # noqa: E402

import cartpole_term  # noqa: F401,E402
from eval_predictor import (  # noqa: E402
    base_rate,
    build_features,
    classification_metrics,
    feature_screening,
    make_inputs,
    median_detection_lead_time,
)
from eval_predictor_generalization import checkpoint_iteration, discover_checkpoints  # noqa: E402

OPERATING_POINTS_FROM_SELECTED = ("struggling", "target", "competent")
FLUENT_THRESHOLD_DEFAULT = 990.0  # out of the 1000-step episode cap (CLAUDE.md S2.2)
ENVS_PER_POLICY_DEFAULT = 1024
TRANSITIONS_PER_POLICY_DEFAULT = 250_000  # x4 stacks = 1,000,000 total -- comparable to ONE gated dataset,
# not the ~9M the previous (pooled-everything) attempt collected. "There's way too much data" was the point.
MAX_STEPS_SAFETY_CAP_PER_POLICY = 500_000  # engineering guard, not expected to ever fire: a pure transitions
# threshold accumulates at ~envs_per_policy per env-step regardless of competence, so ~250 steps is typical.
STRATIFIED_SEED_OFFSET = 1000  # distinct from train/select/collect/generalize/unified/multickpt's own offsets
STRATIFIED_ENV_SEED_OFFSET = 50  # separates the runner-build seed from the rollout env.seed(), mirroring
# collect_frozen.py's own build-seed vs. env.seed(seed + COLLECT_SEED_OFFSET) split.
POOL_EPISODE_ID_OFFSET = 10_000_000  # only used to keep the 4 stacks' episode ids disjoint for the ONE
# pooled/overall test-set lead-time number below; per-stack metrics never need it.


def parse_hidden(spec: str) -> tuple[int, int]:
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"--hidden must be 'H1,H2' (two ints), got {spec!r}")
    return int(parts[0]), int(parts[1])


def select_fluent_checkpoint(log_dir: Path, threshold: float) -> tuple[Path, int, float]:
    eval_history_path = log_dir / "eval_history.json"
    if not eval_history_path.exists():
        raise FileNotFoundError(f"{eval_history_path} not found -- can't select a fluent checkpoint without it.")
    eval_history = json.loads(eval_history_path.read_text())
    by_iteration = {rec["iteration"]: rec["mean_action"]["mean_episode_length"] for rec in eval_history}

    for ckpt in discover_checkpoints(log_dir):  # sorted ascending by iteration -- earliest qualifying wins
        it = checkpoint_iteration(ckpt)
        length = by_iteration.get(it)
        if length is not None and length >= threshold:
            return ckpt, it, length

    raise RuntimeError(
        f"No checkpoint under {log_dir} reached mean episode length >= {threshold} (--fluent-threshold). "
        f"Available (iteration: mean_length): {dict(sorted(by_iteration.items()))}. "
        "Lower --fluent-threshold or train longer."
    )


def _build_dataset(records: list[dict]) -> dict[str, np.ndarray]:
    return {
        "obs": np.stack([r["obs"] for r in records]).astype(np.float32),
        "action": np.stack([r["action"] for r in records]).astype(np.float32),
        "label": np.array([r["label"] for r in records], dtype=np.int64),
        "env_id": np.array([r["env_id"] for r in records], dtype=np.int64),
        "episode_id": np.array([r["episode_id"] for r in records], dtype=np.int64),
        "step_in_episode": np.array([r["step_in_episode"] for r in records], dtype=np.int64),
        "steps_to_termination": np.array([r["steps_to_termination"] for r in records], dtype=np.int64),
        "termination_cause": np.array([r["termination_cause"] for r in records], dtype="U16"),
    }


def collect_parallel(
    checkpoints: dict[str, str],
    seed_base: int,
    envs_per_policy: int,
    transitions_per_policy: int,
    horizon: int,
    device: str,
) -> dict[str, tuple[dict[str, np.ndarray], dict]]:
    from cartpole_term.predictor.labeling import TerminationLabeler
    from cartpole_term.runner import build_runner_from_checkpoint

    sources: dict[str, dict] = {}
    for i, (label, ckpt_path) in enumerate(checkpoints.items()):
        policy_seed = seed_base + i
        runner = build_runner_from_checkpoint(
            ckpt_path, seed=policy_seed, eval_num_envs=envs_per_policy, device=device
        )
        actor_snapshot = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
        runner.alg.eval_mode()
        env = runner._build_eval_env()  # identical dynamics to training, CLAUDE.md S6.2.2
        env.seed(policy_seed + STRATIFIED_ENV_SEED_OFFSET)
        obs, _ = env.reset()
        sources[label] = {
            "checkpoint_path": ckpt_path,
            "runner": runner,
            "actor_snapshot": actor_snapshot,
            "env": env,
            "obs": obs.to(device),
            "labeler": TerminationLabeler(num_envs=envs_per_policy, horizon=horizon),
            "episode_id": np.arange(envs_per_policy, dtype=np.int64),
            "next_episode_id": envs_per_policy,
            "step_in_episode": np.zeros(envs_per_policy, dtype=np.int64),
            "cur_episode_len": np.zeros(envs_per_policy, dtype=np.int64),
            "episode_lengths": [],
            "records": [],
            "terminated_episode_count": 0,
            "total_steps": 0,
            "total_pushed": 0,
            "t0": time.time(),
            "done": False,
        }

    print(
        f"[stratified] collecting from {len(sources)} policies simultaneously "
        f"({envs_per_policy} envs each, target {transitions_per_policy} transitions per policy) ..."
    )

    with torch.inference_mode():
        while any(not s["done"] for s in sources.values()):
            for label, s in sources.items():
                if s["done"]:
                    continue
                runner, env, obs, labeler = s["runner"], s["env"], s["obs"], s["labeler"]
                n = envs_per_policy

                actions = runner.alg.actor(obs, stochastic_output=True)  # S7.3: sample, don't take the mean
                obs_actor_np = obs["actor"].detach().cpu().numpy()
                action_np = actions.detach().cpu().numpy()

                next_obs, _rewards, dones, extras = env.step(actions)
                dones_cpu = dones.bool().cpu()
                time_outs = extras.get("time_outs")
                truncated_cpu = (
                    time_outs.bool().cpu() if time_outs is not None else torch.zeros_like(dones_cpu)
                )
                terminated_cpu = dones_cpu & ~truncated_cpu

                raw_env = env.unwrapped
                pole_hit = raw_env.termination_manager.get_term("pole_angle").cpu()
                cart_hit = raw_env.termination_manager.get_term("cart_pos").cpu()

                payloads = [
                    {
                        "obs": obs_actor_np[i],
                        "action": action_np[i],
                        "env_id": i,
                        "episode_id": int(s["episode_id"][i]),
                        "step_in_episode": int(s["step_in_episode"][i]),
                    }
                    for i in range(n)
                ]
                s["total_pushed"] += n
                emitted = labeler.step(payloads, terminated_cpu.tolist(), truncated_cpu.tolist())
                for sample in emitted:
                    cause = ""
                    if sample.label == 1:
                        env_id = sample.env_id
                        if pole_hit[env_id] and cart_hit[env_id]:
                            cause = "both"
                        elif pole_hit[env_id]:
                            cause = "pole_angle"
                        elif cart_hit[env_id]:
                            cause = "cart_pos"
                    s["records"].append(
                        {
                            **sample.payload,
                            "label": sample.label,
                            "steps_to_termination": (
                                sample.steps_to_termination if sample.steps_to_termination is not None else -1
                            ),
                            "termination_cause": cause,
                        }
                    )

                s["terminated_episode_count"] += int(terminated_cpu.sum().item())
                s["total_steps"] += 1
                s["cur_episode_len"] += 1
                s["step_in_episode"] += 1
                done_np = dones_cpu.numpy()
                if done_np.any():
                    idx = np.nonzero(done_np)[0]
                    s["episode_lengths"].extend(s["cur_episode_len"][idx].tolist())
                    s["cur_episode_len"][idx] = 0
                    s["step_in_episode"][idx] = 0
                    s["episode_id"][idx] = s["next_episode_id"] + np.arange(len(idx))
                    s["next_episode_id"] += len(idx)

                s["obs"] = next_obs.to(device)

                if len(s["records"]) >= transitions_per_policy:
                    s["done"] = True
                elif s["total_steps"] > MAX_STEPS_SAFETY_CAP_PER_POLICY:
                    print(
                        f"[stratified] [{label}] SAFETY CAP hit ({MAX_STEPS_SAFETY_CAP_PER_POLICY} steps) "
                        f"before reaching {transitions_per_policy} transitions "
                        f"({len(s['records'])} collected). Stopping this source."
                    )
                    s["done"] = True

    results: dict[str, tuple[dict[str, np.ndarray], dict]] = {}
    for label, s in sources.items():
        for name, before in s["actor_snapshot"].items():
            after = s["runner"].alg.actor.state_dict()[name]
            assert torch.equal(before, after), f"[{label}] actor.{name} changed during frozen-policy collection"

        wall_time = time.time() - s["t0"]
        dataset = _build_dataset(s["records"])
        rate = float(dataset["label"].mean()) if len(dataset["label"]) > 0 else 0.0
        cause_arr = dataset["termination_cause"]
        cause_counts = {
            c: int((cause_arr == c).sum()) for c in ("pole_angle", "cart_pos", "both") if (cause_arr == c).any()
        }
        lengths_arr = np.array(s["episode_lengths"], dtype=np.int64)
        sidecar = {
            "checkpoint_path": s["checkpoint_path"],
            "operating_point": label,
            "envs_per_policy": envs_per_policy,
            "horizon": horizon,
            "wall_time_s": wall_time,
            "total_env_steps": s["total_steps"],
            "total_transitions_pushed": s["total_pushed"],
            "labelled_transitions": len(dataset["label"]),
            "base_rate": rate,
            "episode_count": len(s["episode_lengths"]),
            "terminated_episode_count": s["terminated_episode_count"],
            "termination_cause_counts": cause_counts,
            "episode_length_mean": float(lengths_arr.mean()) if len(lengths_arr) else None,
            "episode_length_median": float(np.median(lengths_arr)) if len(lengths_arr) else None,
        }
        results[label] = (dataset, sidecar)
    return results


@dataclass
class BalancedTrainResult:
    state_dict: dict
    best_epoch: int
    best_val_macro_auprc: float
    history: list[dict] = field(default_factory=list)


def train_balanced(
    model: nn.Module,
    train_stacks: dict[str, tuple[np.ndarray, np.ndarray]],
    val_stacks: dict[str, tuple[np.ndarray, np.ndarray]],
    device: str = "cpu",
    seed: int = 0,
    batch_size: int = 1024,
    max_epochs: int = 200,
    patience: int = 8,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
) -> BalancedTrainResult:
    labels = list(train_stacks.keys())
    per_source_batch = max(1, batch_size // len(labels))

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    loss_fn = nn.BCEWithLogitsLoss()

    val_tensors = {
        lbl: (torch.as_tensor(x, dtype=torch.float32).to(device), y) for lbl, (x, y) in val_stacks.items()
    }
    steps_per_epoch = max(1, min(len(x) for x, _ in train_stacks.values()) // per_source_batch)

    best_val_macro_auprc = -1.0
    best_state: dict | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            xb_parts, yb_parts = [], []
            for lbl in labels:
                x, y = train_stacks[lbl]
                idx = rng.integers(0, len(x), size=per_source_batch)  # random positions from each stack
                xb_parts.append(x[idx])
                yb_parts.append(y[idx])
            xb = torch.as_tensor(np.concatenate(xb_parts), dtype=torch.float32).to(device)
            yb = torch.as_tensor(np.concatenate(yb_parts), dtype=torch.float32).to(device)

            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        per_source_val_auprc = {}
        with torch.no_grad():
            for lbl, (x_val_t, y_val) in val_tensors.items():
                probs = torch.sigmoid(model(x_val_t).squeeze(-1)).cpu().numpy()
                has_both = len(np.unique(y_val)) > 1
                per_source_val_auprc[lbl] = float(average_precision_score(y_val, probs)) if has_both else 0.0
        macro_auprc = float(np.mean(list(per_source_val_auprc.values())))

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / steps_per_epoch,
                "val_macro_auprc": macro_auprc,
                "per_source_val_auprc": per_source_val_auprc,
            }
        )

        if macro_auprc > best_val_macro_auprc:
            best_val_macro_auprc = macro_auprc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    return BalancedTrainResult(
        state_dict=best_state, best_epoch=best_epoch, best_val_macro_auprc=best_val_macro_auprc, history=history
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selected-checkpoints", type=str, default="outputs/selected_checkpoints.json")
    parser.add_argument("--fluent-threshold", type=float, default=FLUENT_THRESHOLD_DEFAULT)
    parser.add_argument("--envs-per-policy", type=int, default=ENVS_PER_POLICY_DEFAULT)
    parser.add_argument("--transitions-per-policy", type=int, default=TRANSITIONS_PER_POLICY_DEFAULT)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hidden", type=str, default="128,128")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--metrics-out", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to collect checkpoint datasets (CLAUDE.md S0.2).")
    device = "cuda"

    selected_path = Path(args.selected_checkpoints)
    if not selected_path.exists():
        raise FileNotFoundError(f"{selected_path} not found -- run scripts/select_checkpoints.py first.")
    selected = json.loads(selected_path.read_text())
    seed_key = str(args.seed)
    if seed_key not in selected:
        raise KeyError(f"seed {args.seed} not found in {selected_path}")

    checkpoints = {
        point: selected[seed_key][point]["checkpoint_path"] for point in OPERATING_POINTS_FROM_SELECTED
    }
    log_dir = Path(checkpoints["target"]).parent
    fluent_ckpt, fluent_it, fluent_len = select_fluent_checkpoint(log_dir, args.fluent_threshold)
    checkpoints["fluent"] = str(fluent_ckpt)

    print("[stratified] operating points:")
    for label, path in checkpoints.items():
        print(f"  {label}: {path}")
    print(f"  (fluent selected at iteration={fluent_it}, mean_episode_length={fluent_len:.1f})")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    stacks_raw = collect_parallel(
        checkpoints,
        seed_base=args.seed + STRATIFIED_SEED_OFFSET,
        envs_per_policy=args.envs_per_policy,
        transitions_per_policy=args.transitions_per_policy,
        horizon=args.horizon,
        device=device,
    )

    print("[stratified] collection results:")
    for label, (dataset, sidecar) in stacks_raw.items():
        npz_path = outputs_dir / f"frozen_stratified_{label}_seed{args.seed}.npz"
        np.savez_compressed(npz_path, **dataset)
        (outputs_dir / f"frozen_stratified_{label}_seed{args.seed}.json").write_text(json.dumps(sidecar, indent=2))
        print(
            f"  [{label}] {sidecar['labelled_transitions']} transitions, base_rate={sidecar['base_rate']:.4f}, "
            f"terminated_episodes={sidecar['terminated_episode_count']} ({sidecar['wall_time_s']:.1f}s)"
        )

    from cartpole_term.predictor.model import TerminationPredictor
    from cartpole_term.predictor.split import split_episodes

    train_dicts: dict[str, dict[str, np.ndarray]] = {}
    train_stacks, val_stacks, test_stacks = {}, {}, {}
    print("[stratified] episode-level split per stack:")
    for label, (dataset, _sidecar) in stacks_raw.items():
        train_ids, val_ids, test_ids = split_episodes(dataset["episode_id"].tolist(), seed=args.seed)

        def subset(ids: set[int]) -> dict[str, np.ndarray]:
            mask = np.isin(dataset["episode_id"], list(ids))
            return {k: v[mask] for k, v in dataset.items()}

        train, val, test = subset(train_ids), subset(val_ids), subset(test_ids)
        for name, split in (("train", train), ("val", val), ("test", test)):
            if len(split["label"]) == 0:
                raise ValueError(f"[{label}] {name} split is empty -- collected too little data for this stack.")

        train_dicts[label] = train
        train_stacks[label] = (make_inputs(train), train["label"].astype(np.float32))
        val_stacks[label] = (make_inputs(val), val["label"].astype(np.float32))
        test_stacks[label] = test
        print(
            f"  [{label}] train={len(train['label'])} val={len(val['label'])} test={len(test['label'])} "
            f"base_rate(train)={base_rate(train['label']):.4f}"
        )

    hidden = parse_hidden(args.hidden)
    model = TerminationPredictor(history=1, hidden=hidden)
    result = train_balanced(
        model,
        train_stacks,
        val_stacks,
        device="cpu",
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    model.eval()
    print(
        f"[stratified] training converged: best_val_epoch={result.best_epoch} "
        f"best_val_macro_auprc={result.best_val_macro_auprc:.4f}"
    )
    print(f"  per-source val AUPRC at best epoch: {result.history[result.best_epoch]['per_source_val_auprc']}")

    print("[stratified] test results (per stack):")
    per_source_test = {}
    with torch.no_grad():
        for label, test in test_stacks.items():
            x_test, y_test = make_inputs(test), test["label"].astype(np.float32)
            p_test = torch.sigmoid(model(torch.as_tensor(x_test)).squeeze(-1)).numpy()
            m = classification_metrics(y_test, p_test, args.threshold)
            m["lead_time"] = median_detection_lead_time(
                test["episode_id"], test["steps_to_termination"], y_test, p_test, args.threshold
            )
            m["n_test_transitions"] = int(len(y_test))
            per_source_test[label] = m
            auprc_s = f"{m['auprc']:.4f}" if m["auprc"] is not None else "n/a"
            auroc_s = f"{m['auroc']:.4f}" if m["auroc"] is not None else "n/a"
            print(
                f"  [{label}] n={m['n_test_transitions']} base_rate={m['base_rate']:.4f} "
                f"auprc={auprc_s} auroc={auroc_s} lead={m['lead_time']['median_lead_steps']}"
            )

    pooled: dict[str, np.ndarray] = {}
    for k in ("obs", "action", "label", "steps_to_termination"):
        pooled[k] = np.concatenate([test_stacks[l][k] for l in test_stacks], axis=0)
    pooled["episode_id"] = np.concatenate(
        [test_stacks[l]["episode_id"] + i * POOL_EPISODE_ID_OFFSET for i, l in enumerate(test_stacks)], axis=0
    )
    x_pooled, y_pooled = make_inputs(pooled), pooled["label"].astype(np.float32)
    with torch.no_grad():
        p_pooled = torch.sigmoid(model(torch.as_tensor(x_pooled)).squeeze(-1)).numpy()
    overall = classification_metrics(y_pooled, p_pooled, args.threshold)
    overall["lead_time"] = median_detection_lead_time(
        pooled["episode_id"], pooled["steps_to_termination"], y_pooled, p_pooled, args.threshold
    )
    print(
        f"  [overall, pooled] n={len(y_pooled)} base_rate={overall['base_rate']:.4f} "
        f"auprc={overall['auprc']:.4f} auroc={overall['auroc']:.4f} lead={overall['lead_time']['median_lead_steps']}"
    )

    features_train = build_features(
        np.concatenate([train_dicts[l]["obs"] for l in train_dicts], axis=0),
        np.concatenate([train_dicts[l]["action"] for l in train_dicts], axis=0),
    )
    y_train_pooled = np.concatenate([train_dicts[l]["label"] for l in train_dicts], axis=0)
    screening = feature_screening(features_train, y_train_pooled.astype(np.int64))

    out_path = Path(args.out) if args.out else outputs_dir / f"predictor_stratified_seed{args.seed}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"[stratified] wrote {out_path}")

    metrics_path = (
        Path(args.metrics_out) if args.metrics_out else outputs_dir / f"metrics_stratified_seed{args.seed}.json"
    )
    metrics_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "hidden": list(hidden),
                "checkpoints": checkpoints,
                "fluent_iteration": fluent_it,
                "fluent_mean_episode_length": fluent_len,
                "envs_per_policy": args.envs_per_policy,
                "transitions_per_policy": args.transitions_per_policy,
                "best_val_epoch": result.best_epoch,
                "best_val_macro_auprc": result.best_val_macro_auprc,
                "threshold": args.threshold,
                "per_source_test": per_source_test,
                "overall_test": overall,
                "feature_screening": screening,
            },
            indent=2,
        )
    )
    print(f"[stratified] wrote {metrics_path}")


if __name__ == "__main__":
    main()
