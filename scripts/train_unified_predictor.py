"""Phase-2 tool (outside CLAUDE.md's spec, see DEVIATIONS.md): train ONE
predictor on the union of multiple competence regimes, instead of the gated
pipeline's one-predictor-per-operating-point design.

Motivation (from the generalization experiment in
outputs/predictor_generalization_seed3.json / .png): a predictor trained
only on `target`-competence data (~15% base rate) generalizes well to an
unseen policy across most of the competence range, but breaks down sharply
in the most extreme, near-1000-step-ceiling regime (<1% base rate) --
because it never saw states from a policy that tight during training. This
script trains a single network on the union of `struggling` + `target` +
`competent` (already collected in outputs/, CLAUDE.md S7.3) plus an OPTIONAL
freshly-collected "near_ceiling" dataset from a late/converged checkpoint,
so the training distribution actually covers the regime that broke.

Architecture, loss, optimizer, and early-stopping rule are UNCHANGED from
CLAUDE.md S8.2/S8.3 (same TerminationPredictor MLP, same train_predictor
function) -- only the input data is a union instead of one operating
point's dataset. Episode-level split (CLAUDE.md S7.5) still applies to the
union: each source dataset's episode_id namespace starts at 0 independently
(collect_one's own per-collection counter), so they're remapped to a
disjoint range per source before splitting -- otherwise episode 5 from
`struggling` and episode 5 from `competent` would collide under the same
split-membership decision, which is not a leakage risk (splitting is still
internally consistent) but is a real bookkeeping bug: per-episode
aggregates (episode counts, feature-screening class stats, per-source
detection lead time) would silently merge two unrelated episodes under one
id. Fixed with a large fixed offset per source, not a hash, so it's
inspectable and reversible.

Usage, from the repo root, on the GPU machine (only needed if
--near-ceiling-checkpoint is passed; combining+training itself is CPU-only
and cheap):

    # Using only the three already-collected operating-point datasets:
    python scripts/train_unified_predictor.py --seed 0

    # Also collecting a near-ceiling dataset from a late checkpoint first:
    python scripts/train_unified_predictor.py --seed 0 \\
        --near-ceiling-checkpoint logs/rsl_rl/cartpole_term/2026-08-26_17-35-01_seed0/model_20.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from eval_predictor import (  # noqa: E402
    base_rate,
    build_features,
    classification_metrics,
    feature_screening,
    make_inputs,
    median_detection_lead_time,
)

EPISODE_ID_OFFSET = 10_000_000  # per-source, keeps each dataset's own episode_id namespace disjoint
NEAR_CEILING_SEED_OFFSET = 800  # distinct from train/select/collect/generalize's own offsets
NEAR_CEILING_TRANSITIONS = 1_000_000
# Deliberately lower than collect_frozen.py's 1,000 (CLAUDE.md S7.3): that
# target is calibrated for `struggling`/`target`/`competent`'s base rates
# (roughly 40%/15%/4.5%), where 1,000 failures arrive in ~15-20s. A
# near-ceiling checkpoint can have a base rate an order of magnitude below
# `competent`'s (0.2-0.5% observed against seed 3's own late checkpoints,
# DEVIATIONS.md) -- at that rate, collect_one's loop (which runs until BOTH
# the transition AND terminated-episode targets are met) can take many
# minutes just waiting for enough failures, even though the transition
# target alone is satisfied almost instantly. This dataset's purpose is
# only to give the union training set SOME exposure to this regime's
# feature distribution, not to match the primary datasets' statistical
# power one-for-one -- 200 failures is already ~5x smoke's floor and far
# more than needed for that. Override with --near-ceiling-terminated-episodes
# if more rigor is wanted and the wait is acceptable.
NEAR_CEILING_TERMINATED_EPISODES = 200
NEAR_CEILING_NUM_ENVS = 4096


def load_and_tag(npz_path: Path, source: str, source_index: int) -> dict[str, np.ndarray]:
    data = dict(np.load(npz_path, allow_pickle=False))
    data["episode_id"] = data["episode_id"] + source_index * EPISODE_ID_OFFSET
    data["source"] = np.full(len(data["label"]), source, dtype="U16")
    return data


def concat_datasets(datasets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = datasets[0].keys()
    return {k: np.concatenate([d[k] for d in datasets], axis=0) for k in keys}


def collect_near_ceiling(
    checkpoint_path: str,
    seed: int,
    horizon: int,
    num_envs: int = NEAR_CEILING_NUM_ENVS,
    target_transitions: int = NEAR_CEILING_TRANSITIONS,
    target_terminated: int = NEAR_CEILING_TERMINATED_EPISODES,
) -> tuple[dict, dict]:
    import torch

    from collect_frozen import collect_one

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to collect --near-ceiling-checkpoint (CLAUDE.md S0.2).")
    print(
        f"[unified] collect_one has no intermediate progress output and runs until "
        f"BOTH {target_transitions} transitions AND {target_terminated} terminated "
        "episodes are collected -- for a near-ceiling checkpoint this can take "
        "several minutes even though nothing prints in the meantime. This is "
        "expected; collect_frozen.py's own MAX_STEPS_SAFETY_CAP (2,000,000 env "
        "steps) is the eventual bound if the policy essentially never fails."
    )
    return collect_one(
        checkpoint_path=checkpoint_path,
        seed=seed,
        operating_point="near_ceiling",
        num_envs=num_envs,
        target_transitions=target_transitions,
        target_terminated=target_terminated,
        horizon=horizon,
        device="cuda",
    )


def evaluate_with_per_source_breakdown(model, test: dict[str, np.ndarray], threshold: float) -> dict:
    import torch

    x_test, y_test = make_inputs(test), test["label"].astype(np.float32)
    with torch.no_grad():
        p_test = torch.sigmoid(model(torch.as_tensor(x_test)).squeeze(-1)).numpy()

    overall = classification_metrics(y_test, p_test, threshold)
    overall_lead = median_detection_lead_time(
        test["episode_id"], test["steps_to_termination"], y_test, p_test, threshold
    )

    per_source: dict[str, dict] = {}
    for source in sorted(set(test["source"].tolist())):
        mask = test["source"] == source
        y_s, p_s = y_test[mask], p_test[mask]
        m = classification_metrics(y_s, p_s, threshold)
        m["lead_time"] = median_detection_lead_time(
            test["episode_id"][mask], test["steps_to_termination"][mask], y_s, p_s, threshold
        )
        m["n_test_transitions"] = int(mask.sum())
        per_source[source] = m

    return {"overall": overall, "overall_lead_time": overall_lead, "per_source": per_source}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Torch training seed AND episode-split seed.")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of (npz_path, source_label) pairs as 'path:label'. "
        "Default: outputs/frozen_{struggling,target,competent}_seed{--seed}.npz.",
    )
    parser.add_argument(
        "--near-ceiling-checkpoint",
        type=str,
        default=None,
        help="If given, collect a 4th 'near_ceiling' dataset from this checkpoint first "
        "(needs a GPU) and include it in the union.",
    )
    parser.add_argument(
        "--near-ceiling-terminated-episodes",
        type=int,
        default=NEAR_CEILING_TERMINATED_EPISODES,
        help="Lower than collect_frozen.py's 1,000 by default -- a near-ceiling checkpoint's "
        "base rate can be an order of magnitude below `competent`'s, so waiting for 1,000 "
        "failures can take many minutes for no real benefit to this dataset's purpose "
        "(exposing the union to this regime's feature distribution, not matching the "
        "primary datasets' statistical power one-for-one).",
    )
    parser.add_argument("--near-ceiling-transitions", type=int, default=NEAR_CEILING_TRANSITIONS)
    parser.add_argument("--near-ceiling-num-envs", type=int, default=NEAR_CEILING_NUM_ENVS)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5, help="Same caveat as eval_predictor_generalization.py: no separate val-only refit step here beyond training's own early stopping, so pass explicitly rather than assume 0.5 is right for your data.")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--metrics-out", type=str, default=None)
    args = parser.parse_args()

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        pairs = []
        for spec in args.datasets:
            path_str, _, label = spec.partition(":")
            pairs.append((Path(path_str), label or Path(path_str).stem))
    else:
        pairs = [
            (outputs_dir / f"frozen_struggling_seed{args.seed}.npz", "struggling"),
            (outputs_dir / f"frozen_target_seed{args.seed}.npz", "target"),
            (outputs_dir / f"frozen_competent_seed{args.seed}.npz", "competent"),
        ]

    for path, _ in pairs:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run scripts/collect_frozen.py for it first.")

    print("[unified] source datasets:")
    tagged = []
    for i, (path, label) in enumerate(pairs):
        d = load_and_tag(path, label, i)
        print(f"  [{i}] {label}: {path} -- {len(d['label'])} transitions, base_rate={base_rate(d['label']):.4f}")
        tagged.append(d)

    if args.near_ceiling_checkpoint:
        print(f"[unified] collecting near_ceiling dataset from {args.near_ceiling_checkpoint} ...")
        dataset, sidecar = collect_near_ceiling(
            args.near_ceiling_checkpoint,
            seed=args.seed + NEAR_CEILING_SEED_OFFSET,
            horizon=args.horizon,
            num_envs=args.near_ceiling_num_envs,
            target_transitions=args.near_ceiling_transitions,
            target_terminated=args.near_ceiling_terminated_episodes,
        )
        npz_path = outputs_dir / f"frozen_near_ceiling_seed{args.seed}.npz"
        np.savez_compressed(npz_path, **dataset)
        (outputs_dir / f"frozen_near_ceiling_seed{args.seed}.json").write_text(json.dumps(sidecar, indent=2))
        print(
            f"  wrote {npz_path}: {sidecar['labelled_transitions']} transitions, "
            f"base_rate={sidecar['base_rate']:.4f} ({sidecar['wall_time_s']:.1f}s)"
        )
        d = load_and_tag(npz_path, "near_ceiling", len(pairs))
        tagged.append(d)

    combined = concat_datasets(tagged)
    print(
        f"[unified] combined: {len(combined['label'])} transitions across {len(tagged)} sources, "
        f"overall base_rate={base_rate(combined['label']):.4f}"
    )

    from cartpole_term.predictor.model import TerminationPredictor
    from cartpole_term.predictor.split import split_episodes
    from cartpole_term.predictor.train import train_predictor
    import torch

    train_ids, val_ids, test_ids = split_episodes(combined["episode_id"].tolist(), seed=args.seed)

    def subset(ids: set[int]) -> dict[str, np.ndarray]:
        mask = np.isin(combined["episode_id"], list(ids))
        return {k: v[mask] for k, v in combined.items()}

    train, val, test = subset(train_ids), subset(val_ids), subset(test_ids)
    for name, split in (("train", train), ("val", val), ("test", test)):
        if len(split["label"]) == 0:
            raise ValueError(f"{name} split is empty -- combined dataset too small.")
        print(f"  {name}: {len(split['label'])} transitions, base_rate={base_rate(split['label']):.4f}")

    x_train, y_train = make_inputs(train), train["label"].astype(np.float32)
    x_val, y_val = make_inputs(val), val["label"].astype(np.float32)

    torch.manual_seed(args.seed)
    model = TerminationPredictor(history=1)
    result = train_predictor(model, x_train, y_train, x_val, y_val, device="cpu", seed=args.seed)
    model.eval()
    print(f"[unified] training done: best_val_epoch={result.best_epoch} best_val_auprc={result.best_val_auprc:.4f}")

    breakdown = evaluate_with_per_source_breakdown(model, test, args.threshold)
    print("[unified] test results:")
    print(f"  overall: base_rate={breakdown['overall']['base_rate']:.4f} "
          f"auprc={breakdown['overall']['auprc']:.4f} auroc={breakdown['overall']['auroc']:.4f} "
          f"lead={breakdown['overall_lead_time']['median_lead_steps']}")
    for source, m in breakdown["per_source"].items():
        auprc_s = f"{m['auprc']:.4f}" if m["auprc"] is not None else "n/a"
        auroc_s = f"{m['auroc']:.4f}" if m["auroc"] is not None else "n/a"
        print(f"  [{source}] n={m['n_test_transitions']} base_rate={m['base_rate']:.4f} "
              f"auprc={auprc_s} auroc={auroc_s} lead={m['lead_time']['median_lead_steps']}")

    features_train = build_features(train["obs"], train["action"])
    screening = feature_screening(features_train, y_train.astype(np.int64))

    out_path = Path(args.out) if args.out else outputs_dir / f"predictor_unified_seed{args.seed}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"[unified] wrote {out_path}")

    metrics_path = Path(args.metrics_out) if args.metrics_out else outputs_dir / f"metrics_unified_seed{args.seed}.json"
    metrics_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "sources": [label for _, label in pairs] + (["near_ceiling"] if args.near_ceiling_checkpoint else []),
                "near_ceiling_checkpoint": args.near_ceiling_checkpoint,
                "split_sizes": {"train": len(y_train), "val": len(y_val), "test": len(test["label"])},
                "best_val_epoch": result.best_epoch,
                "best_val_auprc": result.best_val_auprc,
                "threshold": args.threshold,
                **breakdown,
                "feature_screening": screening,
            },
            indent=2,
        )
    )
    print(f"[unified] wrote {metrics_path}")


if __name__ == "__main__":
    main()
