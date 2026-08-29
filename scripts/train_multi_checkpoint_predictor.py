"""Phase-2 tool (outside CLAUDE.md's spec, see DEVIATIONS.md): train ONE
predictor on data pooled from N checkpoints spanning a single training run's
entire competence range, instead of the gated pipeline's fixed
struggling/target/competent trio, and instead of train_unified_predictor.py's
union of those three pre-collected datasets plus one optional near-ceiling
add-on.

Motivation: the seed-3 generalization sweep
(outputs/predictor_generalization_seed3.json / .png) showed a target-only
predictor's AUPRC collapsing once policy competence pushes the base rate
below ~1% (iterations 13-14, competence ~1000 steps, base rate 0.5%/0.24%).
train_unified_predictor.py's fix -- mixing one small near-ceiling dataset
into struggling+target+competent -- didn't move the needle (user report:
"same shape curves as before, no good generalizations yet"). The leading
hypothesis (DEVIATIONS.md) is that under plain unweighted BCE (CLAUDE.md
S8.3, unchanged here), a few hundred thousand near-ceiling transitions are
drowned out by several million higher-base-rate transitions from the other
three sources -- gradient signal is dominated by whichever source has the
most examples, regardless of how "important" a source is.

This script tests a more direct fix first: densely sample N checkpoints
ACROSS the competence spectrum (not three fixed named points plus an
afterthought), each contributing a comparably-sized dataset, so no single
regime dominates the pooled training set by sheer transition count and the
network sees the near-ceiling regime's states as often as the early ones.

Checkpoints are drawn from ONE already-completed training run's own log
directory (an --eval.interval-checkpointed run, same convention as every
other phase-2 script here) -- either an explicit list via --checkpoints, or
auto-selected as --num-checkpoints indices evenly spaced across that run's
saved model_*.pt files (first and last always included by construction, so
the earliest near-random policy and the latest near-converged one are both
in the pool).

Network size: the pooled dataset here is roughly num_checkpoints x 1M
transitions (~5M at the default of 5), five times any single CLAUDE.md S7.3
dataset. CLAUDE.md S8.2/S12#16's "no architecture other than the specified
MLP" governs the GATED pipeline (eval_predictor.py) only -- this is an
explicitly out-of-spec phase-2 script, and the user asked to size the
network up given the extra data. Still an MLP, still ELU, still 2 hidden
layers (no RNN/attention/etc, CLAUDE.md's structural prohibition is
respected) -- just wider by default: --hidden 256,256 instead of the
spec's 128,128. Pass --hidden 128,128 to reproduce the spec width exactly.

Generalization is tested exactly like every other predictor here -- point
scripts/eval_predictor_generalization.py (extended with a --predictor-hidden
flag so it can load this wider network) at this script's saved weights and
a SEPARATE, independent training run's log dir; plot with
scripts/plot_predictor_generalization.py. No new eval/plot code needed.

Usage, from the repo root, on the GPU machine:

    # auto-select 5 checkpoints evenly spaced across seed 0's own run:
    python scripts/train_multi_checkpoint_predictor.py --seed 0 \\
        --log-dir logs/rsl_rl/cartpole_term/2026-08-26_17-35-01_seed0

    # explicit checkpoints instead of auto-selection:
    python scripts/train_multi_checkpoint_predictor.py --seed 0 \\
        --checkpoints logs/.../model_0.pt logs/.../model_4.pt logs/.../model_8.pt \\
                      logs/.../model_12.pt logs/.../model_16.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import cartpole_term  # noqa: F401,E402
from collect_frozen import collect_one  # noqa: E402
from eval_predictor import base_rate, build_features, feature_screening, make_inputs  # noqa: E402
from eval_predictor_generalization import CHECKPOINT_ITER_RE, discover_checkpoints  # noqa: E402
from train_unified_predictor import EPISODE_ID_OFFSET, concat_datasets, evaluate_with_per_source_breakdown  # noqa: E402

NUM_CHECKPOINTS_DEFAULT = 5
TRANSITIONS_PER_CHECKPOINT = 1_000_000  # same target as collect_frozen.py's own datasets
# Deliberately lower than collect_frozen.py's 1,000 (CLAUDE.md S7.3), for the
# same reason train_unified_predictor.py's NEAR_CEILING_TERMINATED_EPISODES
# is: a checkpoint drawn near the end of training can have a base rate an
# order of magnitude (or more) below `competent`'s, so waiting for 1,000
# failures can take many minutes for no benefit to this script's purpose
# (giving every competence regime comparable REPRESENTATION in the pooled
# set, not matching each source's statistical power one-for-one). Applies to
# every checkpoint here, not just the last one, since --num-checkpoints
# spans the WHOLE competence range by design.
TERMINATED_PER_CHECKPOINT = 200
NUM_ENVS = 4096
DEFAULT_HIDDEN = "256,256"
COLLECT_SEED_OFFSET = 900  # distinct from train/select/collect/generalize/unified's own offsets


def parse_hidden(spec: str) -> tuple[int, int]:
    parts = spec.split(",")
    if len(parts) != 2:
        raise ValueError(f"--hidden must be 'H1,H2' (two ints), got {spec!r}")
    return int(parts[0]), int(parts[1])


def select_checkpoints(log_dir: Path, num_checkpoints: int) -> list[tuple[Path, int]]:
    all_ckpts = discover_checkpoints(log_dir)  # sorted by iteration, ascending
    if num_checkpoints >= len(all_ckpts):
        chosen = all_ckpts
    else:
        # Evenly spaced INDICES (not competence values -- eval_history.json
        # isn't guaranteed monotonic, e.g. it can plateau/dip at the 1000-step
        # ceiling), first and last always included by np.linspace's endpoints.
        idx = sorted(set(int(round(i)) for i in np.linspace(0, len(all_ckpts) - 1, num_checkpoints)))
        chosen = [all_ckpts[i] for i in idx]
    return [(p, int(CHECKPOINT_ITER_RE.search(p.name).group(1))) for p in chosen]


def tag_dataset(dataset: dict[str, np.ndarray], source: str, source_index: int) -> dict[str, np.ndarray]:
    tagged = dict(dataset)
    tagged["episode_id"] = tagged["episode_id"] + source_index * EPISODE_ID_OFFSET
    tagged["source"] = np.full(len(tagged["label"]), source, dtype="U16")
    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="Torch training seed AND episode-split seed.")
    parser.add_argument("--log-dir", type=str, default=None, help="Auto-select --num-checkpoints from this run's model_*.pt files.")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None, help="Explicit checkpoint paths, overrides --log-dir/--num-checkpoints.")
    parser.add_argument("--num-checkpoints", type=int, default=NUM_CHECKPOINTS_DEFAULT)
    parser.add_argument("--transitions-per-checkpoint", type=int, default=TRANSITIONS_PER_CHECKPOINT)
    parser.add_argument("--terminated-per-checkpoint", type=int, default=TERMINATED_PER_CHECKPOINT)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hidden", type=str, default=DEFAULT_HIDDEN, help="'H1,H2' hidden layer widths (default 256,256; pass 128,128 to match the gated pipeline's spec width).")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--metrics-out", type=str, default=None)
    args = parser.parse_args()

    if not args.checkpoints and not args.log_dir:
        parser.error("pass either --checkpoints or --log-dir")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required to collect checkpoint datasets (CLAUDE.md S0.2).")
    device = "cuda"

    if args.checkpoints:
        selected = [(Path(p), None) for p in args.checkpoints]
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.exists():
            raise FileNotFoundError(f"log dir not found: {log_dir}")
        selected = select_checkpoints(log_dir, args.num_checkpoints)

    print(f"[multickpt] selected {len(selected)} checkpoints:")
    for p, it in selected:
        print(f"  {p}  (iteration={it})")

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[multickpt] each checkpoint is collected until BOTH "
        f"{args.transitions_per_checkpoint} transitions AND {args.terminated_per_checkpoint} "
        "terminated episodes are reached, with no intermediate progress output -- a "
        "near-converged checkpoint's low failure rate can make this take several "
        "minutes of apparent silence per checkpoint. This is expected."
    )

    tagged_sources = []
    source_labels = []
    for i, (ckpt_path, iteration) in enumerate(selected):
        label = f"it{iteration}" if iteration is not None else Path(ckpt_path).stem
        print(f"[multickpt] [{i}] collecting from {ckpt_path} (source={label}) ...")
        t0 = time.time()
        dataset, sidecar = collect_one(
            checkpoint_path=str(ckpt_path),
            seed=args.seed + COLLECT_SEED_OFFSET + i,
            operating_point=f"multickpt_{label}",
            num_envs=args.num_envs,
            target_transitions=args.transitions_per_checkpoint,
            target_terminated=args.terminated_per_checkpoint,
            horizon=args.horizon,
            device=device,
        )
        print(
            f"    {sidecar['labelled_transitions']} transitions, base_rate={sidecar['base_rate']:.4f}, "
            f"terminated_episodes={sidecar['terminated_episode_count']} ({time.time() - t0:.1f}s)"
        )
        npz_path = outputs_dir / f"frozen_multickpt_seed{args.seed}_src{i}_{label}.npz"
        np.savez_compressed(npz_path, **dataset)
        (outputs_dir / f"frozen_multickpt_seed{args.seed}_src{i}_{label}.json").write_text(
            json.dumps(sidecar, indent=2)
        )
        tagged_sources.append(tag_dataset(dataset, label, i))
        source_labels.append({"index": i, "label": label, "checkpoint_path": str(ckpt_path), "iteration": iteration})

    combined = concat_datasets(tagged_sources)
    print(
        f"[multickpt] combined: {len(combined['label'])} transitions across {len(tagged_sources)} sources, "
        f"overall base_rate={base_rate(combined['label']):.4f}"
    )

    from cartpole_term.predictor.model import TerminationPredictor
    from cartpole_term.predictor.split import split_episodes
    from cartpole_term.predictor.train import train_predictor

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

    hidden = parse_hidden(args.hidden)
    torch.manual_seed(args.seed)
    model = TerminationPredictor(history=1, hidden=hidden)
    result = train_predictor(model, x_train, y_train, x_val, y_val, device="cpu", seed=args.seed)
    model.eval()
    print(f"[multickpt] training done: best_val_epoch={result.best_epoch} best_val_auprc={result.best_val_auprc:.4f}")

    breakdown = evaluate_with_per_source_breakdown(model, test, args.threshold)
    print("[multickpt] test results:")
    print(
        f"  overall: base_rate={breakdown['overall']['base_rate']:.4f} "
        f"auprc={breakdown['overall']['auprc']:.4f} auroc={breakdown['overall']['auroc']:.4f} "
        f"lead={breakdown['overall_lead_time']['median_lead_steps']}"
    )
    for source, m in breakdown["per_source"].items():
        auprc_s = f"{m['auprc']:.4f}" if m["auprc"] is not None else "n/a"
        auroc_s = f"{m['auroc']:.4f}" if m["auroc"] is not None else "n/a"
        print(
            f"  [{source}] n={m['n_test_transitions']} base_rate={m['base_rate']:.4f} "
            f"auprc={auprc_s} auroc={auroc_s} lead={m['lead_time']['median_lead_steps']}"
        )

    features_train = build_features(train["obs"], train["action"])
    screening = feature_screening(features_train, y_train.astype(np.int64))

    out_path = Path(args.out) if args.out else outputs_dir / f"predictor_multickpt_seed{args.seed}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"[multickpt] wrote {out_path}")

    metrics_path = (
        Path(args.metrics_out) if args.metrics_out else outputs_dir / f"metrics_multickpt_seed{args.seed}.json"
    )
    metrics_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "hidden": list(hidden),
                "sources": source_labels,
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
    print(f"[multickpt] wrote {metrics_path}")


if __name__ == "__main__":
    main()
