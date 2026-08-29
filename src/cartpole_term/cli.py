"""Shared CLI flag surface (CLAUDE.md S5.5).

A single shared argument-adding function so flag names and defaults cannot
drift between scripts. Each script calls `add_common_flags(parser,
scripts=(...))` with the subset of {"train", "select", "collect", "eval",
"video"} it belongs to (per the CLAUDE.md S5.5 table's "Scripts" column),
then adds any flags of its own.

`--smoke` itself is common to every script; each script is responsible for
applying SMOKE_OVERRIDES to whichever of its own settings that implies (the
override targets differ per script, so there's no one generic apply
function -- see each script's own `--smoke` handling).
"""

from __future__ import annotations

import argparse

POLE_ANGLE_LIMIT_RAD_DEFAULT = 0.20943951023931953  # 12 degrees, CLAUDE.md S3.1
CART_POS_LIMIT_M_DEFAULT = 1.0
CART_CENTER_WEIGHT_DEFAULT = 0.2
CART_CENTER_SIGMA_DEFAULT = 0.5
PREDICTION_HORIZON_DEFAULT = 16

OPERATING_POINTS = ("struggling", "target", "competent")

# CLAUDE.md S11 gate 4 / S5.5 --smoke: "64 envs, 5 iterations, 32 eval
# episodes, 5000 transitions". Single source of truth for these numbers;
# each script applies whichever subset is relevant to it.
SMOKE_NUM_ENVS = 64
SMOKE_MAX_ITERATIONS = 5
SMOKE_EVAL_NUM_ENVS = 32
SMOKE_EVAL_INTERVAL = 2
SMOKE_COLLECT_TRANSITIONS = 5000


def add_common_flags(parser: argparse.ArgumentParser, scripts: tuple[str, ...]) -> None:
    scripts_set = set(scripts)

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end run (CLAUDE.md S11 gate 4): "
        f"{SMOKE_NUM_ENVS} envs, {SMOKE_MAX_ITERATIONS} iterations, "
        f"{SMOKE_EVAL_NUM_ENVS} eval episodes, {SMOKE_COLLECT_TRANSITIONS} transitions.",
    )

    if scripts_set & {"train", "select", "collect", "eval"}:
        parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    if "train" in scripts_set:
        parser.add_argument(
            "--env.scene.num-envs", dest="env_scene_num_envs", type=int, default=4096
        )
        parser.add_argument("--eval.num-envs", dest="eval_num_envs", type=int, default=256)
        parser.add_argument("--eval.interval", dest="eval_interval", type=int, default=10)
        parser.add_argument(
            "--env.reward.cart-center-weight",
            dest="env_reward_cart_center_weight",
            type=float,
            default=CART_CENTER_WEIGHT_DEFAULT,
        )
        parser.add_argument(
            "--env.reward.cart-center-sigma",
            dest="env_reward_cart_center_sigma",
            type=float,
            default=CART_CENTER_SIGMA_DEFAULT,
        )

    if scripts_set & {"train", "collect"}:
        parser.add_argument(
            "--env.termination.angle-limit-rad",
            dest="env_termination_angle_limit_rad",
            type=float,
            default=POLE_ANGLE_LIMIT_RAD_DEFAULT,
        )
        parser.add_argument(
            "--env.termination.cart-limit-m",
            dest="env_termination_cart_limit_m",
            type=float,
            default=CART_POS_LIMIT_M_DEFAULT,
        )

    if scripts_set & {"collect", "eval"}:
        parser.add_argument(
            "--operating-point", dest="operating_point", choices=OPERATING_POINTS, default="target"
        )
        parser.add_argument(
            "--predictor.horizon",
            dest="predictor_horizon",
            type=int,
            default=PREDICTION_HORIZON_DEFAULT,
        )

    if "eval" in scripts_set:
        parser.add_argument("--predictor.history", dest="predictor_history", type=int, default=1)
        parser.add_argument(
            "--predictor.uncertainty",
            dest="predictor_uncertainty",
            choices=("none", "ensemble"),
            default="none",
        )

    if scripts_set & {"collect", "video"}:
        parser.add_argument("--checkpoint-file", dest="checkpoint_file", type=str, default=None)


def build_parser(scripts: tuple[str, ...], description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_common_flags(parser, scripts)
    return parser
