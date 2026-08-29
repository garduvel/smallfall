"""Env cfg for Mjlab-Cartpole-Balance-Term: shipped balance cfg + the two
failure terminations + the cart-centering reward (CLAUDE.md S2.6, S3, S3.3).

Built by calling mjlab's shipped `cartpole_balance_env_cfg` factory and then
adding terms to the returned cfg's `rewards` / `terminations` dicts. The
shipped factory and site-packages are never edited (CLAUDE.md S3.3, S12#18).
"""

from __future__ import annotations

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

from cartpole_term.rewards import cart_centered
from cartpole_term.terminations import cart_pos_out_of_bounds, pole_angle_out_of_bounds

# --- Binding values (CLAUDE.md S3.1). Do not change these defaults. ---
POLE_ANGLE_LIMIT_RAD = 0.20943951023931953  # 12 degrees
CART_POS_LIMIT_M = 1.0
RAIL_LIMIT_M = 1.8  # hard XML joint limit; used for the assertion below only
CART_CENTER_SIGMA = 0.5
CART_CENTER_WEIGHT = 0.2
PREDICTION_HORIZON = 16  # env steps (= 0.8 s at 20 Hz, see _assert_timing)

assert CART_POS_LIMIT_M < RAIL_LIMIT_M - 0.5, (
    f"CART_POS_LIMIT_M={CART_POS_LIMIT_M} must sit at least 0.5 m inside "
    f"RAIL_LIMIT_M={RAIL_LIMIT_M} (CLAUDE.md S3.1)"
)
assert 0 < CART_CENTER_WEIGHT < 1.0, (
    f"CART_CENTER_WEIGHT={CART_CENTER_WEIGHT} must be in (0, 1) (CLAUDE.md S3.1)"
)

TASK_ID = "Mjlab-Cartpole-Balance-Term"

_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider",))
_HINGE_CFG = SceneEntityCfg("cartpole", joint_names=("hinge_1",))


def _assert_timing(cfg) -> None:
    """CLAUDE.md S2.2: the 16-step horizon is only 0.8s if timestep=0.01,
    decimation=5 (20 Hz env step). Fail loudly if the installed shipped cfg
    ever changes this."""
    timestep = cfg.sim.mujoco.timestep
    decimation = cfg.decimation
    if timestep != 0.01 or decimation != 5:
        raise AssertionError(
            "cartpole-term assumes timestep=0.01, decimation=5 (20 Hz env "
            f"step; CLAUDE.md S2.2). Installed shipped cfg has "
            f"timestep={timestep}, decimation={decimation}. This changes "
            "what PREDICTION_HORIZON=16 steps means in wall-clock time -- "
            "stop and log this in DEVIATIONS.md before proceeding."
        )


def cartpole_balance_term_env_cfg(play: bool = False):
    """Shipped balance env cfg + the two failure terminations + cart centering."""
    cfg = cartpole_balance_env_cfg(play=play)
    _assert_timing(cfg)

    cfg.rewards["cart_centered"] = RewardTermCfg(
        func=cart_centered,
        weight=CART_CENTER_WEIGHT,
        params={"asset_cfg": _CART_CFG, "sigma": CART_CENTER_SIGMA},
    )
    cfg.terminations["pole_angle"] = TerminationTermCfg(
        func=pole_angle_out_of_bounds,
        params={"asset_cfg": _HINGE_CFG, "limit_rad": POLE_ANGLE_LIMIT_RAD},
    )
    cfg.terminations["cart_pos"] = TerminationTermCfg(
        func=cart_pos_out_of_bounds,
        params={"asset_cfg": _CART_CFG, "limit_m": CART_POS_LIMIT_M},
    )
    return cfg
