"""Per-robot module/function resolution for the multi-robot phase-2
pipeline (see DEVIATIONS.md's "multi-robot scaffolding" entry). Every
robot subpackage (cartpole_term.go1, cartpole_term.g1) follows the exact
same naming convention -- env_cfg.py's TASK_ID and
{robot}_velocity_term_env_cfg(), rl_cfg.py's
{robot}_velocity_term_ppo_runner_cfg(), and a single *FrozenEvalRunner
class in runner.py. get_robot_backend() discovers these by that
convention rather than every new robot_*.py script re-deriving the same
import/getattr chain, and fails loud with the exact missing attribute(s)
if a robot subpackage doesn't follow it -- never a bare AttributeError.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RobotBackend:
    task_id: str
    env_cfg_factory: Callable[..., Any]
    rl_cfg_factory: Callable[..., Any]
    runner_cls: type


def get_robot_backend(robot: str) -> RobotBackend:
    """Resolves `robot`'s (env_cfg_factory, rl_cfg_factory, runner_cls,
    task_id) and, as a side effect, imports cartpole_term.{robot} itself
    -- which registers the task with mjlab, mirroring every existing
    go1_*.py script's `import cartpole_term.go1` pattern, parameterized."""
    env_cfg_mod = importlib.import_module(f"cartpole_term.{robot}.env_cfg")
    rl_cfg_mod = importlib.import_module(f"cartpole_term.{robot}.rl_cfg")
    runner_mod = importlib.import_module(f"cartpole_term.{robot}.runner")

    from cartpole_term.runner import FrozenEvalRunner

    task_id = getattr(env_cfg_mod, "TASK_ID", None)
    env_cfg_factory = getattr(env_cfg_mod, f"{robot}_velocity_term_env_cfg", None)
    rl_cfg_factory = getattr(rl_cfg_mod, f"{robot}_velocity_term_ppo_runner_cfg", None)
    # Filter by actual subclassing, not a name-suffix guess: runner.py
    # modules import the FrozenEvalRunner base class itself into their own
    # namespace (`from cartpole_term.runner import FrozenEvalRunner`), and
    # that name trivially also ends with "FrozenEvalRunner" -- a naive
    # suffix filter matches it as a false second candidate. Excluding
    # FrozenEvalRunner itself and requiring a strict subclass fixes this
    # without depending on naming coincidence.
    runner_candidates = [
        obj
        for obj in vars(runner_mod).values()
        if isinstance(obj, type) and issubclass(obj, FrozenEvalRunner) and obj is not FrozenEvalRunner
    ]

    missing = []
    if task_id is None:
        missing.append(f"cartpole_term.{robot}.env_cfg.TASK_ID")
    if env_cfg_factory is None:
        missing.append(f"cartpole_term.{robot}.env_cfg.{robot}_velocity_term_env_cfg")
    if rl_cfg_factory is None:
        missing.append(f"cartpole_term.{robot}.rl_cfg.{robot}_velocity_term_ppo_runner_cfg")
    if len(runner_candidates) != 1:
        missing.append(
            f"exactly one *FrozenEvalRunner class in cartpole_term.{robot}.runner "
            f"(found {runner_candidates})"
        )
    if missing:
        raise RuntimeError(
            f"robot {robot!r}'s subpackage doesn't follow the naming convention "
            f"cartpole_term.go1/cartpole_term.g1 both use -- missing/ambiguous: {missing}. "
            "Fix the naming in that subpackage, or this discovery helper in "
            "cartpole_term/common/backend.py, before proceeding."
        )

    importlib.import_module(f"cartpole_term.{robot}")  # registration side effect

    return RobotBackend(
        task_id=task_id,
        env_cfg_factory=env_cfg_factory,
        rl_cfg_factory=rl_cfg_factory,
        runner_cls=runner_candidates[0],
    )
