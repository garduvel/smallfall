"""Registers the Mjlab-Go1-Protective-Fall task (Gate 5, see DEVIATIONS.md).

Importing this package is the registration side effect -- any script that
needs the task must `import cartpole_term.protective` first, mirroring
cartpole_term/go1/__init__.py's own pattern exactly.

Registers with DEFAULT_STATE_BANK_FILE baked in (see env_cfg.py) -- a
script that needs a non-default fall-state bank builds its own env cfg via
go1_protective_fall_env_cfg(state_bank_file=...) directly rather than
relying on this package-level registration (mirrors
scripts/go1_wild_push_predictor.py's build_impulse_env_cfg_factory pattern,
for the same reason: a module-level register_mjlab_task call can't take a
per-run CLI arg).

play_env_cfg is the SAME object as the training env_cfg, same caveat as
cartpole_term.go1's own registration (Go1's real play-cfg loader wasn't
confirmed) -- not used anywhere in this task's training pipeline regardless.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from cartpole_term.protective.env_cfg import TASK_ID, go1_protective_fall_env_cfg
from cartpole_term.protective.rl_cfg import go1_protective_ppo_runner_cfg
from cartpole_term.protective.runner import ProtectiveFrozenEvalRunner

_env_cfg = go1_protective_fall_env_cfg(play=False)

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=_env_cfg,
    play_env_cfg=_env_cfg,
    rl_cfg=go1_protective_ppo_runner_cfg(),
    runner_cls=ProtectiveFrozenEvalRunner,
)
