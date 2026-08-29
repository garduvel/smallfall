"""Registers the Mjlab-Cartpole-Balance-Term task (CLAUDE.md S3.3).

Importing this package is the registration side effect -- any script that
needs the task must `import cartpole_term` first.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from cartpole_term.env_cfg import TASK_ID, cartpole_balance_term_env_cfg
from cartpole_term.rl_cfg import cartpole_term_ppo_runner_cfg
from cartpole_term.runner import FrozenEvalRunner

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=cartpole_balance_term_env_cfg(play=False),
    play_env_cfg=cartpole_balance_term_env_cfg(play=True),
    rl_cfg=cartpole_term_ppo_runner_cfg(),
    runner_cls=FrozenEvalRunner,  # S6: adds the periodic frozen-eval hook
)
