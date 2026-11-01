"""Registers the Mjlab-G1-Velocity-Term task (phase-2, see DEVIATIONS.md's
"multi-robot scaffolding" entry).

Importing this package is the registration side effect -- any script that
needs the task must `import cartpole_term.g1` first, mirroring
cartpole_term.go1's own pattern exactly.

play_env_cfg is currently the SAME object as the training env_cfg, same
unresolved gap as Go1's own play_env_cfg (see
cartpole_term.g1.env_cfg.g1_velocity_term_env_cfg's docstring for why).
This means `uv run play` on this task id would run with training-shaped
randomization rather than mjlab's usual disabled-randomization play mode
-- not used anywhere in the training/frozen-eval/predictor pipeline
(CLAUDE.md S6.2.2 never uses play_env_cfg for frozen eval regardless), so
this only matters once G1 video recording is implemented. See
DEVIATIONS.md.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from cartpole_term.g1.env_cfg import TASK_ID, g1_velocity_term_env_cfg
from cartpole_term.g1.rl_cfg import g1_velocity_term_ppo_runner_cfg
from cartpole_term.g1.runner import G1FrozenEvalRunner

_env_cfg = g1_velocity_term_env_cfg(play=False)

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=_env_cfg,
    play_env_cfg=_env_cfg,  # see module docstring: real play cfg not wired yet
    rl_cfg=g1_velocity_term_ppo_runner_cfg(),
    runner_cls=G1FrozenEvalRunner,
)
