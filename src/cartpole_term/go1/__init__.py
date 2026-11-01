"""Registers the Mjlab-Go1-Velocity-Term task (phase-2, see DEVIATIONS.md).

Importing this package is the registration side effect -- any script that
needs the task must `import cartpole_term.go1` first, mirroring
cartpole_term/__init__.py's own pattern exactly.

play_env_cfg is currently the SAME object as the training env_cfg (see
cartpole_term.go1.env_cfg.go1_velocity_term_env_cfg's docstring: Go1's real
play-cfg loader wasn't confirmed at the time this was written). This means
`uv run play` on this task id would run with training-shaped randomization
rather than mjlab's usual disabled-randomization play mode -- not used
anywhere in the training/frozen-eval/predictor pipeline (CLAUDE.md S6.2.2
never uses play_env_cfg for frozen eval regardless), so this only matters
once Go1 video recording is implemented. See DEVIATIONS.md.
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from cartpole_term.go1.env_cfg import TASK_ID, go1_velocity_term_env_cfg
from cartpole_term.go1.rl_cfg import go1_velocity_term_ppo_runner_cfg
from cartpole_term.go1.runner import Go1FrozenEvalRunner

_env_cfg = go1_velocity_term_env_cfg(play=False)

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=_env_cfg,
    play_env_cfg=_env_cfg,  # see module docstring: real play cfg not wired yet
    rl_cfg=go1_velocity_term_ppo_runner_cfg(),
    runner_cls=Go1FrozenEvalRunner,
)
