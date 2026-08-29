"""PPO runner config: the installed `cartpole_ppo_runner_cfg()` verbatim,
with only the permitted overrides (CLAUDE.md S5.1, S5.3).

Do NOT retype hyperparameters from memory -- call the installed factory and
mutate only the three permitted fields, so nothing here can drift from the
authoritative installed source.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_ppo_runner_cfg


def cartpole_term_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = cartpole_ppo_runner_cfg()
    cfg.experiment_name = "cartpole_term"  # CLAUDE.md S5.3.1
    cfg.logger = "tensorboard"  # CLAUDE.md S5.3.1b -- shipped default is wandb
    cfg.save_interval = 10  # CLAUDE.md S5.3.2
    return cfg
