"""PPO runner config for the Go1 velocity-term task: mjlab's shipped Go1
rl_cfg, unmodified except the same class of overrides CLAUDE.md S5.3
permits for cartpole (own experiment name, tensorboard logger, save_interval
matching eval_interval so every saved checkpoint has a frozen-eval
measurement -- CLAUDE.md S6.2.1's reasoning applies identically here). No
other hyperparameter is touched, same discipline as cartpole_term/rl_cfg.py.
"""

from __future__ import annotations

from cartpole_term.go1.env_cfg import discover_source_task_id


def go1_velocity_term_ppo_runner_cfg():
    from mjlab.tasks.registry import load_rl_cfg

    cfg = load_rl_cfg(discover_source_task_id())
    cfg.experiment_name = "go1_velocity_term"
    cfg.logger = "tensorboard"  # shipped default may be wandb, same override as cartpole
    cfg.save_interval = 10  # matches cartpole_term.rl_cfg's S5.3.2 override
    return cfg
