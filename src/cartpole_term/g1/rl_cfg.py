"""PPO runner config for the G1 velocity-term task: mjlab's shipped G1
rl_cfg, unmodified except the same class of overrides CLAUDE.md S5.3
permits for cartpole and cartpole_term.go1.rl_cfg already applies for Go1
(own experiment name, tensorboard logger, save_interval matching
eval_interval so every saved checkpoint has a frozen-eval measurement --
CLAUDE.md S6.2.1's reasoning applies identically here). No other
hyperparameter is touched, same discipline as every rl_cfg.py in this
project.
"""

from __future__ import annotations

from cartpole_term.g1.env_cfg import discover_source_task_id


def g1_velocity_term_ppo_runner_cfg():
    from mjlab.tasks.registry import load_rl_cfg

    cfg = load_rl_cfg(discover_source_task_id())
    cfg.experiment_name = "g1_velocity_term"
    cfg.logger = "tensorboard"  # shipped default may be wandb, same override as cartpole/Go1
    cfg.save_interval = 10  # matches cartpole_term.go1.rl_cfg's own override
    return cfg
