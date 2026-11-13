"""PPO runner config for the protective-fall task: Go1's own shipped PPO
ALGORITHM hyperparameters (lr, clip_param, entropy_coef, gamma, lam, etc.)
are reused verbatim, unmodified -- CLAUDE.md S5.3's "no hyperparameter
tuning" discipline, extended to this phase-2 task, same as
cartpole_term/go1/rl_cfg.py.

`max_iterations` is NOT one of those algorithm hyperparameters -- CLAUDE.md
S5.3 itself treats it as task-dependent ("max_iterations may be raised if
the policy hasn't converged, but only after reporting the training
curve"), and mjlab's shipped value (10_000, confirmed via
mjlab/tasks/velocity/config/go1/rl_cfg.py) is calibrated for the WALKING
task -- 1000-step episodes, a much harder locomotion skill to learn. This
task's episodes are now variable length, ending once the robot settles
(see env_cfg.py/terminations.py -- revised from an earlier fixed 10-step
design that was cutting episodes off mid-fall, see DEVIATIONS.md), with a
150-step safety-cap budget; at the shipped num_steps_per_env=24, ONE
iteration still covers many episodes per env per iteration across the
default 4096 envs -- reusing 10_000 iterations unmodified would still be a
first, un-thought-through inheritance, not a reasoned choice, and would
waste a large amount of GPU time on a much simpler reward-shaping problem.

500 is a bounded, reasoned STARTING POINT (not independently validated --
neither cited paper's own iteration count was fetched this session),
watched via the same frozen-eval mechanism (CLAUDE.md S6) every other
task in this project uses; raise via --max-iterations if the eval curve
hasn't converged by then, per CLAUDE.md S5.3's own explicit protocol.
"""

from __future__ import annotations

# Round 7: 500 -> 300. Round 6's frozen-eval return was flat from ~iteration
# 200 (-1.62 at 200, -1.60 at 490); see DEVIATIONS.md.
PROTECTIVE_MAX_ITERATIONS = 300


def go1_protective_ppo_runner_cfg():
    from cartpole_term.go1.rl_cfg import go1_velocity_term_ppo_runner_cfg

    cfg = go1_velocity_term_ppo_runner_cfg()
    cfg.experiment_name = "go1_protective_fall"
    cfg.max_iterations = PROTECTIVE_MAX_ITERATIONS
    return cfg


def g1_protective_ppo_runner_cfg():
    """Same overrides on the G1's own shipped PPO cfg (identical network and
    PPO settings to the Go1's, per mjlab 1.6.0's g1/rl_cfg.py)."""
    from cartpole_term.g1.rl_cfg import g1_velocity_term_ppo_runner_cfg

    cfg = g1_velocity_term_ppo_runner_cfg()
    cfg.experiment_name = "g1_protective_fall"
    cfg.max_iterations = PROTECTIVE_MAX_ITERATIONS
    return cfg
