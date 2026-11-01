"""Go1 velocity-term frozen-eval runner (phase-2, see DEVIATIONS.md).

Reuses cartpole_term.runner.FrozenEvalRunner's entire mechanism (CLAUDE.md
S6: periodic frozen evaluation on a separate, identically-configured env,
fixed eval seed, weight-freeze assertions, mean+sampled action modes,
termination-cause breakdown) unmodified -- only the two hooks that were
cartpole-specific are overridden here:

- _default_env_cfg_factory: points at the Go1 env cfg instead of cartpole's.
- _default_extra_scalar_metric_fn: cartpole's version reads mean |cart
  position| for the S2.6 centering ablation, which has no Go1 analog (no
  cart-centering reward exists on this task); returns None (no extra
  metric) rather than guessing a substitute.

Everything else -- the training-loop override, the eval-rollout loop, the
dynamic termination-cause discovery, weight-freeze assertions -- is
inherited verbatim from FrozenEvalRunner.
"""

from __future__ import annotations

from cartpole_term.runner import FrozenEvalRunner


class Go1FrozenEvalRunner(FrozenEvalRunner):
    @staticmethod
    def _default_env_cfg_factory(play: bool = False):
        from cartpole_term.go1.env_cfg import go1_velocity_term_env_cfg

        return go1_velocity_term_env_cfg(play=play)

    @staticmethod
    def _default_extra_scalar_metric_fn(obs):
        return None  # no Go1 analog to cartpole's mean_abs_cart_pos ablation metric


def build_go1_runner_from_checkpoint(
    checkpoint_path: str,
    seed: int,
    eval_num_envs: int = 256,
    eval_interval: int = 10,
    device: str = "cuda",
):
    """Go1-flavored build_runner_from_checkpoint (CLAUDE.md S7.2.4 re-
    measurement, S7.3 collection) -- thin wrapper pinning runner_cls/
    env_cfg_factory/task_id to Go1's, reusing the fully generalized
    cartpole_term.runner.build_runner_from_checkpoint underneath."""
    from cartpole_term.go1.env_cfg import TASK_ID, go1_velocity_term_env_cfg
    from cartpole_term.runner import build_runner_from_checkpoint

    return build_runner_from_checkpoint(
        checkpoint_path,
        seed=seed,
        eval_num_envs=eval_num_envs,
        eval_interval=eval_interval,
        device=device,
        runner_cls=Go1FrozenEvalRunner,
        env_cfg_factory=go1_velocity_term_env_cfg,
        task_id=TASK_ID,
        extra_scalar_metric_fn=None,
    )
