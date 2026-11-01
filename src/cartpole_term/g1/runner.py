"""G1 velocity-term frozen-eval runner (phase-2, see DEVIATIONS.md).

Mirrors cartpole_term.go1.runner.Go1FrozenEvalRunner exactly -- see that
file's docstring for the mechanism being reused unmodified (CLAUDE.md S6's
periodic frozen evaluation, weight-freeze assertions, mean+sampled action
modes, dynamic termination-cause discovery, all inherited verbatim from
cartpole_term.runner.FrozenEvalRunner). Only the two robot-specific hooks
are overridden here, same as Go1's own runner. No G1 analog to cartpole's
cart-centering ablation metric exists either.
"""

from __future__ import annotations

from cartpole_term.runner import FrozenEvalRunner


class G1FrozenEvalRunner(FrozenEvalRunner):
    @staticmethod
    def _default_env_cfg_factory(play: bool = False):
        from cartpole_term.g1.env_cfg import g1_velocity_term_env_cfg

        return g1_velocity_term_env_cfg(play=play)

    @staticmethod
    def _default_extra_scalar_metric_fn(obs):
        return None  # no G1 analog to cartpole's mean_abs_cart_pos ablation metric


def build_g1_runner_from_checkpoint(
    checkpoint_path: str,
    seed: int,
    eval_num_envs: int = 256,
    eval_interval: int = 10,
    device: str = "cuda",
):
    """G1-flavored build_runner_from_checkpoint (CLAUDE.md S7.2.4 re-
    measurement, S7.3 collection) -- thin wrapper pinning runner_cls/
    env_cfg_factory/task_id to G1's, reusing the fully generalized
    cartpole_term.runner.build_runner_from_checkpoint underneath, exactly
    mirroring cartpole_term.go1.runner.build_go1_runner_from_checkpoint."""
    from cartpole_term.g1.env_cfg import TASK_ID, g1_velocity_term_env_cfg
    from cartpole_term.runner import build_runner_from_checkpoint

    return build_runner_from_checkpoint(
        checkpoint_path,
        seed=seed,
        eval_num_envs=eval_num_envs,
        eval_interval=eval_interval,
        device=device,
        runner_cls=G1FrozenEvalRunner,
        env_cfg_factory=g1_velocity_term_env_cfg,
        task_id=TASK_ID,
        extra_scalar_metric_fn=None,
    )
