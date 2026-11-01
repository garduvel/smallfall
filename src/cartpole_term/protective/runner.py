"""Protective-fall frozen-eval runner: mirrors
cartpole_term.go1.runner.Go1FrozenEvalRunner's own minimal-override pattern
exactly -- only the env-cfg-factory hook differs. No Go1-walking analog
exists for the S2.6-style extra-scalar-metric hook, same reasoning
Go1FrozenEvalRunner already gives for its own `None`.
"""

from __future__ import annotations

from cartpole_term.runner import FrozenEvalRunner


class ProtectiveFrozenEvalRunner(FrozenEvalRunner):
    @staticmethod
    def _default_env_cfg_factory(play: bool = False):
        from cartpole_term.protective.env_cfg import go1_protective_fall_env_cfg

        return go1_protective_fall_env_cfg(play=play)

    @staticmethod
    def _default_extra_scalar_metric_fn(obs):
        return None
