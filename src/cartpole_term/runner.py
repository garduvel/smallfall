"""Frozen in-training evaluation (CLAUDE.md S6).

`MjlabOnPolicyRunner` (confirmed via scripts/inspect_runner.py) does not
override `learn()` -- it inherits the entire training loop from
`rsl_rl.runners.on_policy_runner.OnPolicyRunner`, with no callback/hook
mechanism. The only way to add a periodic frozen-eval measurement is to
override `learn()` and reproduce its outer loop, which is what this file
does. The loop body below through `self.save(...)` is the original
rsl_rl loop verbatim (confirmed via scripts/inspect_rsl_rl.py) with exactly
one addition: a call to `_run_frozen_eval` every `eval_interval` iterations.
No PPO math is touched -- every call in the loop still goes through the
untouched `self.alg` (rollout, GAE, PPO update) exactly as rsl_rl wrote it;
only the orchestration around those calls is duplicated. This is the
documented insertion point per CLAUDE.md S6.2, not a PPO reimplementation
and not a monkey-patch.

Scoping note on the termination-cause breakdown (CLAUDE.md S6.2.8): it is
computed from extras["log"]["Episode_Termination/<cause>"], the same
mechanism validated in scripts/check_terminations.py (Gate 1) -- summed
over the whole eval rollout. Because envs run in lockstep and auto-reset
(the default), an env that finishes its first episode early will start a
second episode while slower envs are still finishing their first; that
second episode's termination, if it happens before the rollout stops, also
gets counted here. This is a deliberate simplification: the alternative
(auto_reset=False plus manual per-env partial resets and TensorDict
splicing) needs an unconfirmed ManagerBasedRlEnvCfg.auto_reset field and is
considerably more fragile. Episode length, return, and censored-fraction
statistics are NOT affected by this -- those are computed per-env from the
`dones`/`truncated` tensors returned by step(), which are accurate
regardless of auto-reset, and are strictly scoped to each env's first
episode via the `active` mask.

Phase-2 generalization (outside CLAUDE.md's cartpole-only spec, see
DEVIATIONS.md): the cartpole gated pipeline never constructs this runner via
mjlab's `train` CLI subprocess -- scripts/train.py always constructs
runner_cls(...) directly (CLAUDE.md S2.9's "in-process training must
construct the runner directly" note) -- so there is no fixed-signature
constraint forcing env-cfg/task-id to be hardcoded here. `FrozenEvalRunner`
therefore accepts an optional `env_cfg_factory`/`task_id`/
`extra_scalar_metric_fn` triple, all keyword-only and all defaulting to
exactly cartpole's current values, so every existing cartpole call site
(none of which pass these) is byte-for-byte unaffected. A second task (e.g.
Go1 velocity) passes its own factory/id/metric-fn instead of subclassing.
The termination-cause breakdown itself was ALREADY task-agnostic in
mechanism (it walks extras["log"] by key) but had cartpole's two cause names
hardcoded as the only keys it would ever look for; that dict is now
discovered dynamically from whatever "Episode_Termination/<cause>" keys
actually appear in `log`, which is a strict generalization -- for cartpole
this still finds exactly {"pole_angle", "cart_pos", "time_out"}, same as
before.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from rsl_rl.utils import check_nan

if TYPE_CHECKING:
    from mjlab.rl import RslRlVecEnvWrapper as _EnvType

EVAL_NUM_ENVS_DEFAULT = 256
EVAL_INTERVAL_DEFAULT = 10  # matches save_interval, CLAUDE.md S6.2.1
EVAL_SEED_OFFSET = 100  # CLAUDE.md S6.2.3


class FrozenEvalRunner(MjlabOnPolicyRunner):
    """MjlabOnPolicyRunner + a periodic frozen evaluation on a separate env.

    Accepts the exact same construction signature as MjlabOnPolicyRunner so
    it works with mjlab's generic `runner_cls(env, agent_cfg, log_dir,
    device)` construction call (see mjlab/scripts/train.py); eval_num_envs
    and eval_interval are optional keyword-only overrides for callers that
    construct this runner directly (e.g. our own scripts/train.py), with
    defaults matching CLAUDE.md S6.2 exactly.
    """

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
        *,
        eval_num_envs: int = EVAL_NUM_ENVS_DEFAULT,
        eval_interval: int = EVAL_INTERVAL_DEFAULT,
        early_stop_patience: int | None = None,
        env_cfg_factory=None,
        extra_scalar_metric_fn=None,
    ) -> None:
        """early_stop_patience: not part of CLAUDE.md S6 -- a project-level
        addition (see DEVIATIONS.md, gate 5). If set, `learn()` stops early
        once `early_stop_patience` CONSECUTIVE frozen evals report
        mean_action.mean_episode_length at the env's max_episode_length
        (i.e. the policy is reliably surviving full episodes), rather than
        running out num_learning_iterations regardless. None (default)
        disables this -- matches stock behaviour.

        env_cfg_factory / extra_scalar_metric_fn: phase-2 generalization
        hooks (see module docstring), both keyword-only and both defaulting
        to cartpole's own values when omitted -- existing cartpole call
        sites are unaffected. env_cfg_factory(play: bool) must build the
        SAME dynamics/terminations/reward as the training env (CLAUDE.md
        S6.2.2). extra_scalar_metric_fn(obs) -> (name, per_env_tensor) | None
        is an optional bonus scalar accumulated (masked by `active`) over
        the eval rollout, same role as cartpole's mean_abs_cart_pos
        (CLAUDE.md S2.6 ablation support); None means no extra metric.
        """
        super().__init__(env, train_cfg, log_dir, device)
        self.eval_num_envs = eval_num_envs
        self.eval_interval = eval_interval
        self.early_stop_patience = early_stop_patience
        self.train_seed = self.cfg.get("seed", 0)
        self.eval_seed = self.train_seed + EVAL_SEED_OFFSET
        self._eval_env: "_EnvType | None" = None
        self._eval_history: list[dict] = []
        self._eval_env_build_info: dict | None = None
        self._consecutive_maxed_evals = 0
        self._should_stop_early = False
        self._env_cfg_factory = env_cfg_factory or self._default_env_cfg_factory
        self._extra_scalar_metric_fn = (
            extra_scalar_metric_fn if extra_scalar_metric_fn is not None else self._default_extra_scalar_metric_fn
        )

    @staticmethod
    def _default_env_cfg_factory(play: bool = False):
        from cartpole_term.env_cfg import cartpole_balance_term_env_cfg

        return cartpole_balance_term_env_cfg(play=play)

    @staticmethod
    def _default_extra_scalar_metric_fn(obs):
        # CLAUDE.md S2.6 ablation: mean |cart position|, read from
        # obs["actor"][:, 0] (cart_pos) BEFORE env.step() each iteration --
        # see the long comment in _run_eval_rollout for why pre-step.
        return "mean_abs_cart_pos", obs["actor"][:, 0].abs()

    # ------------------------------------------------------------------
    # Training loop: rsl_rl's OnPolicyRunner.learn(), verbatim, plus one
    # inserted call to _run_frozen_eval. See module docstring.
    # ------------------------------------------------------------------
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            # --- CLAUDE.md S6: frozen evaluation, every eval_interval iterations ---
            if it % self.eval_interval == 0:
                self._run_frozen_eval(it)

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[name-defined]

            if self._should_stop_early:
                print(
                    f"[frozen-eval] early stop at it={it}: {self.early_stop_patience} "
                    "consecutive evals at max episode length (DEVIATIONS.md, gate 5)."
                )
                break

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[name-defined]
            self.logger.stop_logging_writer()

    # ------------------------------------------------------------------
    # Frozen evaluation.
    # ------------------------------------------------------------------
    def _build_eval_env(self) -> "_EnvType":
        t0 = time.time()
        eval_cfg = self._env_cfg_factory(play=False)  # NOT play_env_cfg: S6.2.2
        eval_cfg.scene.num_envs = self.eval_num_envs
        raw_env = ManagerBasedRlEnv(cfg=eval_cfg, device=self.device)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=self.cfg["clip_actions"])
        build_time = time.time() - t0

        mem_gb = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_gb = torch.cuda.memory_allocated() / 1e9
        self._eval_env_build_info = {"build_time_s": build_time, "cuda_memory_allocated_gb": mem_gb}
        print(
            f"[frozen-eval] second env ({self.eval_num_envs} envs) built in "
            f"{build_time:.1f}s, CUDA memory now "
            f"{mem_gb:.2f} GB" if mem_gb is not None else f"{build_time:.1f}s (CPU)"
        )
        return wrapped

    def _run_frozen_eval(self, iteration: int) -> None:
        t0 = time.time()

        actor_snapshot = copy.deepcopy(self.alg.actor.state_dict())
        critic_snapshot = copy.deepcopy(self.alg.critic.state_dict())
        iteration_before = self.current_learning_iteration

        self.alg.eval_mode()
        if self._eval_env is None:
            self._eval_env = self._build_eval_env()

        results = {}
        for mode, stochastic in (("mean", False), ("sampled", True)):
            results[mode] = self._run_eval_rollout(seed=self.eval_seed, stochastic_output=stochastic)

        # CLAUDE.md S6.2.4: assert nothing changed.
        for name, before in actor_snapshot.items():
            after = self.alg.actor.state_dict()[name]
            assert torch.equal(before, after), f"actor.{name} changed during frozen eval"
        for name, before in critic_snapshot.items():
            after = self.alg.critic.state_dict()[name]
            assert torch.equal(before, after), f"critic.{name} changed during frozen eval"
        assert self.current_learning_iteration == iteration_before, (
            "current_learning_iteration changed during frozen eval -- an "
            "optimizer step must have run, which must never happen here."
        )

        self.alg.train_mode()

        record = {
            "iteration": iteration,
            "eval_seed": self.eval_seed,
            "wall_time_s": time.time() - t0,
            "eval_env_build_info": self._eval_env_build_info,
            "mean_action": results["mean"],
            "sampled_action": {
                "num_episodes": results["sampled"]["num_episodes"],
                "mean_episode_length": results["sampled"]["mean_episode_length"],
                "median_episode_length": results["sampled"]["median_episode_length"],
            },
        }
        self._eval_history.append(record)
        self._write_eval_history()

        mean_len = results["mean"]["mean_episode_length"]
        sampled_len = results["sampled"]["mean_episode_length"]
        extra_bits = ""
        if self._extra_scalar_metric_fn is not None:
            for k, v in results["mean"].items():
                if k in (
                    "num_episodes", "mean_episode_length", "median_episode_length",
                    "se_episode_length", "censored_fraction", "mean_return",
                    "termination_cause_breakdown", "termination_cause_counts_raw",
                ):
                    continue
                extra_bits += f"{k}={v:.3f} " if isinstance(v, (int, float)) else f"{k}={v} "
        print(
            f"[frozen-eval] it={iteration} mean_action_len={mean_len:.1f} "
            f"sampled_action_len={sampled_len:.1f} "
            f"{extra_bits}"
            f"censored_frac={results['mean']['censored_fraction']:.3f} "
            f"causes={results['mean']['termination_cause_breakdown']}"
        )

        if self.early_stop_patience is not None:
            max_len = self._eval_env.max_episode_length  # type: ignore[union-attr]
            if mean_len >= max_len:
                self._consecutive_maxed_evals += 1
            else:
                self._consecutive_maxed_evals = 0
            if self._consecutive_maxed_evals >= self.early_stop_patience:
                self._should_stop_early = True

    def evaluate(self, seed: int, stochastic_output: bool = False) -> dict:
        """Public one-off frozen evaluation, for callers outside the
        training loop (scripts/select_checkpoints.py's re-measurement,
        CLAUDE.md S7.2.4; scripts/collect_frozen.py's rollout collection,
        S7.3). Lazily builds the eval env if it doesn't exist yet. Does NOT
        perform the "weights unchanged" snapshot/assert dance that
        `_run_frozen_eval` does around training -- callers here are not
        mid-training, so there is nothing to protect against.
        """
        self.alg.eval_mode()
        if self._eval_env is None:
            self._eval_env = self._build_eval_env()
        result = self._run_eval_rollout(seed=seed, stochastic_output=stochastic_output)
        self.alg.train_mode()
        return result

    def _run_eval_rollout(self, seed: int, stochastic_output: bool) -> dict:
        env = self._eval_env
        assert env is not None
        env.seed(seed)
        # Reset inside inference mode too: the rollout below steps inside it,
        # so any buffer an env term (re)allocates there is an inference tensor,
        # and resetting it in place outside inference mode at the next eval
        # raises. Hit by the G1's contact-sensor history buffer (DEVIATIONS.md,
        # 2026-11-08); same bug class as the earlier settled/impulse crashes.
        with torch.inference_mode():
            obs, _ = env.reset()
        obs = obs.to(self.device)

        num_envs = env.num_envs
        device = self.device

        active = torch.ones(num_envs, dtype=torch.bool, device=device)
        episode_return = torch.zeros(num_envs, device=device)
        recorded_length = torch.zeros(num_envs, dtype=torch.long)
        recorded_return = torch.zeros(num_envs)
        recorded_truncated = torch.zeros(num_envs, dtype=torch.bool)

        # Discovered dynamically from whatever "Episode_Termination/<cause>"
        # keys actually appear in extras["log"] (phase-2 generalization, see
        # module docstring) -- for cartpole this still finds exactly
        # {"pole_angle", "cart_pos", "time_out"}.
        cause_totals: dict[str, float] = {}
        # Extra bonus scalar (CLAUDE.md S2.6 ablation support for cartpole;
        # see _default_extra_scalar_metric_fn / module docstring). Read
        # BEFORE calling env.step() each iteration, i.e. the state the
        # policy is actually acting from -- not a post-step query, which
        # scripts/inspect_reset_behavior.py showed is contaminated by
        # auto-reset for envs that terminate that step.
        extra_metric_name: str | None = None
        extra_metric_sum = 0.0
        extra_metric_count = 0

        max_steps = int(env.max_episode_length)
        with torch.inference_mode():
            for step in range(1, max_steps + 1):
                if not bool(active.any()):
                    break

                if self._extra_scalar_metric_fn is not None:
                    metric = self._extra_scalar_metric_fn(obs)
                    if metric is not None:
                        name, per_env_value = metric
                        extra_metric_name = name
                        extra_metric_sum += float((per_env_value * active.float()).sum().item())
                        extra_metric_count += int(active.sum().item())

                actions = self.alg.actor(obs, stochastic_output=stochastic_output)
                obs, rewards, dones, extras = env.step(actions)
                obs = obs.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device).bool()
                time_outs = extras.get("time_outs")
                truncated = (
                    time_outs.to(device).bool() if time_outs is not None else torch.zeros_like(dones)
                )

                episode_return += rewards * active.float()

                newly_done = active & dones
                if bool(newly_done.any()):
                    idx = newly_done.nonzero(as_tuple=False).squeeze(-1)
                    recorded_length[idx] = step
                    recorded_return[idx] = episode_return[idx].detach().cpu()
                    recorded_truncated[idx] = truncated[idx].detach().cpu()
                    active = active & ~dones

                log = extras.get("log", {})
                for key, val in log.items():
                    if key.startswith("Episode_Termination/"):
                        cause = key[len("Episode_Termination/") :]
                        cause_totals[cause] = cause_totals.get(cause, 0.0) + float(val)

        if bool(active.any()):
            remaining = int(active.sum())
            raise AssertionError(
                f"{remaining}/{num_envs} eval envs never completed an episode "
                f"within max_episode_length={max_steps} steps -- CLAUDE.md "
                "S6.2.5 requires running until all episodes complete."
            )

        lengths = recorded_length.float()
        returns = recorded_return
        censored = recorded_truncated.float()

        n = num_envs
        mean_len = lengths.mean().item()
        median_len = lengths.median().item()
        se_len = (lengths.std(unbiased=True) / (n**0.5)).item() if n > 1 else 0.0
        censored_fraction = censored.mean().item()
        mean_return = returns.mean().item()

        total_cause = sum(cause_totals.values())
        cause_fractions = (
            {k: v / total_cause for k, v in cause_totals.items()}
            if total_cause > 0
            else dict.fromkeys(cause_totals, 0.0)
        )

        result = {
            "num_episodes": n,
            "mean_episode_length": mean_len,
            "median_episode_length": median_len,
            "se_episode_length": se_len,
            "censored_fraction": censored_fraction,
            "mean_return": mean_return,
            "termination_cause_breakdown": cause_fractions,
            "termination_cause_counts_raw": dict(cause_totals),
        }
        if extra_metric_name is not None:
            result[extra_metric_name] = extra_metric_sum / extra_metric_count if extra_metric_count > 0 else None
        return result

    def _write_eval_history(self) -> None:
        path = Path(self.logger.log_dir) / "eval_history.json"
        path.write_text(json.dumps(self._eval_history, indent=2))


def build_runner_from_checkpoint(
    checkpoint_path: str,
    seed: int,
    eval_num_envs: int = EVAL_NUM_ENVS_DEFAULT,
    eval_interval: int = EVAL_INTERVAL_DEFAULT,
    device: str = "cuda",
    *,
    runner_cls: type[FrozenEvalRunner] | None = None,
    env_cfg_factory=None,
    task_id: str | None = None,
    extra_scalar_metric_fn=None,
) -> FrozenEvalRunner:
    """Construct a FrozenEvalRunner with actor weights loaded from a saved
    checkpoint, for one-off evaluation outside a training run
    (scripts/select_checkpoints.py's re-measurement, CLAUDE.md S7.2.4;
    scripts/collect_frozen.py's rollout collection, S7.3).

    Builds a minimal (1-env) training-shaped env purely to satisfy
    OnPolicyRunner.__init__'s shape-inference requirements (it queries
    env.get_observations() to size the actor/critic) -- that env is never
    stepped; all real evaluation happens through FrozenEvalRunner.evaluate(),
    which builds its own eval env internally.

    Loads only the actor (load_cfg={"actor": True}, confirmed via
    scripts/inspect_runner.py's PPO.load() source) -- optimizer/critic state
    is irrelevant for a frozen, non-training evaluation.

    runner_cls / env_cfg_factory / task_id / extra_scalar_metric_fn: phase-2
    generalization hooks (see module docstring), all keyword-only and all
    defaulting to exactly cartpole's own values when omitted -- existing
    cartpole call sites (none of which pass these) are unaffected.
    """
    from dataclasses import asdict

    from mjlab.tasks.registry import load_rl_cfg

    from cartpole_term.env_cfg import TASK_ID, cartpole_balance_term_env_cfg

    runner_cls = runner_cls or FrozenEvalRunner
    env_cfg_factory = env_cfg_factory or cartpole_balance_term_env_cfg
    task_id = task_id or TASK_ID

    rl_cfg = load_rl_cfg(task_id)
    rl_cfg.seed = seed
    agent_cfg = asdict(rl_cfg)

    env_cfg = env_cfg_factory(play=False)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = seed
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)

    runner = runner_cls(
        env,
        agent_cfg,
        log_dir=None,
        device=device,
        env_cfg_factory=env_cfg_factory,
        extra_scalar_metric_fn=extra_scalar_metric_fn,
        eval_num_envs=eval_num_envs,
        eval_interval=eval_interval,
    )
    runner.load(checkpoint_path, load_cfg={"actor": True}, strict=True, map_location=device)
    return runner
