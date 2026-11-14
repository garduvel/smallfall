"""Settle lock (G1 round 3, DEVIATIONS.md 2026-11-14): if a protective fall
has not come to rest on its own, freeze the robot's joints so it does.

The lock replaces the task's joint-position action term. Until an env locks,
it is mjlab's JointPositionAction unchanged (target = default pose + scale x
action). An env locks, once per episode, on the first policy step where
either

  * the episode (= the fall, which starts at the injected state) has run
    `lock_after_s` seconds, or
  * the robot is low and nearly still: root height below `still_max_height_m`
    and root linear / angular speed below `still_lin_mps` / `still_ang_rad_s`
    for `still_hold_s` seconds running.

From then on every joint's position target is the joint's own position at
the lock step, so the normal PD actuators hold the pose ("joints lock"); the
policy's actions are ignored for the rest of the episode. Holding the pose,
rather than going limp, keeps the body as compact as it was when it locked.

mjlab facts used (read from mjlab 1.6.0's source, not guessed):
  * ActionTermCfg.build(env) constructs the term; JointPositionAction keeps
    its processed targets in `_processed_actions` and `apply_actions` sends
    `_processed_actions - encoder_bias` to `set_joint_position_target`, so a
    processed target of `joint_pos_biased` holds the current position.
  * ManagerBasedRlEnv.step calls action_manager.process_action once per
    policy step before physics, and increments episode_length_buf after it;
    episode_length_buf is zeroed on reset. So at process time
    episode_length_buf = policy steps already taken in this episode.
  * ActionManager.reset calls every term's reset(env_ids).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

LOCK_AFTER_S = 3.0
STILL_LIN_MPS = 0.3       # 1.5x the `settled` thresholds (0.2 m/s, 0.5 rad/s)
STILL_ANG_RAD_S = 0.75
STILL_HOLD_S = 0.3
STILL_MAX_HEIGHT_FRACTION = 0.5  # of the robot's standing root height
# Standing root height, mjlab init_state (same numbers as START_HEIGHT_M in
# scripts/robot_fall_size_ablation.py).
STANDING_ROOT_HEIGHT_M = {"go1": 0.278, "g1": 0.76}


@dataclass(kw_only=True)
class SettleLockActionCfg(JointPositionActionCfg):
    lock_after_s: float = LOCK_AFTER_S
    still_lin_mps: float = STILL_LIN_MPS
    still_ang_rad_s: float = STILL_ANG_RAD_S
    still_hold_s: float = STILL_HOLD_S
    still_max_height_m: float = 0.38

    def build(self, env) -> "SettleLockAction":
        return SettleLockAction(self, env)


def lock_update(*, locked, lock_target, lock_step, by_timer, still_count, t, lin, ang, height, joint_pos,
                lock_steps: int, still_steps: int, lin_max: float, ang_max: float, height_max: float):
    """One policy step of the lock logic, out of place. Returns the new
    (locked, lock_target, lock_step, by_timer, still_count)."""
    still = (lin < lin_max) & (ang < ang_max) & (height < height_max)
    still_count = torch.where(still, still_count + 1, torch.zeros_like(still_count))
    timer = t >= lock_steps
    new = ~locked & (timer | (still_count >= still_steps))
    lock_target = torch.where(new.unsqueeze(-1), joint_pos, lock_target)
    lock_step = torch.where(new, t, lock_step)
    by_timer = torch.where(new, timer & (still_count < still_steps), by_timer)
    return locked | new, lock_target, lock_step, by_timer, still_count


class SettleLockAction(JointPositionAction):
    def __init__(self, cfg: SettleLockActionCfg, env):
        super().__init__(cfg=cfg, env=env)
        self._lock_steps = max(1, round(cfg.lock_after_s / env.step_dt))
        self._still_steps = max(1, round(cfg.still_hold_s / env.step_dt))
        n, dev = self.num_envs, self.device
        self._locked = torch.zeros(n, dtype=torch.bool, device=dev)
        self._by_timer = torch.zeros(n, dtype=torch.bool, device=dev)
        self._lock_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._still_count = torch.zeros(n, dtype=torch.long, device=dev)
        self._lock_target = torch.zeros(n, self.action_dim, device=dev)

    @property
    def locked(self) -> torch.Tensor:
        return self._locked

    @property
    def lock_step(self) -> torch.Tensor:
        """Policy step of this episode at which the env locked (-1 = not locked)."""
        return self._lock_step

    @property
    def locked_by_timer(self) -> torch.Tensor:
        return self._by_timer

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        d = self._entity.data
        self._locked, self._lock_target, self._lock_step, self._by_timer, self._still_count = lock_update(
            locked=self._locked, lock_target=self._lock_target, lock_step=self._lock_step,
            by_timer=self._by_timer, still_count=self._still_count,
            t=self._env.episode_length_buf.to(torch.long),
            lin=torch.norm(d.root_link_lin_vel_w, dim=-1), ang=torch.norm(d.root_link_ang_vel_w, dim=-1),
            height=d.root_link_pos_w[:, 2], joint_pos=d.joint_pos_biased[:, self._target_ids],
            lock_steps=self._lock_steps, still_steps=self._still_steps, lin_max=self.cfg.still_lin_mps,
            ang_max=self.cfg.still_ang_rad_s, height_max=self.cfg.still_max_height_m,
        )
        self._processed_actions = torch.where(self._locked.unsqueeze(-1), self._lock_target, self._processed_actions)
        log = self._env.extras.get("log")
        if log is not None:
            log["Metrics/settle_locked_fraction"] = self._locked.float().mean()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        # Out of place, as in terminations.settled: the buffers can be
        # inference tensors under the frozen-eval runner.
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[slice(None) if env_ids is None else env_ids] = True
        self._locked = self._locked & ~mask
        self._by_timer = self._by_timer & ~mask
        self._lock_step = torch.where(mask, torch.full_like(self._lock_step, -1), self._lock_step)
        self._still_count = torch.where(mask, torch.zeros_like(self._still_count), self._still_count)


def settle_lock_action_cfg(base: JointPositionActionCfg, still_max_height_m: float, **overrides) -> SettleLockActionCfg:
    """The task's joint_pos action cfg with the settle lock added (every
    field of `base` copied)."""
    import dataclasses

    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    return SettleLockActionCfg(**fields, still_max_height_m=still_max_height_m, **overrides)


def settle_locked(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Log-only metric: 1 where the env has locked (reduce="max" gives the
    share of episodes that needed the lock)."""
    return env.action_manager.get_term(action_name).locked.float()
