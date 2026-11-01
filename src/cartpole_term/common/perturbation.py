"""Perturbation events that stop once a robot is down.

Drop-in replacements for mjlab's two push events, used by the live hand-off
env (scripts/go1_handoff_eval.py, scripts/go1_handoff_demo_video.py). Each
reads a per-env bool mask `env.perturbations_off` (set by the script every
step; absent = nobody is down) and skips those envs. A robot that has
already fallen to the ground is no longer shoved around: the push models the
disturbance that causes the fall, not something that keeps hitting a fallen
robot. Everything else is mjlab's own behaviour, unchanged.

Confirmed against mjlab 1.6.0's envs/mdp/events.py: `apply_body_impulse`
keeps per-env `_active`, `_time_remaining`, `_interval_time_left` and clears
an impulse with `write_external_wrench_to_sim(zeros, zeros, ...)`;
`push_by_setting_velocity` acts only on the `env_ids` it is given.
"""

from __future__ import annotations

import torch

from mjlab.envs.mdp.events import apply_body_impulse, push_by_setting_velocity

_NEVER_S = 1.0e9  # cooldown long enough that a down env is never eligible again


def perturbations_off_mask(env) -> torch.Tensor | None:
    return getattr(env, "perturbations_off", None)


class apply_body_impulse_until_down(apply_body_impulse):
    """mjlab's apply_body_impulse, except envs flagged in
    `env.perturbations_off` have any impulse in progress cleared at once and
    never start a new one (until the event's own reset on env reset)."""

    def __call__(self, env, env_ids, *args, **kwargs) -> None:
        off = perturbations_off_mask(env)
        if off is not None and bool(off.any()):
            cut = off & self._active
            if bool(cut.any()):
                ids = cut.nonzero(as_tuple=False).squeeze(-1)
                zeros = torch.zeros((len(ids), self._num_bodies, 3), device=self._device)
                self._asset.write_external_wrench_to_sim(zeros, zeros, env_ids=ids, body_ids=self._body_ids)
                self._active[ids] = False
                self._time_remaining[ids] = 0.0
            self._interval_time_left[off] = _NEVER_S
        super().__call__(env, env_ids, *args, **kwargs)


def push_by_setting_velocity_until_down(env, env_ids, velocity_range, **kwargs) -> None:
    """mjlab's push_by_setting_velocity, skipping envs flagged in
    `env.perturbations_off`."""
    off = perturbations_off_mask(env)
    if off is not None and env_ids is not None:
        env_ids = env_ids[~off[env_ids]]
        if len(env_ids) == 0:
            return
    push_by_setting_velocity(env, env_ids, velocity_range, **kwargs)


def stop_perturbations_when_down(cfg) -> None:
    """Swap an env cfg's push events for the until-down versions, in place.
    Fails loudly if an expected event is missing or is not mjlab's own."""
    impulse = cfg.events.get("wild_impulse")
    if impulse is None or impulse.func is not apply_body_impulse:
        raise KeyError("stop_perturbations_when_down: cfg.events['wild_impulse'] is not mjlab's apply_body_impulse")
    impulse.func = apply_body_impulse_until_down
    push = cfg.events.get("push_robot")
    if push is None or push.func is not push_by_setting_velocity:
        raise KeyError("stop_perturbations_when_down: cfg.events['push_robot'] is not mjlab's push_by_setting_velocity")
    push.func = push_by_setting_velocity_until_down
