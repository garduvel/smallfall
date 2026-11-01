"""Reset event: literal fall-state injection for the protective "small-fall"
task (Gate 5 of ~/.claude/plans/snug-booping-gosling.md).

The plan's original Gate 5 design chose a "rollout-then-handoff" init
mechanism specifically BECAUSE literal state injection was believed to need
"an unconfirmed mjlab 'set full state directly' API." That API is real and
confirmed this session: `Entity.write_root_link_pose_to_sim`,
`write_root_link_velocity_to_sim`, and `write_joint_state_to_sim`
(src/mjlab/entity/entity.py, mujocolab/mjlab@main) -- and the calling
convention below (a plain `func(env, env_ids, **params) -> None`, resolving
`env_ids=None` to "all environments", adding `env.scene.env_origins` back
onto a stored env-local position) is copied directly from mjlab's own
shipped `reset_root_state_uniform` (src/mjlab/envs/mdp/events.py), not
guessed. See DEVIATIONS.md for the full writeup superseding the plan's
"rollout-then-handoff" choice.

The state bank itself is produced by
scripts/go1_collect_flagged_states.py's `--capture-state-bank`: for every
real fall the live predictor flagged in advance, one raw physics state
(root pose/velocity + joint pos/velocity) reservoir-sampled uniformly from
[flag_step, termination_step) -- i.e. real "predictor-flagged unsafe
states," the same population SafeFall's own Stage II inits episodes from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


class FallStateBank:
    """Loads a go1_fall_state_bank_seed{N}.npz once and keeps it as
    device-resident tensors for repeated random sampling during training --
    avoids re-reading the file or re-transferring host->device every reset.
    """

    def __init__(self, npz_path: str, device: str):
        data = np.load(npz_path)
        if len(data["episode_id"]) == 0:
            raise ValueError(
                f"Fall-state bank at {npz_path} is empty (0 captured states) -- "
                "was --capture-state-bank actually set when it was collected?"
            )
        self.root_pos_local = torch.as_tensor(data["root_pos_local"], device=device, dtype=torch.float32)
        self.root_quat = torch.as_tensor(data["root_quat"], device=device, dtype=torch.float32)
        self.root_lin_vel = torch.as_tensor(data["root_lin_vel"], device=device, dtype=torch.float32)
        self.root_ang_vel = torch.as_tensor(data["root_ang_vel"], device=device, dtype=torch.float32)
        self.joint_pos = torch.as_tensor(data["joint_pos"], device=device, dtype=torch.float32)
        self.joint_vel = torch.as_tensor(data["joint_vel"], device=device, dtype=torch.float32)
        self.size = len(data["episode_id"])
        self.source_path = npz_path

    def sample(self, n: int, device: str) -> torch.Tensor:
        return torch.randint(0, self.size, (n,), device=device)


def inject_fall_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    state_bank_path: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sequential: bool = False,
) -> None:
    """Reset event (mode="reset"): instead of the default standing pose,
    place the robot in a REAL, previously-recorded mid-fall state, drawn
    uniformly at random from the fall-state bank.

    `sequential=True` instead gives env i bank state (i mod bank size) on
    every reset -- deterministic, so different policies can be compared on
    identical falls (scripts/go1_protective_compare.py). Training leaves it
    False.

    `env_ids=None` means "all environments" -- resolved to a concrete index
    tensor the same way mjlab's own reset_root_state_uniform does (never a
    guessed convention). The state bank's root_pos_local is relative to
    ITS OWN original collection env's origin; env.scene.env_origins[env_ids]
    is re-added here for THIS env's own grid position -- the same
    add-back reset_root_state_uniform performs, for the same reason (fixed
    per-env grid offsets, not physically meaningful positions to reuse
    verbatim across different env instances).
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

    bank = _get_or_load_bank(env, state_bank_path)
    asset: Entity = env.scene[asset_cfg.name]

    if sequential:
        sample_idx = env_ids.long() % bank.size
    else:
        sample_idx = bank.sample(len(env_ids), env.device)

    positions = bank.root_pos_local[sample_idx] + env.scene.env_origins[env_ids]
    orientations = bank.root_quat[sample_idx]
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)

    lin_vel = bank.root_lin_vel[sample_idx]
    ang_vel = bank.root_ang_vel[sample_idx]
    asset.write_root_link_velocity_to_sim(torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids)

    asset.write_joint_state_to_sim(bank.joint_pos[sample_idx], bank.joint_vel[sample_idx], env_ids=env_ids)


_BANK_CACHE: dict[tuple[int, str], FallStateBank] = {}


def _get_or_load_bank(env: ManagerBasedRlEnv, state_bank_path: str) -> FallStateBank:
    """Cached by (id(env), path) so the SAME env instance never reloads the
    bank on every reset call, but two different env instances (e.g. a
    training env and a separate eval env) each get their own -- avoids a
    module-global singleton silently shared across unrelated env objects."""
    key = (id(env), str(Path(state_bank_path).resolve()))
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = FallStateBank(state_bank_path, device=env.device)
    return _BANK_CACHE[key]
