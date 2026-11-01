"""Phase-2 discovery script (not part of CLAUDE.md's cartpole-only spec, see
DEVIATIONS.md): dump the real mjlab API surface for the shipped Unitree Go1
velocity task, so the Go1 registration/scripts in cartpole_term/go1/ don't
have to guess a task id, factory name, obs/action width, or termination-term
name -- same discipline as scripts/inspect_mjlab.py used for cartpole at
gate 0 (CLAUDE.md S0.2: "never guess a signature").

Read-only -- does not modify anything, does not train anything. Needs a real
mjlab install and (for the live-env section) a GPU, same as every other
script in this project.

Run once, on the GPU machine, from the repo root:

    python scripts/inspect_go1_task.py

Then either paste the full output back, or -- since it also writes
outputs/go1_task_inspection.json -- just confirm it ran; the JSON is what
cartpole_term/go1/env_cfg.py's task-id discovery reads at import time, so as
long as this ran once, the rest of the Go1 code path is self-configuring.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path


def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_source(obj, label: str) -> None:
    section(label)
    try:
        print(inspect.getsource(obj))
    except (OSError, TypeError) as e:
        print(f"(source unavailable: {e})")


def main() -> None:
    from mjlab.tasks import registry as registry_mod
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    section("mjlab.tasks.registry -- public names (need: is there a play-cfg loader?)")
    print([n for n in dir(registry_mod) if not n.startswith("_")])
    print_source(registry_mod.register_mjlab_task, "SOURCE: register_mjlab_task (confirms how play_env_cfg is stored/retrieved)")

    section("list_tasks()")
    all_tasks = list_tasks()
    print(all_tasks)

    go1_candidates = [t for t in all_tasks if re.search(r"go1", t, re.IGNORECASE)]
    section("Task ids matching /go1/i")
    print(go1_candidates)

    flat_velocity_candidates = [
        t for t in go1_candidates
        if re.search(r"velocity", t, re.IGNORECASE) and re.search(r"flat", t, re.IGNORECASE)
    ]
    section("Narrowed to /velocity/i + /flat/i (the task we actually want)")
    print(flat_velocity_candidates)

    report: dict = {
        "all_tasks": all_tasks,
        "go1_candidates": go1_candidates,
        "flat_velocity_candidates": flat_velocity_candidates,
        "per_task": {},
    }

    inspect_ids = flat_velocity_candidates or go1_candidates
    if not inspect_ids:
        section("NO Go1 TASK FOUND")
        print(
            "No task id matched /go1/i in list_tasks(). This means either mjlab's "
            "Go1 velocity task isn't registered under the version installed here, "
            "or it's named differently than expected. Full all_tasks list is above "
            "-- inspect it by hand and tell me the actual id."
        )

    for task_id in inspect_ids:
        section(f"=== {task_id} ===")
        entry: dict = {"task_id": task_id}
        try:
            env_cfg = load_env_cfg(task_id)
        except Exception as e:  # noqa: BLE001 -- diagnostic script, report and continue
            print(f"load_env_cfg({task_id!r}) FAILED: {e!r}")
            entry["load_env_cfg_error"] = repr(e)
            report["per_task"][task_id] = entry
            continue

        entry["env_cfg_module"] = type(env_cfg).__module__
        entry["env_cfg_class"] = type(env_cfg).__name__
        print(f"env_cfg type: {entry['env_cfg_module']}.{entry['env_cfg_class']}")

        try:
            entry["decimation"] = env_cfg.decimation
            entry["timestep"] = env_cfg.sim.mujoco.timestep
            entry["episode_length_s"] = env_cfg.episode_length_s
            entry["default_num_envs"] = env_cfg.scene.num_envs
            print(f"decimation={entry['decimation']} timestep={entry['timestep']} "
                  f"episode_length_s={entry['episode_length_s']} "
                  f"default_num_envs={entry['default_num_envs']}")
        except Exception as e:  # noqa: BLE001
            print(f"(timing/scene fields: {e!r})")

        try:
            term_names = list(env_cfg.terminations.keys())
            term_timeouts = {}
            for name, term_cfg in env_cfg.terminations.items():
                term_timeouts[name] = getattr(term_cfg, "time_out", None)
            entry["termination_terms"] = term_names
            entry["termination_time_out_flags"] = term_timeouts
            print(f"termination terms: {term_timeouts}")
        except Exception as e:  # noqa: BLE001
            print(f"(terminations: {e!r})")

        try:
            reward_names = list(env_cfg.rewards.keys())
            reward_weights = {n: env_cfg.rewards[n].weight for n in reward_names}
            entry["reward_terms"] = reward_weights
            print(f"reward terms (weight): {reward_weights}")
        except Exception as e:  # noqa: BLE001
            print(f"(rewards: {e!r})")

        try:
            obs_groups = list(env_cfg.observations.keys())
            entry["observation_groups"] = {}
            for g in obs_groups:
                terms = env_cfg.observations[g]
                term_names_in_group = list(terms.keys()) if hasattr(terms, "keys") else [
                    getattr(t, "func", t) for t in terms
                ]
                entry["observation_groups"][g] = [str(x) for x in term_names_in_group]
            print(f"observation groups: {entry['observation_groups']}")
        except Exception as e:  # noqa: BLE001
            print(f"(observations: {e!r})")

        try:
            rl_cfg = load_rl_cfg(task_id)
            entry["rl_cfg"] = {
                "experiment_name": rl_cfg.experiment_name,
                "logger": rl_cfg.logger,
                "save_interval": rl_cfg.save_interval,
                "max_iterations": rl_cfg.max_iterations,
                "num_steps_per_env": rl_cfg.num_steps_per_env,
            }
            print(f"rl_cfg: {entry['rl_cfg']}")
        except Exception as e:  # noqa: BLE001
            print(f"(rl_cfg: {e!r})")

        report["per_task"][task_id] = entry

    # --- Live-env section: actual obs/action widths + termination-cause
    # attribution API, which only exist once a real env is constructed.
    # Requires GPU (same as every other script here).
    section("LIVE ENV: obs/action widths + termination_manager surface")
    try:
        import torch

        from mjlab.envs import ManagerBasedRlEnv

        if not torch.cuda.is_available():
            print("CUDA not available here -- skipping live-env construction. "
                  "Run this on the GPU machine to fill in this section.")
        else:
            chosen = inspect_ids[0]
            env_cfg = load_env_cfg(chosen)
            env_cfg.scene.num_envs = 4
            raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
            obs, _ = raw_env.reset()
            live: dict = {"task_id": chosen}
            for group_name, group_tensor in obs.items():
                print(f"obs[{group_name!r}].shape = {tuple(group_tensor.shape)}")
                live[f"obs_{group_name}_dim"] = int(group_tensor.shape[-1])

            action_dim = raw_env.action_manager.total_action_dim if hasattr(raw_env, "action_manager") else None
            print(f"action_manager.total_action_dim = {action_dim}")
            live["action_dim"] = action_dim

            tm = raw_env.termination_manager
            print(f"termination_manager public attrs: {[n for n in dir(tm) if not n.startswith('_')]}")
            live["termination_manager_public_attrs"] = [n for n in dir(tm) if not n.startswith("_")]

            active_terms = getattr(tm, "active_terms", None)
            print(f"termination_manager.active_terms = {active_terms}")
            live["termination_manager_active_terms"] = list(active_terms) if active_terms is not None else None

            # Confirm get_term(name) works for each registered term, exactly
            # like collect_frozen.py's cartpole cause-attribution call.
            for name in list(env_cfg.terminations.keys()):
                try:
                    val = tm.get_term(name)
                    print(f"termination_manager.get_term({name!r}) -> tensor shape {tuple(val.shape)}, dtype {val.dtype}")
                except Exception as e:  # noqa: BLE001
                    print(f"termination_manager.get_term({name!r}) FAILED: {e!r}")

            report["live_env"] = live
            raw_env.close()
    except Exception as e:  # noqa: BLE001 -- diagnostic script
        print(f"Live-env section failed: {e!r}")
        report["live_env_error"] = repr(e)

    out_path = Path("outputs")
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "go1_task_inspection.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote outputs/go1_task_inspection.json")


if __name__ == "__main__":
    main()
