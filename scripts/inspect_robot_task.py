"""Phase-2 discovery script (not part of CLAUDE.md's cartpole-only spec,
see DEVIATIONS.md): dump the real mjlab API surface for a given robot's
velocity-flat task, so nothing downstream has to guess a task id, factory
name, obs/action width, termination-term name, or push-perturbation
config -- same discipline scripts/inspect_go1_task.py used for Go1's first
pass (CLAUDE.md S0.2: "never guess a signature"), generalized via
cartpole_term.common.RobotProfile instead of a Go1-only hardcoded script.

Read-only -- does not modify anything, does not train anything. Needs a
real mjlab install and (for the live-env section) a GPU, same as
inspect_go1_task.py.

Run once per robot, on the GPU machine, from the repo root:

    python scripts/inspect_robot_task.py --robot go1
    python scripts/inspect_robot_task.py --robot g1

Writes outputs/{robot}_task_inspection.json.

For go1, this must reproduce every already-known fact unchanged (task id
Mjlab-Velocity-Flat-Unitree-Go1, obs=48/action=12, single cause
fell_over, 50 Hz/1000-step episodes) -- a regression check confirming that
generalizing task discovery (cartpole_term.common.discovery) changed
nothing about Go1's actual behavior.

For g1, every number in this script's output is meant to be read off its
LIVE run, not assumed from source review -- in particular: the live
obs/action tensor widths (which then get copied into
RobotProfile.expected_obs_dim/expected_action_dim in
cartpole_term/common/robot_profile.py as a follow-up edit), whether any
termination term name contains "collision" (resolves the open question of
whether G1's self-collision reward also terminates the episode), and
whether push_robot's params match the same values already confirmed for
Go1 (both tasks are expected to share the exact same push_robot event,
per this session's GitHub source review, but this script confirms that
live rather than trusting the source read alone).
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", required=True, choices=("go1", "g1"))
    args = parser.parse_args()

    from cartpole_term.common.discovery import discover_task_id
    from cartpole_term.common.robot_profile import get_profile

    profile = get_profile(args.robot)

    from mjlab.tasks import registry as registry_mod
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

    section("mjlab.tasks.registry -- public names (need: is there a play-cfg loader?)")
    print([n for n in dir(registry_mod) if not n.startswith("_")])
    print_source(
        registry_mod.register_mjlab_task,
        "SOURCE: register_mjlab_task (confirms how play_env_cfg is stored/retrieved)",
    )

    section("list_tasks()")
    all_tasks = list_tasks()
    print(all_tasks)

    section(f"discover_task_id(profile={profile.name!r})")
    task_id = discover_task_id(profile)
    print(task_id)

    report: dict = {
        "robot": profile.name,
        "all_tasks": all_tasks,
        "discovered_task_id": task_id,
        "per_task": {},
    }

    section(f"=== {task_id} ===")
    entry: dict = {"task_id": task_id}
    try:
        env_cfg = load_env_cfg(task_id)
    except Exception as e:  # noqa: BLE001 -- diagnostic script, report and continue
        print(f"load_env_cfg({task_id!r}) FAILED: {e!r}")
        entry["load_env_cfg_error"] = repr(e)
        report["per_task"][task_id] = entry
        out_path = Path("outputs")
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / f"{profile.name}_task_inspection.json").write_text(json.dumps(report, indent=2, default=str))
        return

    entry["env_cfg_module"] = type(env_cfg).__module__
    entry["env_cfg_class"] = type(env_cfg).__name__
    print(f"env_cfg type: {entry['env_cfg_module']}.{entry['env_cfg_class']}")

    try:
        entry["decimation"] = env_cfg.decimation
        entry["timestep"] = env_cfg.sim.mujoco.timestep
        entry["episode_length_s"] = env_cfg.episode_length_s
        entry["default_num_envs"] = env_cfg.scene.num_envs
        print(
            f"decimation={entry['decimation']} timestep={entry['timestep']} "
            f"episode_length_s={entry['episode_length_s']} "
            f"default_num_envs={entry['default_num_envs']}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"(timing/scene fields: {e!r})")

    term_names: list[str] = []
    try:
        term_names = list(env_cfg.terminations.keys())
        term_timeouts = {name: getattr(term_cfg, "time_out", None) for name, term_cfg in env_cfg.terminations.items()}
        entry["termination_terms"] = term_names
        entry["termination_time_out_flags"] = term_timeouts
        print(f"termination terms: {term_timeouts}")

        collision_terms = [n for n in term_names if re.search(r"collision", n, re.IGNORECASE)]
        entry["collision_termination_terms"] = collision_terms
        print(f"collision-named termination terms: {collision_terms or '(none)'}")
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
            term_names_in_group = (
                list(terms.keys()) if hasattr(terms, "keys") else [getattr(t, "func", t) for t in terms]
            )
            entry["observation_groups"][g] = [str(x) for x in term_names_in_group]
        print(f"observation groups: {entry['observation_groups']}")
    except Exception as e:  # noqa: BLE001
        print(f"(observations: {e!r})")

    try:
        push_cfg = env_cfg.events.get("push_robot")
        if push_cfg is not None:
            entry["push_robot_params"] = {
                "mode": getattr(push_cfg, "mode", None),
                "interval_range_s": getattr(push_cfg, "interval_range_s", None),
                "params": getattr(push_cfg, "params", None),
            }
            print(f"push_robot event: {entry['push_robot_params']}")
        else:
            entry["push_robot_params"] = None
            print("push_robot event: NOT FOUND on this task's events")
    except Exception as e:  # noqa: BLE001
        print(f"(push_robot event: {e!r})")

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

    # --- Live-env section: actual obs/action widths, body names, and
    # termination-cause attribution API, which only exist once a real env
    # is constructed. Requires GPU (same as every other script here).
    section("LIVE ENV: obs/action widths + body names + termination_manager surface")
    try:
        import torch

        from mjlab.envs import ManagerBasedRlEnv

        if not torch.cuda.is_available():
            print(
                "CUDA not available here -- skipping live-env construction. "
                "Run this on the GPU machine to fill in this section."
            )
        else:
            live_cfg = load_env_cfg(task_id)
            live_cfg.scene.num_envs = 4
            raw_env = ManagerBasedRlEnv(cfg=live_cfg, device="cuda")
            obs, _ = raw_env.reset()
            live: dict = {"task_id": task_id}
            for group_name, group_tensor in obs.items():
                print(f"obs[{group_name!r}].shape = {tuple(group_tensor.shape)}")
                live[f"obs_{group_name}_dim"] = int(group_tensor.shape[-1])

            action_dim = raw_env.action_manager.total_action_dim if hasattr(raw_env, "action_manager") else None
            print(f"action_manager.total_action_dim = {action_dim}")
            live["action_dim"] = action_dim

            tm = raw_env.termination_manager
            active_terms = getattr(tm, "active_terms", None)
            print(f"termination_manager.active_terms = {active_terms}")
            live["termination_manager_active_terms"] = list(active_terms) if active_terms is not None else None

            for name in term_names:
                try:
                    val = tm.get_term(name)
                    print(f"termination_manager.get_term({name!r}) -> tensor shape {tuple(val.shape)}, dtype {val.dtype}")
                except Exception as e:  # noqa: BLE001
                    print(f"termination_manager.get_term({name!r}) FAILED: {e!r}")

            try:
                asset = raw_env.scene["robot"]
                body_names = None
                if hasattr(asset, "body_names"):
                    body_names = list(asset.body_names)
                elif hasattr(asset, "data") and hasattr(asset.data, "body_names"):
                    body_names = list(asset.data.body_names)
                if body_names is None:
                    raise AttributeError("no body_names attribute found on asset or asset.data")
                print(f"asset body_names ({len(body_names)}): {body_names}")
                live["body_names"] = body_names
            except Exception as e:  # noqa: BLE001
                public_attrs = [n for n in dir(raw_env.scene["robot"]) if not n.startswith("_")]
                print(f"(body_names: {e!r} -- asset's public attrs, inspect by hand if needed: {public_attrs})")
                live["body_names_error"] = repr(e)

            report["live_env"] = live
            raw_env.close()
    except Exception as e:  # noqa: BLE001 -- diagnostic script
        print(f"Live-env section failed: {e!r}")
        report["live_env_error"] = repr(e)

    out_path = Path("outputs")
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"{profile.name}_task_inspection.json"
    out_file.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
