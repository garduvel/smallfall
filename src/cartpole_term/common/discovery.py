"""Live mjlab task-id discovery for a RobotProfile (phase-2, see
DEVIATIONS.md's "multi-robot scaffolding" entry). Generalizes
cartpole_term.go1.env_cfg's original discover_source_task_id -- same
"verify against list_tasks(), never guess" contract, plus a word-boundary
regex fix: the original Go1-only version used a naive substring search
(re.search(r"go1", ...)) which, while it happens not to collide with
mjlab's actual "...Unitree-G1" humanoid task name today (neither "go1" nor
"g1" is a literal substring of the other's real id), is not guaranteed to
stay collision-free as task names change -- word-boundary matching removes
that risk for free rather than relying on today's exact spelling.
"""

from __future__ import annotations

import re

from cartpole_term.common.robot_profile import RobotProfile

_CACHE: dict[str, str] = {}


def _word_bounded(fragment: str) -> str:
    return rf"(?<![a-zA-Z0-9]){re.escape(fragment)}(?![a-zA-Z0-9])"


def discover_task_id(profile: RobotProfile) -> str:
    """The real mjlab-shipped source task id for `profile`, found by
    checking list_tasks() rather than assumed as a literal (CLAUDE.md
    S0.2's rule, applied here to phase-2 work too). Cached per robot name
    after the first successful call."""
    if profile.name in _CACHE:
        return _CACHE[profile.name]

    from mjlab.tasks.registry import list_tasks

    all_tasks = list_tasks()

    for candidate in profile.task_id_candidates:
        if candidate in all_tasks:
            _CACHE[profile.name] = candidate
            return candidate

    must_patterns = [_word_bounded(f) for f in profile.task_id_must]
    exclude_patterns = [_word_bounded(f) for f in profile.task_id_exclude]
    matches = [
        t
        for t in all_tasks
        if all(re.search(p, t, re.IGNORECASE) for p in must_patterns)
        and not any(re.search(p, t, re.IGNORECASE) for p in exclude_patterns)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one mjlab task id for robot {profile.name!r} matching "
            f"{profile.task_id_must} (excluding {profile.task_id_exclude}), found "
            f"{matches} out of {all_tasks}. Fix RobotProfile.task_id_must/"
            "task_id_candidates in cartpole_term/common/robot_profile.py if mjlab's "
            "naming differs from expected. Not guessing a literal task id (CLAUDE.md S0.2)."
        )
    _CACHE[profile.name] = matches[0]
    return matches[0]
