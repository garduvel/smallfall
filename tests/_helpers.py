"""Shared test utilities.

Importing `cartpole_term.<anything>` via a normal `import` statement runs
`cartpole_term/__init__.py` first (Python always initializes the parent
package), which calls mjlab's `register_mjlab_task` and therefore requires
mjlab -- and, transitively, a real installed cartpole env cfg -- to be
importable. Per CLAUDE.md S0.2.5, the reward/termination/labelling/split
tests must run in a plain CPU sandbox with neither mjlab nor a GPU
installed, so tests load the specific source file directly by path instead
of going through the package. This exercises the exact same code the real
package ships, just without paying the mjlab import cost the package
`__init__.py` otherwise forces on every submodule import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src" / "cartpole_term"


def _load_from(base: Path, relative_path: str, module_name: str) -> ModuleType:
    path = base / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"could not load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_module(relative_path: str, module_name: str) -> ModuleType:
    """Load a module by path relative to src/cartpole_term/."""
    return _load_from(_SRC, relative_path, module_name)


def load_script(relative_path: str, module_name: str) -> ModuleType:
    """Load a module by path relative to the repo root (e.g. "scripts/foo.py")."""
    return _load_from(_REPO_ROOT, relative_path, module_name)
