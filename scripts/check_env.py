"""Gate 0: confirm GPU visible, confirm mjlab imports, record installed versions.

Run on the GPU machine (Colab or equivalent), from the repo root:

    uv run python scripts/check_env.py

(or `python scripts/check_env.py` if the environment was set up with plain
pip rather than uv).

Writes installed package versions and the CUDA driver version into
DEVIATIONS.md under "## Environment". Exits non-zero with a specific message
if the GPU is not visible, mjlab does not import, or a required package is
missing -- there is no fallback path for gate 0.
"""

from __future__ import annotations

import importlib.metadata as im
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEVIATIONS_PATH = REPO_ROOT / "DEVIATIONS.md"
ENV_MARKER = "<!-- gate 0: installed versions go here -->"

# (display label as used in CLAUDE.md, actual pip distribution name)
PACKAGES = [
    ("mjlab", "mjlab"),
    ("mujoco", "mujoco"),
    ("mujoco-warp", "mujoco-warp"),
    ("rsl-rl", "rsl-rl-lib"),
    ("torch", "torch"),
]


def package_version(dist_name: str) -> str:
    try:
        return im.version(dist_name)
    except im.PackageNotFoundError:
        return "NOT INSTALLED"


def cuda_driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"nvidia-smi failed or is not on PATH: {e}") from e
    lines = [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi returned no driver version")
    return lines[0]


def confirm_gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. This project requires an "
            "NVIDIA GPU (CLAUDE.md S0.2 / S5.4). If on Colab: Runtime -> "
            "Change runtime type -> set Hardware accelerator to GPU, then "
            "re-run."
        )
    return torch.cuda.get_device_name(0)


def confirm_mjlab_imports() -> None:
    import mjlab  # noqa: F401


def write_deviations(lines: list[str]) -> None:
    text = DEVIATIONS_PATH.read_text()
    if ENV_MARKER not in text:
        raise RuntimeError(
            f"Expected marker {ENV_MARKER!r} not found in {DEVIATIONS_PATH}. "
            "Refusing to guess where to write -- restore the marker or edit "
            "the file by hand."
        )
    DEVIATIONS_PATH.write_text(text.replace(ENV_MARKER, "\n".join(lines)))


def main() -> None:
    print("Gate 0: environment check")

    device_name = confirm_gpu()
    print(f"GPU visible: {device_name}")

    confirm_mjlab_imports()
    print("mjlab imports: OK")

    driver_version = cuda_driver_version()

    versions: dict[str, str] = {}
    missing: list[str] = []
    for label, dist_name in PACKAGES:
        ver = package_version(dist_name)
        versions[label] = ver
        print(f"{label}: {ver}")
        if ver == "NOT INSTALLED":
            missing.append(label)
    print(f"CUDA driver: {driver_version}")

    if missing:
        raise RuntimeError(
            f"Required package(s) not installed: {', '.join(missing)}. "
            "Gate 0 cannot pass."
        )

    lines = [
        f"- GPU: {device_name}",
        f"- CUDA driver: {driver_version}",
        *(f"- {label}: {ver}" for label, ver in versions.items()),
    ]
    write_deviations(lines)
    print(f"\nWrote versions to {DEVIATIONS_PATH}")
    print("Gate 0 PASSED.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nGate 0 FAILED: {e}", file=sys.stderr)
        sys.exit(1)
