"""Rebuild cartpole_term_code.zip (what the Colab notebooks upload): pyproject.toml
plus everything under scripts/ and src/, without caches. Run from anywhere:

    python3 tools/build_code_zip.py            # also refreshes the copy in every colab_uploads/*/ folder
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cartpole_term_code.zip"


def main() -> None:
    files = [ROOT / "pyproject.toml"] + [
        p for d in ("scripts", "src") for p in sorted((ROOT / d).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    ]
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT).as_posix())
    refreshed = []
    for folder in sorted((ROOT / "colab_uploads").glob("*/")):
        if (folder / OUT.name).exists():
            shutil.copy(OUT, folder / OUT.name)
            refreshed.append(folder.name)
    print(f"{OUT.name}: {len(files)} files, {OUT.stat().st_size / 1e3:.0f} kB; refreshed in colab_uploads/: {refreshed}")


if __name__ == "__main__":
    main()
