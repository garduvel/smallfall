import json
from pathlib import Path
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})
md("""
# Go1 hand-off demo video

Fresh runtime, any GPU (T4 is fine). Only the demo video: the same fall three
ways, side by side (walker alone | limp, motors off | hand-off to the protective
policy); ~15 min.
In step 2 select **all 6 files** in
`colab_uploads/demo_video/` (Ctrl+A). The
protective checkpoint is whichever `round*_model_*.pt` file is in that folder.
""")
md("## 1. Install, confirm GPU")
code("""
!pip install -q mjlab==1.6.0 rsl-rl-lib==5.4.2 scikit-learn "imageio[ffmpeg]" pillow
import torch
assert torch.cuda.is_available(), "No GPU visible -- check Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
""")
md("## 2. Upload everything (6 files)")
code("""
from google.colab import files
import os, re, zipfile, shutil
uploaded = files.upload()
names = {re.sub(r" \\(\\d+\\)(?=\\.\\w+$)", "", n): n for n in uploaded}
DEST = {
    "model_9999.pt": "checkpoints/model_9999.pt",
    "go1_wildpush_predictor_seed0.pt": "outputs/go1_wildpush_predictor_seed0.pt",
    "go1_wildpush_metrics_seed0.json": "outputs/go1_wildpush_metrics_seed0.json",
    "go1_fall_state_bank_seed0.npz": "outputs/go1_fall_state_bank_seed0.npz",
}
protective = [n for n in names if re.fullmatch(r"round\\w+_model_\\d+\\.pt", n)]
assert len(protective) == 1, f"need exactly one round*_model_*.pt protective checkpoint, got {protective}"
PROTECTIVE = f"checkpoints/protective/{protective[0]}"
DEST[protective[0]] = PROTECTIVE
missing = ({"cartpole_term_code.zip"} | set(DEST)) - set(names)
assert not missing, f"missing uploads: {sorted(missing)} -- re-run this cell and select all 6 files"
ROOT = "/content/cartpole-term"
os.makedirs(ROOT, exist_ok=True)
with zipfile.ZipFile(names["cartpole_term_code.zip"]) as zf:
    zf.extractall(ROOT)
for d in ("checkpoints/protective", "outputs"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
for canon, rel in DEST.items():
    shutil.move(names[canon], f"{ROOT}/{rel}")
%cd /content/cartpole-term
""")
md("## 3. Render (rollout seeds from 700 up), show, download")
code("""
VIDEO = "outputs/videos/go1_handoff_demo_seed0.mp4"
print("protective checkpoint:", PROTECTIVE)
!python scripts/go1_handoff_demo_video.py --protective-checkpoint {PROTECTIVE}
assert _exit_code == 0, f"video failed (exit {_exit_code}) -- scroll up for the traceback"
from IPython.display import Video, display
display(Video(VIDEO, embed=True, width=1280))
files.download(VIDEO)
""")
code("files.download(VIDEO.replace('.mp4', '.json'))")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(Path(__file__).resolve().parents[2] / "notebooks" / "go1_handoff_demo_video.ipynb", "w"), indent=1)
print(len(cells), "cells")
