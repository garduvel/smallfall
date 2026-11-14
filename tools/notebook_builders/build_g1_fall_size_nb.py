"""Builds notebooks/g1_fall_size_ablation.ipynb: ten ways of measuring how
small a fall is, applied to seven G1 policies on the same falls, and three
ways of deciding which policy falls smallest overall (DEVIATIONS.md
2026-11-13, metric ablation).

    python3 tools/notebook_builders/build_g1_fall_size_nb.py
"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# G1: which policy falls smallest? (fall-size measure ablation)

Seven policies replay the **same** falls: the chosen SmallFall checkpoint, the
other SmallFall checkpoints from training (iterations 100-300), the walker
continuing, and limp (motors off). Each fall is measured ten ways; then three
methods decide which policy falls smallest overall.

| Step | What it does | Rough time |
|---|---|---|
| 0 | install, clone, Drive, fetch walker, predictor and SmallFall checkpoints | ~4 min |
| S1 | a fresh bank of falls (seed 6, never used before) | ~5 min |
| S2 | replay every fall under all seven policies, measuring each fall ten ways | ~10 min |
| S3 | the ablation: per-measure rankings, agreement, the three methods, resampling | <1 min |
| S4 | plots | <1 min |

**The ten measures** (`src/cartpole_term/protective/fall_size.py`), all
horizontal, from the fall's start, lower = smaller:
ground covered (the project's main measure), max reach, root travel at the end,
root's largest travel, root path length, centroid travel, swept hull area,
hull growth, contact reach, floor area touched.

**The three methods** (`fall_size_ablation.py`, fixed before the data):
- **A. Dominance:** a policy is out if another one is at least as small on every
  measure and smaller on one. What's left can't be beaten without trading.
- **B. Mean rank:** rank the policies on every fall and every measure, average.
- **C. Size relative to limp:** per fall and measure, policy / limp, geometric
  mean. 0.6 = 40% smaller than doing nothing.

**Overall:** if B and C pick the same policy and it survives A, that's the
answer; otherwise the methods disagree and A's survivors are the answer.
The falls are resampled 1,000 times to show how stable each pick is.

This uses a fresh bank, so the SmallFall test bank (seed 4) stays used once.
""")

md("## 0a. Install (pinned versions), confirm GPU")
code("""
!pip install -q mjlab==1.6.0 rsl-rl-lib==5.4.2 scikit-learn scipy tensorboard "imageio[ffmpeg]" pillow matplotlib
import torch
assert torch.cuda.is_available(), "No GPU visible -- check Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
%env MUJOCO_GL=egl
%env PYTORCH_JIT=0
""")

md("## 0b. Get the code (public repo, newest version)")
code("""
import glob, json, os, re, shutil, subprocess, zipfile

REPO_URL = "https://github.com/garduvel/smallfall.git"
REPO_REF = "main"
if not os.path.exists("/content/smallfall/.git"):
    !git clone -q {REPO_URL} /content/smallfall
%cd /content/smallfall
!git fetch -q origin && git checkout -q {REPO_REF} && (git pull -q --ff-only origin {REPO_REF} 2>/dev/null || true)
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code at commit", COMMIT)
for d in ("checkpoints/protective", "outputs"):
    os.makedirs(d, exist_ok=True)
""")

md("## 0c. Settings, Drive, inputs")
code("""
RUN_NAME = "sizeablation1"
WALKER_RUN, PREDICTOR_RUN, SMALLFALL_RUN = "walker2", "predictor1", "smallfall1"
SEED = 0
BANK = 6                                    # fresh: not train (5), selection (3) or test (4)
CANDIDATES = (100, 150, 200, 250, 300)

DRIVE = "/content/drive/MyDrive/cartpole_term_runs"
WALKER = f"checkpoints/g1_walker_{WALKER_RUN}_model_9999.pt"
PREDICTOR = f"outputs/g1_wildpush_predictor_seed{SEED}.pt"
METRICS = f"outputs/g1_wildpush_metrics_seed{SEED}.json"
SELECTED = "outputs/g1_selected_checkpoint.json"
def candidate(it): return f"checkpoints/protective/g1_seed{SEED}_model_{it}.pt"
STATE_BANK = f"outputs/g1_flag_state_bank_seed{BANK}.npz"
FLAGGED = f"outputs/g1_flagged_states_seed{BANK}.json"
COMPARE = "outputs/g1_size_ablation.json"
REPORT = "outputs/g1_size_ablation_report.json"

from google.colab import drive, files
drive.mount("/content/drive")
SAVE_DIR = f"{DRIVE}/g1_fall_size/{RUN_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

def on_drive(path):
    return os.path.join(SAVE_DIR, path)

def save(*paths):
    for p in paths:
        os.makedirs(os.path.dirname(on_drive(p)), exist_ok=True)
        shutil.copy(p, on_drive(p))
        print("saved to Drive:", p)

def restore(*paths):
    if not all(os.path.exists(p) or os.path.exists(on_drive(p)) for p in paths):
        return False
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            shutil.copy(on_drive(p), p)
    return True

sf = f"{DRIVE}/g1_smallfall/{SMALLFALL_RUN}"
inputs = [(WALKER, f"{DRIVE}/g1_walker/{WALKER_RUN}/{WALKER}"),
          (PREDICTOR, f"{DRIVE}/g1_predictor/{PREDICTOR_RUN}/{PREDICTOR}"),
          (METRICS, f"{DRIVE}/g1_predictor/{PREDICTOR_RUN}/{METRICS}"),
          (SELECTED, f"{sf}/{SELECTED}")] + [(candidate(it), f"{sf}/{candidate(it)}") for it in CANDIDATES]
for local, src in inputs:
    assert os.path.exists(src), f"missing {src} -- check the *_RUN names (SmallFall must have finished C4)"
    shutil.copy(src, local)
    print("input:", local)

CHOSEN_IT = int(json.load(open(SELECTED))["selected_label"])
print("chosen SmallFall checkpoint: iteration", CHOSEN_IT)

versions = {"gpu": torch.cuda.get_device_name(0), "commit": COMMIT}
for pkg in ("mjlab", "mujoco-warp", "rsl-rl-lib", "torch", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_g1_fall_size.json", "w"), indent=2)
save("outputs/versions_g1_fall_size.json")
""")

md("""
## S1. A fresh bank of falls (seed 6)
Walker + 200 N push + live predictor; the robot's state the moment the
predictor fires, for 500 falls. Same procedure as the SmallFall banks.
""")
code("""
if restore(STATE_BANK, FLAGGED):
    print("bank already collected")
else:
    !python scripts/go1_collect_flagged_states.py --robot g1 --checkpoint-file {WALKER} --predictor-file {PREDICTOR} --metrics-file {METRICS} --seed {BANK} --capture-state-bank --capture-at-flag
    assert _exit_code == 0, f"bank collection failed (exit {_exit_code}) -- scroll up"
    save(STATE_BANK, FLAGGED)
f = json.load(open(FLAGGED))
print(f"{f['num_lead_time_trials']} flagged falls; live recall {f['live_recall']}")
""")

md("""
## S2. Replay every fall under all seven policies
`protective` is the chosen checkpoint; `sf_it<N>` are the other SmallFall
checkpoints. The usual SmallFall-vs-walker/limp table is printed too.
""")
code("""
extras = " ".join(f"--extra-policy sf_it{it}={candidate(it)}" for it in CANDIDATES if it != CHOSEN_IT)
NPZ = COMPARE.replace(".json", ".npz")
if not restore(COMPARE, NPZ):
    !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {candidate(CHOSEN_IT)} --walker-checkpoint {WALKER} --state-bank-file {STATE_BANK} --seed {BANK} --output {COMPARE} {extras}
    assert _exit_code == 0, f"replay failed (exit {_exit_code}) -- scroll up"
    save(COMPARE, NPZ)
""")

md("## S3. The ablation")
code("""
# Cheap and deterministic, so it simply runs again after a disconnect.
!python scripts/robot_fall_size_ablation.py --compare-npz {NPZ} --robot g1 --output {REPORT}
assert _exit_code == 0, f"ablation failed (exit {_exit_code}) -- scroll up"
save(REPORT)
rep = json.load(open(REPORT))
""")

md("""
## S4. Plots
Left: how often a fall would hit an object at each distance (max reach).
Middle: each policy's rank on each measure (1 = smallest). Right: how much
the measures agree fall by fall (Spearman).
""")
code("""
import numpy as np
import matplotlib.pyplot as plt

pols, mets = rep["policies"], rep["metrics"]
short = ["ground", "reach", "root end", "root max", "root path", "centroid", "hull", "hull grow", "contact", "touched"]
fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
radii = rep["reach_curves"]["radii_m"]
for p in pols:
    ax[0].plot(radii, rep["reach_curves"][f"{p}|max_reach_m"], label=p, lw=2.5 if p == "protective" else 1.3)
ax[0].set_xlabel("object distance from the fall's start (m)"); ax[0].set_ylabel("share of falls that reach it")
ax[0].set_title("Clear radius needed (lower = smaller)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

ranks = np.array([[rep["rank_by_metric"][m].index(p) + 1 for m in mets] for p in pols])
im = ax[1].imshow(ranks, cmap="viridis_r", aspect="auto")
ax[1].set_xticks(range(len(mets)), short, rotation=45, ha="right"); ax[1].set_yticks(range(len(pols)), pols)
for i in range(len(pols)):
    for j in range(len(mets)):
        ax[1].text(j, i, ranks[i, j], ha="center", va="center", color="w", fontsize=8)
ax[1].set_title("Rank on each measure (1 = smallest median)")

S = np.eye(len(mets))
for k, v in rep["per_fall_spearman"].items():
    a, b = k.split("|"); i, j = mets.index(a), mets.index(b); S[i, j] = S[j, i] = v
im2 = ax[2].imshow(S, vmin=-1, vmax=1, cmap="RdBu")
ax[2].set_xticks(range(len(mets)), short, rotation=45, ha="right"); ax[2].set_yticks(range(len(mets)), short)
ax[2].set_title("Do the measures agree? (per-fall Spearman)"); fig.colorbar(im2, ax=ax[2], fraction=.046)
plt.tight_layout()
plt.savefig("outputs/g1_size_ablation.png", dpi=120); save("outputs/g1_size_ablation.png")
plt.show()
""")

md("## Z. Download this run's Drive folder as one zip")
code("""
bundle = shutil.make_archive(f"/content/g1_fall_size_{RUN_NAME}", "zip", SAVE_DIR)
with zipfile.ZipFile(bundle) as zf:
    assert zf.testzip() is None
    print(len(zf.namelist()), "files,", f"{os.path.getsize(bundle)/1e6:.1f} MB")
files.download(bundle)
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4", "machine_shape": "hm", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).resolve().parents[2] / "notebooks" / "g1_fall_size_ablation.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
