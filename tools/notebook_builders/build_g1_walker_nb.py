"""Builds notebooks/g1_walker.ipynb: stage 1 of the G1 humanoid port
(docs/G1_PORT_PLAN.md), the walker, trained for 10,000 iterations and
resumable across Colab sessions from Google Drive.

    python3 tools/notebook_builders/build_g1_walker_nb.py
"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# G1 humanoid: walker (stage 1 of the humanoid port)

Trains the walking policy for mjlab's Unitree G1 humanoid
(`Mjlab-Velocity-Flat-Unitree-G1`, 29 joints), the first step of repeating
the Go1 SmallFall pipeline on a humanoid (`docs/G1_PORT_PLAN.md`). The
predictor, protective policy and evaluation come in later notebooks and
start from the checkpoint this one produces.

| Step | What it does | Rough A100 time |
|---|---|---|
| 0 | install, clone the repo, mount Drive | ~3 min |
| A1 | inspect the task (confirms 99 obs / 29 actions) | ~2 min |
| A2 | 3-iteration smoke test (first session only) | ~3 min |
| A3 | **train the walker, 10,000 iterations** (mjlab ships 30,000; see `DEVIATIONS.md`) | many hours, unmeasured; expect longer than the Go1's |
| A4 | frozen-evaluation curve | seconds |
| A5 | video of the trained walker | ~3 min |
| Z | download the Drive folder as a zip | ~1 min |

**Disconnects.** Training writes straight into this run's Drive folder, with
a checkpoint every `CHECKPOINT_EVERY` iterations. If the runtime drops,
reconnect and **run all** again with the same `RUN_NAME`: the newest readable
checkpoint is found and training continues from it to 10,000. Keep the tab
open (or use Colab Pro background execution); idle sessions are cut off.

**Output:** `checkpoints/g1_walker_<RUN_NAME>_model_9999.pt` here and on Drive.
""")

md("## 0a. Install (pinned versions), confirm GPU")
code("""
!pip install -q mjlab==1.6.0 rsl-rl-lib==5.4.2 scikit-learn "imageio[ffmpeg]" pillow matplotlib
import torch
assert torch.cuda.is_available(), "No GPU visible -- check Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
# Headless rendering, and mjlab's jit-scripted utils crash on this Colab image (DEVIATIONS.md).
%env MUJOCO_GL=egl
%env PYTORCH_JIT=0
""")

md("## 0b. Get the code (public repo)")
code("""
import glob, json, os, re, shutil, subprocess, zipfile
from IPython.display import Image, Video, display

REPO_URL = "https://github.com/garduvel/smallfall.git"
REPO_REF = "main"   # branch, tag or commit
if not os.path.exists("/content/smallfall/.git"):
    !git clone -q {REPO_URL} /content/smallfall
%cd /content/smallfall
# Always pick up the newest pushed code (fixes land on main), even on a reused runtime.
!git fetch -q origin && git checkout -q {REPO_REF} && (git pull -q --ff-only origin {REPO_REF} 2>/dev/null || true)
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code at commit", COMMIT)
for d in ("checkpoints", "outputs", "logs/rsl_rl"):
    os.makedirs(d, exist_ok=True)
""")

md("## 0c. Settings, Google Drive, package versions")
code("""
RUN_NAME = "walker2"           # Drive folder for this run; a new name starts a fresh run
#   walker1 = first attempt, air_time 0.25, learned to sit (DEVIATIONS.md 2026-11-12)
TOTAL_ITERATIONS = 10_000      # user's budget (mjlab ships 30,000 for the G1)
CHECKPOINT_EVERY = 500         # also the frozen-evaluation interval
SEED = 0
NUM_ENVS = 4096
EXPERIMENT = "g1_velocity_term"            # set in src/cartpole_term/g1/rl_cfg.py
FINAL_NAME = f"model_{TOTAL_ITERATIONS - 1}.pt"   # iterations are numbered from 0
WALKER = f"checkpoints/g1_walker_{RUN_NAME}_{FINAL_NAME}"

from google.colab import drive, files
drive.mount("/content/drive")
SAVE_DIR = f"/content/drive/MyDrive/cartpole_term_runs/g1_walker/{RUN_NAME}"
DRIVE_LOGS = os.path.join(SAVE_DIR, "logs", EXPERIMENT)   # training writes here directly
os.makedirs(DRIVE_LOGS, exist_ok=True)

def on_drive(path):
    return os.path.join(SAVE_DIR, path)

def save(*paths):
    \"\"\"Copy result files to this run's Drive folder, keeping their relative paths.\"\"\"
    for p in paths:
        os.makedirs(os.path.dirname(on_drive(p)), exist_ok=True)
        shutil.copy(p, on_drive(p))
        print("saved to Drive:", p)

def restore(*paths):
    \"\"\"True if every path exists here or can be copied back from Drive.\"\"\"
    if not all(os.path.exists(p) or os.path.exists(on_drive(p)) for p in paths):
        return False
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            shutil.copy(on_drive(p), p)
    return True

def checkpoints_newest_first(run_dir):
    paths = glob.glob(run_dir + "/model_*.pt")
    return sorted(paths, key=lambda p: int(re.search(r"model_(\\d+)\\.pt$", p).group(1)), reverse=True)

def readable(path):
    \"\"\"A checkpoint cut off mid-write by a disconnect fails to load; skip it.\"\"\"
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception as e:
        print(f"unreadable checkpoint {path}: {type(e).__name__}: {e}")
        return False

def drive_run_dirs():
    return sorted(glob.glob(os.path.join(DRIVE_LOGS, f"*_seed{SEED}")))

versions = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "commit": COMMIT,
            "total_iterations": TOTAL_ITERATIONS, "num_envs": NUM_ENVS, "seed": SEED}
for pkg in ("mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib", "torch", "warp-lang", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_g1_walker.json", "w"), indent=2)
print(json.dumps(versions, indent=2))
save("outputs/versions_g1_walker.json")
""")

md("""
# Part A: walker

## A1. Inspect the task
Fails loudly if the installed mjlab's G1 task no longer matches what the code
expects (99 observations, 29 actions, `time_out` / `fell_over` endings).
""")
code("""
!python scripts/inspect_robot_task.py --robot g1
assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up for the traceback"
save("outputs/g1_task_inspection.json")
""")

md("""
## A2. Smoke test (first session only)
Three iterations on a few robots, to catch setup errors before committing
hours of GPU time. Skipped once Drive holds a training run. Its logs are
deleted afterwards so they cannot be mistaken for the real run.
""")
code("""
local_logs = f"logs/rsl_rl/{EXPERIMENT}"
if drive_run_dirs() or os.path.exists(WALKER) or os.path.exists(on_drive(WALKER)):
    print("a training run already exists on Drive -- smoke test skipped")
else:
    if os.path.islink(local_logs):
        os.unlink(local_logs)
    !python scripts/robot_train.py --robot g1 --smoke
    assert _exit_code == 0, f"smoke test failed (exit {_exit_code}) -- scroll up"
    shutil.rmtree(local_logs, ignore_errors=True)
    print("smoke test passed")
""")

md("""
## A3. Train the walker, 10,000 iterations

Checked in this order: a finished walker is already at `WALKER` or on Drive
(nothing to do); a run on Drive has a readable checkpoint (resume from the
newest); otherwise start fresh. Logs, checkpoints, TensorBoard events and
`eval_history.json` go straight to Drive through a link, so a disconnect loses
at most `CHECKPOINT_EVERY` iterations. The cell prints which case it took.
""")
code("""
local_logs = f"logs/rsl_rl/{EXPERIMENT}"
if os.path.islink(local_logs):
    os.unlink(local_logs)
shutil.rmtree(local_logs, ignore_errors=True)
os.symlink(DRIVE_LOGS, local_logs)

common = (f"--robot g1 --seeds {SEED} --max-iterations {TOTAL_ITERATIONS} "
          f"--eval.interval {CHECKPOINT_EVERY} --env.scene.num-envs {NUM_ENVS}")
runs = drive_run_dirs()
if len(runs) > 1:
    print("WARNING: several runs in", DRIVE_LOGS, "-- using the newest:", runs[-1])
run_dir = runs[-1] if runs else None

if restore(WALKER):
    print("finished walker already present:", WALKER)
elif run_dir and os.path.exists(os.path.join(run_dir, FINAL_NAME)) and readable(os.path.join(run_dir, FINAL_NAME)):
    print("training had finished; copying", FINAL_NAME)
else:
    resume = next((p for p in checkpoints_newest_first(run_dir) if readable(p)), None) if run_dir else None
    if resume:
        print("RESUMING from", resume)
        !python scripts/robot_train.py {common} --resume-from {resume}
    else:
        print(f"TRAINING FROM SCRATCH: {TOTAL_ITERATIONS:,} iterations, many hours.")
        !python scripts/robot_train.py {common}
    assert _exit_code == 0, f"walker training failed (exit {_exit_code}) -- re-run this cell to resume"
    run_dir = drive_run_dirs()[-1]

if not os.path.exists(WALKER):
    final = os.path.join(run_dir, FINAL_NAME)
    assert os.path.exists(final), f"{final} missing -- training did not reach {TOTAL_ITERATIONS} iterations"
    shutil.copy(final, WALKER)
    save(WALKER)
!ls -la {WALKER}
""")

md("""
## A4. Training curves: survival and velocity tracking
Left: episode length from the frozen evaluation every `CHECKPOINT_EVERY`
iterations (256 robots, deterministic actions). **Length alone cannot tell
walking from sitting**: a seated robot never tips past 70 degrees, so it
also lasts the full 1,000 steps. Right: the velocity-tracking reward from
training, which only rises if the robot actually moves as commanded. Can be
run while training is paused or after a disconnect to check progress.
""")
code("""
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
runs = drive_run_dirs()
if not runs:
    print("no training run on Drive (walker was uploaded, not trained here)")
else:
    run = runs[-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    hist_path = os.path.join(run, "eval_history.json")
    if os.path.exists(hist_path):
        hist = json.load(open(hist_path))
        it = [h["iteration"] for h in hist]
        length = [h["mean_action"]["mean_episode_length"] for h in hist]
        ax[0].plot(it, length)
        print(f"frozen eval at iteration {it[-1]}: mean episode length {length[-1]:.0f} of 1000 steps")
        shutil.copy(hist_path, "outputs/g1_walker_eval_history.json")
    ax[0].set(xlabel="iteration", ylabel="frozen-eval episode length (steps)", ylim=(0, 1050))
    acc = EventAccumulator(run, size_guidance={"scalars": 0}); acc.Reload()
    tracking = {}
    for tag in ("Episode_Reward/track_linear_velocity", "Episode_Reward/air_time", "Episode_Reward/upright"):
        if tag in acc.Tags()["scalars"]:
            ev = acc.Scalars(tag)
            ax[1].plot([e.step for e in ev], [e.value for e in ev], label=tag.split("/")[1])
            tracking[tag] = ev[-1].value
    ax[1].set(xlabel="iteration", ylabel="reward per episode (training)"); ax[1].legend(frameon=False)
    for a in ax: a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("outputs/g1_walker_curves.png", dpi=150); plt.show()
    print("latest training rewards:", {k.split("/")[1]: round(v, 3) for k, v in tracking.items()})
    save("outputs/g1_walker_curves.png", *[p for p in ["outputs/g1_walker_eval_history.json"] if os.path.exists(p)])
""")


md("""
## A5. Walker video (three episodes, forward commands)
Camera set back for a 1.3 m robot. Check it walks steadily and tracks the
commanded speed before moving on to the push stage.
""")
code("""
!python scripts/robot_watch_episodes.py --robot g1 --checkpoint-file {WALKER} --episodes 3 --camera-distance 3.5 --seed {SEED}
assert _exit_code == 0, f"video failed (exit {_exit_code}) -- scroll up"
VIDEO = f"outputs/g1_watch_seed{SEED}_ep3.mp4"
save(VIDEO)
display(Video(VIDEO, embed=True, width=640))
""")

md("## Z. Download this run's Drive folder as one zip (checkpoints, logs, curve, video)")
code("""
bundle = shutil.make_archive(f"/content/g1_walker_{RUN_NAME}", "zip", SAVE_DIR)
with zipfile.ZipFile(bundle) as zf:
    assert zf.testzip() is None
    print(len(zf.namelist()), "files,", f"{os.path.getsize(bundle)/1e6:.1f} MB")
files.download(bundle)
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = Path(__file__).resolve().parents[2] / "notebooks" / "g1_walker.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
