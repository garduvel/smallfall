"""Builds notebooks/g1_predictor.ipynb: stages 2-3 of the G1 humanoid port
(docs/G1_PORT_PLAN.md): the push check and the fall predictor, starting from
the walker trained by notebooks/g1_walker.ipynb.

    python3 tools/notebook_builders/build_g1_predictor_nb.py
"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# G1 humanoid: push and fall predictor (stages 2-3 of the humanoid port)

Uses the walker from `g1_walker.ipynb` (read from Drive). Same method as the
Go1 (`DEVIATIONS.md`), with the push scaled to the G1's mass.

| Step | What it does | Rough time |
|---|---|---|
| 0 | install, clone the repo, mount Drive, fetch the walker | ~4 min |
| B1 | push sweep: falls per 1,000 robot-steps at 200-600 N, compared with the Go1 | ~5 min |
| B2 | video of the walker under the chosen push (390 N) | ~3 min |
| B3 | ~1M labelled steps under the push, train the predictor | ~10-20 min |
| B4 | predictor confidence intervals and tests (bootstrap) | ~5-10 min |
| Z | download the Drive folder as a zip | ~1 min |

**The push (fixed before this runs, `DEVIATIONS.md` 2026-11-13).** The Go1's
150 N came from scaling a published force range by body mass. The same rule
for the 33.3 kg G1 gives 392 N, so **390 N per axis on `torso_link`**: the
same acceleration as the Go1's push. B1 checks the resulting fall rate is
within 0.25x-4x of the Go1's (1.07 falls per 1,000 robot-steps) and stops if
not; the other forces are shown for context only.

**What a good predictor looks like** (Go1 for comparison): AUPRC well above
the logistic-regression baseline and far above chance (= the fraction of
positive steps, about 0.02), with a median warning of a few tenths of a
second.

**If the runtime disconnects:** run all again with the same `RUN_NAME`;
finished steps are restored from Drive and skipped.
""")

md("## 0a. Install (pinned versions), confirm GPU")
code("""
!pip install -q mjlab==1.6.0 rsl-rl-lib==5.4.2 scikit-learn scipy "imageio[ffmpeg]" pillow matplotlib
import torch
assert torch.cuda.is_available(), "No GPU visible -- check Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
%env MUJOCO_GL=egl
%env PYTORCH_JIT=0
""")

md("## 0b. Get the code (public repo, newest version)")
code("""
import glob, json, os, re, shutil, subprocess, zipfile
from IPython.display import Video, display

REPO_URL = "https://github.com/garduvel/smallfall.git"
REPO_REF = "main"
if not os.path.exists("/content/smallfall/.git"):
    !git clone -q {REPO_URL} /content/smallfall
%cd /content/smallfall
!git fetch -q origin && git checkout -q {REPO_REF} && (git pull -q --ff-only origin {REPO_REF} 2>/dev/null || true)
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code at commit", COMMIT)
for d in ("checkpoints", "outputs/stats"):
    os.makedirs(d, exist_ok=True)
""")

md("## 0c. Settings, Google Drive, the walker")
code("""
RUN_NAME = "predictor1"        # Drive folder for this notebook's results
WALKER_RUN = "walker2"         # RUN_NAME used in g1_walker.ipynb
SEEDS = (0,)                   # one predictor per seed; add seeds later for the multi-run statistics
SWEEP_FORCES = (200, 300, 390, 500, 600)   # N per axis; 390 is the chosen push, the rest are context
PUSH_N = 390

WALKER = f"checkpoints/g1_walker_{WALKER_RUN}_model_9999.pt"
WALKER_ON_DRIVE = f"/content/drive/MyDrive/cartpole_term_runs/g1_walker/{WALKER_RUN}/{WALKER}"

def predictor(s): return f"outputs/g1_wildpush_predictor_seed{s}.pt"
def metrics(s): return f"outputs/g1_wildpush_metrics_seed{s}.json"
def collection(s): return f"outputs/g1_wildpush_frozen_seed{s}.json"
def dataset(s): return f"outputs/g1_wildpush_frozen_seed{s}.npz"
def predictor_stats(s): return f"outputs/stats/g1_predictor_seed{s}/predictor_stats.json"
SWEEP = "outputs/g1_impulse_sweep_seed0.json"
PUSH_VIDEO = f"outputs/g1_watch_push{PUSH_N}N_seed0_ep3.mp4"

from google.colab import drive, files
drive.mount("/content/drive")
SAVE_DIR = f"/content/drive/MyDrive/cartpole_term_runs/g1_predictor/{RUN_NAME}"
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

assert os.path.exists(WALKER_ON_DRIVE), f"walker not found at {WALKER_ON_DRIVE} -- finish g1_walker.ipynb first, or fix WALKER_RUN"
shutil.copy(WALKER_ON_DRIVE, WALKER)
print("walker:", WALKER)

versions = {"gpu": torch.cuda.get_device_name(0), "commit": COMMIT, "walker": WALKER_ON_DRIVE, "push_n": PUSH_N}
for pkg in ("mjlab", "mujoco-warp", "rsl-rl-lib", "torch", "scikit-learn", "scipy", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_g1_predictor.json", "w"), indent=2)
save("outputs/versions_g1_predictor.json")
""")

md("""
# Part B: push and predictor

## B1. Push sweep
Each force: 4,096 robots for 262 steps from a reset (the same exposure as the
Go1's collections), counting falls. Stops if 390 N falls outside the pre-set
0.25x-4x of the Go1's rate.
""")
code("""
if restore(SWEEP):
    print("sweep already done")
else:
    forces = " ".join(str(f) for f in SWEEP_FORCES)
    !python scripts/measure_push_failure_rate.py --robot g1 --checkpoint-file {WALKER} --mechanism impulse --impulse-forces {forces}
    assert _exit_code == 0, f"sweep failed (exit {_exit_code}) -- scroll up"
    save(SWEEP)
sweep = json.load(open(SWEEP))
print(f"{'force (N)':>10} {'falls per 1,000 steps':>22}   (Go1 at 150 N: {sweep['go1_150n_falls_per_1000_steps']})")
for r in sweep["results"]:
    print(f"{r['force_n']:>10.0f} {r['falls_per_1000_steps']:>22.2f}")
check = sweep["chosen_force_check"]
print(f"\\n{PUSH_N} N: {check['ratio_to_go1']:.2f}x the Go1's fall rate")
assert check["within_quarter_to_4x"], (
    f"STOP: {PUSH_N} N gives {check['ratio_to_go1']:.2f}x the Go1's fall rate, outside the pre-set 0.25x-4x. "
    "Do not pick another force here; the choice needs a decision logged in DEVIATIONS.md first.")
""")

md("""
## B2. The walker under the push (three episodes)
Check the falls look like falls from a shove: stumbling and going down, not
being launched. (Fallen episodes restart.)
""")
code("""
if not restore(PUSH_VIDEO):
    !python scripts/robot_watch_episodes.py --robot g1 --checkpoint-file {WALKER} --episodes 3 --camera-distance 3.5 --impulse-force {PUSH_N} --natural-commands
    assert _exit_code == 0, f"video failed (exit {_exit_code}) -- scroll up"
    save(PUSH_VIDEO)
display(Video(PUSH_VIDEO, embed=True, width=640))
""")

md("""
## B3. Push dataset and predictor
For each seed: collect about a million labelled steps (and at least 1,000 falls) under the 390 N
push, label a step positive if a fall follows within 16 steps (0.32 s), split
by episode 70/15/15, train the MLP (128 inputs: 99 observations + 29
actions) and the logistic-regression baseline.
""")
code("""
for s in SEEDS:
    if restore(predictor(s), metrics(s)):
        print(f"seed {s}: predictor already done")
        continue
    print(f"\\n==================== seed {s}: dataset + predictor ====================")
    !python scripts/go1_wild_push_predictor.py --robot g1 --checkpoint-file {WALKER} --seed {s}
    assert _exit_code == 0, f"seed {s}: predictor script failed (exit {_exit_code}) -- scroll up"
    save(predictor(s), metrics(s), collection(s))

for s in SEEDS:
    m = json.load(open(metrics(s)))
    c = m["collection_sidecar"]
    print(f"seed {s}: {c['labelled_transitions']:,} labelled steps, {c['terminated_episode_count']:,} falls, "
          f"positive fraction (chance AUPRC) {m['split_base_rates']['test']:.4f}")
    print(f"   MLP AUPRC {m['mlp']['auprc']:.3f} | logistic regression "
          f"{m['baselines']['logistic_regression_all_inputs']['auprc']:.3f} | AUROC {m['mlp']['auroc']:.3f} | "
          f"median warning {m['mlp_lead_time']['median_lead_seconds']:.2f} s | threshold {m['mlp']['threshold']:.3f}")
""")

md("""
## B4. Confidence intervals and tests
Episode bootstrap (1,000 resamples) for AUPRC, AUROC and the MLP-minus-baseline
differences, same script as the Go1. Needs the dataset from B3 on this
runtime; after a disconnect B3 re-collects it only if this step is not done.
""")
code("""
for s in SEEDS:
    st = predictor_stats(s)
    if restore(st):
        print(f"seed {s}: stats already done")
    else:
        if not os.path.exists(dataset(s)):
            raise RuntimeError(f"{dataset(s)} is not on this runtime (it is not saved to Drive). "
                               f"Delete {on_drive(predictor(s))} and run B3 again to re-collect it.")
        sd = os.path.dirname(st)
        !python scripts/go1_predictor_stats.py --dataset {dataset(s)} --predictor-file {predictor(s)} --metrics-file {metrics(s)} --seed {s} --live-json none --live-npz none --out-dir {sd} --quiet
        assert _exit_code == 0, f"seed {s}: stats failed (exit {_exit_code}) -- scroll up"
        save(st, st.replace(".json", ".md"))
    print(open(st.replace(".json", ".md")).read()[:3000])
""")

md("## Z. Download this run's Drive folder as one zip")
code("""
bundle = shutil.make_archive(f"/content/g1_predictor_{RUN_NAME}", "zip", SAVE_DIR)
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
out = Path(__file__).resolve().parents[2] / "notebooks" / "g1_predictor.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
