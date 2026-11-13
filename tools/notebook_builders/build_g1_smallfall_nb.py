"""Builds notebooks/g1_smallfall.ipynb: stages 4-8 of the G1 humanoid port
(docs/G1_PORT_PLAN.md): fall banks, reward calibration, SmallFall training,
checkpoint selection, and the one-time test against the walker and limp.

    python3 tools/notebook_builders/build_g1_smallfall_nb.py
"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# G1 humanoid: SmallFall policy (stages 4-8 of the humanoid port)

Starts from the walker (`g1_walker.ipynb`) and predictor (`g1_predictor.ipynb`)
on Drive. Same procedure as the Go1's final run (round 8b), with the
robot-specific numbers re-derived by the Go1's own rules (`DEVIATIONS.md`,
2026-11-13).

| Step | What it does | Rough time |
|---|---|---|
| 0 | install, clone, Drive, fetch walker + predictor | ~4 min |
| C1 | training bank: the robot's state the moment the predictor fires (500 falls), plus the live predictor check | ~5 min |
| C2 | calibration: 11 training iterations with the Go1 weights, then re-size four reward weights | ~3 min |
| C3 | train SmallFall, 301 iterations | ~15-30 min |
| C4 | selection bank; replay five candidate checkpoints vs walker and limp; pick one by the Go1's fixed rule | ~10 min |
| C5 | fresh test bank; the chosen checkpoint once vs walker and limp | ~3 min |
| V | video of SmallFall on eight falls | ~3 min |
| Z | download the Drive folder | ~1 min |

**Fixed before any G1 data (`DEVIATIONS.md`):**
- **Banks:** one seed per role, as for the Go1: training 5, selection 3, test 4.
- **Episode budget:** 5 s, the Go1's 3 s scaled by the square root of the
  start height.
- **Reward:** the Go1's round-8 reward. The four Go1-calibrated weights
  (joint torque, joint limits, ground covered, peak force) are re-sized so
  each contributes what it did on the Go1 in its first 11 training
  iterations. The literature weights are kept.
- **Checkpoint:** chosen from iterations 100/150/200/250/300 on the
  selection bank, by the Go1's rule (`select_checkpoint` in
  `protective/compare.py`). Eligible means it comes to rest in at least 99% of
  falls and has a lower median than the walker on ground covered, trunk
  force and torque. Preferred means eligible and also lower any-body force.
  Among the preferred (else the eligible) it takes the least ground covered.
  If none is eligible it says so.
- **Test:** the chosen checkpoint is tested once on the test bank against the
  walker and limp.

**Disconnects:** run all again with the same `RUN_NAME`; finished steps are
restored from Drive.
""")

md("## 0a. Install (pinned versions), confirm GPU")
code("""
!pip install -q mjlab==1.6.0 rsl-rl-lib==5.4.2 scikit-learn scipy tensorboard "imageio[ffmpeg]" pillow
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
for d in ("checkpoints/protective", "outputs"):
    os.makedirs(d, exist_ok=True)
""")

md("## 0c. Settings, Drive, inputs")
code("""
RUN_NAME = "smallfall1"
WALKER_RUN, PREDICTOR_RUN = "walker2", "predictor1"
SEED = 0                                   # predictor seed and training seed
TRAIN_BANK, SELECT_BANK, TEST_BANK = 5, 3, 4   # bank seeds, one per role
TRAIN_ITERATIONS = 301                     # checkpoints model_0 ... model_300, every 10
CANDIDATES = (100, 150, 200, 250, 300)

DRIVE = "/content/drive/MyDrive/cartpole_term_runs"
WALKER = f"checkpoints/g1_walker_{WALKER_RUN}_model_9999.pt"
PREDICTOR = f"outputs/g1_wildpush_predictor_seed{SEED}.pt"
METRICS = f"outputs/g1_wildpush_metrics_seed{SEED}.json"
WEIGHTS = "outputs/g1_protective_weights.json"
def bank(seed): return f"outputs/g1_flag_state_bank_seed{seed}.npz"
def flagged(seed): return f"outputs/g1_flagged_states_seed{seed}.json"
def candidate(it): return f"checkpoints/protective/g1_seed{SEED}_model_{it}.pt"
def select_compare(it): return f"outputs/g1_select_it{it}.json"
SELECTED = "outputs/g1_selected_checkpoint.json"
TEST = "outputs/g1_protective_compare_test.json"
HISTORY = "outputs/g1_protective_eval_history.json"

from google.colab import drive, files
drive.mount("/content/drive")
SAVE_DIR = f"{DRIVE}/g1_smallfall/{RUN_NAME}"
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

for local, src in ((WALKER, f"{DRIVE}/g1_walker/{WALKER_RUN}/{WALKER}"),
                   (PREDICTOR, f"{DRIVE}/g1_predictor/{PREDICTOR_RUN}/{PREDICTOR}"),
                   (METRICS, f"{DRIVE}/g1_predictor/{PREDICTOR_RUN}/{METRICS}")):
    assert os.path.exists(src), f"missing {src} -- check WALKER_RUN / PREDICTOR_RUN"
    shutil.copy(src, local)
    print("input:", local)

versions = {"gpu": torch.cuda.get_device_name(0), "commit": COMMIT, "walker_run": WALKER_RUN, "predictor_run": PREDICTOR_RUN}
for pkg in ("mjlab", "mujoco-warp", "rsl-rl-lib", "torch", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_g1_smallfall.json", "w"), indent=2)
save("outputs/versions_g1_smallfall.json")

def collect_bank(seed):
    \"\"\"Live: walker + 200 N push + predictor; saves the state at the moment
    the predictor fires, for 500 falls that followed a flag.\"\"\"
    if restore(bank(seed), flagged(seed)):
        print(f"bank {seed}: already collected")
    else:
        !python scripts/go1_collect_flagged_states.py --robot g1 --checkpoint-file {WALKER} --predictor-file {PREDICTOR} --metrics-file {METRICS} --seed {seed} --capture-state-bank --capture-at-flag
        assert _exit_code == 0, f"bank {seed}: collection failed (exit {_exit_code}) -- scroll up"
        save(bank(seed), flagged(seed))
    f = json.load(open(flagged(seed)))
    print(f"bank {seed}: {f['num_lead_time_trials']} flagged falls; live recall {f['live_recall']}; "
          f"median warning {f['median_lead_seconds']:.2f} s")
""")

md("""
# Part C: SmallFall

## C1. Training bank (and the live predictor check)
The walker walks under the 200 N push with the predictor running live. Each
time the predictor fires and a fall follows, the robot's full state at the
moment of firing is saved. That state is where SmallFall takes over. The
printout includes the live recall (falls the predictor caught) and the live
warning time.
""")
code("""
collect_bank(TRAIN_BANK)
""")

md("""
## C2. Reward calibration
11 iterations with the Go1's weights, then each of the four Go1-calibrated
terms is re-sized to contribute what it did on the Go1 over the same
iterations (`src/cartpole_term/protective/calibration.py`). The run is kept
under `logs/rsl_rl/g1_protective_fall/calibration/`.
""")
code("""
if restore(WEIGHTS):
    print("calibration already done")
else:
    !python scripts/go1_train_protective.py --robot g1 --seeds {SEED} --state-bank-file {bank(TRAIN_BANK)} --max-iterations 11 --eval.interval 10
    assert _exit_code == 0, f"calibration training failed (exit {_exit_code}) -- scroll up"
    cal_run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_seed{SEED}"))[-1]
    shutil.move(cal_run, "logs/rsl_rl/g1_protective_fall/calibration")
    !python scripts/robot_calibrate_protective.py --run-dir logs/rsl_rl/g1_protective_fall/calibration --output {WEIGHTS}
    assert _exit_code == 0, f"calibration failed (exit {_exit_code}) -- scroll up"
    save(WEIGHTS)
w = json.load(open(WEIGHTS))
for term, r in w["terms"].items():
    note = " (clamped)" if r["clamped"] else (" (kept)" if r["kept_base"] else "")
    print(f"{term:28s} Go1 weight {r['base_weight']:>10.3g} -> G1 weight {r['weight']:>10.3g}  (x{r['factor']:.3f}){note}")
""")

md("""
## C3. Train SmallFall (301 iterations)
4,096 robots, each episode starting from a training-bank state, with the
calibrated reward. A checkpoint and a frozen evaluation every 10 iterations.
""")
code("""
if all(restore(candidate(it)) for it in CANDIDATES) and restore(HISTORY):
    print("SmallFall already trained")
else:
    !python scripts/go1_train_protective.py --robot g1 --seeds {SEED} --state-bank-file {bank(TRAIN_BANK)} --weights-json {WEIGHTS} --max-iterations {TRAIN_ITERATIONS}
    assert _exit_code == 0, f"SmallFall training failed (exit {_exit_code}) -- scroll up"
    run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_seed{SEED}"))[-1]
    for it in CANDIDATES:
        shutil.copy(f"{run}/model_{it}.pt", candidate(it))
    shutil.copy(f"{run}/eval_history.json", HISTORY)
    save(*[candidate(it) for it in CANDIDATES], HISTORY)
hist = json.load(open(HISTORY))
for h in hist[::5] + [hist[-1]]:
    m = h["mean_action"]
    print(f"iteration {h['iteration']:>3}: frozen-eval return {m['mean_return']:8.3f}, "
          f"mean episode length {m['mean_episode_length']:6.1f} steps, came to rest {m['termination_cause_breakdown'].get('settled', 0):.0%}")
""")

md("""
## C4. Selection bank and checkpoint choice
A second bank (seed 3). Each candidate replays every fall in it next to the
walker continuing and limp (motors off), then the Go1's fixed rule picks one.
""")
code("""
collect_bank(SELECT_BANK)
for it in CANDIDATES:
    out = select_compare(it)
    if restore(out, out.replace(".json", ".npz")):
        continue
    !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {candidate(it)} --walker-checkpoint {WALKER} --state-bank-file {bank(SELECT_BANK)} --seed {SELECT_BANK} --output {out}
    assert _exit_code == 0, f"selection replay for iteration {it} failed (exit {_exit_code}) -- scroll up"
    save(out, out.replace(".json", ".npz"))
if not restore(SELECTED):
    !python scripts/go1_select_protective_checkpoint.py --compare-glob "outputs/g1_select_it*.json" --output {SELECTED}
    assert _exit_code == 0, f"selection failed (exit {_exit_code}) -- scroll up"
    save(SELECTED)
sel = json.load(open(SELECTED))
print(f"{'iteration':>10s}{'came to rest':>14s}{'ground m':>10s}{'any-body N':>12s}  eligible  preferred")
for r in sel["candidates"]:
    print(f"{r['label']:>10s}{r['settled']:14.3f}{r['envelope_growth_median']:10.3f}{r['any_force_median']:12.1f}  "
          f"{str(r['eligible']):8s}  {r['preferred']}")
print(f"\\nchosen: iteration {sel['selected_label']} (eligible: {sel['selected_is_eligible']}, "
      f"also lowers any-body force: {sel['selected_also_reduces_any_force']})")
""")

md("""
## C5. Test: the chosen checkpoint once, on a fresh bank
Every fall in the test bank (seed 4) under SmallFall, the walker continuing,
and limp. Medians per policy and the verdict against the walker.
""")
code("""
label = sel["selected_label"]           # e.g. "200"
CHOSEN_IT = int(label)
CHOSEN = candidate(CHOSEN_IT)
print("chosen checkpoint:", CHOSEN)
collect_bank(TEST_BANK)
if not restore(TEST, TEST.replace(".json", ".npz")):
    !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {CHOSEN} --walker-checkpoint {WALKER} --state-bank-file {bank(TEST_BANK)} --seed {TEST_BANK} --output {TEST}
    assert _exit_code == 0, f"test replay failed (exit {_exit_code}) -- scroll up"
    save(TEST, TEST.replace(".json", ".npz"))
t = json.load(open(TEST))
rows = [("ground covered (m)", "envelope_growth_m"), ("trunk/pelvis peak force (N)", "peak_force_critical_n"),
        ("any-body peak force (N)", "peak_force_any_n"), ("peak joint torque (Nm)", "peak_torque_nm"),
        ("steps until at rest", "steps")]
print(f"{'median':30s}{'SmallFall':>12s}{'walker':>12s}{'limp':>12s}")
for name, key in rows:
    print(f"{name:30s}" + "".join(f"{t['summaries'][arm][key]['median']:12.2f}" for arm in ("protective", "walker", "limp")))
print(f"{'came to rest (share)':30s}" + "".join(f"{t['summaries'][arm]['settled']['mean']:12.2f}" for arm in ("protective", "walker", "limp")))
print("\\nbeats the walker on all four (trunk force, any-body force, torque, ground covered):", t["verdict"]["pass"])
print("lower than limp:", {m: r["candidate_lower"] for m, r in t["vs_limp"].items()})
""")

md("## V. SmallFall on eight falls from the training bank")
code("""
VIDEO = f"outputs/g1_protective_watch_seed0_ep8.mp4"
if not restore(VIDEO):
    !python scripts/go1_watch_protective_episodes.py --robot g1 --checkpoint-file {CHOSEN} --state-bank-file {bank(TRAIN_BANK)} --episodes 8 --camera-distance 3.5
    assert _exit_code == 0, f"video failed (exit {_exit_code}) -- scroll up"
    save(VIDEO)
display(Video(VIDEO, embed=True, width=640))
""")

md("## Z. Download this run's Drive folder as one zip")
code("""
bundle = shutil.make_archive(f"/content/g1_smallfall_{RUN_NAME}", "zip", SAVE_DIR)
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
out = Path(__file__).resolve().parents[2] / "notebooks" / "g1_smallfall.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
