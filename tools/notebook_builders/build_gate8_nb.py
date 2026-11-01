"""Generates notebooks/go1_end_to_end.ipynb: the whole Go1 pipeline, with
everything after the walker repeated once per seed in SEEDS."""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Go1 fall pipeline, end to end, repeated over several seeds

One walker, then everything after it once per seed in `SEEDS`: push dataset and
fall predictor -> fall banks -> protective "small-fall" policy -> live hand-off
-> three-way comparison (hand-off, walker continues, limp) -> statistics.
Every step is a `scripts/*.py` call; the cells only pass paths between them.
The decisions behind the fixed choices are in `DEVIATIONS.md`.

| Part | What it does | Runs | Rough A100 time |
|---|---|---|---|
| A | inspect the task, train the walker (10,000 iterations), walker video | once | several hours (or skipped) |
| B | 150 N push dataset, predictor, its confidence intervals | per seed | ~8 min per seed |
| C | flag-moment training bank, protective policy (round-8 reward, iteration 200) | per seed | ~8 min per seed |
| D | live hand-off check (1,024 robots, two arms) | per seed | ~4 min per seed |
| E | fresh test bank, three-way replay, verdict | per seed | ~4 min per seed |
| F | three more fresh banks per seed (about 1,500 falls) and that seed's own significance tests; at the very end, mean and SD across the seeds and the box plots | per seed, then once | ~12 min per seed |
| V | videos (walker, predictor overlay, protective, three-panel demo) | first seed only | ~20 min |

With the default five seeds the back half takes about three hours.

**Skipping the walker training:** tick `SKIP_WALKER_TRAINING` in step 0c **before
running that cell**, then upload `model_9999.pt` when step A2 asks.

**If the runtime disconnects:** run all again with the same `RUN_NAME`. Every
finished step of every seed is restored from Drive and skipped.

**Seeds.** Seed `s` trains its own predictor and protective policy and uses its
own fall banks (bank seed = 1000 x s + role: 5 trains, 4 tests, 6-8 evaluate),
so nothing is shared between seeds except the walker. `SEEDS = (0,)` is the
original single run.
""")
md("## 0a. Install (pinned versions), confirm GPU")
code("""
!pip install -q mjlab==1.6.0 rsl-rl-lib==5.4.2 scikit-learn "imageio[ffmpeg]" pillow
import torch
assert torch.cuda.is_available(), "No GPU visible -- check Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
# Before any script runs: headless rendering, and mjlab's jit-scripted utils crash on this Colab image (DEVIATIONS.md).
%env MUJOCO_GL=egl
%env PYTORCH_JIT=0
""")
md("## 0b. Upload and unpack the code (`cartpole_term_code.zip`)")
code("""
from google.colab import files
from IPython.display import Image, Video, display
import glob, json, os, re, shutil, subprocess, zipfile

uploaded = files.upload()
name = list(uploaded.keys())[0]
os.makedirs("/content/cartpole-term", exist_ok=True)
with zipfile.ZipFile(name) as zf:
    zf.extractall("/content/cartpole-term")
%cd /content/cartpole-term
for d in ("checkpoints/protective", "outputs/videos", "outputs/stats"):
    os.makedirs(d, exist_ok=True)
""")
md("## 0c. Settings, Google Drive, package versions")
code("""
RUN_NAME = "multiseed1"         # results folder on Drive; use a new name for a fresh run
SKIP_WALKER_TRAINING = False  #@param {type:"boolean"}
# ^ tick the box (or write True) to upload an existing model_9999.pt instead of training for hours.
#   This cell must be RUN after changing it: the value is read when the cell runs.
SEEDS = (0, 1, 2, 3, 4)        # one full back half per seed; (0,) is the original single run
PROTECTIVE_ITERATION = 200     # fixed for every seed (chosen on seed 0's selection bank, DEVIATIONS.md)
PEAK_FORCE_WEIGHT = -0.02      # round-8 reward
# Iterations are numbered from 0, so checkpoint model_200.pt exists after 201 iterations.
PROTECTIVE_TRAIN_ITERATIONS = PROTECTIVE_ITERATION + 1
WALKER = "checkpoints/model_9999.pt"
VIDEO_SEED = SEEDS[0]          # videos are made for this seed only
TRAIN_ROLE, TEST_ROLE, EVAL_ROLES = 5, 4, (6, 7, 8)

# Per-seed file names. Nothing below is shared between seeds.
def predictor(s): return f"outputs/go1_wildpush_predictor_seed{s}.pt"
def metrics(s): return f"outputs/go1_wildpush_metrics_seed{s}.json"
def dataset(s): return f"outputs/go1_wildpush_frozen_seed{s}.npz"
def predictor_stats(s): return f"outputs/stats/predictor_seed{s}/predictor_stats.json"
def bank_seed(s, role): return 1000 * s + role
def flag_bank(s, role): return f"outputs/go1_flag_state_bank_seed{bank_seed(s, role)}.npz"
def protective(s): return f"checkpoints/protective/seed{s}_round8b_model_{PROTECTIVE_ITERATION}.pt"
def recovery_stats(s): return f"outputs/stats/recovery_seed{s}/recovery_stats.json"
def handoff(s): return f"outputs/go1_handoff_eval_seed{s}.json"
def test_compare(s): return f"outputs/go1_protective_compare_test_seed{s}.json"
def eval_compare(s, role): return f"outputs/go1_recovery_eval_seed{s}_flag{role}.json"
def npz(path): return path.replace(".json", ".npz")

from google.colab import drive
drive.mount("/content/drive")
SAVE_DIR = f"/content/drive/MyDrive/cartpole_term_runs/go1_end_to_end/{RUN_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

def on_drive(path):
    return os.path.join(SAVE_DIR, path)

def save(*paths):
    \"\"\"Copy result files to this run's Drive folder, keeping their relative paths.\"\"\"
    for p in paths:
        os.makedirs(os.path.dirname(on_drive(p)), exist_ok=True)
        shutil.copy(p, on_drive(p))
        print("saved to Drive:", p)

def restore(*paths):
    \"\"\"True if every path exists here, or could all be brought back from this
    run's Drive folder (an earlier, interrupted attempt). Lets a step be skipped.\"\"\"
    if not all(os.path.exists(p) or os.path.exists(on_drive(p)) for p in paths):
        return False
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            shutil.copy(on_drive(p), p)
    return True

def newest_checkpoint(run_dir):
    return max(glob.glob(run_dir + "/model_*.pt"), key=lambda p: int(re.search(r"model_(\\d+)\\.pt$", p).group(1)))

versions = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "seeds": list(SEEDS)}
for pkg in ("mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib", "torch", "warp-lang", "scipy", "numpy", "scikit-learn"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_go1_end_to_end.json", "w"), indent=2)
print(json.dumps(versions, indent=2))
save("outputs/versions_go1_end_to_end.json")
""")

md("# Part A: walker (once)")
md("## A1. Inspect the task")
code("""
!python scripts/inspect_robot_task.py --robot go1
assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up for the traceback"
""")
md("""
## A2. Train the walker, 10,000 iterations

No training happens if any of these holds, checked in this order: a walker is
already at `checkpoints/model_9999.pt`; this `RUN_NAME` has one saved on Drive;
`SKIP_WALKER_TRAINING` is True (you are then asked to upload `model_9999.pt`).
The cell prints which case it took.
""")
code("""
print(f"SKIP_WALKER_TRAINING = {SKIP_WALKER_TRAINING}  (set in step 0c; re-run 0c after changing it)")
if os.path.exists(WALKER):
    print("using the walker already at", WALKER)   # e.g. dragged into checkpoints/ with Colab's file browser
    save(WALKER)
elif restore(WALKER):
    print("reusing the walker saved on Drive for this RUN_NAME")
elif SKIP_WALKER_TRAINING:
    up = files.upload()
    shutil.move(list(up.keys())[0], WALKER)
    save(WALKER)
else:
    print("TRAINING THE WALKER: 10,000 iterations, several hours. To skip: stop this cell, "
          "set SKIP_WALKER_TRAINING = True in step 0c, re-run 0c, then re-run this cell.")
    !python scripts/robot_train.py --robot go1 --seeds 0 --eval.interval 1000
    assert _exit_code == 0, f"walker training failed (exit {_exit_code}) -- scroll up"
    run_dir = sorted(glob.glob("logs/rsl_rl/go1_velocity_term/*_seed0"))[-1]
    shutil.copy(newest_checkpoint(run_dir), WALKER)
    save(WALKER)
!ls -la {WALKER}
""")

md("# Part B: push dataset and fall predictor (per seed)")
md("""
Collects about a million labelled steps under the 150 N push with collection
seed `s`, trains predictor `s`, and writes its confidence intervals
(`outputs/stats/predictor_seed{s}/`). The dataset stays on this runtime only.
""")
code("""
for s in SEEDS:
    if restore(predictor(s), metrics(s), predictor_stats(s)):
        print(f"seed {s}: predictor already done, skipping")
        continue
    print(f"\\n==================== seed {s}: predictor ====================")
    !python scripts/go1_wild_push_predictor.py --checkpoint-file {WALKER} --mechanism impulse --impulse-force 150 --seed {s}
    assert _exit_code == 0, f"seed {s}: predictor script failed (exit {_exit_code}) -- scroll up"
    p, m, d, sd = predictor(s), metrics(s), dataset(s), os.path.dirname(predictor_stats(s))
    !python scripts/go1_predictor_stats.py --dataset {d} --predictor-file {p} --metrics-file {m} --seed {s} --live-json none --live-npz none --out-dir {sd} --quiet
    assert _exit_code == 0, f"seed {s}: predictor stats failed (exit {_exit_code}) -- scroll up"
    save(p, m, predictor_stats(s), predictor_stats(s).replace(".json", ".md"))
    print(f"seed {s}: threshold {json.load(open(m))['mlp']['threshold']:.4f}, test AUPRC {json.load(open(m))['mlp']['auprc']:.4f}")
""")

md("# Part C: protective policy (per seed)")
md("## C1. Training bank: the robot's state at the moment predictor `s` fires")
code("""
for s in SEEDS:
    bank = flag_bank(s, TRAIN_ROLE)
    if restore(bank):
        print(f"seed {s}: training bank already done, skipping")
        continue
    p, m, bs = predictor(s), metrics(s), bank_seed(s, TRAIN_ROLE)
    !python scripts/go1_collect_flagged_states.py --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
    assert _exit_code == 0, f"seed {s}: bank collection failed (exit {_exit_code}) -- scroll up"
    save(bank)
""")
md("## C2. Train the protective policy (one smoke test, then each seed to the fixed iteration)")
code("""
if not all(restore(protective(s)) for s in SEEDS):
    smoke_bank = flag_bank(SEEDS[0], TRAIN_ROLE)
    !python scripts/go1_train_protective.py --smoke --state-bank-file {smoke_bank}
    assert _exit_code == 0, f"smoke failed (exit {_exit_code}) -- scroll up"
for s in SEEDS:
    history = f"outputs/protective_eval_history_seed{s}.json"
    if restore(protective(s), history):
        print(f"seed {s}: protective policy already trained, skipping")
        continue
    print(f"\\n==================== seed {s}: protective training ====================")
    bank = flag_bank(s, TRAIN_ROLE)
    !python scripts/go1_train_protective.py --seeds {s} --state-bank-file {bank} --max-iterations {PROTECTIVE_TRAIN_ITERATIONS} --peak-force-weight {PEAK_FORCE_WEIGHT}
    assert _exit_code == 0, f"seed {s}: protective training failed (exit {_exit_code}) -- scroll up"
    prot_run = sorted(glob.glob(f"logs/rsl_rl/go1_protective_fall/*_seed{s}"))[-1]
    shutil.copy(f"{prot_run}/model_{PROTECTIVE_ITERATION}.pt", protective(s))
    shutil.copy(f"{prot_run}/eval_history.json", history)
    save(protective(s), history)
""")

md("# Part D: live hand-off check (per seed)")
md("""
Walker under the push, predictor `s` scoring every step, permanent switch to
protective policy `s`. Checks that nothing changes weights or switches back,
and records recall, precision, and warning time.
""")
code("""
for s in SEEDS:
    if restore(handoff(s), npz(handoff(s))):
        print(f"seed {s}: hand-off check already done, skipping")
        continue
    print(f"\\n==================== seed {s}: live hand-off ====================")
    pr, p, m, bank = protective(s), predictor(s), metrics(s), flag_bank(s, TRAIN_ROLE)
    !python scripts/go1_handoff_eval.py --protective-checkpoint {pr} --walker-checkpoint {WALKER} --predictor-file {p} --metrics-file {m} --state-bank-file {bank} --seed {s}
    assert _exit_code == 0, f"seed {s}: hand-off eval failed (exit {_exit_code}) -- scroll up"
    save(handoff(s), npz(handoff(s)))
""")

md("# Part E: test on a fresh bank (per seed)")
md("## E1. Fresh flag-moment bank, replayed under hand-off, walker, and limp")
code("""
for s in SEEDS:
    out = test_compare(s)
    if restore(out, npz(out)):
        print(f"seed {s}: test already done, skipping")
        continue
    print(f"\\n==================== seed {s}: test ====================")
    pr, p, m, bs, bank = protective(s), predictor(s), metrics(s), bank_seed(s, TEST_ROLE), flag_bank(s, TEST_ROLE)
    !python scripts/go1_collect_flagged_states.py --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
    assert _exit_code == 0, f"seed {s}: test bank failed (exit {_exit_code}) -- scroll up"
    !python scripts/go1_protective_compare.py --protective-checkpoint {pr} --walker-checkpoint {WALKER} --state-bank-file {bank} --output {out}
    assert _exit_code == 0, f"seed {s}: compare failed (exit {_exit_code}) -- scroll up"
    save(out, npz(out))
""")
md("""
## E2. Verdict per seed (space first)

PASS = less ground covered than both the walker and limp, comes to rest
(>= 99%), and lower trunk force and torque than the walker.
""")
code("""
g = "envelope_growth_m"
print(f"{'seed':>4s}{'ground: hand-off':>18s}{'walker':>9s}{'limp':>8s}{'trunk N':>9s}{'any-body N':>12s}{'settled':>9s}   space goal   any-body force below walker")
for s in SEEDS:
    q = json.load(open(test_compare(s)))["summaries"]
    med = lambda arm, k: q[arm][k]["median"]
    settled = q["protective"]["settled"]["mean"]
    ok = (med("protective", g) < med("walker", g) and med("protective", g) < med("limp", g) and settled >= 0.99
          and med("protective", "peak_force_critical_n") < med("walker", "peak_force_critical_n")
          and med("protective", "peak_torque_nm") < med("walker", "peak_torque_nm"))
    print(f"{s:4d}{med('protective', g):18.3f}{med('walker', g):9.3f}{med('limp', g):8.3f}"
          f"{med('protective', 'peak_force_critical_n'):9.0f}{med('protective', 'peak_force_any_n'):12.0f}{settled:9.3f}"
          f"   {'PASS' if ok else 'FAIL':10s}   {med('protective', 'peak_force_any_n') < med('walker', 'peak_force_any_n')}")
""")

md("# Part F: statistical evaluation")
md("""
## F1. Per seed: three more fresh banks (about 1,500 paired falls), then that seed's own significance tests

Each seed's tests are run on its own falls and saved
(`outputs/stats/recovery_seed{s}/recovery_stats.json`). Nothing is combined here.
""")
code("""
for s in SEEDS:
    for role in EVAL_ROLES:
        out = eval_compare(s, role)
        if restore(out, npz(out)):
            print(f"seed {s}, bank {role}: already done, skipping")
            continue
        print(f"\\n==================== seed {s}: evaluation bank {role} ====================")
        pr, p, m, bs, bank = protective(s), predictor(s), metrics(s), bank_seed(s, role), flag_bank(s, role)
        !python scripts/go1_collect_flagged_states.py --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
        assert _exit_code == 0, f"seed {s}: evaluation bank {role} failed (exit {_exit_code}) -- scroll up"
        !python scripts/go1_protective_compare.py --protective-checkpoint {pr} --walker-checkpoint {WALKER} --state-bank-file {bank} --output {out}
        assert _exit_code == 0, f"seed {s}: compare failed (exit {_exit_code}) -- scroll up"
        save(out, npz(out))
    stats_json = recovery_stats(s)
    if restore(stats_json):
        print(f"seed {s}: significance tests already done, skipping")
        continue
    seed_npz = " ".join(npz(eval_compare(s, role)) for role in EVAL_ROLES)
    sdir = os.path.dirname(stats_json)
    !python scripts/go1_recovery_stats.py --compare-npz {seed_npz} --out-dir {sdir} --no-figures --quiet
    assert _exit_code == 0, f"seed {s}: significance tests failed (exit {_exit_code}) -- scroll up"
    save(stats_json, stats_json.replace(".json", ".md"))
""")
md("""
## F2. All seeds together: the four box plots (displayed, saved to Drive, downloaded)

Only now are the seeds combined: the mean and standard deviation, across the
seeds, of every statistic each seed saved (medians, differences, effect sizes,
p-values). Boxes: every fall from every seed. Under each name: mean and SD of
the per-seed medians. Brackets: mean and SD of the per-seed p-values and how
many seeds were significant. All tables are written to
`outputs/stats/multiseed/multiseed_stats.md` and go into the final zip.
""")
code("""
# Run with subprocess, not "!": the templates contain a literal {seed} that the
# script fills in per seed, and "!" would try to substitute it itself.
stats_cmd = ["python", "scripts/go1_multiseed_stats.py", "--seeds", *[str(s) for s in SEEDS],
             "--recovery-stats-template", "outputs/stats/recovery_seed{seed}/recovery_stats.json",
             "--compare-npz-template", "outputs/go1_recovery_eval_seed{seed}_flag*.npz",
             "--predictor-stats-template", "outputs/stats/predictor_seed{seed}/predictor_stats.json",
             "--handoff-json-template", "outputs/go1_handoff_eval_seed{seed}.json",
             "--out-dir", "outputs/stats/multiseed", "--quiet"]
done = subprocess.run(stats_cmd, capture_output=True, text=True)
print(done.stdout[-3000:], done.stderr[-6000:])
assert done.returncode == 0, f"multi-seed stats failed (exit {done.returncode}) -- see the output above"
BOXPLOTS = "outputs/stats/multiseed/fig_boxplots.png"
save("outputs/stats/multiseed/multiseed_stats.json", "outputs/stats/multiseed/multiseed_stats.md", BOXPLOTS)
display(Image(BOXPLOTS))
files.download(BOXPLOTS)
""")

md("# Part V: videos (first seed only)")
md("## V1. The walker at a fixed 2 m/s straight ahead")
code("""
WATCH_SPEED_MPS = 2.0
!python scripts/robot_watch_episodes.py --robot go1 --checkpoint-file {WALKER} --episodes 5 --commanded-speed {WATCH_SPEED_MPS}
assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up"
save("outputs/go1_watch_seed0_ep5.mp4")
display(Video("outputs/go1_watch_seed0_ep5.mp4", embed=True, width=640))
""")
md("## V2. Predictor overlay: the five falls only, the last 4 s before each")
code("""
vp, vm = predictor(VIDEO_SEED), metrics(VIDEO_SEED)
!python scripts/go1_wildpush_video.py --checkpoint-file {WALKER} --predictor-weights {vp} --predictor-metrics {vm} --mechanism impulse --impulse-force 150 --min-failures 5 --seed {VIDEO_SEED}
assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up"
OVERLAY = f"outputs/go1_wildpush_overlay_impulse150.0N_seed{VIDEO_SEED}_take0.mp4"
save(OVERLAY)
display(Video(OVERLAY, embed=True, width=640))
""")
md("## V3. The protective policy on eight falls")
code("""
vpr, vbank = protective(VIDEO_SEED), flag_bank(VIDEO_SEED, TRAIN_ROLE)
!python scripts/go1_watch_protective_episodes.py --checkpoint-file {vpr} --state-bank-file {vbank} --episodes 8
assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up"
save("outputs/go1_protective_watch_seed0_ep8.mp4")
display(Video("outputs/go1_protective_watch_seed0_ep8.mp4", embed=True, width=640))
""")
md("## V4. Demo: the same fall three ways (walker alone | limp | hand-off)")
code("""
DEMO = "outputs/videos/go1_handoff_demo_seed0.mp4"
!python scripts/go1_handoff_demo_video.py --protective-checkpoint {vpr} --walker-checkpoint {WALKER} --predictor-file {vp} --metrics-file {vm} --state-bank-file {vbank}
assert _exit_code == 0, f"video failed (exit {_exit_code}) -- scroll up for the traceback"
save(DEMO, DEMO.replace(".mp4", ".json"))
display(Video(DEMO, embed=True, width=1280))
""")

md("## G. Download everything saved to Drive as one zip")
code("""
bundle = shutil.make_archive(f"/content/go1_end_to_end_{RUN_NAME}", "zip", SAVE_DIR)
with zipfile.ZipFile(bundle) as zf:
    assert zf.testzip() is None
    print(len(zf.namelist()), "files,", f"{os.path.getsize(bundle)/1e6:.1f} MB")
files.download(bundle)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(Path(__file__).resolve().parents[2] / "notebooks" / "go1_end_to_end.ipynb", "w"), indent=1)
print(len(cells), "cells")
