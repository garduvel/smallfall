"""Builds notebooks/g1_round3.ipynb: round 3 of G1 SmallFall -- the settle
lock. The reward ablation's top four rewards by size score are retrained
with the lock on (2 runs each, 600 iterations); checkpoint and reward are
chosen on the selection bank; one test on a fresh bank (seed 8) that also
measures what the lock does to existing policies when added without
retraining (DEVIATIONS.md 2026-11-14).

    python3 tools/notebook_builders/build_g1_round3_nb.py
"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# G1 SmallFall, round 3: the settle lock

**The lock:** if a fall has not come to rest on its own, the joints hold
their pose (the policy is cut off) once the fall has run 3.0 s, or earlier
once the robot is low (pelvis below 0.38 m) and nearly still for 0.3 s.
"Came to rest" is still the real physics check, so the lock cannot fake it.

| Step | What it does | Rough time (A100 / L4) |
|---|---|---|
| 0 | install, clone, Drive, inputs | ~4 min |
| T | 4 rewards x 2 runs, 600 iterations each, lock on (resumes per run) | ~4 h / ~7 h |
| S | each run picks its checkpoint (400/500/600) on the selection bank, lock on; then one reward and one policy | ~40 min |
| X | the test on a fresh bank (seed 8), once: round-3 runs, plus existing policies with and without the lock | ~30 min |
| V | video: the chosen policy on eight test falls, lock on | ~3 min |

**Rewards:** the reward ablation's top four by size score, each in place of
SmallFall's ground-covered term: hull growth, centroid travel, hull area,
floor touched.

**Lock without retraining (test only):** the original SmallFall, round 2's
hullgrow_s1 and hullgrow_s2, and the four reward-ablation policies, each
replayed with and without the lock on the same falls.

**Fixed before any data (`DEVIATIONS.md`):** same size score and safety as
round 2; the rest bar is raised to **99%**. Pass (chosen policy, test bank):
comes to rest in 99% or more, safe, smaller than the original SmallFall
(paired, p < 0.05) and than limp, ground covered below limp and walker, and
size score no more than 1.05x round 2's chosen policy on the same falls.

**Disconnects:** run all again; finished runs are restored from Drive.
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
import glob, json, os, re, shutil, subprocess, time, zipfile
from IPython.display import Video, display

REPO_URL = "https://github.com/garduvel/smallfall.git"
REPO_REF = "main"
if not os.path.exists("/content/smallfall/.git"):
    !git clone -q {REPO_URL} /content/smallfall
%cd /content/smallfall
!git fetch -q origin && git checkout -q {REPO_REF} && (git pull -q --ff-only origin {REPO_REF} 2>/dev/null || true)
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code at commit", COMMIT)
for d in ("checkpoints/protective", "checkpoints/round2", "checkpoints/reward_ablation", "checkpoints/round3", "outputs"):
    os.makedirs(d, exist_ok=True)
""")

md("## 0c. Settings, Drive, inputs")
code("""
RUN_NAME = "round3"
WALKER_RUN, PREDICTOR_RUN, SMALLFALL_RUN, ABLATION_RUN, ROUND2_RUN = "walker2", "predictor1", "smallfall1", "rewardablation1", "round2"
SEEDS = (0, 1)
TRAIN_BANK, SELECT_BANK, TEST_BANK = 5, 3, 8
TRAIN_ITERATIONS = 601
CANDIDATES = (400, 500, 600)
MIN_SETTLED = 0.99
REWARDS = {   # round-3 label: fall-size measure (the reward ablation's top four by size score)
    "hullgrow": "hull_growth_m2",
    "centroid": "centroid_travel_end_m",
    "hull": "swept_hull_area_m2",
    "touched": "touched_area_m2",
}

DRIVE = "/content/drive/MyDrive/cartpole_term_runs"
WALKER = f"checkpoints/g1_walker_{WALKER_RUN}_model_9999.pt"
PREDICTOR = f"outputs/g1_wildpush_predictor_seed0.pt"
METRICS = f"outputs/g1_wildpush_metrics_seed0.json"
WEIGHTS = "outputs/g1_protective_weights.json"
SIZE_WEIGHTS = "outputs/g1_fall_size_weights.json"
SELECTED = "outputs/g1_selected_checkpoint.json"
def bank(seed): return f"outputs/g1_flag_state_bank_seed{seed}.npz"
def flagged(seed): return f"outputs/g1_flagged_states_seed{seed}.json"
def run_label(reward, seed): return f"{reward}_s{seed}"
def ckpt(label, it): return f"checkpoints/round3/{label}_model_{it}.pt"
def history(label): return f"outputs/g1_r3_{label}_eval_history.json"
def sel_compare(label): return f"outputs/g1_r3sel_{label}.json"
def sel_choice(label): return f"outputs/g1_r3sel_{label}_choice.json"
CHOICE = "outputs/g1_r3_choice.json"
TEST = "outputs/g1_r3_test.json"
SUMMARY = "outputs/g1_r3_summary.json"
TABLE = "outputs/g1_r3_measures.json"

from google.colab import drive, files
drive.mount("/content/drive")
SAVE_DIR = f"{DRIVE}/g1_round3/{RUN_NAME}"
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
abl_dir = f"{DRIVE}/g1_reward_ablation/{ABLATION_RUN}"
r2_dir = f"{DRIVE}/g1_round2/{ROUND2_RUN}"
inputs = [(WALKER, f"{DRIVE}/g1_walker/{WALKER_RUN}/{WALKER}"),
          (PREDICTOR, f"{DRIVE}/g1_predictor/{PREDICTOR_RUN}/{PREDICTOR}"),
          (METRICS, f"{DRIVE}/g1_predictor/{PREDICTOR_RUN}/{METRICS}"),
          (WEIGHTS, f"{sf}/{WEIGHTS}"), (SELECTED, f"{sf}/{SELECTED}"),
          (bank(TRAIN_BANK), f"{sf}/{bank(TRAIN_BANK)}"), (bank(SELECT_BANK), f"{sf}/{bank(SELECT_BANK)}"),
          (SIZE_WEIGHTS, f"{abl_dir}/{SIZE_WEIGHTS}")]
# Existing policies replayed with and without the lock (test only).
ABL_TAGS = ("hullgrow", "centroid", "hull", "touched")
for tag in ABL_TAGS:
    c = f"outputs/g1_rsel_{tag}_choice.json"
    inputs.append((c, f"{abl_dir}/{c}"))
R2_CKPTS = {"r2s1": "checkpoints/round2/hullgrow_s1_model_600.pt", "r2s2": "checkpoints/round2/hullgrow_s2_model_600.pt"}
inputs += [(p, f"{r2_dir}/{p}") for p in R2_CKPTS.values()]
for local, src in inputs:
    assert os.path.exists(src), f"missing {src} -- check the *_RUN names"
    shutil.copy(src, local)
    print("input:", local)
SF_CHOSEN = f"checkpoints/protective/g1_seed0_model_{int(json.load(open(SELECTED))['selected_label'])}.pt"
shutil.copy(f"{sf}/{SF_CHOSEN}", SF_CHOSEN)
EXISTING = {"sf": SF_CHOSEN, **R2_CKPTS}      # label -> checkpoint
for tag in ABL_TAGS:
    p = json.load(open(f"outputs/g1_rsel_{tag}_choice.json"))["checkpoint"]
    shutil.copy(f"{abl_dir}/{p}", p)
    EXISTING[f"abl{tag}"] = p
for k, v in EXISTING.items():
    print(f"existing policy {k}: {v}")

versions = {"gpu": torch.cuda.get_device_name(0), "commit": COMMIT}
for pkg in ("mjlab", "mujoco-warp", "rsl-rl-lib", "torch", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_g1_round3.json", "w"), indent=2)
save("outputs/versions_g1_round3.json")
""")

md("""
## T. Train 4 rewards x 2 runs with the lock on (600 iterations each)
Each finished run is saved to Drive straight away. The log line
`Episode_Metrics/settle_locked` is the share of training falls that needed the lock.
""")
code("""
for seed in SEEDS:                     # seed-major: one run of every reward first
    for reward, measure in REWARDS.items():
        label = run_label(reward, seed)
        files_ = [ckpt(label, it) for it in CANDIDATES] + [history(label)]
        if restore(*files_):
            print(f"{label}: already trained")
            continue
        t0 = time.time()
        !python scripts/go1_train_protective.py --robot g1 --seeds {seed} --state-bank-file {bank(TRAIN_BANK)} --weights-json {WEIGHTS} --fall-size-weights-json {SIZE_WEIGHTS} --fall-size-measure {measure} --settle-lock --run-tag r3_{reward} --max-iterations {TRAIN_ITERATIONS}
        assert _exit_code == 0, f"training {label} failed (exit {_exit_code}) -- scroll up"
        run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_r3_{reward}_seed{seed}"))[-1]
        for it in CANDIDATES:
            shutil.copy(f"{run}/model_{it}.pt", ckpt(label, it))
        shutil.copy(f"{run}/eval_history.json", history(label))
        save(*files_)
        print(f"{label}: trained in {(time.time() - t0) / 60:.0f} min")
for seed in SEEDS:
    for reward in REWARDS:
        h = json.load(open(history(run_label(reward, seed))))
        g = lambda i: h[i]["mean_action"]
        print(f"{run_label(reward, seed):12s} frozen-eval return it300 {g(30)['mean_return']:7.3f} -> it600 {g(-1)['mean_return']:7.3f}; "
              f"came to rest {g(-1)['termination_cause_breakdown'].get('settled', 0):.0%}")
""")

md("""
## S. Choose: checkpoint per run, then the reward and the policy (selection bank, lock on)
""")
code("""
for seed in SEEDS:
    for reward in REWARDS:
        label = run_label(reward, seed)
        out, choice = sel_compare(label), sel_choice(label)
        if not restore(out, out.replace(".json", ".npz")):
            extra = " ".join(f"--extra-policy it{it}={ckpt(label, it)}" for it in CANDIDATES[:-1])
            locked = ",".join(["protective"] + [f"it{it}" for it in CANDIDATES[:-1]])
            !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {ckpt(label, CANDIDATES[-1])} --walker-checkpoint {WALKER} --state-bank-file {bank(SELECT_BANK)} --seed {SELECT_BANK} --output {out} {extra} --locked-arms {locked}
            assert _exit_code == 0, f"selection replay {label} failed (exit {_exit_code}) -- scroll up"
            save(out, out.replace(".json", ".npz"))
        print(f"--- {label}")
        cands = f"--candidate protective={ckpt(label, CANDIDATES[-1])} " + " ".join(f"--candidate it{it}={ckpt(label, it)}" for it in CANDIDATES[:-1])
        !python scripts/robot_select_safe_small.py --compare-npz {out.replace(".json", ".npz")} {cands} --min-settled {MIN_SETTLED} --output {choice}
        assert _exit_code == 0, f"selection {label} failed (exit {_exit_code}) -- scroll up"
        save(choice)
print("=== reward and policy")
!python scripts/robot_round2_choose.py --choices "outputs/g1_r3sel_*_choice.json" --output {CHOICE}
assert _exit_code == 0, "choice failed -- scroll up"
save(CHOICE)
choice = json.load(open(CHOICE))
""")

md("""
## X. The test (bank 8, once)
`protective` = the original SmallFall without the lock. Round-3 runs are
replayed with the lock (as trained). Every existing policy is replayed twice:
`<name>` without the lock, `<name>_lock` with it.
""")
code("""
collected = restore(bank(TEST_BANK), flagged(TEST_BANK))
if not collected:
    !python scripts/go1_collect_flagged_states.py --robot g1 --checkpoint-file {WALKER} --predictor-file {PREDICTOR} --metrics-file {METRICS} --seed {TEST_BANK} --capture-state-bank --capture-at-flag
    assert _exit_code == 0, f"bank collection failed (exit {_exit_code}) -- scroll up"
    save(bank(TEST_BANK), flagged(TEST_BANK))
NPZ = TEST.replace(".json", ".npz")
if not restore(TEST, NPZ):
    runs = json.load(open(CHOICE))["runs"]
    arms = {label: r["checkpoint"] for label, r in runs.items()}
    for name, p in EXISTING.items():
        if name != "sf":
            arms[name] = p
        arms[f"{name}_lock"] = p
    locked = [label for label in runs] + [f"{name}_lock" for name in EXISTING]
    extra = " ".join(f"--extra-policy {label}={p}" for label, p in arms.items())
    !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {SF_CHOSEN} --walker-checkpoint {WALKER} --state-bank-file {bank(TEST_BANK)} --seed {TEST_BANK} --output {TEST} {extra} --locked-arms {",".join(locked)}
    assert _exit_code == 0, f"test replay failed (exit {_exit_code}) -- scroll up"
    save(TEST, NPZ)
!python scripts/robot_round3_summary.py --test-npz {NPZ} --choice-json {CHOICE} --output {SUMMARY}
assert _exit_code == 0, "summary failed -- scroll up"
!python scripts/robot_fall_size_ablation.py --compare-npz {NPZ} --robot g1 --output {TABLE}
assert _exit_code == 0, "ten-measure table failed -- scroll up"
save(SUMMARY, TABLE)
""")

md("## V. The chosen policy on eight test falls (lock on)")
code("""
choice = json.load(open(CHOICE))
VIDEO = f"outputs/g1_r3_{choice['chosen_policy']}_watch.mp4"
if not restore(VIDEO):
    !python scripts/go1_watch_protective_episodes.py --robot g1 --checkpoint-file {choice['checkpoint']} --state-bank-file {bank(TEST_BANK)} --episodes 8 --camera-distance 3.5 --settle-lock
    assert _exit_code == 0, f"video failed (exit {_exit_code}) -- scroll up"
    shutil.move(sorted(glob.glob("outputs/g1_protective_watch_seed*_ep8.mp4"), key=os.path.getmtime)[-1], VIDEO)
    save(VIDEO)
display(Video(VIDEO, embed=True, width=640))
""")

md("## Z. Download this run's Drive folder as one zip")
code("""
bundle = shutil.make_archive(f"/content/g1_round3_{RUN_NAME}", "zip", SAVE_DIR)
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
out = Path(__file__).resolve().parents[2] / "notebooks" / "g1_round3.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
