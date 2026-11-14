"""Builds notebooks/g1_reward_ablation.ipynb: one SmallFall policy per way of
measuring fall size (ten rewards), then every policy scored on every
measure (DEVIATIONS.md 2026-11-13, reward ablation).

    python3 tools/notebook_builders/build_g1_reward_ablation_nb.py
"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# G1 reward ablation: one SmallFall policy per fall-size measure

SmallFall's reward penalises **ground covered**. Here, ten policies are
trained, each with a different fall-size measure in that slot, then every
policy is scored on every measure on the same fresh falls.

| Step | What it does | Rough time (A100 / L4) |
|---|---|---|
| 0 | install, clone, Drive, fetch walker, predictor, SmallFall inputs | ~4 min |
| R1 | size the ten weights: one 11-iteration run logging every measure | ~3 min |
| R2 | train ten policies, 301 iterations each (resumes per policy) | ~2.5 h / ~4-5 h |
| R3 | each policy picks its checkpoint (200/250/300) by **its own** measure, on the selection bank | ~30 min |
| R4 | all ten + original SmallFall + walker + limp replay the same fresh falls (bank 6) | ~15 min |
| R5 | the ablation: matrix, own-measure wins, the three combining methods, plots | ~1 min |

**Fixed before any data (`DEVIATIONS.md`):**
- **Everything except the fall-size term** is the G1 SmallFall reward
  (calibrated weights), with the same training bank (5), selection bank (3),
  iterations (301) and seed (0).
- **The fall-size term** pays, each step, the increase in its measure, so an
  episode's total is that measure for the fall.
- **Weights:** each measure's weight makes its episode total weigh what
  ground covered does, on the same early episodes (iterations 0-10 of R1).
  Ground covered keeps exactly SmallFall's weight.
- **Checkpoint:** among iterations 200/250/300, the lowest median of the
  policy's own measure on the selection bank, among those that come to rest in
  at least 99% of falls.
- **Scoring:** a fresh bank (seed 6). The question "which falls smallest
  overall" uses the three combining methods from the fall-size ablation.

**Two pairs differ only by a constant:** max reach = ground covered + the
robot's starting size, and hull area = hull growth + the starting hull. The
reward differs per episode only by a constant the policy can't change, but
the weights differ, so the pairs are trained separately and show what the
weight scale alone does. `r_ground` is also a second training run of
SmallFall's own reward, so `r_ground` vs the original SmallFall shows how much
two runs of the same reward differ.

**Disconnects:** run all again; finished policies are restored from Drive.
`MEASURES` in 0c can be trimmed to run fewer.
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

REPO_URL = "https://github.com/garduvel/smallfall.git"
REPO_REF = "main"
if not os.path.exists("/content/smallfall/.git"):
    !git clone -q {REPO_URL} /content/smallfall
%cd /content/smallfall
!git fetch -q origin && git checkout -q {REPO_REF} && (git pull -q --ff-only origin {REPO_REF} 2>/dev/null || true)
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code at commit", COMMIT)
for d in ("checkpoints/protective", "checkpoints/reward_ablation", "outputs"):
    os.makedirs(d, exist_ok=True)
""")

md("## 0c. Settings, Drive, inputs")
code("""
RUN_NAME = "rewardablation1"
WALKER_RUN, PREDICTOR_RUN, SMALLFALL_RUN = "walker2", "predictor1", "smallfall1"
SEED = 0
TRAIN_BANK, SELECT_BANK, SCORE_BANK = 5, 3, 6
TRAIN_ITERATIONS = 301
CANDIDATES = (200, 250, 300)
TAGS = {"envelope_growth_m": "ground", "max_reach_m": "reach", "root_travel_end_m": "rootend",
        "root_travel_max_m": "rootmax", "root_path_m": "rootpath", "centroid_travel_end_m": "centroid",
        "swept_hull_area_m2": "hull", "hull_growth_m2": "hullgrow", "contact_reach_m": "contact",
        "touched_area_m2": "touched"}
MEASURES = list(TAGS)                       # trim to run fewer

DRIVE = "/content/drive/MyDrive/cartpole_term_runs"
WALKER = f"checkpoints/g1_walker_{WALKER_RUN}_model_9999.pt"
PREDICTOR = f"outputs/g1_wildpush_predictor_seed{SEED}.pt"
METRICS = f"outputs/g1_wildpush_metrics_seed{SEED}.json"
WEIGHTS = "outputs/g1_protective_weights.json"
SELECTED = "outputs/g1_selected_checkpoint.json"
SIZE_WEIGHTS = "outputs/g1_fall_size_weights.json"
def bank(seed): return f"outputs/g1_flag_state_bank_seed{seed}.npz"
def flagged(seed): return f"outputs/g1_flagged_states_seed{seed}.json"
def sf_candidate(it): return f"checkpoints/protective/g1_seed{SEED}_model_{it}.pt"
def ckpt(tag, it): return f"checkpoints/reward_ablation/r_{tag}_model_{it}.pt"
def history(tag): return f"outputs/g1_r_{tag}_eval_history.json"
def sel_compare(tag): return f"outputs/g1_rsel_{tag}.json"
def sel_choice(tag): return f"outputs/g1_rsel_{tag}_choice.json"
SCORE = "outputs/g1_reward_ablation_score.json"
REPORT = "outputs/g1_reward_ablation_report.json"

from google.colab import drive, files
drive.mount("/content/drive")
SAVE_DIR = f"{DRIVE}/g1_reward_ablation/{RUN_NAME}"
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
          (WEIGHTS, f"{sf}/{WEIGHTS}"), (SELECTED, f"{sf}/{SELECTED}"),
          (bank(TRAIN_BANK), f"{sf}/{bank(TRAIN_BANK)}"), (bank(SELECT_BANK), f"{sf}/{bank(SELECT_BANK)}")]
for local, src in inputs:
    assert os.path.exists(src), f"missing {src} -- check the *_RUN names (SmallFall must have finished C4)"
    shutil.copy(src, local)
    print("input:", local)
SF_IT = int(json.load(open(SELECTED))["selected_label"])
SF_CHOSEN = sf_candidate(SF_IT)
shutil.copy(f"{sf}/{SF_CHOSEN}", SF_CHOSEN)
print("original SmallFall: iteration", SF_IT)

versions = {"gpu": torch.cuda.get_device_name(0), "commit": COMMIT}
for pkg in ("mjlab", "mujoco-warp", "rsl-rl-lib", "torch", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_g1_reward_ablation.json", "w"), indent=2)
save("outputs/versions_g1_reward_ablation.json")
""")

md("""
## R1. Size the ten weights
11 iterations of SmallFall's own reward with every measure logged (no effect
on the reward). Each measure's weight = SmallFall's ground-covered weight x
(ground covered total / that measure's total).
""")
code("""
if restore(SIZE_WEIGHTS):
    print("weights already sized")
else:
    !python scripts/go1_train_protective.py --robot g1 --seeds {SEED} --state-bank-file {bank(TRAIN_BANK)} --weights-json {WEIGHTS} --log-fall-sizes --run-tag sizecal --max-iterations 11 --eval.interval 10
    assert _exit_code == 0, f"sizing run failed (exit {_exit_code}) -- scroll up"
    run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_sizecal_seed{SEED}"))[-1]
    !python scripts/robot_calibrate_fall_size.py --run-dir {run} --weights-json {WEIGHTS} --output {SIZE_WEIGHTS}
    assert _exit_code == 0, f"sizing failed (exit {_exit_code}) -- scroll up"
    save(SIZE_WEIGHTS)
for m, r in json.load(open(SIZE_WEIGHTS))["terms"].items():
    note = " (clamped)" if r["clamped"] else (" (kept)" if r["kept_base"] else "")
    print(f"{m:24s} episode total {r['episode_sum']:9.4f}   weight {r['weight']:10.4g}{note}")
""")

md("""
## R2. Train one policy per measure (301 iterations each)
Each finished policy is saved to Drive straight away, so a disconnect only
loses the one in progress.
""")
code("""
for m in MEASURES:
    tag = TAGS[m]
    files_ = [ckpt(tag, it) for it in CANDIDATES] + [history(tag)]
    if restore(*files_):
        print(f"r_{tag}: already trained")
        continue
    t0 = time.time()
    !python scripts/go1_train_protective.py --robot g1 --seeds {SEED} --state-bank-file {bank(TRAIN_BANK)} --weights-json {WEIGHTS} --fall-size-measure {m} --fall-size-weights-json {SIZE_WEIGHTS} --run-tag r_{tag} --max-iterations {TRAIN_ITERATIONS}
    assert _exit_code == 0, f"training r_{tag} failed (exit {_exit_code}) -- scroll up"
    run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_r_{tag}_seed{SEED}"))[-1]
    for it in CANDIDATES:
        shutil.copy(f"{run}/model_{it}.pt", ckpt(tag, it))
    shutil.copy(f"{run}/eval_history.json", history(tag))
    save(*files_)
    print(f"r_{tag}: trained in {(time.time() - t0) / 60:.0f} min")
for m in MEASURES:
    h = json.load(open(history(TAGS[m])))[-1]["mean_action"]
    print(f"r_{TAGS[m]:9s} final frozen-eval: return {h['mean_return']:8.3f}, "
          f"came to rest {h['termination_cause_breakdown'].get('settled', 0):.0%}")
""")

md("""
## R3. Each policy picks its checkpoint by its own measure (selection bank)
""")
code("""
for m in MEASURES:
    tag = TAGS[m]
    out, choice = sel_compare(tag), sel_choice(tag)
    if not restore(out, out.replace(".json", ".npz")):
        extra = " ".join(f"--extra-policy it{it}={ckpt(tag, it)}" for it in CANDIDATES[:-1])
        !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {ckpt(tag, CANDIDATES[-1])} --walker-checkpoint {WALKER} --state-bank-file {bank(SELECT_BANK)} --seed {SELECT_BANK} --output {out} {extra}
        assert _exit_code == 0, f"selection replay r_{tag} failed (exit {_exit_code}) -- scroll up"
        save(out, out.replace(".json", ".npz"))
    if not restore(choice):
        cands = f"--candidate protective={ckpt(tag, CANDIDATES[-1])} " + " ".join(f"--candidate it{it}={ckpt(tag, it)}" for it in CANDIDATES[:-1])
        !python scripts/robot_select_by_measure.py --compare-npz {out.replace(".json", ".npz")} --measure {m} {cands} --output {choice}
        assert _exit_code == 0, f"selection r_{tag} failed (exit {_exit_code}) -- scroll up"
        save(choice)
    c = json.load(open(choice))
    print(f"r_{tag:9s} -> {c['checkpoint']}" + ("" if c["selected_is_eligible"] else "   (none comes to rest in 99%)"))
""")

md("""
## R4. Score: every policy on the same fresh falls (bank 6)
`protective` here is the **original** SmallFall (from `g1_smallfall.ipynb`).
""")
code("""
if not restore(bank(SCORE_BANK), flagged(SCORE_BANK)):
    other = f"{DRIVE}/g1_fall_size/sizeablation1"   # reuse bank 6 if the fall-size notebook made it
    if os.path.exists(f"{other}/{bank(SCORE_BANK)}"):
        for p in (bank(SCORE_BANK), flagged(SCORE_BANK)):
            shutil.copy(f"{other}/{p}", p)
    else:
        !python scripts/go1_collect_flagged_states.py --robot g1 --checkpoint-file {WALKER} --predictor-file {PREDICTOR} --metrics-file {METRICS} --seed {SCORE_BANK} --capture-state-bank --capture-at-flag
        assert _exit_code == 0, f"bank collection failed (exit {_exit_code}) -- scroll up"
    save(bank(SCORE_BANK), flagged(SCORE_BANK))
NPZ = SCORE.replace(".json", ".npz")
if not restore(SCORE, NPZ):
    extra = " ".join(f"--extra-policy r_{TAGS[m]}={json.load(open(sel_choice(TAGS[m])))['checkpoint']}" for m in MEASURES)
    !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {SF_CHOSEN} --walker-checkpoint {WALKER} --state-bank-file {bank(SCORE_BANK)} --seed {SCORE_BANK} --output {SCORE} {extra}
    assert _exit_code == 0, f"scoring replay failed (exit {_exit_code}) -- scroll up"
    save(SCORE, NPZ)
""")

md("## R5. The ablation")
code("""
!python scripts/robot_fall_size_ablation.py --compare-npz {NPZ} --robot g1 --output {REPORT}
assert _exit_code == 0, f"ablation failed (exit {_exit_code}) -- scroll up"
save(REPORT)
rep = json.load(open(REPORT))
""")

md("""
Left: each trained policy's median on each measure, relative to limp
(rows = what it was trained on; boxed = its own measure). Right: share of
falls that would reach an object at each distance.
""")
code("""
import numpy as np
import matplotlib.pyplot as plt

t = rep["trained_on"]
pols, mets = list(t["ratio_to_reference"]), rep["metrics"]
short = ["ground", "reach", "root end", "root max", "root path", "centroid", "hull", "hull grow", "contact", "touched"]
R = np.array([[t["ratio_to_reference"][p][m] for m in mets] for p in pols])
fig, ax = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
im = ax[0].imshow(R, cmap="RdYlGn_r", vmin=0.3, vmax=1.7, aspect="auto")
ax[0].set_xticks(range(len(mets)), short, rotation=45, ha="right"); ax[0].set_yticks(range(len(pols)), pols)
for i, p in enumerate(pols):
    for j, m in enumerate(mets):
        ax[0].text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8)
        if m == t["own_measure"][p]["measure"]:
            ax[0].add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=2.5, ec="k"))
ax[0].set_title("Median size relative to limp (lower = smaller)"); fig.colorbar(im, ax=ax[0], fraction=.03)
radii = rep["reach_curves"]["radii_m"]
for p in rep["policies"]:
    ax[1].plot(radii, rep["reach_curves"][f"{p}|max_reach_m"], label=p,
               lw=2.5 if p in ("protective", "limp", "walker") else 1.2, ls="--" if p in ("limp", "walker") else "-")
ax[1].set_xlabel("object distance from the fall's start (m)"); ax[1].set_ylabel("share of falls that reach it")
ax[1].set_title("Clear radius needed"); ax[1].legend(fontsize=7, ncol=2); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("outputs/g1_reward_ablation.png", dpi=120); save("outputs/g1_reward_ablation.png")
plt.show()
""")

md("## Z. Download this run's Drive folder as one zip")
code("""
bundle = shutil.make_archive(f"/content/g1_reward_ablation_{RUN_NAME}", "zip", SAVE_DIR)
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
out = Path(__file__).resolve().parents[2] / "notebooks" / "g1_reward_ablation.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
