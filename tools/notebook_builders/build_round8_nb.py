import json
from pathlib import Path
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Round 8: redo of the protective policy after the Gate 7 fail (Go1)

Fresh runtime, A100, about an hour, unattended after the uploads.
Pre-registered in DEVIATIONS.md ("Gate 5/7 redo: round 8").

1. Train two variants with the new peak-impact penalty: **8a** on the usual
   fall bank, **8b** on flag-moment states (how the hand-off really starts).
2. A fixed rule picks one checkpoint on the selection bank (flag seed 3);
   round 7's checkpoint competes, so round 8 only wins if it is better.
3. That one checkpoint is tested **once** on a brand-new flag bank (seed 4).
   The pass condition is the space goal; force is reported beside it.
4. Side-by-side demo video with the selected checkpoint.

**One upload:** in step 2 select **all 7 files** in
`/home/grgott537/cartpole-term/colab_uploads/round8/` (Ctrl+A).
Everything is copied to `MyDrive/cartpole_term_runs/round8/`; one results zip
downloads at the end.
""")
md("## 1. Install (pinned), confirm GPU")
code("""
!pip install -q mjlab==1.6.0 rsl-rl-lib==5.4.2 scikit-learn "imageio[ffmpeg]" pillow
import torch
assert torch.cuda.is_available(), "No GPU visible -- check Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
# mjlab's jit-scripted utils crash on this Colab image (DEVIATIONS.md); speed-only, no effect on results.
%env PYTORCH_JIT=0
""")
md("## 2. Upload everything (select all 7 files in `colab_uploads/round8/`)")
code("""
from google.colab import files
import glob, json, os, re, shutil, subprocess, zipfile

uploaded = files.upload()
names = {re.sub(r" \\(\\d+\\)(?=\\.\\w+$)", "", n): n for n in uploaded}  # strip Colab's " (1)"
DEST = {
    "model_9999.pt": "checkpoints/model_9999.pt",
    "go1_wildpush_predictor_seed0.pt": "outputs/go1_wildpush_predictor_seed0.pt",
    "go1_wildpush_metrics_seed0.json": "outputs/go1_wildpush_metrics_seed0.json",
    "go1_fall_state_bank_seed0.npz": "outputs/go1_fall_state_bank_seed0.npz",
    "go1_flag_state_bank_seed3.npz": "outputs/go1_flag_state_bank_seed3.npz",
    "round7_model_120.pt": "checkpoints/protective/round7_model_120.pt",
}
missing = ({"cartpole_term_code.zip"} | set(DEST)) - set(names)
assert not missing, f"missing uploads: {sorted(missing)} -- re-run this cell and select all 7 files"
ROOT = "/content/cartpole-term"
os.makedirs(ROOT, exist_ok=True)
with zipfile.ZipFile(names["cartpole_term_code.zip"]) as zf:
    zf.extractall(ROOT)
for d in ("checkpoints/protective", "outputs/videos"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
for canon, rel in DEST.items():
    shutil.move(names[canon], f"{ROOT}/{rel}")
%cd /content/cartpole-term
""")
md("## 2b. Google Drive (one permission click), versions, shared settings")
code("""
from google.colab import drive
drive.mount("/content/drive")
SAVE_DIR = "/content/drive/MyDrive/cartpole_term_runs/round8"
os.makedirs(SAVE_DIR, exist_ok=True)
RESULTS = []  # every file that goes in the final zip

def save(*paths):
    for p in paths:
        print("saved to Drive:", shutil.copy(p, SAVE_DIR))
        RESULTS.append(p)

WALKER = "checkpoints/model_9999.pt"
COLLECT = ("python scripts/go1_collect_flagged_states.py --checkpoint-file checkpoints/model_9999.pt "
           "--predictor-file outputs/go1_wildpush_predictor_seed0.pt --metrics-file outputs/go1_wildpush_metrics_seed0.json "
           "--capture-state-bank --capture-at-flag")
TRAIN_BANK = {"8a": "outputs/go1_fall_state_bank_seed0.npz", "8b": "outputs/go1_flag_state_bank_seed5.npz"}
SELECT_BANK = "outputs/go1_flag_state_bank_seed3.npz"
TEST_BANK = "outputs/go1_flag_state_bank_seed4.npz"
CANDIDATE_ITERATIONS = [80, 100, 120, 150, 200, 299]

versions = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
for pkg in ("mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib", "torch", "warp-lang"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_round8.json", "w"), indent=2)
print(json.dumps(versions, indent=2))
save("outputs/versions_round8.json")
""")
md("## 3. Build variant 8b's training bank: flag-moment states, seed 5")
code("""
import numpy as np
!{COLLECT} --seed 5
assert _exit_code == 0, f"bank collection failed (exit {_exit_code}) -- scroll up"
b = np.load(TRAIN_BANK["8b"])
assert np.array_equal(b["captured_step"], b["flag_step"]), "states were not captured at the flag step"
print(len(b["flag_step"]), "flag-moment training states")
save(TRAIN_BANK["8b"])
""")
md("## 4. Smoke test (~1-2 min): the new reward term wires up and PPO runs")
code("""
!python scripts/go1_train_protective.py --smoke
assert _exit_code == 0, f"smoke failed (exit {_exit_code}) -- scroll up for the traceback"
""")
md("""
## 5. Train both variants (300 iterations each, ~8 min each)

Each run's logs (TensorBoard events, eval history, params; without the
checkpoints) are zipped to Drive and added to the final download. The full
run folder with every checkpoint is also zipped to Drive.
""")
code("""
RUN = {}
for variant in ("8a", "8b"):
    print(f"\\n==================== training {variant} on {TRAIN_BANK[variant]} ====================")
    bank = TRAIN_BANK[variant]
    !python scripts/go1_train_protective.py --seeds 0 --state-bank-file {bank}
    assert _exit_code == 0, f"training {variant} failed (exit {_exit_code}) -- scroll up"
    RUN[variant] = sorted(glob.glob("logs/rsl_rl/go1_protective_fall/*"))[-1]
    assert os.path.exists(f"{RUN[variant]}/eval_history.json"), f"{variant}: no eval_history.json"
    full = shutil.make_archive(f"/content/round{variant}_full_run", "zip", RUN[variant])
    print("full run saved to Drive:", shutil.copy(full, SAVE_DIR))
    lite_dir = f"/content/round{variant}_logs"
    shutil.copytree(RUN[variant], lite_dir, ignore=shutil.ignore_patterns("model_*.pt"), dirs_exist_ok=True)
    lite = shutil.make_archive(lite_dir, "zip", lite_dir)
    assert os.path.getsize(lite) > 50_000, f"{variant}: logs zip is suspiciously small"
    save(lite)
print(RUN)
""")
md("""
## 6. Selection: 13 candidates on the selection bank, then the fixed rule

12 round-8 checkpoints plus round 7's iteration 120. Rule: must settle and beat
the walker on ground covered, trunk force and torque; among those, prefer ones
that also beat the walker on any-body force; pick the least ground covered.
""")
code("""
CKPT = {"7_it120": "checkpoints/protective/round7_model_120.pt"}  # round 7's checkpoint competes too
for variant in ("8a", "8b"):
    for it in CANDIDATE_ITERATIONS:
        CKPT[f"{variant}_it{it}"] = f"{RUN[variant]}/model_{it}.pt"
for label, ckpt in CKPT.items():
    out = f"outputs/round8_select_{label}.json"
    print(f"\\n---------- {label} ----------")
    assert os.path.exists(ckpt), f"missing checkpoint {ckpt}"
    !python scripts/go1_protective_compare.py --protective-checkpoint {ckpt} --walker-checkpoint {WALKER} --state-bank-file {SELECT_BANK} --output {out}
    assert _exit_code == 0, f"compare failed for {label} (exit {_exit_code}) -- scroll up"
    save(out)
!python scripts/go1_select_protective_checkpoint.py --compare-glob "outputs/round8_select_*.json" --output outputs/round8_selected_checkpoint.json
assert _exit_code == 0, f"selection failed (exit {_exit_code}) -- scroll up"
sel = json.load(open("outputs/round8_selected_checkpoint.json"))
SEL_VARIANT, SEL_IT = sel["selected_label"].split("_it")
PROTECTIVE = f"checkpoints/protective/round{SEL_VARIANT}_model_{SEL_IT}.pt"
if os.path.abspath(CKPT[sel["selected_label"]]) != os.path.abspath(PROTECTIVE):
    shutil.copy(CKPT[sel["selected_label"]], PROTECTIVE)
print("PROTECTIVE =", PROTECTIVE, "| eligible on the selection bank:", sel["selected_is_eligible"],
      "| also reduces any-body force there:", sel["selected_also_reduces_any_force"])
save("outputs/round8_selected_checkpoint.json", PROTECTIVE)
""")
md("## 7. Fresh test bank (flag moments, seed 4) and checks")
code("""
!{COLLECT} --seed 4
assert _exit_code == 0, f"test bank collection failed (exit {_exit_code}) -- scroll up"
state = lambda b: np.concatenate([b["root_pos_local"], b["root_quat"], b["joint_pos"], b["joint_vel"]], axis=1)
test = np.load(TEST_BANK)
assert np.array_equal(test["captured_step"], test["flag_step"]), "test states were not captured at the flag step"
St = state(test)
for other in (TRAIN_BANK["8a"], TRAIN_BANK["8b"], SELECT_BANK):
    nearest = np.min(np.linalg.norm(St[:, None, :] - state(np.load(other))[None, :, :], axis=-1), axis=1)
    print(f"vs {os.path.basename(other)}: nearest distance min={nearest.min():.3f} median={np.median(nearest):.2f}")
    assert nearest.min() > 1e-4, f"a test state duplicates a state in {other} -- stop"
print(len(St), "test flag moments")
save(TEST_BANK, "outputs/go1_flagged_states_seed4.json")
""")
md("## 8. The test: selected checkpoint, once, on the fresh bank")
code("""
OUT = "outputs/go1_protective_compare_round8_flagstate_seed4.json"
!python scripts/go1_protective_compare.py --protective-checkpoint {PROTECTIVE} --walker-checkpoint {WALKER} --state-bank-file {TEST_BANK} --output {OUT}
assert _exit_code == 0, f"compare failed (exit {_exit_code}) -- scroll up"
save(OUT, OUT.replace(".json", ".npz"))
""")
md("""
### 8b. Verdict (pre-registered, amended to the stated goal: space first)

PASS = less ground covered than BOTH the walker and limp (motors off), comes to rest
(>= 99%), and lower trunk force and torque than the walker. Whether any-body
force is also lower than the walker's is reported separately.
**Paste this cell's output back.**
""")
code("""
s = json.load(open(OUT))["summaries"]
METRICS = ("peak_force_critical_n", "peak_force_any_n", "peak_torque_nm", "envelope_growth_m")
print("selected checkpoint:", sel["selected_label"], "| eligible on selection bank:", sel["selected_is_eligible"])
print(f"{'metric':24s}{'hand-off':>12s}{'walker':>12s}{'limp':>13s}  hand-off < walker")
beats = {}
for m in METRICS:
    beats[m] = s["protective"][m]["median"] < s["walker"][m]["median"]
    print(f"{m:24s}{s['protective'][m]['median']:12.3f}{s['walker'][m]['median']:12.3f}{s['limp'][m]['median']:13.3f}  {beats[m]}")
settled = s["protective"]["settled"]["mean"]
print(f"settled rate: hand-off {settled:.3f}, walker {s['walker']['settled']['mean']:.3f}, "
      f"limp {s['limp']['settled']['mean']:.3f}")
g = "envelope_growth_m"
space_vs_limp = s["protective"][g]["median"] < s["limp"][g]["median"]
print(f"ground covered: {100 * (1 - s['protective'][g]['median'] / s['walker'][g]['median']):.0f}% less than the walker, "
      f"{100 * (1 - s['protective'][g]['median'] / s['limp'][g]['median']):.0f}% less than limp")
print("GATE 7 PASS (space goal):", beats[g] and space_vs_limp and settled >= 0.99
      and beats["peak_force_critical_n"] and beats["peak_torque_nm"])
print("any-body peak force also below the walker:", beats["peak_force_any_n"])
""")
md("## 9. Demo video with the selected checkpoint (640x480 halves, ~10 min)")
code("""
VIDEO = "outputs/videos/go1_handoff_demo_seed0.mp4"
!python scripts/go1_handoff_demo_video.py --protective-checkpoint {PROTECTIVE}
assert _exit_code == 0, f"video failed (exit {_exit_code}) -- scroll up for the traceback"
save(VIDEO, VIDEO.replace(".mp4", ".json"))
from IPython.display import Video, display
display(Video(VIDEO, embed=True, width=960))
""")
md("## 10. Download the results as one zip (everything is already on Drive)")
code("""
bundle_dir = "/content/round8_results"
os.makedirs(bundle_dir, exist_ok=True)
for p in RESULTS:
    shutil.copy(p, bundle_dir)
bundle = shutil.make_archive(bundle_dir, "zip", bundle_dir)
with zipfile.ZipFile(bundle) as zf:
    assert zf.testzip() is None
    print(len(zf.namelist()), "files,", f"{os.path.getsize(bundle)/1e6:.1f} MB")
print("saved to Drive:", shutil.copy(bundle, SAVE_DIR))
files.download(bundle)
""")
md("## 11. Retry the download (only if step 10's didn't arrive)")
code("files.download(bundle)")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(Path(__file__).resolve().parents[2] / "notebooks" / "go1_round8_redo.ipynb", "w"), indent=1)
print(len(cells), "cells")
