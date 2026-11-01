import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Confirmatory evaluation of the three recovery policies (Go1)

Fresh runtime, A100 (L4 also fine), ~20 min. No training. Pre-registered in
DEVIATIONS.md ("Statistical evaluation: analysis plan").

1. Collect three fresh flag-moment banks (seeds 6, 7, 8; about 1,500 falls) and
   one fresh mid-fall bank (seed 9).
2. Replay every state under: hand-off to the protective policy (round 8b,
   iteration 200), the walker continuing, and limp (motors off).
3. Replay the seed-6 bank a second time (how much the simulator's
   non-determinism moves the numbers).
4. Run the paired statistical tests and draw the box plots (one figure, four panels).

**One upload:** in step 2 select **all 9 files** in
`/home/grgott537/cartpole-term/colab_uploads/recovery_eval/` (Ctrl+A).
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
md("## 2. Upload everything (select all 9 files in `colab_uploads/recovery_eval/`)")
code("""
from google.colab import files
import json, os, re, shutil, subprocess, zipfile
import numpy as np

uploaded = files.upload()
names = {re.sub(r" \\(\\d+\\)(?=\\.\\w+$)", "", n): n for n in uploaded}  # strip Colab's " (1)"
DEST = {
    "model_9999.pt": "checkpoints/model_9999.pt",
    "round8b_model_200.pt": "checkpoints/protective/round8b_model_200.pt",
    "go1_wildpush_predictor_seed0.pt": "outputs/go1_wildpush_predictor_seed0.pt",
    "go1_wildpush_metrics_seed0.json": "outputs/go1_wildpush_metrics_seed0.json",
    "go1_fall_state_bank_seed0.npz": "outputs/go1_fall_state_bank_seed0.npz",
    "go1_flag_state_bank_seed3.npz": "outputs/go1_flag_state_bank_seed3.npz",
    "go1_flag_state_bank_seed4.npz": "outputs/go1_flag_state_bank_seed4.npz",
    "go1_flag_state_bank_seed5.npz": "outputs/go1_flag_state_bank_seed5.npz",
}
missing = ({"cartpole_term_code.zip"} | set(DEST)) - set(names)
assert not missing, f"missing uploads: {sorted(missing)} -- re-run this cell and select all 9 files"
ROOT = "/content/cartpole-term"
os.makedirs(ROOT, exist_ok=True)
with zipfile.ZipFile(names["cartpole_term_code.zip"]) as zf:
    zf.extractall(ROOT)
for d in ("checkpoints/protective", "outputs/recovery_eval"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
for canon, rel in DEST.items():
    shutil.move(names[canon], f"{ROOT}/{rel}")
%cd /content/cartpole-term

from google.colab import drive
drive.mount("/content/drive")
SAVE_DIR = "/content/drive/MyDrive/cartpole_term_runs/recovery_eval"
os.makedirs(SAVE_DIR, exist_ok=True)
OUT = "outputs/recovery_eval"
PROTECTIVE, WALKER = "checkpoints/protective/round8b_model_200.pt", "checkpoints/model_9999.pt"
COLLECT = ("python scripts/go1_collect_flagged_states.py --checkpoint-file checkpoints/model_9999.pt "
           "--predictor-file outputs/go1_wildpush_predictor_seed0.pt --metrics-file outputs/go1_wildpush_metrics_seed0.json "
           "--capture-state-bank")
versions = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
for pkg in ("mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib", "torch", "warp-lang", "scipy", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open(f"{OUT}/versions_recovery_eval.json", "w"), indent=2)
print(json.dumps(versions, indent=2))
""")
md("## 3. Collect the fresh banks: flag moments (seeds 6, 7, 8) and mid-fall states (seed 9)")
code("""
FLAG_SEEDS, FALL_SEED = (6, 7, 8), 9
for seed in FLAG_SEEDS:
    !{COLLECT} --capture-at-flag --seed {seed}
    assert _exit_code == 0, f"flag bank seed {seed} failed (exit {_exit_code}) -- scroll up"
!{COLLECT} --seed {FALL_SEED}
assert _exit_code == 0, f"fall bank failed (exit {_exit_code}) -- scroll up"

BANKS = {f"flag_seed{s}": f"outputs/go1_flag_state_bank_seed{s}.npz" for s in FLAG_SEEDS}
BANKS[f"fall_seed{FALL_SEED}"] = f"outputs/go1_fall_state_bank_seed{FALL_SEED}.npz"
state = lambda b: np.concatenate([b["root_pos_local"], b["root_quat"], b["joint_pos"], b["joint_vel"]], axis=1)
OLD = [f"outputs/go1_flag_state_bank_seed{s}.npz" for s in (3, 4, 5)] + ["outputs/go1_fall_state_bank_seed0.npz"]
loaded = {name: np.load(path) for name, path in BANKS.items()}
for name, b in loaded.items():
    if name.startswith("flag"):
        assert np.array_equal(b["captured_step"], b["flag_step"]), f"{name}: states were not captured at the flag step"
    others = [state(np.load(p)) for p in OLD] + [state(o) for n, o in loaded.items() if n != name]
    nearest = min(float(np.min(np.linalg.norm(state(b)[:, None, :] - o[None, :, :], axis=-1))) for o in others)
    print(f"{name}: {len(b['flag_step'])} states, nearest state in any other bank at distance {nearest:.3f}")
    assert nearest > 1e-4, f"{name} shares a state with another bank -- stop"
    shutil.copy(BANKS[name], OUT)
""")
md("""
## 4. Replay every state under the three policies

Four banks, plus the seed-6 bank a second time. The script stops if the limp
arm's motors produce any torque.
""")
code("""
RUNS = [(name, path, f"{OUT}/compare_{name}.json") for name, path in BANKS.items()]
RUNS.append(("flag_seed6_repeat", BANKS["flag_seed6"], f"{OUT}/compare_flag_seed6_repeat.json"))
for name, bank, out in RUNS:
    print(f"\\n==================== {name} ====================")
    !python scripts/go1_protective_compare.py --protective-checkpoint {PROTECTIVE} --walker-checkpoint {WALKER} --state-bank-file {bank} --output {out}
    assert _exit_code == 0, f"compare failed for {name} (exit {_exit_code}) -- scroll up"
""")
md("""
## 5. Statistics: the confirmatory test (pooled flag banks), then the two robustness checks

Displays and downloads the four box plots for the pooled falls. All tables are
written to `recovery_stats.md` in each stats folder and are in the final zip.
""")
code("""
pooled = " ".join(f"{OUT}/compare_flag_seed{s}.npz" for s in FLAG_SEEDS)
!python scripts/go1_recovery_stats.py --compare-npz {pooled} --out-dir {OUT}/stats_confirmatory_flag_pooled --quiet
assert _exit_code == 0, f"stats failed (exit {_exit_code}) -- scroll up"
from IPython.display import Image, display
BOXPLOTS = f"{OUT}/stats_confirmatory_flag_pooled/fig_boxplots.png"
display(Image(BOXPLOTS))
files.download(BOXPLOTS)
""")
code("""
print("=========== robustness 1: late hand-off (mid-fall states, seed 9) ===========")
!python scripts/go1_recovery_stats.py --compare-npz {OUT}/compare_fall_seed{FALL_SEED}.npz --out-dir {OUT}/stats_robustness_midfall --quiet
assert _exit_code == 0, f"stats failed (exit {_exit_code}) -- scroll up"
print("=========== robustness 2: the seed-6 bank replayed a second time ===========")
!python scripts/go1_recovery_stats.py --compare-npz {OUT}/compare_flag_seed6_repeat.npz --out-dir {OUT}/stats_repeat_seed6 --no-figures --quiet
assert _exit_code == 0, f"stats failed (exit {_exit_code}) -- scroll up"
""")
md("## 6. Download everything as one zip (also copied to Drive)")
code("""
bundle = shutil.make_archive("/content/recovery_eval_results", "zip", OUT)
with zipfile.ZipFile(bundle) as zf:
    assert zf.testzip() is None
    print(len(zf.namelist()), "files,", f"{os.path.getsize(bundle)/1e6:.1f} MB")
print("saved to Drive:", shutil.copy(bundle, SAVE_DIR))
files.download(bundle)
""")
code("files.download(bundle)  # retry, only if the download above did not arrive")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(Path(__file__).resolve().parents[2] / "notebooks" / "go1_recovery_eval.ipynb", "w"), indent=1)
print(len(cells), "cells")
