import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Hand-off vs walker vs limp (motors off) (Go1)

Fresh runtime, A100 (L4 also fine), ~10 min. No training. Replays the same 500
flag moments as the reported round-8 table (flag bank seed 4) under three
policies: the protective policy (round 8b, iteration 200), the walker
continuing, and **limp** (motors off, zero torque at every joint). Pre-registered in DEVIATIONS.md ("Limp baseline").

**One upload:** in step 2 select **all 5 files** in
`/home/grgott537/cartpole-term/colab_uploads/limp_baseline/` (Ctrl+A).
""")
md("## 1. Install (pinned), confirm GPU")
code("""
!pip install -q mjlab==1.6.0 rsl-rl-lib==5.4.2 scikit-learn "imageio[ffmpeg]" pillow
import torch
assert torch.cuda.is_available(), "No GPU visible -- check Runtime > Change runtime type"
print(torch.cuda.get_device_name(0))
""")
md("## 2. Upload everything (select all 5 files in `colab_uploads/limp_baseline/`)")
code("""
from google.colab import files
import json, os, re, shutil, subprocess, zipfile

uploaded = files.upload()
names = {re.sub(r" \\(\\d+\\)(?=\\.\\w+$)", "", n): n for n in uploaded}  # strip Colab's " (1)"
DEST = {
    "model_9999.pt": "checkpoints/model_9999.pt",
    "round8b_model_200.pt": "checkpoints/protective/round8b_model_200.pt",
    "go1_fall_state_bank_seed0.npz": "outputs/go1_fall_state_bank_seed0.npz",
    "go1_flag_state_bank_seed4.npz": "outputs/go1_flag_state_bank_seed4.npz",
}
missing = ({"cartpole_term_code.zip"} | set(DEST)) - set(names)
assert not missing, f"missing uploads: {sorted(missing)} -- re-run this cell and select all 5 files"
ROOT = "/content/cartpole-term"
os.makedirs(ROOT, exist_ok=True)
with zipfile.ZipFile(names["cartpole_term_code.zip"]) as zf:
    zf.extractall(ROOT)
for d in ("checkpoints/protective", "outputs"):
    os.makedirs(f"{ROOT}/{d}", exist_ok=True)
for canon, rel in DEST.items():
    shutil.move(names[canon], f"{ROOT}/{rel}")
%cd /content/cartpole-term
versions = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}
for pkg in ("mjlab", "mujoco", "mujoco-warp", "rsl-rl-lib", "torch", "warp-lang"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_limp_baseline.json", "w"), indent=2)
print(json.dumps(versions, indent=2))
""")
md("""
## 3. Replay the 500 flag moments under the three policies

The script stops with an error if the limp arm's motors produce any torque.
""")
code("""
OUT = "outputs/go1_protective_compare_limp_flagstate_seed4.json"
!python scripts/go1_protective_compare.py --protective-checkpoint checkpoints/protective/round8b_model_200.pt --walker-checkpoint checkpoints/model_9999.pt --state-bank-file outputs/go1_flag_state_bank_seed4.npz --output {OUT}
assert _exit_code == 0, f"compare failed (exit {_exit_code}) -- scroll up for the traceback"
""")
md("""
## 4. Verdict (space first), against each baseline

**Paste this cell's output back.**
""")
code("""
s = json.load(open(OUT))["summaries"]
ROWS = ("envelope_growth_m", "peak_force_critical_n", "peak_force_any_n", "peak_torque_nm", "steps")
ARMS = ("protective", "walker", "limp")
print(f"{'median':24s}" + "".join(f"{a:>13s}" for a in ARMS))
for m in ROWS:
    print(f"{m:24s}" + "".join(f"{s[a][m]['median']:13.3f}" for a in ARMS))
print(f"{'settled rate':24s}" + "".join(f"{s[a]['settled']['mean']:13.3f}" for a in ARMS))
g, settled = "envelope_growth_m", s["protective"]["settled"]["mean"]
for base in ARMS[1:]:
    less = 100 * (1 - s["protective"][g]["median"] / s[base][g]["median"])
    print(f"ground covered vs {base}: {less:.0f}% less | SPACE PASS vs {base}:",
          s["protective"][g]["median"] < s[base][g]["median"] and settled >= 0.99)
for m in ("peak_force_critical_n", "peak_force_any_n"):
    print(f"{m} below limp:", s["protective"][m]["median"] < s["limp"][m]["median"])
""")
md("## 5. Download the results (one zip)")
code("""
bundle_dir = "/content/limp_baseline_results"
os.makedirs(bundle_dir, exist_ok=True)
for f in (OUT, OUT.replace(".json", ".npz"), "outputs/versions_limp_baseline.json"):
    shutil.copy(f, bundle_dir)
bundle = shutil.make_archive(bundle_dir, "zip", bundle_dir)
with zipfile.ZipFile(bundle) as zf:
    assert zf.testzip() is None and len(zf.namelist()) == 3, zf.namelist()
files.download(bundle)
""")
code("files.download(bundle)  # retry, only if the download above did not arrive")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(Path(__file__).resolve().parents[2] / "notebooks" / "go1_limp_baseline.ipynb", "w"), indent=1)
print(len(cells), "cells")
