"""Builds notebooks/g1_end_to_end.ipynb: the whole G1 humanoid pipeline in one
notebook, the counterpart of notebooks/go1_end_to_end.ipynb. One walker, then
everything after it once per seed in SEEDS (three, not the Go1's five), with
the final G1 SmallFall recipe: hull-growth reward, 600 iterations, settle
lock at run time (DEVIATIONS.md 2026-11-14).

    python3 tools/notebook_builders/build_g1_end_to_end_nb.py
"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# G1 humanoid fall pipeline, end to end, repeated over three seeds

The Go1 end-to-end notebook, for the G1: one walker, then everything after it
once per seed in `SEEDS`: push dataset and fall predictor -> fall banks ->
SmallFall policies -> live hand-off -> comparison against the walker and limp
-> statistics. Every step is a `scripts/*.py` call.

Each seed trains **four new SmallFall policies**, one per fall-size reward
(hull growth, centroid travel, hull area, floor touched: the reward
ablation's top four), plus **the original SmallFall** (ground-covered
reward). Hull growth is the recipe's policy: it is the one handed control
live (D), tested (E) and filmed (V). All seven (four new, original, walker,
limp) are replayed on the same evaluation falls (F), compared with each
other in box plots and paired tests.

| Part | What it does | Runs | Rough A100 time |
|---|---|---|---|
| A | inspect the task; the walker (reused from Drive by default, or trained: 10,000 iterations) | once | ~2 min (reuse) / many hours (train) |
| B | 200 N push dataset, predictor, its confidence intervals | per seed | ~15 min per seed |
| C | reward weights (reused or calibrated, once); flag-moment training bank; four new SmallFall policies (600 iterations each) and the original SmallFall (250) | per seed | ~2.3 h per seed |
| D | live hand-off check (1,024 robots, two arms), settle lock on | per seed | ~8 min per seed |
| E | fresh test bank, three-way replay, verdict | per seed | ~5 min per seed |
| F | three more fresh banks per seed (about 1,500 falls), all seven policies replayed on them; that seed's significance tests; then mean and SD across seeds and the box plots | per seed, then once | ~35 min per seed |
| V | videos (walker, predictor overlay, each of the five SmallFall policies captioned by name, three-panel demo) | first seed only | ~40 min |

With three seeds and the walker reused: about 10-11 hours.

**The new SmallFall policies** (fixed by rounds 2 and 3, `DEVIATIONS.md`): the
G1 SmallFall reward with one fall-size measure as its size term; 600
iterations, checkpoint `model_600.pt`; trained without the settle lock, run
with it (joints hold their pose 3 s after the fall starts or the hand-off, or
once the robot is low and still). **The original SmallFall:** the
ground-covered term, 250 iterations (the checkpoint chosen in the first G1
run), no settle lock, as it was.

**Seeds.** Seed `s` trains its own predictor and SmallFall and uses its own
fall banks (bank seed = 1000 x (s + 1) + role: 5 trains, 4 tests, 6-8
evaluate), so nothing is shared between seeds except the walker and the
reward weights. The `+ 1` keeps every bank away from those used while the
recipe was developed (seeds 3-8).

**If the runtime disconnects:** run all again with the same `RUN_NAME`.
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
from IPython.display import Image, Video, display

REPO_URL = "https://github.com/garduvel/smallfall.git"
REPO_REF = "main"
if not os.path.exists("/content/smallfall/.git"):
    !git clone -q {REPO_URL} /content/smallfall
%cd /content/smallfall
!git fetch -q origin && git checkout -q {REPO_REF} && (git pull -q --ff-only origin {REPO_REF} 2>/dev/null || true)
COMMIT = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
print("code at commit", COMMIT)
for d in ("checkpoints/protective", "outputs/videos", "outputs/stats", "logs/rsl_rl"):
    os.makedirs(d, exist_ok=True)
""")

md("""
## 0c. Settings, Google Drive, package versions
`REUSE_WALKER_RUN` / `REUSE_WEIGHTS_FROM`: earlier runs on Drive to take the
walker and the reward weights from. Set either to `None` to train or
calibrate them here instead.
""")
code("""
RUN_NAME = "e2e1"                 # results folder on Drive; use a new name for a fresh run
SEEDS = (0, 1, 2)                 # one full back half per seed
REUSE_WALKER_RUN = "walker2"      # g1_walker/<run> on Drive; None = train the walker here (many hours)
REUSE_WEIGHTS_FROM = {"protective": "g1_smallfall/smallfall1", "fall_size": "g1_reward_ablation/rewardablation1"}  # None = calibrate
REWARDS = {                       # new SmallFall policies: label -> fall-size measure (first = the recipe's policy)
    "hullgrow": "hull_growth_m2",
    "centroid": "centroid_travel_end_m",
    "hull": "swept_hull_area_m2",
    "touched": "touched_area_m2",
}
LABELS = {"hullgrow": "Hull growth", "centroid": "Centroid travel", "hull": "Hull area", "touched": "Floor touched"}
SIZE_MEASURE = REWARDS["hullgrow"]
PROTECTIVE_ITERATION = 600        # fixed for every seed and every new policy
SF_ITERATION = 250                # the original SmallFall: checkpoint chosen in the first G1 run (smallfall1)
PROTECTIVE_TRAIN_ITERATIONS = PROTECTIVE_ITERATION + 1   # iterations are numbered from 0
WALKER_ITERATIONS = 10_000
VIDEO_SEED = SEEDS[0]
TRAIN_ROLE, TEST_ROLE, EVAL_ROLES = 5, 4, (6, 7, 8)

DRIVE = "/content/drive/MyDrive/cartpole_term_runs"
WALKER = "checkpoints/g1_walker_model_9999.pt"
WEIGHTS = "outputs/g1_protective_weights.json"
SIZE_WEIGHTS = "outputs/g1_fall_size_weights.json"

# Per-seed file names. Nothing below is shared between seeds.
def predictor(s): return f"outputs/g1_wildpush_predictor_seed{s}.pt"
def metrics(s): return f"outputs/g1_wildpush_metrics_seed{s}.json"
def dataset(s): return f"outputs/g1_wildpush_frozen_seed{s}.npz"
def predictor_stats(s): return f"outputs/stats/predictor_seed{s}/predictor_stats.json"
def bank_seed(s, role): return 1000 * (s + 1) + role
def flag_bank(s, role): return f"outputs/g1_flag_state_bank_seed{bank_seed(s, role)}.npz"
def policy(s, r): return f"checkpoints/protective/g1_seed{s}_{r}_model_{PROTECTIVE_ITERATION}.pt"
def policy_history(s, r): return f"outputs/g1_protective_eval_history_seed{s}.json" if r == "hullgrow" else f"outputs/g1_{r}_eval_history_seed{s}.json"
def protective(s): return policy(s, "hullgrow")   # the recipe's policy
def original_sf(s): return f"checkpoints/protective/g1_seed{s}_original_model_{SF_ITERATION}.pt"
def recovery_stats(s): return f"outputs/stats/recovery_seed{s}/recovery_stats.json"
def handoff(s): return f"outputs/g1_handoff_eval_seed{s}.json"
def test_compare(s): return f"outputs/g1_protective_compare_test_seed{s}.json"
def eval_compare(s, role): return f"outputs/g1_policies_eval_seed{s}_flag{role}.json"   # all seven policies
def npz(path): return path.replace(".json", ".npz")

from google.colab import drive, files
drive.mount("/content/drive")
SAVE_DIR = f"{DRIVE}/g1_end_to_end/{RUN_NAME}"
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

versions = {"gpu": torch.cuda.get_device_name(0), "commit": COMMIT, "seeds": list(SEEDS),
            "reuse_walker_run": REUSE_WALKER_RUN, "reuse_weights_from": REUSE_WEIGHTS_FROM}
for pkg in ("mjlab", "mujoco-warp", "rsl-rl-lib", "torch", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_g1_end_to_end.json", "w"), indent=2)
save("outputs/versions_g1_end_to_end.json")
""")

md("# Part A: walker (once)")
md("## A1. Inspect the task")
code("""
!python scripts/inspect_robot_task.py --robot g1
assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up for the traceback"
""")
md("""
## A2. The walker
In order: already here; saved in this run's Drive folder; copied from
`REUSE_WALKER_RUN`; otherwise trained here for 10,000 iterations. Training
logs go straight to Drive, so after a disconnect this cell resumes from the
newest readable checkpoint.
""")
code("""
EXPERIMENT = "g1_velocity_term"
if restore(WALKER):
    print("using the walker in this run:", WALKER)
elif REUSE_WALKER_RUN:
    src = f"{DRIVE}/g1_walker/{REUSE_WALKER_RUN}/checkpoints/g1_walker_{REUSE_WALKER_RUN}_model_9999.pt"
    assert os.path.exists(src), f"missing {src} -- check REUSE_WALKER_RUN, or set it to None to train"
    shutil.copy(src, WALKER)
    save(WALKER)
    print("reused walker from", src)
else:
    drive_logs = on_drive(f"logs/{EXPERIMENT}")
    os.makedirs(drive_logs, exist_ok=True)
    local_logs = f"logs/rsl_rl/{EXPERIMENT}"
    if os.path.islink(local_logs):
        os.unlink(local_logs)
    shutil.rmtree(local_logs, ignore_errors=True)
    os.symlink(drive_logs, local_logs)

    def readable(path):
        try:
            torch.load(path, map_location="cpu", weights_only=False)
            return True
        except Exception as e:
            print(f"unreadable checkpoint {path}: {type(e).__name__}")
            return False

    ckpts = sorted(glob.glob(f"{local_logs}/*_seed0/model_*.pt"), key=lambda p: int(re.search(r"model_(\\d+)\\.pt$", p).group(1)), reverse=True)
    resume = next((c for c in ckpts if readable(c)), None)
    TRAIN = f"--robot g1 --seeds 0 --eval.interval 500 --max-iterations {WALKER_ITERATIONS}"
    if resume:
        print("TRAINING THE WALKER: resuming from", resume)
        !python scripts/robot_train.py {TRAIN} --resume-from {resume}
    else:
        print("TRAINING THE WALKER: 10,000 iterations from scratch, many hours")
        !python scripts/robot_train.py {TRAIN}
    assert _exit_code == 0, f"walker training failed (exit {_exit_code}) -- run this cell again to resume"
    final = sorted(glob.glob(f"{local_logs}/*_seed0/model_{WALKER_ITERATIONS - 1}.pt"))[-1]
    shutil.copy(final, WALKER)
    save(WALKER)
!ls -la {WALKER}
""")

md("# Part B: push dataset and fall predictor (per seed)")
md("""
Collects labelled steps under the 200 N push (the G1's push, matched to the
Go1's fall rate) with collection seed `s`, trains predictor `s`, and writes
its confidence intervals. The dataset stays on this runtime only.
""")
code("""
for s in SEEDS:
    if restore(predictor(s), metrics(s), predictor_stats(s)):
        print(f"seed {s}: predictor already done, skipping")
        continue
    print(f"\\n==================== seed {s}: predictor ====================")
    !python scripts/go1_wild_push_predictor.py --robot g1 --checkpoint-file {WALKER} --seed {s}
    assert _exit_code == 0, f"seed {s}: predictor script failed (exit {_exit_code}) -- scroll up"
    p, m, d, sd = predictor(s), metrics(s), dataset(s), os.path.dirname(predictor_stats(s))
    !python scripts/go1_predictor_stats.py --dataset {d} --predictor-file {p} --metrics-file {m} --seed {s} --live-json none --live-npz none --out-dir {sd} --quiet
    assert _exit_code == 0, f"seed {s}: predictor stats failed (exit {_exit_code}) -- scroll up"
    save(p, m, predictor_stats(s), predictor_stats(s).replace(".json", ".md"))
    print(f"seed {s}: threshold {json.load(open(m))['mlp']['threshold']:.4f}, test AUPRC {json.load(open(m))['mlp']['auprc']:.4f}")
""")

md("# Part C: SmallFall (per seed)")
md("## C1. Training bank: the robot's state at the moment predictor `s` fires")
code("""
for s in SEEDS:
    bank = flag_bank(s, TRAIN_ROLE)
    if restore(bank):
        print(f"seed {s}: training bank already done, skipping")
        continue
    p, m, bs = predictor(s), metrics(s), bank_seed(s, TRAIN_ROLE)
    !python scripts/go1_collect_flagged_states.py --robot g1 --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
    assert _exit_code == 0, f"seed {s}: bank collection failed (exit {_exit_code}) -- scroll up"
    save(bank)
""")

md("""
## C2. Reward weights (once, shared by every seed)
Reused from the runs that developed the recipe, or, with
`REUSE_WEIGHTS_FROM = None`, calibrated here on the first seed's training
bank: 11 iterations with the Go1 weights re-size four terms
(`protective/calibration.py`), then 11 more size the fall-size weight
(`scripts/robot_calibrate_fall_size.py`).
""")
code("""
if restore(WEIGHTS, SIZE_WEIGHTS):
    print("reward weights already in this run")
elif REUSE_WEIGHTS_FROM:
    for local, run in ((WEIGHTS, REUSE_WEIGHTS_FROM["protective"]), (SIZE_WEIGHTS, REUSE_WEIGHTS_FROM["fall_size"])):
        src = f"{DRIVE}/{run}/{local}"
        assert os.path.exists(src), f"missing {src} -- check REUSE_WEIGHTS_FROM, or set it to None to calibrate"
        shutil.copy(src, local)
    save(WEIGHTS, SIZE_WEIGHTS)
else:
    s0, bank = SEEDS[0], flag_bank(SEEDS[0], TRAIN_ROLE)
    !python scripts/go1_train_protective.py --robot g1 --seeds {s0} --state-bank-file {bank} --max-iterations 11 --eval.interval 10 --run-tag calibration
    assert _exit_code == 0, f"calibration training failed (exit {_exit_code}) -- scroll up"
    run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_calibration_seed{s0}"))[-1]
    !python scripts/robot_calibrate_protective.py --run-dir {run} --output {WEIGHTS}
    assert _exit_code == 0, f"calibration failed (exit {_exit_code}) -- scroll up"
    !python scripts/go1_train_protective.py --robot g1 --seeds {s0} --state-bank-file {bank} --weights-json {WEIGHTS} --log-fall-sizes --run-tag sizecal --max-iterations 11 --eval.interval 10
    assert _exit_code == 0, f"sizing run failed (exit {_exit_code}) -- scroll up"
    run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_sizecal_seed{s0}"))[-1]
    !python scripts/robot_calibrate_fall_size.py --run-dir {run} --weights-json {WEIGHTS} --output {SIZE_WEIGHTS}
    assert _exit_code == 0, f"sizing failed (exit {_exit_code}) -- scroll up"
    save(WEIGHTS, SIZE_WEIGHTS)
print("hull-growth weight:", json.load(open(SIZE_WEIGHTS))["weights"][SIZE_MEASURE])
""")

md("## C3. Train the four new SmallFall policies, each seed to iteration 600 (settle lock off in training)")
code("""
for s in SEEDS:
    for r, measure in REWARDS.items():
        if restore(policy(s, r), policy_history(s, r)):
            print(f"seed {s}, {r}: already trained, skipping")
            continue
        print(f"\\n==================== seed {s}: SmallFall, {LABELS[r]} reward ====================")
        t0 = time.time()
        bank = flag_bank(s, TRAIN_ROLE)
        !python scripts/go1_train_protective.py --robot g1 --seeds {s} --state-bank-file {bank} --weights-json {WEIGHTS} --fall-size-weights-json {SIZE_WEIGHTS} --fall-size-measure {measure} --run-tag e2e_{r} --max-iterations {PROTECTIVE_TRAIN_ITERATIONS}
        assert _exit_code == 0, f"seed {s}, {r}: training failed (exit {_exit_code}) -- scroll up"
        run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_e2e_{r}_seed{s}"))[-1]
        shutil.copy(f"{run}/model_{PROTECTIVE_ITERATION}.pt", policy(s, r))
        shutil.copy(f"{run}/eval_history.json", policy_history(s, r))
        save(policy(s, r), policy_history(s, r))
        print(f"seed {s}, {r}: trained in {(time.time() - t0) / 60:.0f} min")
""")

md("## C4. Train the original SmallFall (ground-covered reward), each seed to iteration 250")
code("""
for s in SEEDS:
    history = f"outputs/g1_original_eval_history_seed{s}.json"
    if restore(original_sf(s), history):
        print(f"seed {s}: original SmallFall already trained, skipping")
        continue
    print(f"\\n==================== seed {s}: original SmallFall ====================")
    bank = flag_bank(s, TRAIN_ROLE)
    !python scripts/go1_train_protective.py --robot g1 --seeds {s} --state-bank-file {bank} --weights-json {WEIGHTS} --run-tag e2e_original --max-iterations {SF_ITERATION + 1}
    assert _exit_code == 0, f"seed {s}: original SmallFall training failed (exit {_exit_code}) -- scroll up"
    run = sorted(glob.glob(f"logs/rsl_rl/g1_protective_fall/*_e2e_original_seed{s}"))[-1]
    shutil.copy(f"{run}/model_{SF_ITERATION}.pt", original_sf(s))
    shutil.copy(f"{run}/eval_history.json", history)
    save(original_sf(s), history)
""")

md("# Part D: live hand-off check (per seed)")
md("""
Walker under the push, predictor `s` scoring every step, permanent switch to
SmallFall `s`, settle lock from the switch. Checks that nothing changes
weights or switches back, and records recall, precision and warning time.
""")
code("""
for s in SEEDS:
    if restore(handoff(s), npz(handoff(s))):
        print(f"seed {s}: hand-off check already done, skipping")
        continue
    print(f"\\n==================== seed {s}: live hand-off ====================")
    pr, p, m, bank = protective(s), predictor(s), metrics(s), flag_bank(s, TRAIN_ROLE)
    !python scripts/go1_handoff_eval.py --robot g1 --settle-lock --protective-checkpoint {pr} --walker-checkpoint {WALKER} --predictor-file {p} --metrics-file {m} --state-bank-file {bank} --seed {s} --output {handoff(s)}
    assert _exit_code == 0, f"seed {s}: hand-off eval failed (exit {_exit_code}) -- scroll up"
    save(handoff(s), npz(handoff(s)))
""")

md("# Part E: test on a fresh bank (per seed)")
md("## E1. Fresh flag-moment bank, replayed under SmallFall (lock on), walker, and limp")
code("""
for s in SEEDS:
    out = test_compare(s)
    if restore(out, npz(out)):
        print(f"seed {s}: test already done, skipping")
        continue
    print(f"\\n==================== seed {s}: test ====================")
    pr, p, m, bs, bank = protective(s), predictor(s), metrics(s), bank_seed(s, TEST_ROLE), flag_bank(s, TEST_ROLE)
    !python scripts/go1_collect_flagged_states.py --robot g1 --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
    assert _exit_code == 0, f"seed {s}: test bank failed (exit {_exit_code}) -- scroll up"
    !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {pr} --walker-checkpoint {WALKER} --state-bank-file {bank} --seed {bs} --output {out} --locked-arms protective
    assert _exit_code == 0, f"seed {s}: compare failed (exit {_exit_code}) -- scroll up"
    save(out, npz(out))
""")

md("""
## E2. Verdict per seed (space first)
PASS = less ground covered than both the walker and limp, comes to rest
(>= 99%), trunk/pelvis force no higher than the walker's, and any-body force
below limp's (the G1 rounds' safety rule).
""")
code("""
g = "envelope_growth_m"
print(f"{'seed':>4s}{'ground: SmallFall':>19s}{'walker':>9s}{'limp':>8s}{'trunk N':>9s}{'any-body N':>12s}{'limp any N':>12s}{'settled':>9s}   verdict")
for s in SEEDS:
    q = json.load(open(test_compare(s)))["summaries"]
    med = lambda arm, k: q[arm][k]["median"]
    settled = q["protective"]["settled"]["mean"]
    ok = (med("protective", g) < med("walker", g) and med("protective", g) < med("limp", g) and settled >= 0.99
          and med("protective", "peak_force_critical_n") <= med("walker", "peak_force_critical_n")
          and med("protective", "peak_force_any_n") < med("limp", "peak_force_any_n"))
    print(f"{s:4d}{med('protective', g):19.3f}{med('walker', g):9.3f}{med('limp', g):8.3f}"
          f"{med('protective', 'peak_force_critical_n'):9.0f}{med('protective', 'peak_force_any_n'):12.0f}"
          f"{med('limp', 'peak_force_any_n'):12.0f}{settled:9.3f}   {'PASS' if ok else 'FAIL'}")
""")

md("# Part F: statistical evaluation")
md("""
## F1. Per seed: three more fresh banks (about 1,500 paired falls), all seven policies on every fall
`protective` = hull growth; `centroid`, `hull`, `touched` = the other new
policies (all four with the settle lock); `original` = the original
SmallFall (no lock); walker; limp. Then that seed's own significance tests
for the recipe's policy against walker and limp.
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
        !python scripts/go1_collect_flagged_states.py --robot g1 --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
        assert _exit_code == 0, f"seed {s}: evaluation bank {role} failed (exit {_exit_code}) -- scroll up"
        extra = " ".join(f"--extra-policy {r}={policy(s, r)}" for r in REWARDS if r != "hullgrow") + f" --extra-policy original={original_sf(s)}"
        locked = ",".join(["protective"] + [r for r in REWARDS if r != "hullgrow"])
        !python scripts/go1_protective_compare.py --robot g1 --protective-checkpoint {pr} --walker-checkpoint {WALKER} --state-bank-file {bank} --seed {bs} --output {out} {extra} --locked-arms {locked}
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
## F2. All seeds together: the box plots (displayed, saved to Drive, downloaded)
Only now are the seeds combined: mean and SD across seeds of every statistic
each seed saved. Tables: `outputs/stats/multiseed/multiseed_stats.md`.
""")
code("""
# subprocess, not "!": the templates contain a literal {seed} the script fills in.
stats_cmd = ["python", "scripts/go1_multiseed_stats.py", "--seeds", *[str(s) for s in SEEDS],
             "--recovery-stats-template", "outputs/stats/recovery_seed{seed}/recovery_stats.json",
             "--compare-npz-template", "outputs/g1_policies_eval_seed{seed}_flag*.npz",
             "--predictor-stats-template", "outputs/stats/predictor_seed{seed}/predictor_stats.json",
             "--handoff-json-template", "outputs/g1_handoff_eval_seed{seed}.json",
             "--robot-label", "G1", "--out-dir", "outputs/stats/multiseed", "--quiet"]
done = subprocess.run(stats_cmd, capture_output=True, text=True)
print(done.stdout[-3000:], done.stderr[-6000:])
assert done.returncode == 0, f"multi-seed stats failed (exit {done.returncode}) -- see the output above"
BOXPLOTS = "outputs/stats/multiseed/fig_boxplots.png"
save("outputs/stats/multiseed/multiseed_stats.json", "outputs/stats/multiseed/multiseed_stats.md", BOXPLOTS)
display(Image(BOXPLOTS))
files.download(BOXPLOTS)
""")

md("""
## F3. All seven policies against each other (displayed, saved to Drive, downloaded)
Boxes: every fall from every seed. Each seed's tests are paired on its own
falls (Wilcoxon, Holm within the seed): every new policy against the
original SmallFall and against limp, and the four new policies against each
other. Tables: `outputs/stats/policies/policy_stats.md`.
""")
code("""
arm_args = [f"--arm=protective={LABELS['hullgrow']}"] + [f"--arm={r}={LABELS[r]}" for r in REWARDS if r != "hullgrow"] \\
    + ["--arm=original=Original SmallFall", "--arm=walker=Walker", "--arm=limp=Limp"]
cmd = ["python", "scripts/robot_policy_boxplots.py", "--runs", "outputs/g1_policies_eval_seed{seed}_flag*.npz",
       "--seeds", *[str(s) for s in SEEDS], *arm_args,
       "--new", "protective", *[r for r in REWARDS if r != "hullgrow"], "--reference", "original",
       "--title", "G1: four new SmallFall policies (settle lock on) vs the original SmallFall, walker and limp",
       "--out-dir", "outputs/stats/policies"]
done = subprocess.run(cmd, capture_output=True, text=True)
print(done.stdout[-3000:], done.stderr[-6000:])
assert done.returncode == 0, f"policy box plots failed (exit {done.returncode}) -- see the output above"
POLICY_PLOTS = "outputs/stats/policies/fig_policy_boxplots.png"
save("outputs/stats/policies/policy_stats.json", "outputs/stats/policies/policy_stats.md", POLICY_PLOTS)
display(Image(POLICY_PLOTS))
files.download(POLICY_PLOTS)
""")

md("# Part V: videos (first seed only)")
md("## V1. The walker")
code("""
V1 = f"outputs/g1_watch_seed{VIDEO_SEED}_ep3.mp4"
if not restore(V1):
    !python scripts/robot_watch_episodes.py --robot g1 --checkpoint-file {WALKER} --episodes 3 --camera-distance 3.5 --seed {VIDEO_SEED}
    assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up"
    save(V1)
display(Video(V1, embed=True, width=640))
""")
md("## V2. Predictor overlay: five falls, the last 4 s before each")
code("""
vp, vm = predictor(VIDEO_SEED), metrics(VIDEO_SEED)
V2 = f"outputs/g1_wildpush_overlay_impulse200.0N_seed{VIDEO_SEED}_take0.mp4"   # 200 N = the G1's push
if not restore(V2):
    !python scripts/go1_wildpush_video.py --robot g1 --checkpoint-file {WALKER} --predictor-weights {vp} --predictor-metrics {vm} --mechanism impulse --min-failures 5 --seed {VIDEO_SEED} --camera-distance 3.5
    assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up"
    save(V2)
display(Video(V2, embed=True, width=640))
""")
md("""
## V3. Each SmallFall policy on the same eight falls, one captioned video per policy
The caption names the policy on screen. The four new policies run with the
settle lock, the original SmallFall without it (as evaluated in F).
""")
code("""
vbank = flag_bank(VIDEO_SEED, TEST_ROLE)
V3 = {}
for r, path, lock in [(r, policy(VIDEO_SEED, r), True) for r in REWARDS] + [("original", original_sf(VIDEO_SEED), False)]:
    label = "SmallFall: " + (f"{LABELS[r]} reward" if r in REWARDS else "original (ground-covered reward)")
    V3[r] = f"outputs/g1_protective_watch_{r}_seed{VIDEO_SEED}_ep8.mp4"
    if not restore(V3[r]):
        lock_flag = "--settle-lock" if lock else ""
        !python scripts/go1_watch_protective_episodes.py --robot g1 {lock_flag} --checkpoint-file {path} --state-bank-file {vbank} --episodes 8 --camera-distance 3.5 --seed {VIDEO_SEED} --policy-label "{label}" --policy-tag {r}
        assert _exit_code == 0, f"{r}: script failed (exit {_exit_code}) -- scroll up"
        save(V3[r])
for r, path in V3.items():
    print(r)
    display(Video(path, embed=True, width=640))
""")
md("## V4. Demo: the same fall three ways (walker alone | limp | hand-off to the hull-growth SmallFall with the settle lock)")
code("""
vpr = protective(VIDEO_SEED)
DEMO = f"outputs/videos/g1_handoff_demo_hullgrow_seed{VIDEO_SEED}.mp4"
if not restore(DEMO, DEMO.replace(".mp4", ".json")):
    !python scripts/go1_handoff_demo_video.py --robot g1 --settle-lock --protective-checkpoint {vpr} --walker-checkpoint {WALKER} --predictor-file {vp} --metrics-file {vm} --state-bank-file {vbank} --camera-distance 3.5 --seed {VIDEO_SEED} --policy-label "SmallFall ({LABELS['hullgrow']})" --policy-tag hullgrow
    assert _exit_code == 0, f"video failed (exit {_exit_code}) -- scroll up for the traceback"
    save(DEMO, DEMO.replace(".mp4", ".json"))
display(Video(DEMO, embed=True, width=1280))
""")

md("## G. Download everything saved to Drive as one zip")
code("""
bundle = shutil.make_archive(f"/content/g1_end_to_end_{RUN_NAME}", "zip", SAVE_DIR)
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
out = Path(__file__).resolve().parents[2] / "notebooks" / "g1_end_to_end.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
