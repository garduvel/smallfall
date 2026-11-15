"""Builds notebooks/go1_end_to_end.ipynb: the whole Go1 pipeline in one
notebook, laid out exactly like notebooks/g1_end_to_end.ipynb
(build_g1_end_to_end_nb.py): one walker, then everything after it once per
seed in SEEDS (three), with the four new SmallFall policies (the G1 reward
ablation's top four fall-size rewards) next to the original Go1 SmallFall
(round-8 reward, iteration 200), all replayed on the same falls and compared
in one set of box plots (DEVIATIONS.md 2026-11-15).

The previous, hand-edited version of the notebook (five seeds, original
SmallFall only; the multiseed1 run) is kept at
notebooks/archive/go1_end_to_end_multiseed1.ipynb.

    python3 tools/notebook_builders/build_go1_end_to_end_nb.py
"""

import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# Go1 fall pipeline, end to end, repeated over three seeds

One walker, then everything after it once per seed in `SEEDS`: push dataset
and fall predictor -> fall banks -> SmallFall policies -> live hand-off ->
comparison against the walker and limp -> statistics. Every step is a
`scripts/*.py` call. Laid out like `g1_end_to_end.ipynb`, part for part.

Each seed trains **four new SmallFall policies**, one per fall-size reward
(hull growth, centroid travel, hull area, floor touched: the G1 reward
ablation's top four), plus **the original SmallFall** (the Go1's round-8
reward, ground covered). Hull growth is the recipe's policy: it is the one
handed control live (D), tested (E) and filmed (V). All seven (four new,
original, walker, limp) are replayed on the same evaluation falls (F),
compared with each other in box plots and paired tests.

| Part | What it does | Runs | Rough A100 time |
|---|---|---|---|
| A | inspect the task; the walker (reused from Drive by default, or trained: 10,000 iterations) | once | ~2 min (reuse) / several hours (train) |
| B | 150 N push dataset, predictor, its confidence intervals | per seed | ~8 min per seed |
| C | fall-size reward weights (calibrated once); flag-moment training bank; four new SmallFall policies (600 iterations each) and the original SmallFall (200) | per seed | ~1.4 h per seed |
| D | live hand-off check (1,024 robots, two arms), settle lock on | per seed | ~4 min per seed |
| E | fresh test bank, three-way replay, verdict | per seed | ~4 min per seed |
| F | three more fresh banks per seed (about 1,500 falls), all seven policies replayed on them; that seed's significance tests; then mean and SD across seeds and the box plots | per seed, then once | ~25 min per seed |
| V | videos (walker, predictor overlay, SmallFall, three-panel demo) | first seed only | ~20 min |

Runs **seed 0 only by default** (about 2.5 hours with the walker reused). To
add seeds 1-2, set `SEEDS = (0, 1, 2)` in step 0c, keep the same `RUN_NAME`
and run all again: seed 0's finished steps are restored from Drive and
skipped, and the box plots are redrawn over all three (about 4 more hours).

**The new SmallFall policies:** the Go1 round-8 SmallFall reward with its
ground-covered term replaced by one fall-size measure, sized on this robot
(C2); 600 iterations, checkpoint `model_600.pt` (the G1 recipe's count, not
re-chosen for the Go1); trained without the settle lock, run with it (joints
hold their pose 1.8 s after the fall starts or the hand-off, or once the
robot is low and still; 1.8 s is the G1's 3 s scaled to the Go1's 3 s
episode cap). **The original SmallFall:** round-8 reward, 200 iterations (the
checkpoint chosen on seed 0's selection bank), no settle lock, as it was.

**Seeds.** Seed `s` trains its own predictor and SmallFall policies and uses
its own fall banks (bank seed = 1000 x s + role: 5 trains, 4 tests, 6-8
evaluate), so nothing is shared between seeds except the walker and the
reward weights. These are the same bank seeds as the five-seed `multiseed1`
run, so seeds 0-2 replay the same falls it did.

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
`REUSE_WALKER_RUN` / `REUSE_SIZE_WEIGHTS_FROM`: earlier runs on Drive to take
the walker and the fall-size weights from. Set either to `None` to train or
calibrate them here instead (the default for the weights: no Go1 run has
made them yet).
""")
code("""
RUN_NAME = "e2e1"                 # results folder on Drive; use a new name for a fresh run
SEEDS = (0,)                      # seed 0 first (~2.5 h); then (0, 1, 2) with the same RUN_NAME runs only seeds 1-2
REUSE_WALKER_RUN = "multiseed1"   # go1_end_to_end/<run> on Drive; None = train the walker here (several hours)
REUSE_SIZE_WEIGHTS_FROM = None    # e.g. "go1_end_to_end/e2e1"; None = calibrate in C2
REWARDS = {                       # new SmallFall policies: label -> fall-size measure (first = the recipe's policy)
    "hullgrow": "hull_growth_m2",
    "centroid": "centroid_travel_end_m",
    "hull": "swept_hull_area_m2",
    "touched": "touched_area_m2",
}
LABELS = {"hullgrow": "Hull growth", "centroid": "Centroid travel", "hull": "Hull area", "touched": "Floor touched"}
SIZE_MEASURE = REWARDS["hullgrow"]
PROTECTIVE_ITERATION = 600        # fixed for every seed and every new policy (the G1 recipe's count)
SF_ITERATION = 200                # the original SmallFall: chosen on seed 0's selection bank (DEVIATIONS.md)
PEAK_FORCE_WEIGHT = -0.02         # round-8 reward (env_cfg.PEAK_CONTACT_FORCE_WEIGHT)
ENVELOPE_WEIGHT = -30.0           # the Go1's ground-covered weight (env_cfg.FALL_ENVELOPE_WEIGHT), the sizing anchor
PROTECTIVE_TRAIN_ITERATIONS = PROTECTIVE_ITERATION + 1   # iterations are numbered from 0
WALKER_ITERATIONS = 10_000
IMPULSE_N = 150                   # the Go1's push
WATCH_SPEED_MPS = 2.0
VIDEO_SEED = SEEDS[0]
TRAIN_ROLE, TEST_ROLE, EVAL_ROLES = 5, 4, (6, 7, 8)

DRIVE = "/content/drive/MyDrive/cartpole_term_runs"
WALKER = "checkpoints/model_9999.pt"
SIZE_WEIGHTS = "outputs/go1_fall_size_weights.json"

# Per-seed file names. Nothing below is shared between seeds.
def predictor(s): return f"outputs/go1_wildpush_predictor_seed{s}.pt"
def metrics(s): return f"outputs/go1_wildpush_metrics_seed{s}.json"
def dataset(s): return f"outputs/go1_wildpush_frozen_seed{s}.npz"
def predictor_stats(s): return f"outputs/stats/predictor_seed{s}/predictor_stats.json"
def bank_seed(s, role): return 1000 * s + role
def flag_bank(s, role): return f"outputs/go1_flag_state_bank_seed{bank_seed(s, role)}.npz"
def policy(s, r): return f"checkpoints/protective/go1_seed{s}_{r}_model_{PROTECTIVE_ITERATION}.pt"
def policy_history(s, r): return f"outputs/go1_protective_eval_history_seed{s}.json" if r == "hullgrow" else f"outputs/go1_{r}_eval_history_seed{s}.json"
def protective(s): return policy(s, "hullgrow")   # the recipe's policy
def original_sf(s): return f"checkpoints/protective/go1_seed{s}_original_model_{SF_ITERATION}.pt"
def recovery_stats(s): return f"outputs/stats/recovery_seed{s}/recovery_stats.json"
def handoff(s): return f"outputs/go1_handoff_eval_seed{s}.json"
def test_compare(s): return f"outputs/go1_protective_compare_test_seed{s}.json"
def eval_compare(s, role): return f"outputs/go1_policies_eval_seed{s}_flag{role}.json"   # all seven policies
def npz(path): return path.replace(".json", ".npz")

from google.colab import drive, files
drive.mount("/content/drive")
SAVE_DIR = f"{DRIVE}/go1_end_to_end/{RUN_NAME}"
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
            "reuse_walker_run": REUSE_WALKER_RUN, "reuse_size_weights_from": REUSE_SIZE_WEIGHTS_FROM}
for pkg in ("mjlab", "mujoco-warp", "rsl-rl-lib", "torch", "numpy"):
    out = subprocess.run(["pip", "show", pkg], capture_output=True, text=True).stdout
    m = re.search(r"^Version: (.+)$", out, re.M)
    versions[pkg] = m.group(1) if m else None
json.dump(versions, open("outputs/versions_go1_end_to_end.json", "w"), indent=2)
save("outputs/versions_go1_end_to_end.json")
""")

md("# Part A: walker (once)")
md("## A1. Inspect the task")
code("""
!python scripts/inspect_robot_task.py --robot go1
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
EXPERIMENT = "go1_velocity_term"
if restore(WALKER):
    print("using the walker in this run:", WALKER)
elif REUSE_WALKER_RUN:
    src = f"{DRIVE}/go1_end_to_end/{REUSE_WALKER_RUN}/{WALKER}"
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
    TRAIN = f"--robot go1 --seeds 0 --eval.interval 1000 --max-iterations {WALKER_ITERATIONS}"
    if resume:
        print("TRAINING THE WALKER: resuming from", resume)
        !python scripts/robot_train.py {TRAIN} --resume-from {resume}
    else:
        print("TRAINING THE WALKER: 10,000 iterations from scratch, several hours")
        !python scripts/robot_train.py {TRAIN}
    assert _exit_code == 0, f"walker training failed (exit {_exit_code}) -- run this cell again to resume"
    final = sorted(glob.glob(f"{local_logs}/*_seed0/model_{WALKER_ITERATIONS - 1}.pt"))[-1]
    shutil.copy(final, WALKER)
    save(WALKER)
!ls -la {WALKER}
""")

md("# Part B: push dataset and fall predictor (per seed)")
md("""
Collects about a million labelled steps under the 150 N push with collection
seed `s`, trains predictor `s`, and writes its confidence intervals. The
dataset stays on this runtime only.
""")
code("""
for s in SEEDS:
    if restore(predictor(s), metrics(s), predictor_stats(s)):
        print(f"seed {s}: predictor already done, skipping")
        continue
    print(f"\\n==================== seed {s}: predictor ====================")
    !python scripts/go1_wild_push_predictor.py --robot go1 --checkpoint-file {WALKER} --mechanism impulse --impulse-force {IMPULSE_N} --seed {s}
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
    !python scripts/go1_collect_flagged_states.py --robot go1 --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
    assert _exit_code == 0, f"seed {s}: bank collection failed (exit {_exit_code}) -- scroll up"
    save(bank)
""")

md("""
## C2. Fall-size reward weights (once, shared by every seed)
The Go1 keeps its own reward weights (no re-sizing, unlike the G1). Only the
four fall-size terms need a weight on this robot: 11 iterations of the
round-8 task on the first seed's training bank log every measure's episode
total, and each weight is set so its term costs as much per fall as the
ground-covered term (weight -30) does (`scripts/robot_calibrate_fall_size.py`,
the G1's rule). Reused instead if `REUSE_SIZE_WEIGHTS_FROM` is set.
""")
code("""
if restore(SIZE_WEIGHTS):
    print("fall-size weights already in this run")
elif REUSE_SIZE_WEIGHTS_FROM:
    src = f"{DRIVE}/{REUSE_SIZE_WEIGHTS_FROM}/{SIZE_WEIGHTS}"
    assert os.path.exists(src), f"missing {src} -- check REUSE_SIZE_WEIGHTS_FROM, or set it to None to calibrate"
    shutil.copy(src, SIZE_WEIGHTS)
    save(SIZE_WEIGHTS)
else:
    s0, bank = SEEDS[0], flag_bank(SEEDS[0], TRAIN_ROLE)
    !python scripts/go1_train_protective.py --robot go1 --seeds {s0} --state-bank-file {bank} --peak-force-weight {PEAK_FORCE_WEIGHT} --log-fall-sizes --run-tag sizecal --max-iterations 11 --eval.interval 10
    assert _exit_code == 0, f"sizing run failed (exit {_exit_code}) -- scroll up"
    run = sorted(glob.glob(f"logs/rsl_rl/go1_protective_fall/*_sizecal_seed{s0}"))[-1]
    !python scripts/robot_calibrate_fall_size.py --run-dir {run} --envelope-weight {ENVELOPE_WEIGHT} --output {SIZE_WEIGHTS}
    assert _exit_code == 0, f"sizing failed (exit {_exit_code}) -- scroll up"
    save(SIZE_WEIGHTS)
table = json.load(open(SIZE_WEIGHTS))["weights"]
for r, measure in REWARDS.items():
    print(f"{LABELS[r]:16s} {measure:24s} weight {table[measure]:9.3f}")
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
        !python scripts/go1_train_protective.py --robot go1 --seeds {s} --state-bank-file {bank} --peak-force-weight {PEAK_FORCE_WEIGHT} --fall-size-weights-json {SIZE_WEIGHTS} --fall-size-measure {measure} --run-tag e2e_{r} --max-iterations {PROTECTIVE_TRAIN_ITERATIONS}
        assert _exit_code == 0, f"seed {s}, {r}: training failed (exit {_exit_code}) -- scroll up"
        run = sorted(glob.glob(f"logs/rsl_rl/go1_protective_fall/*_e2e_{r}_seed{s}"))[-1]
        shutil.copy(f"{run}/model_{PROTECTIVE_ITERATION}.pt", policy(s, r))
        shutil.copy(f"{run}/eval_history.json", policy_history(s, r))
        save(policy(s, r), policy_history(s, r))
        print(f"seed {s}, {r}: trained in {(time.time() - t0) / 60:.0f} min")
""")

md("## C4. Train the original SmallFall (round-8 reward, ground covered), each seed to iteration 200")
code("""
for s in SEEDS:
    history = f"outputs/go1_original_eval_history_seed{s}.json"
    if restore(original_sf(s), history):
        print(f"seed {s}: original SmallFall already trained, skipping")
        continue
    print(f"\\n==================== seed {s}: original SmallFall ====================")
    bank = flag_bank(s, TRAIN_ROLE)
    !python scripts/go1_train_protective.py --robot go1 --seeds {s} --state-bank-file {bank} --peak-force-weight {PEAK_FORCE_WEIGHT} --run-tag e2e_original --max-iterations {SF_ITERATION + 1}
    assert _exit_code == 0, f"seed {s}: original SmallFall training failed (exit {_exit_code}) -- scroll up"
    run = sorted(glob.glob(f"logs/rsl_rl/go1_protective_fall/*_e2e_original_seed{s}"))[-1]
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
    !python scripts/go1_handoff_eval.py --robot go1 --settle-lock --protective-checkpoint {pr} --walker-checkpoint {WALKER} --predictor-file {p} --metrics-file {m} --state-bank-file {bank} --seed {s} --output {handoff(s)}
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
    !python scripts/go1_collect_flagged_states.py --robot go1 --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
    assert _exit_code == 0, f"seed {s}: test bank failed (exit {_exit_code}) -- scroll up"
    !python scripts/go1_protective_compare.py --robot go1 --protective-checkpoint {pr} --walker-checkpoint {WALKER} --state-bank-file {bank} --seed {bs} --output {out} --locked-arms protective
    assert _exit_code == 0, f"seed {s}: compare failed (exit {_exit_code}) -- scroll up"
    save(out, npz(out))
""")

md("""
## E2. Verdict per seed (space first)
PASS = less ground covered than both the walker and limp, comes to rest
(>= 99%), and lower trunk force and torque than the walker (the Go1's pass
rule, unchanged).
""")
code("""
g = "envelope_growth_m"
print(f"{'seed':>4s}{'ground: SmallFall':>19s}{'walker':>9s}{'limp':>8s}{'trunk N':>9s}{'any-body N':>12s}{'settled':>9s}   verdict   any-body force below walker")
for s in SEEDS:
    q = json.load(open(test_compare(s)))["summaries"]
    med = lambda arm, k: q[arm][k]["median"]
    settled = q["protective"]["settled"]["mean"]
    ok = (med("protective", g) < med("walker", g) and med("protective", g) < med("limp", g) and settled >= 0.99
          and med("protective", "peak_force_critical_n") < med("walker", "peak_force_critical_n")
          and med("protective", "peak_torque_nm") < med("walker", "peak_torque_nm"))
    print(f"{s:4d}{med('protective', g):19.3f}{med('walker', g):9.3f}{med('limp', g):8.3f}"
          f"{med('protective', 'peak_force_critical_n'):9.0f}{med('protective', 'peak_force_any_n'):12.0f}{settled:9.3f}"
          f"   {'PASS' if ok else 'FAIL':7s}   {med('protective', 'peak_force_any_n') < med('walker', 'peak_force_any_n')}")
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
        !python scripts/go1_collect_flagged_states.py --robot go1 --checkpoint-file {WALKER} --predictor-file {p} --metrics-file {m} --seed {bs} --capture-state-bank --capture-at-flag
        assert _exit_code == 0, f"seed {s}: evaluation bank {role} failed (exit {_exit_code}) -- scroll up"
        extra = " ".join(f"--extra-policy {r}={policy(s, r)}" for r in REWARDS if r != "hullgrow") + f" --extra-policy original={original_sf(s)}"
        locked = ",".join(["protective"] + [r for r in REWARDS if r != "hullgrow"])
        !python scripts/go1_protective_compare.py --robot go1 --protective-checkpoint {pr} --walker-checkpoint {WALKER} --state-bank-file {bank} --seed {bs} --output {out} {extra} --locked-arms {locked}
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
             "--compare-npz-template", "outputs/go1_policies_eval_seed{seed}_flag*.npz",
             "--predictor-stats-template", "outputs/stats/predictor_seed{seed}/predictor_stats.json",
             "--handoff-json-template", "outputs/go1_handoff_eval_seed{seed}.json",
             "--robot-label", "Go1", "--out-dir", "outputs/stats/multiseed", "--quiet"]
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
cmd = ["python", "scripts/robot_policy_boxplots.py", "--runs", "outputs/go1_policies_eval_seed{seed}_flag*.npz",
       "--seeds", *[str(s) for s in SEEDS], *arm_args,
       "--new", "protective", *[r for r in REWARDS if r != "hullgrow"], "--reference", "original",
       "--title", "Go1: four new SmallFall policies (settle lock on) vs the original SmallFall, walker and limp",
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
md("## V1. The walker at a fixed 2 m/s straight ahead")
code("""
V1 = f"outputs/go1_watch_seed{VIDEO_SEED}_ep5.mp4"
if not restore(V1):
    !python scripts/robot_watch_episodes.py --robot go1 --checkpoint-file {WALKER} --episodes 5 --commanded-speed {WATCH_SPEED_MPS} --seed {VIDEO_SEED}
    assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up"
    save(V1)
display(Video(V1, embed=True, width=640))
""")
md("## V2. Predictor overlay: five falls, the last 4 s before each")
code("""
vp, vm = predictor(VIDEO_SEED), metrics(VIDEO_SEED)
V2 = f"outputs/go1_wildpush_overlay_impulse{float(IMPULSE_N)}N_seed{VIDEO_SEED}_take0.mp4"
if not restore(V2):
    !python scripts/go1_wildpush_video.py --robot go1 --checkpoint-file {WALKER} --predictor-weights {vp} --predictor-metrics {vm} --mechanism impulse --impulse-force {IMPULSE_N} --min-failures 5 --seed {VIDEO_SEED}
    assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up"
    save(V2)
display(Video(V2, embed=True, width=640))
""")
md("## V3. SmallFall on eight falls (settle lock on)")
code("""
vpr, vbank = protective(VIDEO_SEED), flag_bank(VIDEO_SEED, TEST_ROLE)
V3 = f"outputs/go1_protective_watch_seed{VIDEO_SEED}_ep8.mp4"
if not restore(V3):
    !python scripts/go1_watch_protective_episodes.py --robot go1 --settle-lock --checkpoint-file {vpr} --state-bank-file {vbank} --episodes 8 --seed {VIDEO_SEED}
    assert _exit_code == 0, f"script failed (exit {_exit_code}) -- scroll up"
    save(V3)
display(Video(V3, embed=True, width=640))
""")
md("## V4. Demo: the same fall three ways (walker alone | limp | hand-off with the settle lock)")
code("""
DEMO = f"outputs/videos/go1_handoff_demo_seed{VIDEO_SEED}.mp4"
if not restore(DEMO, DEMO.replace(".mp4", ".json")):
    !python scripts/go1_handoff_demo_video.py --robot go1 --settle-lock --protective-checkpoint {vpr} --walker-checkpoint {WALKER} --predictor-file {vp} --metrics-file {vm} --state-bank-file {vbank} --seed {VIDEO_SEED}
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
out = Path(__file__).resolve().parents[2] / "notebooks" / "go1_end_to_end.ipynb"
out.write_text(json.dumps(nb, indent=1) + "\n")
print("wrote", out)
