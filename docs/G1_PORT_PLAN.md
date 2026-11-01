# Plan: the Go1 fall pipeline on the G1 humanoid

Status: plan only, written 2026-10-03. Nothing here is implemented or run.
It assumes the Go1 end-to-end run (`notebooks/go1_end_to_end.ipynb`) completes.

## 1. Goal

Run the same procedure on mjlab's Unitree G1 humanoid that was run on the Go1:
walker, push, fall predictor, protective "small-fall" policy, hand-off, the
three-arm comparison (hand-off, walker continues, limp), the demo video, and
the statistics.

"The same procedure" means the same steps, the same rules for choosing things,
and the same tests. It does not mean the same numbers. Every number the Go1
work took from Go1 measurements has to be measured again on the G1 with the
same method (section 4).

## 2. What already exists for the G1

Confirmed live on Colab on 2026-09-13 and recorded in
`src/cartpole_term/common/robot_profile.py`:

- Task: `Mjlab-Velocity-Flat-Unitree-G1`, registered here as
  `Mjlab-G1-Velocity-Term`.
- Observation 99 values, action 29 values (Go1: 48 and 12). 50 Hz, 20 s
  episodes, same as Go1.
- Episodes end on `time_out` or `fell_over` (torso tilted more than 70
  degrees), same as Go1. No collision-based ending.
- The built-in mild push event is identical to Go1's.
- 30 body names, sorted into vulnerability tiers. The model has no head or
  hand bodies, so the "protect this" tier is pelvis, torso, and waist.

Working code that already takes `--robot g1`:

- `scripts/inspect_robot_task.py`, `scripts/robot_train.py` (with
  `--resume-from`), `scripts/robot_watch_episodes.py`,
  `scripts/measure_push_failure_rate.py`.
- `src/cartpole_term/g1/` (task registration, runner, training config).

Read from mjlab 1.6.0's source for this plan:

- G1 total mass 33.34 kg (Go1: 12.74 kg). Standing pelvis height 0.76 m.
- Every G1 joint motor is the same motor model as the Go1's
  (`BuiltinPositionActuator`). The limp baseline's motors-off mechanism
  (`dr.pd_gains` scaled to zero) therefore applies unchanged.
- mjlab's own G1 training recipe is 30,000 iterations (Go1: 10,000), same
  network sizes and PPO settings.

Nothing past the walker stage exists for the G1: no trained walker, no
predictor, no protective task.

## 3. What carries over unchanged

- The predictor: architecture, loss, labelling (16-step horizon), episode
  split, baselines, threshold rule. Its input width becomes 99 + 29 = 128.
- The live flag-then-fall collection and both kinds of bank (mid-fall states
  and flag-moment states), including the bank-per-role rule: one seed each for
  training, selection, and test.
- The protective task's structure: start from injected bank states, end when
  the robot is at rest, same reward terms.
- The hand-off logic, the "pushes stop once the robot is down" rule, the
  inference-mode rule for resets.
- The three-arm comparison, the space-first pass rule, the checkpoint
  selection rule, and all of the statistics code (it is robot-agnostic).
- The pre-registration habit: write the rule before the data exists.

## 4. What must be measured again on the G1

Each of these was set from Go1 data. The method stays; the value is re-derived.

| Item | Go1 value | How it was set | What to do for G1 |
|---|---|---|---|
| Push force | 150 N per axis | Literature range 50-850 N for a 35 kg humanoid, scaled by mass, then a failure-rate sweep and a look at the video | The G1 is 33 kg, so the literature range applies almost directly (about 48-810 N). Mass-scaling Go1's 150 N gives about 390 N as a starting point. Sweep with `measure_push_failure_rate.py`, pick by the same criteria. |
| Push target body | `trunk` | The robot's main body | `torso_link` |
| Protective episode budget | 3 s | Go1 post-fall data showed motion still present at 0.6 s | Measure how long an uncontrolled G1 fall takes to come to rest; a humanoid falls from higher and may need 4-5 s. |
| Settle thresholds | 0.2 m/s, 0.5 rad/s, held 0.5 s | Common at-rest cutoffs, not Go1-tuned | Keep; confirm in a log-only round that an at-rest G1 passes them. |
| Torque penalty weight | -5e-4 | Sized from Go1's measured raw torque so the term was comparable to the others | G1 motors reach 139 Nm (Go1: 35.5), so the raw value will be many times larger. Measure in a log-only round, size by the same rule. |
| Joint-limit weight | -10 | Sized from Go1's measured raw value | Measure, size by the same rule. |
| Ground-covered weight | -30 | Sized from Go1's measured envelope | Measure. A standing humanoid that ends up lying down covers about its own body length whatever it does, so expect a larger unavoidable floor. |
| Peak-force weight | -0.02 | Sized against a 1100 N Go1 peak | Measure. A robot 2.6 times heavier will produce larger peaks. |
| Contact-force, velocity, action-rate weights | from arXiv:2512.01336 Table I | Literature values for a humanoid | Keep. They were written for a humanoid in the first place. |
| "Down" height | half the starting height | Our definition | Keep the rule; the number follows from the G1's 0.76 m. |
| Protective checkpoint | round 8b, iteration 200 | Fixed rule on a selection bank | Re-run the rule on G1 candidates. |
| Demo camera | azimuth 90, elevation -20 | Looked right for a 0.3 m robot | Set distance and height for a 1.3 m robot. |

Two Go1 findings should be applied from the start instead of rediscovered:
train the protective policy from flag-moment states (variant 8b), and include
the peak-force term (round 8). That skips Go1's rounds 1-7.

## 5. Code work

Almost all of the Go1 scripts name the Go1 directly (its task, its runner, its
body names, a 150 N default, `go1_` file names). Eleven scripts and the
`protective/` package need a robot parameter.

Recommended approach: make them take `--robot`, driven by `RobotProfile`.

1. Extend `RobotProfile` with the fields now hard-coded: push body, default
   push force, mass, protective task id, protective episode length, reward
   weights, camera settings.
2. Give `protective/` one factory, `protective_fall_env_cfg(profile, ...)`, and
   register one protective task per robot. The reward and termination code is
   already robot-agnostic.
3. Turn each `go1_*.py` script into a `robot_*.py` script with `--robot`
   (default `go1`), reading everything robot-specific from the profile and
   tagging output files with the robot's name. Keep `go1_*.py` as thin
   wrappers so the Go1 notebook and README keep working.
4. Generate `notebooks/g1_end_to_end.ipynb` from the same builder as the Go1
   notebook, with the robot as a parameter.

Regression gate before any G1 run: re-run the Go1 end-to-end notebook with the
existing walker (about an hour) and confirm it still completes and gives the
same results within run-to-run noise. The Go1 pipeline is the validated
reference; the refactor must not change it.

Unit tests carry over. New tests: the profile-driven protective factory builds
for both robots (with mjlab stubbed), and file tags never collide across
robots.

The alternative is to copy the eleven scripts to `g1_*.py` and edit them.
That is faster to start and leaves two copies to keep in step. Not recommended.

## 6. Stages, in order

Each stage ends with a check written down before it is run.

| # | Stage | What runs | Check | GPU time (estimate) |
|---|---|---|---|---|
| 0 | Refactor and regression | Section 5, then the Go1 notebook with the existing walker | Go1 results reproduce | ~1 h |
| 1 | G1 walker | `robot_train.py --robot g1`, 30,000 iterations, resumed across sessions from Drive | Tracks commanded speed; walks in the video | 15-25 h, unmeasured |
| 2 | Push sweep | `measure_push_failure_rate.py --robot g1` over a force grid | Failure rate in the 10-40% band; falls look like falls, not launches | ~30 min |
| 3 | Predictor | Collect at least 1M labelled steps and 1,000 falls; train; evaluate | Beats logistic regression and chance on AUPRC; a usable warning time | ~30 min |
| 4 | Fall-timing measurement | Live collection with the post-fall hold | Sets the protective episode budget | ~10 min |
| 5 | Calibration round | Protective task with new terms log-only | Raw magnitudes recorded; weights set by the Go1 sizing rule | ~15 min |
| 6 | Protective training | Flag-moment bank (train), 300 iterations | Converges; settles | ~20 min |
| 7 | Selection | Candidates on a selection bank, fixed rule | One checkpoint | ~20 min |
| 8 | Test | Selected checkpoint, once, on a fresh bank: hand-off vs walker vs limp | Space-first pass rule | ~10 min |
| 9 | Live hand-off | 1,024 live trials | Recall at least 0.95; latch holds; weights unchanged | ~10 min |
| 10 | Statistics and video | About 1,500 fresh paired falls; predictor intervals; three-panel video | Reported as it comes out | ~40 min |
| 11 | Single notebook | `g1_end_to_end.ipynb` | Runs through | - |

The walker is the only long stage. It will not fit in one Colab session, so it
has to be trained in resumed chunks with checkpoints saved to Drive.

## 7. Risks specific to the humanoid

- **Walker training time and quality.** 30,000 iterations at 29 joints is the
  largest single cost and has never been run here. A walker that does not walk
  well makes every later stage meaningless, as it would have for the Go1.
- **A humanoid may not be able to fall small.** A 1.3 m body that ends up on
  the ground covers roughly its own length. The margin over limp may be
  smaller than the Go1's 21%, or absent. The protective policy might instead
  learn to crouch or sit, which would count.
- **Limp may be a stronger baseline.** A limp humanoid folds at the knees and
  waist. The Go1 result that the protective policy beats limp on space does
  not transfer by assumption.
- **No head in the model.** The usual reason to protect a humanoid's fall is
  its head and sensors. This model can only protect pelvis, torso, and waist,
  and that limits what the result says about a real G1.
- **Forces are larger.** Peak forces will be several times the Go1's. Reward
  weights sized for the Go1 would let one term dominate, which is what Go1's
  rounds 1-3 ran into; hence stage 5.
- **Self-collision.** The G1 task penalizes self-collision in the walker's
  reward. A falling humanoid's arms and legs will hit each other. The
  protective reward has no self-collision term today; whether it needs one is
  a decision for stage 5.
- **Predictor lead time may differ.** A humanoid's fall develops more slowly
  than a quadruped's. The 16-step horizon is kept for comparability; the
  horizon sweep script can check it if the lead looks capped.
- **Simulator non-determinism** applies as before: paired comparisons replay
  captured states, never two live runs.
- **Single seed**, as for the Go1, unless more are budgeted.

## 8. Decisions needed before starting

1. **Refactor or copy** (section 5). Recommended: refactor, with the Go1
   regression run as the gate.
2. **Walker budget.** mjlab's full 30,000 iterations, or stop earlier if the
   tracking reward has plateaued? Go1 used mjlab's full count.
3. **How the push force is chosen.** Same criteria as Go1 (failure-rate band,
   then a look at the video), with the choice written down before the
   predictor is trained on it.
4. **Pass rule.** Keep the Go1 rule (less ground covered than both the walker
   and limp, comes to rest, lower trunk force and torque than the walker), or
   set a different one for a robot that cannot avoid covering its own length?
5. **Self-collision** in the protective reward: add a term, or only measure it.
6. **Seeds.** One, as for the Go1, or more.
