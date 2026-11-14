# Deviations from CLAUDE.md

One entry per departure: what changed, why, and the evidence (error text or
source snippet). An empty list below is the expected state.

## Environment
- GPU: Tesla T4
- CUDA driver: 580.82.07
- mjlab: 1.6.0
- mujoco: 3.11.0
- mujoco-warp: 3.11.0
- rsl-rl: 5.4.2
- torch: 2.11.0+cu128

## Deviations

### S7.4 worked-example numbers corrected (documentation inconsistency, not a code deviation)

CLAUDE.md S7.4's terminated-episode worked example reads: "16 positives
(steps 24-39, zero-indexed) and 23 negatives (steps 0-22). Step 23 is the
last positive; the terminal step 39 emits nothing." These numbers are
internally inconsistent: 0-22 (23 steps) + 24-39 (16 steps) skips step 23
entirely, and "24-39" includes step 39, contradicting "the terminal step 39
emits nothing" one sentence later.

Resolution, derived from the formal definition in the same section
(y=1 iff terminated at t+k, k in {1,...,16}) and cross-checked against the
section's OWN truncated-episode example ("steps 23-39 are discarded", which
only holds if step 22 is the last negative): the self-consistent numbers are
negatives = steps 0-22 (23), positives = steps 23-38 (16), terminal step 39
emits nothing. "24-39" and "last positive" both appear to be transcription
errors for "23-38" and "first positive" / "22 is the last negative".

tests/test_labeling.py implements and asserts these corrected numbers.
Flagged to the user in conversation; not contradicted.

### S6.2.8 termination-cause breakdown: aggregate, not strictly per-episode

S6.2 requires a per-evaluation termination-cause breakdown (pole_angle /
cart_pos / time_out) alongside episode length, return, and censored
fraction, computed over exactly the 256 episodes (one per env).

Implementation note (src/cartpole_term/runner.py, FrozenEvalRunner.
_run_eval_rollout): episode length, return, and censored fraction ARE
computed exactly per-env, scoped strictly to each env's first episode
(via an `active` mask over the per-step `dones`/`truncated` tensors, which
are accurate regardless of auto-reset). The cause breakdown, however, is
computed from extras["log"]["Episode_Termination/<cause>"] summed over the
whole rollout -- the same mechanism validated in scripts/check_terminations.
py (Gate 1). Because envs run in lockstep with auto_reset=True (the
default) and finish their first episode at different times, an env that
finishes early can start and finish a second episode while slower envs are
still on their first; if that happens, the second episode's cause is also
counted in the aggregate.

The alternative (auto_reset=False, manual per-env partial resets, splicing
partial TensorDict updates) would give exact per-episode attribution, but
needs an unconfirmed ManagerBasedRlEnvCfg.auto_reset field and is
considerably more fragile than the already-validated aggregate-log
approach. Per CLAUDE.md S1.2's deference rule ("where this document is
silent... choose the plainest standard option and log it"), the aggregate
approach was chosen. In practice this only matters when the frozen policy
is not yet competent (long tail of fast episodes racing ahead of slow
ones); the S7 frozen-checkpoint protocol runs eval on a policy that has
converged to a specific operating point, where episode lengths across the
256 eval envs are far more homogeneous and the effect is expected to be
small. Confirmed working end to end via scripts/check_frozen_eval.py.

### S11 gate 4 --smoke: select_checkpoints.py bypasses the real operating-point windows

CLAUDE.md S7.1's windows (80+/-15, 150+/-25, 400+/-60 steps) are unreachable
in a 5-iteration --smoke training run -- the smoke policy is barely
trained. Requiring the real windows would make the gate-4 smoke test fail
for a reason unrelated to whether the pipeline works.

scripts/select_checkpoints.py: when --smoke is set, `select_any_for_smoke`
picks whichever checkpoint is available (the last one) for all three
operating-point slots, instead of `select_from_history`'s real
target/tolerance check. Only active under --smoke; the real (non-smoke)
path is untouched. Confirmed working end to end (Gate 4 smoke run,
2026-08-20): train -> select -> collect -> eval_predictor all completed in
~90s, well under the 5-minute budget, all artifact types produced.

### Bug found and fixed during Gate 4: collect_frozen.py's censored_step_count conflated two different discards

scripts/collect_frozen.py originally computed
`censored_count = total_pushed - len(records) - pending_at_end`, intending
to measure S7.3's "censored-step count" (transitions discarded because a
time-out cut the episode short, per S7.4's truncation-censoring rule).

This silently also counted a second, unrelated thing: every TERMINATED
episode drops exactly one entry -- the fresh, age-0 push from its own
terminal step, which never falls in the emitted [1, horizon] positive
range (S7.4: "the terminal step itself is not a training example"). That
discard is correct and expected, and provable from
TerminationLabeler.step()'s structure (exactly one entry per terminated
episode), but it is NOT truncation-censoring.

Caught because the Gate 4 smoke run reported `censored=132` exactly equal
to `terminated_episodes=132`, with the frozen-eval's own censored_frac at
0.000 (i.e. no time-outs occurred in that run at all) -- so the entire
reported "censored" count was actually just terminal-step discards.

Fixed: `terminal_step_discard_count` (= terminated_episode_count, proven
from the algorithm) is now subtracted out and reported as its own sidecar
field, separate from `censored_step_count` (now correctly zero when there
are no truncations). The underlying labelled dataset was never wrong --
only this one summary statistic in the JSON sidecar was mislabeled.
Re-verified via a second Gate 4 smoke re-run: `censored_step_count: 0`.

### Bug found and fixed during Gate 4: eval_predictor.py's --seeds did not control MLP weight initialization

CLAUDE.md S8.3: "Fixed torch seed per run; report it." scripts/
eval_predictor.py constructed the model (`model_cls(history=...)`, which
randomly initializes weights at construction time) BEFORE calling
predictor/train.py's `train_predictor`, whose `torch.manual_seed(seed)` is
the first line of ITS body -- too late to affect the weights that had
already been randomly initialized moments earlier. Net effect: `--seeds 0`
never actually made weight initialization reproducible; only whatever
happened after training started (batch shuffling, etc.) was seeded.

Caught because a Gate 4 re-run (fixing the censored_step_count bug above)
produced a different MLP AUPRC (0.5700 -> 0.4246) from a byte-identical
frozen dataset and the same seed.

Fixed: scripts/eval_predictor.py's evaluate_one_dataset now calls
`torch.manual_seed(seed)` immediately before constructing the model.
Verified deterministic across repeated calls with a new regression test,
tests/test_eval_predictor.py::test_evaluate_one_dataset_is_deterministic_given_same_seed.

### S11 gate 5: dense eval/save interval + early stopping, instead of the shipped max_iterations=500

CLAUDE.md S5.1 says reuse the installed `cartpole_ppo_runner_cfg()`
verbatim (confirmed max_iterations=500); S5.3 anticipates only *raising*
max_iterations if the policy hasn't converged, not lowering it or changing
save_interval/eval_interval beyond the single 10-iteration value S5.3
already fixes. These are deliberate user decisions for the seed-0 gate-5
run, not tuning-for-better-results choices.

First attempt (`--max-iterations 100`, superseded below): a first seed-0
run capped at 100 iterations showed mean episode length under the
frozen-eval mean action reaching 1000 (the time-out cap) by iteration 100.
But because eval only fires every `save_interval=10` iterations (S6.2.1),
and this task converges from near-zero to the 1000-step cap within roughly
two such checkpoints, the sparse 10-iteration measurement grid never
actually landed a measurement inside the `struggling` (80+/-15) or
`competent` (340-460) windows -- eval_history.json showed `target`
(125-175) crossed at it=10, then straight to 1000 by it=90, with
`struggling` and `competent` never measured. This isn't evidence the
policy skipped those windows; the measurement grid is simply too coarse to
resolve a policy that saturates this fast at 4096 envs.

Final approach: rather than accept unmeasured operating points, the
measurement grid itself was densified.

1. `--eval.interval 2` (was 10). Fixed a latent bug in the same change:
   scripts/train.py now sets `rl_cfg.save_interval = eval_interval`
   unconditionally, so a saved checkpoint always exists at every measured
   iteration (previously the two could silently drift apart even in the
   default case, though the default value happened to coincide with S5.3's
   10 either way).
2. `FrozenEvalRunner` gained an optional `early_stop_patience` constructor
   parameter (src/cartpole_term/runner.py) and `--early-stop-patience`
   CLI flag (scripts/train.py): once `early_stop_patience` CONSECUTIVE
   frozen evals report mean_action.mean_episode_length at the eval env's
   max_episode_length, `learn()` breaks out of its loop early rather than
   continuing to num_learning_iterations regardless. Neither flag is part
   of the S5.5 table -- both are project-level additions, default to
   `None`/disabled, matching stock behaviour unless explicitly passed.

Run used `--seeds 0 --eval.interval 2 --early-stop-patience 2`.

Confirmed outcome (run 2026-08-20_19-36-35_seed0, 9 eval records total):
struggling (80+/-15) first crossed at it=4 (68.2 steps); target (150+/-25)
first crossed at it=10 (131.1 steps); competent (340-460) first crossed at
it=12 (430.6 steps); mean_action_len reached the 1000-step cap at it=14 and
held at it=16, triggering early stop there. All three operating-point
windows measured directly -- S11 gate 5's pass condition (>=400-step
frozen-eval mean; all three windows crossed) is satisfied with real
evidence, not inference. Total wall-clock for the run: ~9s of training
compute -- this task converges almost instantly at 4096 envs.

### S11 gate 9: |theta_wrapped| is NOT the strongly predictive feature at `target` -- investigated, not a bug

S9.3 states the feature-screening check as a gate-9 pass condition:
"`|theta_wrapped|` must come out strongly predictive... If it doesn't, the
labels are wrong; stop and debug." At `target` (seed 0), it came out weak
(AUC-ROC 0.584, McFadden R2 0.016) while `|cart_pos|` dominated (AUC-ROC
0.929, R2 0.487) -- the opposite of what S9.3 assumes.

Investigated rather than assumed either way (bug, or dismissed as fine).
Evidence gathered, all internally consistent:

1. `outputs/frozen_target_seed0.json`'s termination_cause_counts:
   `{pole_angle: 15471, cart_pos: 131120, both: 176}` -- 89.3% of actual
   collected terminations at `target` are cart_pos-caused, only 10.5%
   pole_angle. The feature-screening result directly tracks which cause
   actually dominates real terminations in the dataset.
2. Per-class separation, computed directly from frozen_target_seed0.npz:
   `|theta_wrapped|` positives=0.0605+/-0.044 vs negatives=0.0475+/-0.037
   (1.27x, both means are ~3 degrees, nowhere near the 12-degree failure
   threshold). `|cart_pos|` positives=0.6966+/-0.210 vs
   negatives=0.2228+/-0.198 (3.13x, closing in on the 1.0m boundary). States
   heading toward failure at `target` are unambiguously characterized by
   cart drift, not pole tilt.
3. The same breakdown across all three operating points shows a
   non-monotonic, mechanistically coherent story (not noise):
   struggling (it=4): pole_angle 64.7% / cart_pos 33.9% (pole_angle
   majority -- an early, still-unstable policy fails at balancing itself).
   target (it=10): pole_angle 10.5% / cart_pos 89.3% (pole-balancing is
   essentially solved; the only way left to fail is drifting off the rail).
   competent (it=12): pole_angle 27.3% / cart_pos 71.9% (cart_pos share
   partially recedes as absolute failures become rare and control
   precision keeps improving on both fronts).

Conclusion: not a labelling bug. The labelling pipeline is independently
validated (tests/test_labeling.py, S2, all edge cases) and every cross-check
here (cause counts, steps-to-termination, per-class separation) is
internally consistent. The mechanism is real and specific to this project:
S2.6 adds an explicit cart-centering reward term on top of the shipped
reward specifically to suppress cart drift, which makes cart-position
dynamics more actively shaped and more informative here than in a "vanilla"
cartpole -- once pole-balancing is learned at all (which happens fast, see
the gate-5 entry above), cart drift is the dominant remaining failure mode,
and `|cart_pos|`, not `|theta_wrapped|`, is the feature that legitimately
carries the near-term-failure signal.

Tooling improvement made as a result (scripts/eval_predictor.py,
feature_screening): per-class (positive vs negative) mean/std for every
screened feature is now included in the output alongside McFadden R2 and
AUC-ROC. Not spec-mandated, but this is exactly the diagnostic that had to
be hand-rolled to investigate this finding -- making it a standing part of
the output means the same question answers itself directly from
metrics_*.json on any future run (including seeds 1/2, other operating
points, or entirely new experiments), rather than needing an ad hoc query
each time. seed 0's three already-collected metrics_*.json files predate
this change; the manual diagnostics above supplement them for the record.

### S11 gate 10: seed 1's eval.interval=2 grid still too coarse for `target` -- per-seed instance of the gate-5 densification finding

`--seeds 1 2 --eval.interval 2 --early-stop-patience 2` (the exact settings
that worked for seed 0 at gate 5) produced, for seed 1:
`[(0, 11.9), (2, 35.1), (4, 60.7), (6, 69.7), (8, 84.2), (10, 115.0),
(12, 342.1), (14, 1000.0), (16, 1000.0)]` (iteration, mean_action mean
episode length). The `target` window (150+/-25) falls strictly between the
it=10 and it=12 measurements (115.0 -> 342.1) and was never measured, so no
checkpoint exists there to select -- `select_checkpoints.py` correctly
raised (CLAUDE.md S7.2.3: "do not silently substitute") rather than picking
a neighboring checkpoint outside the window.

Not a bug: this is the same phenomenon as the gate-5 densification entry
above, one level deeper. Seed 1's random init happens to converge even
faster/steeper between it=10 and it=12 than seed 0's did between whatever
its own equivalent pair was, so the same eval.interval=2 grid that resolved
seed 0's three windows is too coarse for seed 1's specific trajectory.
Grid density needed to resolve a crossing is a property of how steep a
given seed's convergence curve is, not a fixed constant across seeds.

Fix: retrain seeds 1 and 2 together with `--eval.interval 1` (save_interval
tracks it, per scripts/train.py's existing fix) instead of just retrying
seed 1 alone -- seed 2 had not yet been reached by select_checkpoints.py
(it raises per-seed, in order, and stops at the first failure), so its grid
adequacy was unknown and retraining both up front avoids a second round
trip if seed 2 turns out to have the same issue.

### Bug found and fixed: select_checkpoints.py overwrote selected_checkpoints.json instead of merging

Discovered chasing a `collect_frozen.py` failure ("seed 1 not found in
outputs/selected_checkpoints.json") after the gate-10 retry above.
`select_checkpoints.py --seeds N` was run once for `--seeds 0` (gate 7) and
again for `--seeds 1 2` (gate 10) -- two separate invocations, the pattern
CLAUDE.md's per-seed CLI surface (S5.4: "3 seeds default... via --seeds")
explicitly enables. But `main()` built `all_results` from scratch each
call and wrote it unconditionally at the end, so a successful `--seeds 1 2`
run would have silently dropped seed 0's already-selected entry, and a run
that raised partway through left the file exactly as it was before (seed 0
only) -- which is what produced the immediate "seed 1 not found" error.

Fix: `all_results` now loads any existing outputs/selected_checkpoints.json
first and merges into it (scripts/select_checkpoints.py), and the file is
rewritten after each seed completes rather than only once at the end of the
whole batch, so a later seed's failure can no longer discard an earlier
seed's already-computed selection within the same invocation either.

### Automatic eval-grid densification on a bracketed operating-point window (scripts/train.py)

Manually re-running seed 1 at `--eval.interval 1` (the fix for the entry
above) worked, but required a human to notice the specific failure pattern
(a window's target+/-tolerance band sits strictly between two consecutive
frozen-eval measurements) and manually choose a finer grid. Automated this
rather than requiring it be diagnosed by hand each time it recurs -- and it
will recur, since grid density needed is a property of each seed's own
convergence steepness (established at gate 5 for seed 0, confirmed here for
seed 1), not a fixed constant.

`scripts/train.py` gained `_find_bracketed_windows(history)`: for each S7.1
operating point without a qualifying checkpoint, checks whether two
consecutive measurements straddle the window (one below, one above) --
distinct from a window the policy never approaches at all, which no amount
of resampling fixes and should still surface as `select_checkpoints.py`'s
ordinary RuntimeError. `train_one_seed` now calls this after training and,
if any window is bracketed and eval_interval > 1, automatically retrains
that seed once at eval_interval=1 (the finest possible grid -- naturally
bounds the recursion to a single retry, since 1 can't be subdivided
further) before returning. If a window is still bracketed at
eval_interval=1, that's logged as a genuine, non-fixable-by-resolution gap
and left for select_checkpoints.py to report normally.

Not a PPO hyperparameter search (CLAUDE.md S5.3's prohibition): eval_interval
only controls how often the already-fixed, already-training policy is
measured, never what it learns or how -- identical to the gate-5 grid
densification, just automated and bounded rather than hand-tuned per seed.
Verified by hand-tracing against seed 1's actual eval_history.json numbers
(0..16 by 2: ...,84.2, 115.0, 342.1,...) before relying on it: `target`
(150+/-25) is correctly flagged bracketed (114.99 < 125 lo, 342.06 > 175
hi), while `struggling` (69.7 at it=6, inside [65,95]) and `competent`
(342.06 at it=12, inside [340,460]) are correctly left alone since both
already had a qualifying checkpoint in that same data.

### Second escalation tier: perturbed-RNG-seed retry when a window is skipped even at eval_interval=1

The eval_interval densification above resolves a window skipped by a
COARSE grid, but the same seed-1 numbers show the underlying per-iteration
jump is large enough (~110+ steps/iteration through the 50-step-wide
`target` band, it=10->12: 115.0->342.1 over just 2 iterations) that a
window could in principle be skipped even at eval_interval=1 -- there is no
timestep finer than one PPO iteration to measure at, so once that's not
enough, no further "measure more finely" option exists.

The only remaining lever, given the user explicitly authorized retraining
as needed ("I want a solution for it to work no matter what, even if
training has to rerun"): try a genuinely different random draw for the
same nominal seed slot. `scripts/train.py`'s `train_one_seed` gained a
third parameter, `_perturbation_attempt`, and a module-level
`actual_rng_seed = seed if _perturbation_attempt == 0 else seed +
SEED_PERTURBATION_STRIDE * _perturbation_attempt` (stride 1000, so
perturbed seeds never collide with a real `--seeds` value). If a window is
still bracketed after the eval_interval=1 retry, training retries with
`actual_rng_seed` instead of `seed` feeding both `env_cfg.seed` and
`rl_cfg.seed`, capped at `MAX_SEED_PERTURBATION_ATTEMPTS = 5` (arbitrary
but sane: each attempt costs ~9s of compute, so 5 is generous without
risking a runaway loop). If all 5 are exhausted and a window is still
skipped, `train_one_seed` now raises a RuntimeError itself rather than
silently returning a seed that can never resolve.

Honest caveat, logged because it is NOT free: if this tier ever actually
fires, whatever gets called "seed 1" no longer corresponds to literally
seeding the RNG with the integer 1 -- it's a different, though still fully
deterministic and reproducible, random draw for that seed slot.
`resolved_args.json` records both `seed` (the nominal --seeds label used
for all file/directory naming) and `actual_rng_seed` (what was actually
fed to env_cfg.seed/rl_cfg.seed) plus `seed_perturbation_attempt`, so this
is always fully transparent and auditable after the fact, never silent.
Whether this tier actually fired for any of seeds 0/1/2 in the reported
results must be checked via resolved_args.json before treating "3 seeds"
in the README as literally seeds {0, 1, 2} rather than {0, 1, 2}-with-
possible-perturbation.

### S11 gate 11: camera angle -- shipped cartpole viewer config looks straight down the rail

User feedback after watching the first policy_final.mp4: the rail's
back-and-forth motion was hard to see -- the camera was aligned WITH the
rail rather than perpendicular to it. Confirmed rather than guessed: read
`mjlab.viewer.viewer_config.ViewerConfig`'s real source and the shipped
cartpole config's actual instantiated values directly (via `cfg.viewer` on
`cartpole_balance_term_env_cfg(play=True)`). The shipped config overrides
`azimuth=0.0` (class default is `azimuth=90.0`) with `origin_type=
ASSET_BODY` tracking the `cart` body, `distance=4.0`, `elevation=-15.0`.
Since the rail is the slider joint's `axis="1 0 0"` (CLAUDE.md S2.1), and
azimuth is rotation about the vertical axis, `azimuth=0.0` looks straight
down that axis -- rail motion then only changes apparent depth, not
left-right screen position. `azimuth=90.0` (mjlab's own class default, not
an arbitrary choice) rotates 90 degrees around the vertical axis to a
side-on view instead.

`scripts/record_video.py` gained `--camera-azimuth` (default 90.0,
overridable) applied via a shared `_build_video_env_cfg(args)` helper used
by both video functions, rather than hardcoding the value -- not yet
visually confirmed against an actual render (I cannot view video/image
output directly), so kept adjustable in case 90 needs further tuning once
seen. Confirmed working end to end: user reported the side-on angle looks
correct at azimuth=90.0 (after an earlier false-negative report traced to
stale re-uploaded code, not the angle itself -- a `[video] viewer config:
...` diagnostic print was added to `_build_video_env_cfg` specifically so
this class of confusion is verifiable going forward rather than argued
about).

Follow-up: `--camera-elevation` added the same way (default -25.0, nudged
from the shipped -15.0) after a request to raise the camera slightly so
the pole reads more centered in frame; confirmed applied correctly
(`elevation=-25.0` in the diagnostic print) after a second stale-code
false start, same class of issue as azimuth's.

Follow-up: `--min-failures` (default 10, floor enforced at
MIN_FAILURE_EPISODES_FLOOR=3 -- CLAUDE.md S10 item 2's actual requirement)
replaces the hardcoded 3-failure stop after the resulting clip (174 steps,
8.7s) was found too short to be demonstrative. 3 remains the enforced
minimum; 10 is a project-level default for a longer clip, not a spec
value.

### S11 gate 11: predictor_overlay.mp4 implementation notes

Design questions resolved before writing this (raised in conversation,
confirmed against the cited literature rather than assumed):

1. **Sampled, not mean, actions.** CLAUDE.md S7.3 defines the label under
   the behaviour policy ("the label is defined under the behaviour policy,
   so the behaviour policy is what runs"), and the predictor was trained
   exclusively on transitions from `collect_frozen.py`'s sampled-action
   rollouts -- using the mean action here would evaluate it out of its own
   training distribution. This is also not just a consistency nicety:
   training logs already show `sampled_action_len` running far below
   `mean_action_len` at convergence (e.g. `mean_action_len=1000.0
   sampled_action_len=850.4`), meaning the mean action may rarely or never
   fail at all -- under it, capturing 3 failures (S10 item 2) could be
   impossible in any reasonable rollout length. Matches STATE-NAV's own
   framing too: its fixed ALIP controller is not deliberately perturbed to
   fail: the predictor watches its natural behaviour.
2. **Single rollout pass, not two.** The predictor's probability at step t
   only needs (obs_t, action_t) -- known immediately -- but the
   ground-truth label needs steps t+1..t+16, which are not known until
   later. Rather than buffering, storing every rendered frame in memory
   and labelling in a second pass after the rollout completes (bounded by
   MAX_OVERLAY_STEPS=20,000, small enough this is not a memory concern)
   reuses `predictor.labeling.TerminationLabeler` verbatim instead of
   reimplementing S7.4's delayed-labelling logic a second time.
3. **Stop on failure-episode count, not a frame budget.** Unlike
   policy_final.mp4 (fixed >=20s), S10 item 2 requires >=3 failure
   episodes, and target-competence base rate makes a fixed frame count
   unreliable for guaranteeing that. `MAX_OVERLAY_STEPS` is an engineering
   safety cap (mirrors collect_frozen.py's MAX_STEPS_SAFETY_CAP), not a
   spec requirement; the script warns rather than silently under-delivers
   if the cap is hit first.
4. Frames whose 16-step horizon window never resolves by the time the
   rollout stops (last <=16 frames, or any truncation-censored window) are
   left as `label: pending` in the overlay rather than guessed at --
   correctly refusing to render a censored value as fact, per the same
   S7.4 principle enforced everywhere else in this project.

### S11 gate 11: eval_predictor.py now persists trained predictor weights (needed for predictor_overlay.mp4)

CLAUDE.md S10 item 2's overlay video needs a trained predictor's actual
weights to compute live probabilities frame-by-frame, but `eval_predictor.py`
trained the MLP in-memory (`evaluate_one_dataset`) and never saved it --
only the aggregate metrics JSON was written. Not spec-mandated by S8/S9
directly, but required to build the S10 item 2 deliverable without
retraining a second time (which would still be reproducible given the
already-proven seed determinism, but duplicates compute and logic for no
benefit).

`evaluate_one_dataset` gained an optional `model_save_path` parameter
(default `None`, so existing callers/tests are unaffected); when given, the
trained model's `state_dict()` is saved there right after training,
mirroring `build_runner_from_checkpoint`'s actor-only load pattern (no
optimizer state, no extra metadata -- architecture is fully fixed by
CLAUDE.md S8.2, so `TerminationPredictor(history=predictor_history)`
reconstructs it exactly before `load_state_dict`). `main()` now writes
`outputs/predictor_{point}_seed{N}.pt` alongside each existing
`metrics_{point}_seed{N}.json`.

`tests/test_eval_predictor.py` gained a round-trip test: reload the saved
weights into a fresh model, recompute AUPRC on the same test split, assert
it exactly equals the AUPRC already reported in the returned metrics --
not just "does torch.save/load not raise."

### S11 gate 11: silent EGL failure on first real run -- fixed per reference/'s documented workaround

First real Colab run of `record_video.py` died silently: output stopped
mid-construction (right after the video env's "Base Environment" table,
before its manager tables), no traceback, cell reported "done." This is
exactly `ManagerBasedRlEnv.__init__`'s `OffscreenRenderer(...).initialize()`
call, which runs immediately after that print -- and exactly the failure
mode CLAUDE.md S10 itself warns about: "EGL failures are silent."

`reference/old_notebook_setup.md` (CLAUDE.md S4.1's permitted scope:
"MUJOCO_GL / EGL configuration... known platform quirks") documents this
exact problem from the prior debugged session, with two components the
script was missing:
1. Only `MUJOCO_GL=egl` was set (in the Colab cell); `PYOPENGL_PLATFORM=egl`
   was not, and per the reference note "MUJOCO_GL only takes effect on the
   first `mujoco` import per process."
2. Colab sometimes lacks the NVIDIA EGL vendor ICD file
   (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`) entirely, producing a
   non-functional GL context with no error raised.

Fix: `scripts/record_video.py` now sets both env vars and writes the
vendor ICD file itself (`_ensure_egl_configured()`), called at true
module-import time -- before `mujoco` is ever imported anywhere in the
process (before even `import torch`/`import cartpole_term`), rather than
relying on a Colab cell to set `os.environ` correctly beforehand. This is
directly reusing reference/'s documented workaround (permitted scope, not
methodology) rather than copying its code file verbatim, and is baked into
the script so the fix travels with it rather than depending on the operator
remembering a specific cell order every session.

### S11 gate 11 prep: two confirmed mjlab API facts for record_video.py, verified before writing any code

Per CLAUDE.md S0.2, `scripts/inspect_video_and_termmgr.py` was run (Colab)
before writing `scripts/record_video.py`, rather than guessing at the
video/render API surface from the spec text alone. Two facts confirmed
that the literal S10 text either got slightly wrong or left ambiguous:

1. **`ManagerBasedRlEnv.render()` uses mjlab's own internal
   `OffscreenRenderer`, not raw `mujoco.Renderer`.** S10 item 2 says to
   "render with `mujoco.Renderer`" for the predictor-overlay video. The
   confirmed `ManagerBasedRlEnv.__init__`/`.render()` source shows
   `render_mode="rgb_array"` wires up an `OffscreenRenderer` internally
   and `env.render()` already returns a plain `np.ndarray` frame -- no
   manual `mujoco.Renderer(model, ...)` setup is needed or should be
   built by hand. `policy_final.mp4` (this entry's script) already uses
   `env.render()` for its renderer sanity check per S10's own item 1 text,
   which matches this exactly. The (not-yet-written) overlay script should
   do the same -- call `env.render()` directly on an unwrapped env (bypassing
   `VideoRecorder`, whose `_record_frame` is single-purpose and not built
   for external compositing) rather than reach for `mujoco.Renderer`.

2. **The raw `ManagerBasedRlEnv.step()` (as opposed to
   `RslRlVecEnvWrapper.step()`) returns a 5-tuple, not the 4-tuple
   documented in CLAUDE.md S2.9.** S2.9's 4-tuple fact (`obs, rewards,
   dones, extras`, with `truncated` folded into `extras["time_outs"]`) is
   specific to `RslRlVecEnvWrapper`, used everywhere else in this project
   (training, frozen eval, collect_frozen.py). `VideoRecorder` wraps the
   RAW env directly (its constructor type-hints `env: ManagerBasedRlEnv`,
   not the wrapped type), and the confirmed source shows its `step()`
   delegates straight to `self._wrapped_env.step(action)`, returning
   `(obs, reward, terminated, truncated, info)` -- five values. `record_video.py`
   unpacks this 5-tuple directly; reusing the 4-tuple-handling pattern from
   `runner.py`/`collect_frozen.py` here would have been a real bug.

3. **`play=True` disables randomization and sets episode length to
   (near-)infinite**, confirmed by CLAUDE.md's own S6.2.2 note (the reason
   frozen eval must not use `play_env_cfg`) rather than re-derived from
   `inspect_video_and_termmgr.py`'s output directly. Consequence for
   `VideoRecorder`: its default "stop recording when the tracked episode
   ends" behaviour (`video_length=None`) would never fire under
   `play=True`, since the episode never ends. `record_video.py` sets
   `video_length` explicitly (25s of frames -- 5s of margin over S10's 20s
   minimum) to bound the recording directly instead of relying on an
   episode boundary that may not exist.

`policy_final.mp4`'s checkpoint choice: S10 item 1 says "the fully trained
policy" without naming an operating point; item 2 explicitly names
`target`. Read this as item 1 meaning `competent` (the most-trained
point this project defines) and hardcoded it internally rather than
exposing `--operating-point` on this script -- CLAUDE.md S5.5's flag table
scopes `--operating-point` to `collect, eval` only, not `video`.

### MAX_SEED_PERTURBATION_ATTEMPTS raised from 5 to 25 after seed 1 exhausted it on `competent`

seed 1 (actual_rng_seed values 1, 1001, 2001, 3001, 4001, 5001 -- all 5
perturbation attempts) never landed a frozen-eval measurement inside the
`competent` window (340-460), confirmed via `resolved_args.json`
(seed_perturbation_attempt: 5) and by re-running `_find_bracketed_windows`
directly against the real eval_history numbers from that run (correctly
returns `['competent']`, so the detection logic itself was not the bug --
5 attempts was genuinely just not enough tries for this seed slot).

Context: seed 0 landed `competent` on its very first attempt (it=12:
430.6, gate 5). seed 1 missed it across 6 different random draws. The
window is 120 steps wide; observed single-iteration jumps near saturation
have ranged from ~110 to ~670 steps, so a plausible per-attempt hit
probability well under 50% is not surprising, and 6 misses in a row at
that rate is unlucky but not implausible. Raised the cap to 25 (still
bounded, still ~9s of compute per attempt so ~4 minutes worst case) rather
than leave a seed slot that needs a bigger net to fail outright. Per-user
direction: "I want a solution for it to work no matter what, even if
training has to rerun."

### S11 gate 6: ablation also needed seed-perturbation retries

The S2.6 ablation run (seed 0, weight=0.0) needed 6 perturbation attempts
(actual_rng_seed=6000) before all three operating-point windows resolved
-- an earlier attempt=0 run also exists in `logs/` from a first invocation
that was interrupted partway rather than left to complete; the final,
complete run is `logs/rsl_rl/cartpole_term/2026-08-22_05-44-34_seed0`,
used for every ablation number in README.md S5. Consistent with this
same mechanism already firing for seeds 1 and 2 (S11 gate 10 entries
above) -- removing the centering term does not make checkpoint selection
any more reliably easy than having it, evidence against a hypothesis that
the perturbation mechanism was somehow specific to the centered reward.

### S11 gate 12: README.md written from real downloaded data, not summary snippets

All of README.md's tables and five figures (`docs/figures/*.png`) were
generated by two scripts run locally against the actual `outputs/` +
`logs/` directories (downloaded from Colab and extracted into the repo
root, both gitignored, per the user's own request to ground the README in
real data rather than pasted terminal snippets -- see the conversation
around this gate). Notably, this process is what surfaced the missing
S2.6 ablation (gate 6) as a real gap: a direct grep across every
`resolved_args.json` in the downloaded `logs/` showed zero runs with
`cart_center_weight=0.0` prior to this, contradicting an initial claim
that the ablation had "already worked at one point" -- the user then
re-ran it (see the entry above) rather than the README reporting on data
that didn't actually exist. Figures for the reliability diagram and PR
curve (README.md S10) are real, recomputed directly from the saved
`predictor_target_seed0.pt` weights against the actual test split, not
reconstructed from the scalar metrics alone.

## Phase 2 (post-gate-12, outside CLAUDE.md's spec)

Everything below this line is new work the user asked for after all 12
gates were already complete and verified. It intentionally does not touch
any file CLAUDE.md's spec governs (`src/cartpole_term/cli.py`,
`scripts/{train,select_checkpoints,collect_frozen,eval_predictor}.py`) --
those stay exactly as they were at gate 12 so the original reproduction
commands in README.md S1 keep working unmodified.

### `scripts/eval_predictor_generalization.py` -- new script, cross-checkpoint predictor generalization

User request: test an already-trained predictor against checkpoints it
never trained on -- either every checkpoint saved during one training
run, or a checkpoint from a completely new run -- without retraining the
predictor and without the predictor influencing training in any way
(CLAUDE.md S12#4/#6 still hold; this only adds inference-time
*evaluation*).

Design: reuses `collect_frozen.collect_one` unchanged (same rollout,
sampling, and labelling code the gated pipeline already uses) to build a
fresh, smaller labelled dataset from an arbitrary checkpoint path, loads
an already-trained predictor's saved weights (`torch.load` +
`load_state_dict`, architecture fixed by S8.2 so no extra metadata is
needed -- same pattern `record_video.py` already uses), and runs it in
`eval()`/`no_grad` inference mode over the *entire* freshly collected set.
No train/val/test split is needed here: none of this data was used to fit
the predictor, so there is no leakage to guard against by holding part of
it out.

Threshold: unlike `eval_predictor.py`, there is no validation split here
to fit a decision threshold on (S9.1's rule against fitting on test data
doesn't have an analogue to violate, but there's also no val set to fit
*to*), so `--threshold` must be passed explicitly for F1/precision/
recall/confusion/lead-time metrics; default 0.5, undocumented by CLAUDE.md
since this script is outside its scope. AUPRC/AUROC/Brier/ECE/McFadden's
R^2 don't need a threshold and are reported regardless.

"Alongside a completely new training run" is implemented as
post-hoc-per-checkpoint, not literally live inside the training loop:
`--log-dir` points at a run's `logs/rsl_rl/cartpole_term/<run_dir>/` and
the script evaluates every `model_*.pt` already written there (every
`--eval.interval` iterations, same cadence CLAUDE.md S5.3 already uses).
Deliberately not hooking into `runner.py`'s training loop itself --
CLAUDE.md S1 explicitly scopes online predictor training out ("Do not
implement replay buffers, prior correction, prequential evaluation, or
base-rate tripwires"), and the whole point of this tool is to measure
generalization of a *frozen* predictor, so keeping it a fully separate
process that only reads checkpoints off disk after `train.py` writes them
preserves that separation cleanly rather than blurring it.

Not part of any gate; not required to pass anything. Uses `--smoke` for a
quick sanity check (own numbers, not CLAUDE.md S5.5's -- this script's
flag surface is intentionally kept out of `cartpole_term/cli.py` so gate
1-12 scripts' flags can't drift).

### Follow-up: redirected to whole-second-run generalization + heavy eval + plotting

Per-user redirect: rather than testing the predictor against every
checkpoint of the *same* run it was trained on (S1's original framing),
test it against **every checkpoint of a second, completely independent
training run** -- a real out-of-distribution generalization check, since
that run's random seed, trajectory, and (if desired) hyperparameters
never touched the predictor's training data. `--log-dir` already
supported this unchanged; no script logic needed to change for the
redirect itself, only the defaults and a joining/plotting step:

- `eval_predictor_generalization.py`'s per-checkpoint collection defaults
  were raised from a lightweight probe (50k transitions/100 terminated
  episodes/1024 envs) to the same targets `collect_frozen.py` uses (1M
  transitions/1,000 terminated episodes/4096 envs) -- collection is cheap
  here (~15-20s per 1M transitions, per the gate-8 timings already in
  README.md), so a per-checkpoint probe run across ~10-20 checkpoints has
  no real reason to be less rigorous than the original gated datasets.
  `--smoke` still exists for a fast sanity pass at the old lightweight
  numbers.
- New `scripts/plot_predictor_generalization.py`: joins
  `predictor_generalization_*.json` (this tool's output) against the
  second run's own `eval_history.json` (CLAUDE.md S6's frozen in-training
  eval, already written by `train.py`/`runner.py` every `--eval.interval`
  iterations) on `iteration`, so the x-axis is the second run's own
  measured competence (frozen-eval mean episode length) rather than raw
  iteration number -- consistent with how every other competence-vs-
  performance figure in this project is framed (README.md S11). Pure
  json + matplotlib, no mjlab/torch/GPU dependency, so it also runs
  locally after downloading both JSON files, the same way the gate-12
  README figures were generated.

### `scripts/train_unified_predictor.py` -- new script, one predictor across (most of) the competence range

Result from the seed-3 generalization run: a `target`-trained predictor
generalized well from competence ~70 through ~380 steps (AUROC 0.94-0.98)
but broke down sharply past that -- AUPRC 0.78->0.33->0.056 and median
lead time collapsing to 1 step at the two checkpoints saturating the
1000-step ceiling (base rate 0.55% and 0.24%, an order of magnitude more
extreme than `competent`'s own 4.5%). User's follow-up ask: one predictor
network that does well across essentially the whole competence range,
excluding only the first few near-random learning iterations (competence
~15-40 steps, AUROC 0.67-0.75 in the seed-3 sweep -- inherently close to
unrecoverable, since a near-random policy's failures are sudden with
almost no informative lead-up).

Design: train the SAME architecture/loss/optimizer/early-stopping
(CLAUDE.md S8.2/S8.3, via the same `TerminationPredictor` +
`train_predictor` used everywhere else) on the UNION of `struggling` +
`target` + `competent` (already-collected seed-0 datasets) plus an
optional freshly-collected `near_ceiling` dataset from a late/converged
checkpoint -- directly covering the regime that broke, rather than hoping
a single operating point's data happens to extrapolate. Episode ids are
remapped by a fixed per-source offset (`i * 10_000_000`) before
concatenating: `collect_one` numbers each collection's episodes starting
from 0 independently, so combining datasets without remapping would let
unrelated episodes from different sources collide under the same integer
id -- not a train/test leakage risk (the split is still internally
consistent either way), but it would silently merge two unrelated
episodes' per-episode aggregates (counts, feature-screening class stats,
detection lead time), which is a real correctness bug worth avoiding even
though it's a one-line fix.

Reports a per-source test-set breakdown (not just one pooled number) so
it's visible whether the union actually helps every regime or just
dilutes toward the middle -- the open question this whole exercise exists
to answer, not something to assume in the training code. Same
"no CLAUDE.md flag-table entanglement" boundary as the other phase-2
scripts: entirely outside `cartpole_term/cli.py`, its own argparse.

### Follow-up: near_ceiling collection defaults lowered after a real stall

First run against `model_16.pt` (seed 0's own latest checkpoint, past even
`competent`'s iteration 12) never finished the near_ceiling collection step
-- the user interrupted it (Ctrl-C) after several minutes with no output,
reasonably assuming it had hung. It had not: `collect_one`'s loop runs
until BOTH the transition AND terminated-episode targets are met, prints
nothing in between, and `model_16.pt`'s failure rate is apparently low
enough that reaching 1,000 terminated episodes (collect_frozen.py's own
target, reused unchanged as this script's original default) would have
taken far longer than the ~15-20s the three primary datasets took at their
much higher base rates.

Fixed: `NEAR_CEILING_TERMINATED_EPISODES` default lowered from 1,000 to
200, plus `--near-ceiling-terminated-episodes` / `--near-ceiling-transitions`
/ `--near-ceiling-num-envs` overrides and an explicit print before
collection starts, warning that several minutes of silence is expected.
200 is still ~5x smoke's floor and plenty for this dataset's actual job
(giving the union training set some exposure to the regime's feature
distribution) -- it was never meant to match the primary three datasets'
statistical power one-for-one.

### scripts/train_multi_checkpoint_predictor.py -- new script, densely-sampled checkpoints instead of 3 fixed points + 1 add-on

Follow-up to the unified-predictor experiment above: the user reported the
seed-3 generalization curve looked unchanged after training on
struggling+target+competent+near_ceiling ("same shape curves as before, no
good generalizations yet"). Leading hypothesis discussed with the user:
plain unweighted BCE (CLAUDE.md S8.3, unchanged) means gradient signal is
dominated by whichever source dataset has the most transitions, and the
~1M-transition struggling/target/competent sets simply outnumber a single
~200-episode near_ceiling add-on badly enough that its signal barely
registers, even though it's nominally "in" the training set.

Rather than reweighting the loss (which S8.3 explicitly rules out because
it would distort calibration, a reported metric), this script tries a
different fix: sample N=5 checkpoints evenly spaced by INDEX across one
training run's entire saved-checkpoint sequence (first and last always
included), collect a comparably-sized dataset from each, and pool them.
No single competence regime gets a numerical-majority vote in the pooled
set just because it happens to have a much higher base rate, and the
near-ceiling regime is represented by roughly the same volume of data as
the near-random one instead of a small bolted-on extra.

Reuses `collect_one` (collect_frozen.py), `concat_datasets` /
`EPISODE_ID_OFFSET` / `evaluate_with_per_source_breakdown`
(train_unified_predictor.py, imported rather than duplicated -- both are
phase-2 scripts, no CLAUDE.md boundary crossed) and
`discover_checkpoints` / `CHECKPOINT_ITER_RE`
(eval_predictor_generalization.py). Per-checkpoint collection uses the
same lowered `TERMINATED_PER_CHECKPOINT = 200` default as the near_ceiling
fix above, and for the same reason -- ANY of the 5 sampled checkpoints
here can land near the competence ceiling, not just a deliberately-chosen
last one, so the stall risk applies to the whole sweep, not one step.

Network size: CLAUDE.md S8.2/S12#16 fix the MLP architecture (2 hidden
layers, ELU, no RNN/attention/etc) for the GATED pipeline
(scripts/eval_predictor.py) specifically. This script is phase-2 and pools
roughly 5x a single gated dataset's transition count, so at the user's
request it defaults to `--hidden 256,256` instead of the spec's 128,128
-- still an MLP, still 2 ELU hidden layers, just wider. `--hidden 128,128`
reproduces the spec width exactly if a like-for-like comparison is wanted.

`scripts/eval_predictor_generalization.py` was extended (not a new file)
with a `--predictor-hidden` flag (default `128,128`, backward compatible
with every predictor saved before this change) so it can load a
wider-than-spec network's state dict; `load_frozen_predictor` now takes a
`hidden` parameter instead of hardcoding `(128, 128)`. This is
`eval_predictor_generalization.py`'s own file, not one of the CLAUDE.md-
governed scripts, so editing it is in-bounds per the phase-2 boundary
already established at the top of this section.

### scripts/train_stratified_predictor.py -- new script, replaces train_multi_checkpoint_predictor.py (ground-up rewrite per user request)

The multi-checkpoint attempt's own per-source test breakdown produced a
decisive negative result: `it16` (near-ceiling) ended up with 767,534 of
the 1,372,887 pooled test transitions -- 56%, ~5x every other source --
because `collect_one`'s stop rule (transitions AND terminated-episode
targets) runs far longer for a low-base-rate policy and therefore collects
far MORE data, not less. It still scored AUPRC 0.0056 against its own
0.0004 base rate -- essentially no better than chance for that regime. This
falsifies "the near-ceiling regime just needs more representation": it had
the most data of any source and still failed. The user asked for a
ground-up restart rather than another parameter tweak; this script is
that rewrite, replacing train_multi_checkpoint_predictor.py in the
notebook's Part F. train_unified_predictor.py and
train_multi_checkpoint_predictor.py are left in place (not deleted) as
documented negative-result experiments -- both scripts and this trail of
reasoning are the actual output of that line of investigation.

Structural changes, per the user's explicit design (this is a direct
transcription of their instructions, not a judgment call made on their
behalf):

1. **Collection stops on ONE simple per-policy transitions threshold, no
   terminated-episode target.** This makes collection time roughly
   base-rate-independent (transitions accumulate at ~envs_per_policy per
   env-step regardless of competence) and guarantees every stack is the
   same size by construction rather than by luck -- directly fixing the
   5x-imbalance mechanism identified above. Also addresses "there's way too
   much data": default total volume drops from the previous attempt's ~9M
   pooled transitions to 4 x 250,000 = 1,000,000, comparable to a single
   CLAUDE.md S7.3 dataset instead of ~9x one.
2. **All 4 policies collected simultaneously**, not sequentially: one env
   instance per checkpoint (struggling/target/competent/fluent), stepped
   round-robin inside one outer loop so every policy's rollout advances
   each iteration. Wall time is bounded by whichever policy is slowest to
   reach ITS OWN threshold, not by the sum of all four run one after
   another -- this is what "run the 4 frozen policies in parallel... until
   a certain threshold" asked for. Reuses `build_runner_from_checkpoint`
   (cartpole_term/runner.py) and `TerminationLabeler`
   (cartpole_term/predictor/labeling.py, CLAUDE.md S7.4, unchanged/already
   tested) once per policy; the round-robin stepping loop itself is new
   code (collect_one's loop is written for a single policy and isn't
   decomposable into "one step of one policy" without editing the gated
   file, which is out of bounds).
3. **Stratified minibatch sampling + macro-averaged early stopping.**
   Training keeps the 4 stacks as SEPARATE arrays (no pooling, no
   episode-id-offset bookkeeping needed during training) and every
   minibatch is built by drawing `batch_size // 4` positions at random from
   each stack independently (`train_balanced()`, this file -- deliberately
   NOT scripts/eval_predictor.py's or predictor/train.py's DataLoader,
   since that shuffles one pooled array and reproduces exactly the
   natural-frequency dominance this rewrite exists to avoid). Convergence
   is judged on the MEAN of the 4 stacks' own validation AUPRC each epoch,
   not one pooled number -- a pooled metric is exactly what let the
   multi-checkpoint attempt's headline AUPRC (0.8229) look fine while one
   entire regime had collapsed underneath it. Loss (plain BCE, no
   reweighting), optimizer (Adam, lr=1e-3), and architecture (2-hidden-layer
   ELU MLP, CLAUDE.md S8.2's structural constraint) are unchanged --
   only the batch CONSTRUCTION differs, which is the specific mechanism the
   user asked to fix, not the loss function itself.
4. **`fluent` (the 4th point, ~1000-step average) is a new, project-specific
   name** for a checkpoint CLAUDE.md's S7.1 table doesn't define (its three
   operating points top out at `competent`/400 steps). Selected the same
   mechanical way as S7.2 -- earliest checkpoint whose frozen-eval mean
   episode length clears `--fluent-threshold` (default 990 of the 1000-step
   cap) -- read from the run's own `eval_history.json`, keeping the
   "earliest, not closest, on the improving limb" reproducibility argument
   S7.2 makes for the other three points. This does NOT edit
   scripts/select_checkpoints.py; struggling/target/competent are read
   directly from its already-written outputs/selected_checkpoints.json.
5. **Network width reverted to the CLAUDE.md S8.2 default (128, 128)**
   (train_multi_checkpoint_predictor.py had bumped it to 256,256 to match
   a ~5x larger pooled dataset). With total volume back down to ~1M, that
   justification no longer applies; `--hidden` still overrides it.

`scripts/eval_predictor_generalization.py`'s `--predictor-hidden` flag
(added for the previous script) is reused unchanged here with `128,128`.

### scripts/record_predictor_stratified_video.py -- new script, one video showing the stratified predictor across all 4 of its own checkpoints

User request: a video of the stratified predictor (train_stratified_predictor.py's
output) "working on all four checkpoints." Straightforward extension of
scripts/record_video.py's record_predictor_overlay() (CLAUDE.md S10 item 2,
a completed gated deliverable, NOT modified here) from 1 checkpoint/1
operating-point-matched predictor to 4 checkpoints/1 shared predictor,
concatenated into a single mp4 with a per-segment title overlay
(struggling/target/competent/fluent) so it's visible at a glance which
regime is on screen.

Reuses record_video.py's already-tested `_build_video_env_cfg`,
`_sanity_check_renderer`, and `_composite_overlay_frame` via import rather
than duplicating them. The one deliberate exception is
`_ensure_egl_configured` -- copied verbatim instead of imported, because it
must run before mujoco/mjlab's first import anywhere in the process, and
depending on "importing record_video.py happens to trigger it early
enough" is a fragile ordering assumption; a plain top-of-file copy removes
that risk entirely.

Checkpoint paths and the decision threshold are read directly from
outputs/metrics_stratified_seed{seed}.json (written by
train_stratified_predictor.py) rather than re-derived (e.g. re-running
select_fluent_checkpoint), so the video is guaranteed to show the exact
checkpoints the loaded predictor was actually trained on, not a
possibly-drifted re-selection.

Each of the 4 segments is bounded by BOTH a failure-count target
(--min-failures-per-checkpoint, default 2) AND a wall-clock cap
(--max-seconds-per-checkpoint, default 20s) -- whichever comes first, same
"OR of stop conditions" structure as record_video.py's own overlay loop.
This matters specifically for `fluent`: its whole point is a base rate
near zero, so waiting for it to fail organically could run a very long
time or never terminate at all (the same silent-stall shape as every
near-ceiling collection issue logged above). Bounding it by time instead
means that segment simply shows ~20s of stable balancing with the
predictor's probability staying low throughout if it doesn't fail --
itself the correct, informative result for that regime, not a bug worked
around by force.

### Environment upgrade to Unitree Go1 velocity-flat -- new phase-2 excursion (outside CLAUDE.md's cartpole-only spec entirely)

User request, after confirming the cartpole termination predictor worked:
upgrade to a harder environment that's still cheap to train and fails
naturally, "just maybe with more training" (i.e. same methodology, not a
redesign). Chose mjlab's shipped Unitree Go1 flat-velocity task over the
Unitree G1 humanoid alternative: both ship with a built-in fall termination
(`bad_orientation`, confirmed via scripts/inspect_go1_task.py's discovery
run) and a tracking reward, but Go1 (quadruped, 12 DOF) is cheaper than G1
(humanoid) to train, and the user has an A100 rather than a Colab T4.

Everything below is genuinely outside CLAUDE.md's scope -- that document
specifies the cartpole project exclusively. The prime-directive discipline
(don't invent an mjlab API, verify against the real install, log every
deviation) is carried over as the working method for this excursion, not
because CLAUDE.md requires it here, but because it's the thing that made the
cartpole pipeline trustworthy and there's no reason to abandon it.

**1. `src/cartpole_term/predictor/model.py`: `TerminationPredictor` gained an
`obs_action_dim` constructor parameter**, defaulting to `OBS_ACTION_DIM = 6`
(cartpole's own 5 obs + 1 action). Every existing call site
(`TerminationPredictor(history=1)`, `(history=1, hidden=hidden)`, etc.)
constructs the exact same network as before -- verified by re-running the
full pure-Python test suite (`tests/test_labeling.py`, `test_split.py`,
`test_eval_predictor.py`, 22 tests) after the change, all still passing.
Go1's actor obs + action concatenate to roughly 60 dimensions (vs cartpole's
6), so a literal `6` could no longer be hardcoded once a second task needed
this file.

**2. `src/cartpole_term/runner.py`: `FrozenEvalRunner` gained three optional,
keyword-only constructor hooks** -- `env_cfg_factory`, `extra_scalar_metric_fn`,
plus a `task_id` param on the free function `build_runner_from_checkpoint` --
all defaulting to exactly cartpole's current values when omitted. This file
is the CLAUDE.md S6 frozen-eval mechanism itself; the user explicitly asked
that in-training eval keep running for Go1 too ("keep evaluation intuitive
... run the in-training eval runs every now and then"), and the only way to
reuse the SAME mechanism (not reimplement it) for a second task without
CLAUDE.md's S6.2's fixed-signature concern applying was to make the two
genuinely cartpole-specific parts pluggable:
  - `_build_eval_env` used to hardcode an import of
    `cartpole_balance_term_env_cfg`; it now calls `self._env_cfg_factory(play=False)`,
    which defaults to a staticmethod returning exactly that import (unchanged
    behaviour for cartpole).
  - The termination-cause breakdown used to accumulate into a hardcoded
    `{"pole_angle": 0.0, "cart_pos": 0.0, "time_out": 0.0}` dict. It's now
    discovered dynamically from whatever `"Episode_Termination/<cause>"` keys
    actually appear in `extras["log"]` each step -- a strict generalization
    (for cartpole this still finds exactly the same three keys, verified by
    reading through the accumulation logic; no live env was available to
    re-run the frozen-eval rollout itself in this sandbox, which has no
    mjlab/GPU -- flagging this as the one piece of the runner.py refactor
    that should get a real eval run on the GPU machine, not just re-read, to
    fully confirm cartpole's output is byte-identical).
  - `mean_abs_cart_pos` (the S2.6 ablation-support metric, meaningless for a
    task with no cart-centering reward) is now behind `extra_scalar_metric_fn`,
    an optional `(obs) -> (name, per_env_tensor) | None` hook; cartpole's
    default still returns `("mean_abs_cart_pos", obs["actor"][:, 0].abs())`
    unchanged, Go1's subclass returns `None`.
  - `build_runner_from_checkpoint`'s free-function default arguments
    (`runner_cls`, `env_cfg_factory`, `task_id` all `None` -> resolved to
    cartpole's values) mean every existing caller
    (`scripts/select_checkpoints.py`, `scripts/collect_frozen.py`) is
    unaffected; they don't pass the new keyword arguments at all.

This was judged safe to edit in place (rather than duplicated into a new
file, the pattern used for every OTHER phase-2 addition this session)
because it is a mechanism-preserving generalization with byte-identical
default behaviour for every existing cartpole call site, not a behavioural
change -- and because forking it would mean the S6 frozen-eval loop itself
(the highest-trust-required piece of the whole project after the labelling
code) exists as two copies that could drift. `scripts/train.py`,
`scripts/select_checkpoints.py`, and `scripts/collect_frozen.py` -- the
actual CLAUDE.md-gated cartpole deliverables -- were NOT touched.

**3. `src/cartpole_term/go1/` -- new subpackage, registers
`Mjlab-Go1-Velocity-Term`.** Mirrors `cartpole_term/__init__.py` +
`env_cfg.py` + `rl_cfg.py`'s pattern exactly, but with nothing to add to the
shipped env cfg (Go1's flat-velocity task already has its own fall
termination and reward -- there is no cartpole-style custom
terminations.py/rewards.py here). The one open gap: `register_mjlab_task`
requires both an `env_cfg` and a distinct `play_env_cfg`; mjlab's API for
loading a task's *play*-specific cfg (as opposed to its training cfg)
wasn't confirmed by scripts/inspect_go1_task.py at the time this was
written, so `play_env_cfg` is currently registered as the SAME object as
the training cfg. This doesn't affect training/frozen-eval/collection (S6.2.2's
rule against using play_env_cfg for frozen eval holds regardless), only
Go1 video recording later -- `go1_velocity_term_env_cfg(play=True)` raises
`NotImplementedError` with a pointer to fix this once the registry's real
play-cfg API is confirmed.

**4. The Go1 source task id is discovered by regex against
`mjlab.tasks.registry.list_tasks()`** (`/go1/i` + `/velocity/i` + `/flat/i`,
excluding `/getup/i`), never hardcoded as a guessed literal string --
CLAUDE.md S0.2's "never invent an API/name that hasn't been confirmed"
applied to task-id discovery specifically, since the exact registered string
couldn't be confirmed from documentation alone in this CPU-only sandbox.
Fails loudly, listing every registered task, if the match isn't exactly one.

**5. `scripts/inspect_go1_task.py` -- new discovery script**, same role as
`scripts/inspect_mjlab.py` / `scripts/inspect_video_and_termmgr.py` played
for cartpole at gate 0: read-only, dumps the real task id, timing
(decimation/timestep/episode_length_s), termination term names + time_out
flags, reward term names, observation group structure, and (in a live-env
section, GPU-only) actual obs/action tensor widths and confirms
`termination_manager.get_term(name)` works the same way it does for
cartpole. Must be run once on the GPU machine before trusting anything
downstream that depends on exact names/widths.

**6. `scripts/go1_train.py` / `go1_select_checkpoints.py` /
`go1_collect_frozen.py` / `go1_eval_predictor.py` -- new scripts, parallel
to the cartpole gated ones, none of which were edited.** Reuse every
dimension-agnostic piece directly: `cartpole_term.predictor.labeling`,
`.split`, `.train` unchanged; `scripts/select_checkpoints.py`'s
`select_from_history`/`find_latest_run_dir`/`select_any_for_smoke` imported
directly (no cartpole-specific logic in them); `scripts/eval_predictor.py`'s
`classification_metrics`/`mcfadden_r2`/`expected_calibration_error`/
`best_threshold_by_f1`/`median_detection_lead_time`/`split_dataset_by_split_fn`/
`make_inputs` imported directly (same reuse pattern
`eval_predictor_generalization.py` already established this session). What's
genuinely new per script:
  - `go1_select_checkpoints.py`: Go1's struggling/target/competent/fluent
    windows don't exist yet (no prior Go1 training curve). Defaults reuse
    cartpole's exact numbers (80/150/400/990 steps) as a starting point --
    not an arbitrary guess, since Go1's task ALSO fails via termination and
    ALSO caps at 1000 env steps per episode (confirmed via
    inspect_go1_task.py: `episode_length_s=20.0`, `decimation=4`,
    `timestep=0.005` -> 50 Hz -> 1000 steps), so "earliest checkpoint whose
    survival time crosses a window" is the same valid competence proxy. But
    contact-rich locomotion learns on a different curve than cartpole
    balance, so these are explicitly flagged provisional and overridable
    per-point via `--struggling/--target/--competent/--fluent TARGET
    TOLERANCE`, to be corrected after reading the first real
    `eval_history.json`.
  - `go1_collect_frozen.py`: failure-cause attribution no longer hardcodes
    `"pole_angle"`/`"cart_pos"`. It discovers every termination term
    registered with `time_out=False` on the live env cfg (any such term maps
    to `terminated`, never `truncated` -- CLAUDE.md S3.2.3's rule, which is a
    general mjlab convention, not cartpole-specific) and reads
    `termination_manager.get_term(name)` for each, joining multiple
    simultaneous causes with `"+"` the same way cartpole's `"both"` case
    worked. Expected to resolve to a single cause for Go1
    (`"fell_over"`-ish), but nothing assumes that count.
  - `go1_eval_predictor.py`: cartpole's `build_features()` hardcodes named
    semantic features (`cart_pos`, `cos_theta`, ...) at known obs indices.
    Go1's actor obs concatenates several IMU/joint/command terms whose exact
    order wasn't confirmed by inspect_go1_task.py (mjlab's
    ObservationManager decides concatenation order internally, and that
    script only lists term NAMES per group, not their final flattened
    order). Rather than guess an order and silently mislabel which raw
    dimension is which in the S9.3-analogue feature-screening table, this
    screens every raw obs/action dimension BY INDEX (`obs_0`, `obs_1`, ...,
    `action_0`, ...) plus two generic derived magnitudes (obs L2 norm,
    action L2 norm), and picks the single-best-feature baseline (CLAUDE.md
    S9.2 item 4's analogue) by whichever index actually screens best rather
    than assuming a name like cartpole's `|theta|`. Still answers the actual
    diagnostic question (S9.3's real purpose: "is anything in this dataset
    predictive, and does the labelling look sane") without assuming
    unverified semantics.
  - `go1_train.py` deliberately does NOT port `scripts/train.py`'s
    bracketed-window auto-densify/seed-perturbation retry logic (the
    mechanism that re-trains at a finer eval_interval, or a perturbed seed,
    if an operating-point window gets skipped between two measurements). It
    depends on knowing the real operating-point windows first, which is
    exactly what this first Go1 run is FOR -- adding retry logic tuned to
    windows that don't exist yet would be guessing twice. Can be ported once
    Go1's own windows are set from real data, if it turns out to be needed.

**7. Local verification actually performed, without mjlab or a GPU** (this
sandbox has neither, same constraint as cartpole's own CLAUDE.md S0.2): (a)
re-ran the full pure-Python test suite after the model.py change, all 22
tests still pass; (b) fabricated a synthetic Go1-shaped dataset (48-dim obs,
12-dim action, 400 episodes, one deliberately-injected predictive raw
dimension) and ran it through `go1_eval_predictor.py`'s
`evaluate_one_dataset` end-to-end on CPU, confirming: the 60-dim
`obs_action_dim` is correctly inferred and threaded through
`TerminationPredictor`; the MLP clearly beats the base-rate baseline
(AUPRC 0.251 vs 0.066 baseline); the injected-signal dimension (`obs_3`) is
correctly identified as the top-screening feature (AUC-ROC 0.84) and
selected for the single-best-feature baseline; the dynamically-discovered
cause name (`"fell_over"`) correctly appears in the per-cause breakdown.
This does not touch anything mjlab-specific (env cfg discovery, live obs
widths, termination_manager behaviour, actual PPO training) -- those remain
unverified until `scripts/inspect_go1_task.py` and `scripts/go1_train.py`
are run on the GPU machine, which this sandbox cannot do.

**Known gaps, to be resolved from `scripts/inspect_go1_task.py`'s real
output rather than guessed further:** (a) `play_env_cfg` wiring (item 3
above); (b) whether `termination_manager.get_term(name)` behaves for Go1
exactly as it does for cartpole (the API itself is already confirmed to
exist via cartpole's own collect_frozen.py; only the Go1-specific term
names and per-env boolean shape need confirming); (c) Go1's actual
struggling/target/competent/fluent windows, currently cartpole's numbers
reused as a provisional starting point (item 6 above); (d) GPU memory/step-
time cost of 4096 Go1 envs with contacts enabled on the A100 -- CLAUDE.md's
own S6.2's "verify memory and kernel-compile cost on the first run and log
it" directive applies here too, and `go1_train.py` logs CUDA memory
before/after `learn()` for exactly this reason.

### go1_select_checkpoints.py: selection metric switched from mean_action to sampled_action (found on the first real Go1 training run)

Cartpole's checkpoint selection (S7.2) uses `mean_action.mean_episode_length`
(deterministic, zero exploration noise) as the competence axis, and that's
what `go1_select_checkpoints.py` copied initially. On the first real Go1
run (seed 0, 100 iterations, 4096 envs), this immediately failed: at
iteration 0 `mean_action_len` was already 737.9/1000, rising further from
there -- the `struggling`(80)/`target`(150)/`competent`(400) windows were
literally unreachable (the curve starts above all three and only goes up),
raising `select_from_history`'s "no checkpoint qualifies" error correctly
and loudly, exactly as designed.

Root cause: Go1's action space is a joint-position OFFSET from a default
standing pose (`use_default_offset=True`, confirmed via
scripts/inspect_go1_task.py). Under the deterministic mean-action eval
(near-zero net output early in training), the policy effectively just holds
that default pose -- which happens to be a fairly stable "do nothing"
standing behaviour for a quadruped, independent of whether it has learned
anything about the actual task (tracking commanded linear/angular
velocity). Confirmed via the training log: `Episode_Reward/track_linear_velocity`
stayed ~0.0003-0.03 across the same iterations where `mean_action_len` was
already 740-980 -- i.e. high survival with ~zero task reward, proving the
policy wasn't "competent," it was just standing still and not falling over.
Cartpole has no equivalent trivial-stable-state: gravity pulls the pole
down immediately, so surviving under mean_action genuinely required having
learned to balance. That structural difference doesn't hold for Go1.

Fix: `go1_select_checkpoints.py` now selects (and re-measures, S7.2.4) on
`sampled_action.mean_episode_length` instead -- the STOCHASTIC eval, which
does NOT have this problem (full Gaussian exploration noise on top of a
12-DOF joint-position controller makes "do nothing" fall over almost
immediately, so it actually reflects learned competence) and, not
coincidentally, is the exact same distribution `scripts/go1_collect_frozen.py`
always samples under for the actual predictor dataset anyway -- so it's
arguably the more correct axis for this project's purpose regardless of the
mean_action issue. The re-measurement step was changed to match
(`stochastic_output=True`, not cartpole's deterministic re-measurement) --
selecting on one distribution and re-measuring on a different, saturated
one would make every operating point's reported remeasurement read ~950+
regardless of which was picked, which defeats the point of re-measuring at
all.

`fluent`'s default window was also updated from the initial guess
(990 +- 10) to 950 +- 30, per the user's read of the actual plateau -- Go1's
shipped task has ongoing random pushes and domain randomization (unlike
cartpole once mastered), so even a well-trained policy plateaus somewhat
below the full 1000-step ceiling rather than converging to it exactly.

`scripts/select_checkpoints.py` (the cartpole gated file) was NOT edited --
`select_from_history`'s hardcoded `"mean_action"` field lookup is correct
for cartpole and stays that way. `go1_select_checkpoints.py` has its own
local `select_from_history_by_field(history, target, tolerance, field)`
instead of importing the cartpole one, keyed by a module-level
`SELECTION_FIELD = "sampled_action"` constant.

### scripts/go1_record_video.py -- new script, visual sanity check for the AUPRC~0.98 result

User request, prompted by the Go1 predictor's AUPRC being consistently
~0.98 across all four operating points ("these seem too good to be true"):
a video, same purpose as record_video.py's predictor_overlay.mp4, so the
result can be checked by eye instead of taken on faith from the metrics
JSON alone. Same mechanism (probability bar/threshold/ground-truth-label
overlay per frame, stop once --min-failures real falls are captured),
reusing record_video.py's already-tested `_composite_overlay_frame` and
`_sanity_check_renderer` via import; record_video.py itself is not
modified.

One real gap surfaced by writing this: `go1_velocity_term_env_cfg(play=True)`
still raises `NotImplementedError` (see the earlier Go1-upgrade entry above)
-- this script uses `play=False` instead, so domain randomization and
random pushes are visible in the video. Not a bug, and arguably a more
honest demo (perturbations happening on camera), but noted so the visual
result isn't misread as "the task has no randomization."

### scripts/go1_watch_episodes.py -- new script, plain rollout video with no predictor dependency

User wanted a training-only run (1000 iterations, no dataset collection, no
predictor) followed immediately by watching 5 episodes. `go1_record_video.py`
can't serve this: it hard-requires a trained predictor's `.pt`/metrics
`.json` (for the overlay) and stops on a *failure count*, not an *episode
count* -- neither applies here. Wrote a stripped-down sibling instead:
takes `--checkpoint-file` directly (no `go1_selected_checkpoints.json`
dependency either), renders N episodes back to back with the deterministic
mean action (matching CLAUDE.md S10.1's `policy_final.mp4` convention for a
clean demo; `--stochastic` opts into the sampled/behaviour-policy action
instead), and writes a plain mp4 with no probability overlay. Reuses
`record_video.py`'s `_sanity_check_renderer` via import; does not need
`_composite_overlay_frame` since there is nothing to composite.

Episode boundary detection copies `go1_record_video.py`'s already-proven
pattern exactly: keep stepping and check `terminated`/`truncated` per step,
no manual `env.reset()` between episodes (mjlab's `ManagerBasedRlEnv`
auto-resets internally on the same step a `done` fires -- confirmed by
`go1_record_video.py` never resetting between its counted failure
episodes, which only works if this is true).

Also relevant to this same request: `scripts/go1_train.py` sets
`rl_cfg.save_interval = args.eval_interval` (so every saved checkpoint has
a matching frozen-eval measurement, S6.2.1's reasoning) -- meaning a large
`--eval.interval` (to skip the periodic frozen eval entirely, since this
run doesn't need operating-point windows) also skips most periodic
checkpoint saves. Confirmed safe by reading `FrozenEvalRunner.learn()`
(`src/cartpole_term/runner.py`): the loop does one *unconditional* final
`self.save(...)` after the `for` loop regardless of `save_interval`, so
`--eval.interval 1000 --max-iterations 1000` still guarantees a final
checkpoint (`model_999.pt`) on disk, plus exactly one (nearly-free, at
iteration 0) frozen eval instead of the usual 100.

### scripts/go1_watch_episodes.py -- added per-episode commanded-velocity printout

After the 1000-iteration run, the recorded policy visibly walked sideways.
Two candidate explanations that look identical on video but aren't: (a) the
shipped Go1 task tracks a full omnidirectional command `[vx, vy, wz]` (the
last 3 of the 48 actor obs dims, confirmed layout per
`outputs/go1_task_inspection.json`), so a nonzero sampled `vy` makes
sideways motion the *correct* response, not a defect; (b) the policy hasn't
learned to couple heading with commanded direction, and is shuffling
sideways regardless of what it's told -- plausible given `foot_clearance`
carries a `-2.0` penalty while `air_time` (the term that would otherwise
reward a proper alternating swing/stance rhythm) ships at weight `0.0` in
mjlab's own default Go1 reward, and only 1000 of the shipped
`max_iterations=10000` default has run.

Added `_print_command(obs, prefix)` -- prints `(vx, vy, wz)` from
`obs["actor"][0, 45:48]` at the start of every episode -- so the next
recording directly shows which of (a)/(b) is happening, without needing a
full video overlay. No change to the reward/termination/observation
pipeline itself; this is instrumentation only.

### cartpole_term/go1/env_cfg.py -- air_time reward enabled at weight 0.25 (first real Go1 reward deviation)

User asked, after seeing the sideways-walking video, to research the
optimal way to train Go1 and set real hyperparameters rather than guess.
Read mjlab's actual Go1 velocity source on GitHub
(mujocolab/mjlab@main, `src/mjlab/tasks/velocity/config/go1/env_cfgs.py`
and `.../velocity_env_cfg.py`) rather than relying on memory:

- Confirmed every previously-assumed fact still matches (task id pattern,
  48/12 obs/action dims, single `fell_over` failure cause, 50 Hz timing,
  all 13 reward terms and their shipped weights) -- strong cross-check that
  the earlier GPU-side `inspect_go1_task.py` findings are accurate.
- Confirmed `rl_cfg` (hidden_dims=(512,256,128) actor/critic, ELU,
  num_steps_per_env=24, lr=1e-3 adaptive, gamma=0.99, lam=0.95,
  entropy_coef=0.01, clip=0.2, 5 epochs/4 minibatches, max_iterations=10_000)
  is already loaded verbatim via `load_rl_cfg(TASK_ID)` -- nothing to tune
  here; this already IS mjlab's own validated recipe for this exact task,
  same "use the library's tested implementation" principle CLAUDE.md applies
  to cartpole's PPO config.
- Found the actual mechanism behind the sideways-walking report:
  `track_linear_velocity` compares against `asset.data.root_link_lin_vel_b`
  -- BODY-frame velocity -- while `reset_base`'s spawn event randomizes yaw
  uniformly over the full circle every episode
  (`pose_range["yaw"] = (-3.14, 3.14)`). A policy walking correctly
  "forward" relative to itself can therefore point in an arbitrary
  WORLD-frame direction each episode; viewed through mjlab's world-fixed
  demo camera, this alone can look like sideways/wandering motion
  independent of gait quality. Addressed in `scripts/go1_watch_episodes.py`
  (pins spawn yaw for the demo only, prints actual vs. commanded body-frame
  velocity) rather than in the shared env cfg -- randomizing yaw during
  training/collection/eval is correct and intentional (it's what makes the
  policy heading-agnostic) and must not be touched.
- Found a second, real contributor: mjlab's own Go1 config sets `air_time`
  (`mdp.feet_air_time` -- rewards a foot for time spent airborne between
  contacts within `[threshold_min, threshold_max]` seconds, i.e. an
  alternating gait rhythm) to weight `0.0` for BOTH the flat and rough Go1
  variants -- present in the reward manager but inactive. Checked whether
  any other mjlab robot config uses a nonzero value (G1 humanoid also ships
  it at `0.0`) -- no in-repo precedent. Checked mjlab's own GitHub
  Discussion #220 (a direct mjlab-vs-IsaacLab performance comparison for
  this exact flat Go1 task) instead: it reports IsaacLab's own reward
  config for the equivalent task uses `air_time` weight **0.25**. That is
  the value used here -- not invented, not swept, IsaacLab's own default
  for the reference implementation mjlab's Go1 port is based on.

Change: `go1_velocity_term_env_cfg()` (`cartpole_term/go1/env_cfg.py`) now
overrides `cfg.rewards["air_time"].weight = 0.25` (constant
`GO1_AIR_TIME_WEIGHT`) after loading mjlab's shipped cfg, with a loud
`RuntimeError` if a future mjlab version removes the `"air_time"` key
rather than silently no-op'ing. Every other reward term/weight,
termination, observation, command range, and domain-randomization event is
untouched. This is the first and (as of this entry) only reward-weight
deviation from mjlab's shipped Go1 config anywhere in this project; logged
per this project's own discipline of never silently retuning a reward.

Also discovered, from the same Discussion #220: real-hardware precedent for
wall-clock cost at `max_iterations=10_000`, 4096 envs -- roughly 1.5 hours.
Used to set the user's expectations for the next training run rather than
guessing at timing.

### scripts/go1_horizon_sweep.py -- new script, sweeps the label horizon for one fixed checkpoint

User question: is the current default `PREDICTION_HORIZON`-equivalent for
Go1 (16 steps, inherited from `--predictor.horizon`'s default in
`go1_collect_frozen.py`/`go1_eval_predictor.py`, = 0.32s at Go1's confirmed
50 Hz) actually close to the earliest a fall can be predicted, or would a
longer horizon buy meaningfully more lead time (in seconds) without
sacrificing AUPRC/AUROC? Not answerable from the existing per-operating-
point runs, since all of them (`struggling`/`target`/`competent`/`fluent`)
were collected at the same fixed horizon=16 -- horizon and policy
competence were never varied independently.

This is explicitly NOT a case of "tuning the predictor by trying variants
and keeping the best on the test split" (CLAUDE.md S12#17, which this
project treats as good practice even for phase-2 work): the point isn't to
search for the best-scoring config and silently adopt it, it's to measure
lead-time-vs-accuracy as a function of one variable (horizon) for a FIXED
checkpoint, print every point, and let the user pick after seeing the
tradeoff -- same reporting posture as `go1_select_checkpoints.py`'s
mechanical-but-visible checkpoint selection, not a hidden model-selection
loop.

Design: reuses `go1_collect_frozen.collect_one` (takes `horizon` as a
plain argument already, no changes needed) and
`go1_eval_predictor.evaluate_one_dataset` (same TerminationPredictor
architecture/loss/optimizer/early-stopping, same AUPRC/AUROC/median-
lead-time functions used everywhere else) completely unmodified, called
once per horizon value in a list, each producing its own frozen dataset +
trained predictor + metrics row. A fresh rollout per horizon (rather than
one rollout relabelled post hoc at multiple horizons) was chosen because
`collect_one`'s stopping condition and the ring-buffer labeller are both
horizon-dependent and reusing the already-tested function outweighs the
wall-clock cost of re-collecting (cheap regardless, ~15-20s/1M
transitions per the gate-8 timings already in README.md).

Defaults to a lightweight probe (1024 envs / 50k transitions / 100
terminated episodes) rather than the full gate-8 targets, since the goal
is a quick first-pass comparison across ~8 horizon values; `--full` opts
into the same 4096-env/1M-transition/1000-episode targets
`go1_collect_frozen.py` uses, for confirming a shortlist. Default horizon
grid (`[4, 8, 16, 24, 40, 60, 80, 120]` steps = `[0.08, 0.16, 0.32, 0.48,
0.80, 1.20, 1.60, 2.40]` seconds) brackets the current default on both
sides and reaches past the 1.6s figure that prompted the question, rather
than only testing values on one side of it.

Does not touch `PREDICTION_HORIZON`/`GO1_AIR_TIME_WEIGHT`/any other
constant -- it is a standalone measurement tool. Whatever horizon the user
picks after reviewing `outputs/go1_horizon_sweep_{label}_seed{N}.json`
would be adopted as a documented, evidence-based change to
`go1_collect_frozen.py`'s and `go1_eval_predictor.py`'s
`--predictor.horizon` default in a follow-up edit -- not automatically by
this script.

### Follow-up: sweep both `target` and `competent`, not `competent` alone

First version of the plan pointed the sweep at only the `competent`
checkpoint, reasoning it dodges `struggling`'s near-random abruptness and
`fluent`'s collection-stall risk. User's follow-up question (rightly)
noted `target` dodges the same two problems just as well, so restricting
to one point was arbitrary rather than principled -- and this project's
own stated approach (CLAUDE.md S1: "policy competence is swept, not
guessed... yields predictor performance as a function of policy
competence") argues for measuring the horizon tradeoff at more than one
competence level too, since there's no a priori reason the AUPRC/lead-time
curve vs. horizon has to look the same at both.

Fixed: `go1_horizon_sweep.py` gained a `--label` argument (tags every
intermediate dataset/model file and the final output JSON, default
`"sweep"`) specifically so the same checkpoint-sweep mechanism can be run
twice -- once per operating point -- without one run's files silently
overwriting the other's. `go1_horizon_sweep.ipynb` now selects both
`target` and `competent` from the same 200-iteration run (free, since
`go1_select_checkpoints.py` computes all four points in one pass
regardless) and runs the sweep against each, producing
`go1_horizon_sweep_target_seed0.json` and
`go1_horizon_sweep_competent_seed0.json` for review side by side.
`struggling` stays excluded -- its near-random, abrupt failures are a
different regime (see the entry above this one), not a missing data
point for the same question. `fluent` stays excluded for the
already-documented collection-stall reason.

### Follow-up: --eval.interval 10 too coarse to catch `target` -- real training curve moves faster with air_time added

Actual seed-0 run (`--max-iterations 200 --eval.interval 10`, this
session's `air_time=0.25` reward config) produced:
`(0, 120.7), (10, 136.1), (20, 464.7), (30, 822.6), (40, 884.2), (50,
919.0), (60, 936.7) ... plateau ~930-985 through iteration 190`.
`go1_select_checkpoints.py` correctly found `struggling` (it=0) but raised
its designed loud error for `target` (window `[140,300]`): no measured
iteration ever fell there -- the curve jumps from 136.1 (it=10) straight
to 464.7 (it=20), skipping the entire window in one 10-iteration gap. Not
a tolerance problem (`--target TARGET TOLERANCE` can't select a value that
was never measured) -- a resolution problem. Confirms the pre-air_time
baseline run's convergence speed (near-990 by ~iteration 99, per
NEXT_STEPS_GO1.md) does not carry over now that `air_time` is in the
reward: this run is already near-plateau by iteration ~60, roughly 40%
faster.

Fixed: lowered the notebook's default to `--max-iterations 100
--eval.interval 2` (50 measurements, matching the original NEXT_STEPS_GO1.md
100-iteration/interval-5 run's spirit but finer, since this run's curve
moves faster and the earlier interval-10 attempt proved 10 iterations can
already skip a whole competence window). 100 iterations remains enough
given the observed plateau by ~60. If `target` still isn't found at this
resolution, the next step is finer still (`--eval.interval 1`), not a
wider tolerance -- same reasoning.

### Follow-up: --max-iterations 100 wastes GPU time evaluating an already-plateaued policy

User (rightly) pushed back that `--eval.interval 2` across the full 100
iterations "would take so long." CLAUDE.md S6.2's frozen eval runs every
env to full episode completion (not a fixed step budget), for BOTH the
mean-action and sampled-action distributions, every single eval call --
so eval cost scales directly with episode length, and once this run's
curve plateaus (~iteration 60 onward, episode length ~930-990 per the
data above), each of those evals is close to the maximum possible cost
for a policy that has already stopped changing qualitatively. Evaluating
every 2 iterations through that whole plateau bought nothing.

Fixed: lowered further to `--max-iterations 50 --eval.interval 2` (26
measurements, not 50) -- `competent`'s window was already satisfied by
iteration 20 (464.7) in the interval-10 run, so nothing past ~50 is
needed for `target`/`competent`. Kept at 50 rather than cutting to 30
because `go1_select_checkpoints.py` requires ALL FOUR operating points
(including `fluent`, window `[850,1000]`) to succeed or it raises --
iteration 30 only reached 822.6 (short of 850) in the interval-10 run,
but iteration 50 reached 919.0, so 50 keeps real margin against run-to-run
PPO variance for a point this particular sweep doesn't even use, but
`go1_select_checkpoints.py`'s all-or-nothing selection loop still
requires.

## Multi-robot scaffolding: Go1 + G1 humanoid inflicted-failure + protective-fall extension (Gate 0)

New phase, planned and approved this session (see the approved plan at
implementation time -- not reproduced here in full; this entry covers
only Gate 0's actual code). End goal: one Colab notebook, parameterized
by `--robot {go1,g1}`, that trains a real walking policy for each robot,
inflicts failures via environment perturbation, feeds the existing
predictor, then trains a separate "protective fall" policy that takes
over once the predictor fires. Rollout order: Go1 fully end-to-end first,
then G1. This entry covers Gate 0 only: replacing the audit-identified
Go1-only hardcoding with one parameterized mechanism shared by both
robots, before any training happens.

**Audit findings that motivated this** (read directly from the real code
this session, not assumed): `cartpole_term/go1/env_cfg.py`'s
`discover_source_task_id()` used a naive `re.search(r"go1", ...)` with no
word boundary; every `scripts/go1_*.py` output path hardcoded a literal
`"go1_"` string prefix rather than deriving it from the task/robot;
`scripts/go1_watch_episodes.py` hardcoded Go1's exact 48-dim obs-slice
layout. None of these would have generalized safely to a second robot
without either duplicating-with-edits (fragile, drifts over time) or a
shared parameterized mechanism.

**New:**
- `src/cartpole_term/common/robot_profile.py` -- frozen `RobotProfile`
  dataclass (name, registered task id, task-id discovery
  candidates/regex fragments, reward overrides, expected obs/action
  dims, obs layout, vulnerability regions for the later protective-policy
  work, step Hz, max episode steps). `GO1`/`G1` instances; `PROFILES`
  dict; `get_profile(name)`. `G1`'s `expected_obs_dim`/`expected_action_dim`
  are deliberately `None` -- NOT pre-filled with the ~286/29 figures this
  session's GitHub source research produced, since those are static-source
  estimates, not confirmed against a real install. They get filled in via
  a follow-up edit to this file only after
  `scripts/inspect_robot_task.py --robot g1` has run live on the GPU
  machine (CLAUDE.md S0.2: never invent an API/value that hasn't been
  confirmed against the installed version).
- `src/cartpole_term/common/discovery.py` -- `discover_task_id(profile)`:
  tries `profile.task_id_candidates` as exact matches against
  `list_tasks()` first, else falls back to a **word-boundary** regex AND
  of `task_id_must` fragments minus `task_id_exclude`
  (`(?<![a-zA-Z0-9])frag(?![a-zA-Z0-9])`), fixing the audit's flagged
  risk. Verified locally (without mjlab, by stubbing `list_tasks()`) that
  the word-boundary patterns correctly distinguish `"go1"` from `"g1"` in
  both directions against realistic task-id strings, and that the
  exact-candidate fast path resolves each robot correctly. Fails loud,
  listing every registered task id, on 0 or >1 matches -- same contract
  Go1's original function had.
- `src/cartpole_term/g1/{__init__.py,env_cfg.py,rl_cfg.py,runner.py}` --
  mirrors `cartpole_term/go1/`'s four files exactly, generalized via
  `common/discovery.py` + `RobotProfile.reward_overrides` instead of a
  second hand-copied regex. G1's `air_time` reward override (0.25) is
  declared in `RobotProfile.reward_overrides` on the strength of this
  session's GitHub source review (G1 ships `air_time` at weight 0.0, same
  as Go1) -- `g1_velocity_term_env_cfg()` still asserts the term actually
  exists on the live-loaded cfg before mutating it (loud `RuntimeError`
  otherwise), same defensive pattern Go1's own override uses.
  `play_env_cfg` is the same unresolved-gap placeholder Go1 already has
  (same training cfg reused, `play=True` raises `NotImplementedError`).
- `scripts/inspect_robot_task.py --robot {go1,g1}` -- generalizes
  `scripts/inspect_go1_task.py`. Beyond the original's dumps
  (terminations/rewards/observations/rl_cfg/live obs+action widths), this
  version explicitly: (a) flags any termination term whose name contains
  `"collision"` (resolves the open question of whether G1's
  `self_collisions` reward also terminates the episode -- unconfirmed
  either way by source review alone), (b) prints `push_robot`'s params
  verbatim (confirms live that Go1 and G1 really do share the identical
  event, rather than trusting the source read alone), (c) dumps the
  asset's live body-name list (needed to fill `vulnerability_regions`
  with real names for the later protective-policy reward, not guessed
  ones). The body-name lookup is wrapped in its own try/except (mjlab's
  exact attribute path for this wasn't confirmed by this session's
  research) so a wrong guess there doesn't take down the rest of the
  script's independently-useful output, matching every other section's
  existing per-block try/except resilience.

**Edited:** `src/cartpole_term/go1/env_cfg.py`'s `discover_source_task_id`
body replaced by a call into `common/discovery.py` with Go1's
`RobotProfile` -- must reproduce the exact same result
(`Mjlab-Velocity-Flat-Unitree-Go1`) as before this edit; verify by
re-running `scripts/inspect_go1_task.py` on the GPU machine as a
regression check before trusting any new Go1 run downstream of this edit.

**Verified in this sandbox (no GPU/mjlab needed):** all new/edited files
pass `python3 -m py_compile`. `RobotProfile`/`discovery.py`'s logic
(word-boundary matching, exact-candidate fast path, `get_profile`'s
loud-fail on an unknown robot name) tested directly by loading the two
modules standalone and stubbing `mjlab.tasks.registry.list_tasks()` --
real mjlab behavior still needs live confirmation on the GPU machine, this
only proves the pure-Python matching logic itself is correct. Full
existing test suite (`.venv/bin/python -m pytest tests/`, 33 tests) passes
unchanged, confirming this edit didn't disturb the shared predictor code
path.

**Not yet done (subsequent gates, not part of this entry):** push-
perturbation tuning, full-convergence training runs, the 4-point
checkpoint/dataset/predictor sweep under the tuned push, the protective
"small-fall" policy, the predictor-to-protective-policy handoff, and the
final Colab notebook. All require the GPU machine and have not been
attempted yet.

### Gate 0 closed: both robots live-confirmed on the GPU machine (A100), 2026-09-13

`scripts/inspect_robot_task.py --robot go1` and `--robot g1` both ran to
completion. Results, all live not estimated:

- **Go1 regression check passed**: task id `Mjlab-Velocity-Flat-Unitree-Go1`,
  actor obs (4,48), action_dim 12, terminations `{time_out: True,
  fell_over: False}`, push_robot params as expected -- generalizing
  discovery via `cartpole_term.common.discovery` changed nothing about
  Go1's actual behavior.
- **G1's obs dim is 99, NOT the ~286 this session's earlier GitHub source
  review estimated.** That estimate wrongly assumed the flat-terrain
  variant carries a 187-ray `height_scan` term; it doesn't -- height_scan
  is rough-terrain-only, and `Mjlab-Velocity-Flat-Unitree-G1` has none.
  Live actor group: `base_lin_vel(3) + base_ang_vel(3) +
  projected_gravity(3) + joint_pos(29) + joint_vel(29) + actions(29) +
  command(3) = 99`. Action dim 29 confirmed. This is exactly the kind of
  mistake CLAUDE.md S0.2 exists to catch -- a static-source estimate
  stood in for a live-confirmed fact for one review cycle and was wrong;
  `RobotProfile.G1.expected_obs_dim`/`expected_action_dim` are now set to
  the live numbers (99/29), not the estimate.
- **No collision-named termination exists on G1** -- terminations are
  `{time_out: True, fell_over: False}`, identical structure to Go1.
  `self_collisions` is reward-only (weight -1.0), never terminates the
  episode. Resolves Gate 0's open question: Gate 5's protective-policy
  episode handling needs no special-casing for a self-collision
  termination on G1, because there isn't one.
- **`push_robot`'s params are byte-identical between Go1 and G1**, live-
  confirmed (not just from GitHub source): `mode="interval",
  interval_range_s=(1.0,3.0)`, same six-axis `velocity_range`. Confirms
  the plan's assumption that both robots share the exact same shipped
  perturbation event.
- **mjlab's shipped `rl_cfg.max_iterations` is 30,000 for G1, vs. Go1's
  10,000** -- a real, live-confirmed difference, not assumed equal.
  Directly affects Gate 1's per-robot training budget: G1 needs 3x Go1's
  iteration count by mjlab's own shipped recipe, before any early-stop
  consideration.
- **Real body names, no head or hand link on either robot**: Go1 has 13
  bodies (`trunk` + 4 legs x `{hip,thigh,calf}`, no separate foot/ankle
  body); G1 has 30 (`pelvis`, legs `{hip_pitch,hip_roll,hip_yaw,knee,
  ankle_pitch,ankle_roll} x2`, `waist_yaw_link`/`waist_roll_link`/
  `torso_link`, arms `{shoulder_pitch,shoulder_roll,shoulder_yaw,elbow,
  wrist_roll,wrist_pitch,wrist_yaw} x2`). The SafeFall/arXiv:2512.01336
  literature's head/torso/pelvis vulnerability-region template assumed a
  head and hands exist -- neither robot's model has one, so
  `RobotProfile.vulnerability_regions` for Gate 5 is adapted to real names
  rather than copied verbatim: Go1 high=`trunk`, medium=hips+thighs,
  low=calves (nearest normal ground contact, standing in for "feet");
  G1 high=pelvis/torso/waist (standing in for "head" as the core/critical
  region since none exists), medium=hips/knees/arms, low=ankles.
- Go1's critic obs group carries `foot_contact_forces` (12-dim, 4 feet x
  3-axis) and G1's carries the same (6-dim, 2 feet x 3-axis) -- both
  already exposed as observation terms, not just internal sensor state;
  worth remembering as a possibly-simpler alternative to raw
  `ContactSensor` queries for Gate 5's reward if the actor/critic obs
  already carries what's needed.
- Both robots' static (non-live) observation-group dump hit
  `TypeError("'ObservationGroupCfg' object is not iterable")` in
  `inspect_robot_task.py`'s cfg-level section -- a pre-existing,
  non-fatal bug inherited verbatim from `inspect_go1_task.py`'s original
  code (same try/except-caught pattern), not a regression from
  generalizing it. The LIVE env section (real `ObservationManager` dump)
  already gives the full, correct obs-term breakdown for both robots, so
  this is cosmetic and left unfixed.

Gate 0 is closed for both robots. Next: Gate 1 (full-convergence training,
Go1 first, mjlab's stock unmodified `push_robot`).

## Gate 1: robot_train.py -- generalized full-convergence training, both robots, mjlab's stock push

New: `src/cartpole_term/common/backend.py` (`get_robot_backend(robot)`,
`RobotBackend`) -- discovers a robot subpackage's `TASK_ID`,
`{robot}_velocity_term_env_cfg`, `{robot}_velocity_term_ppo_runner_cfg`,
and its single `*FrozenEvalRunner` class by the naming convention
`cartpole_term.go1`/`cartpole_term.g1` both already follow, instead of
every future `robot_*.py` script re-deriving the same import/getattr
chain by hand. Fails loud, naming every missing/ambiguous attribute, if a
robot subpackage doesn't follow the convention -- never a bare
`AttributeError`. Verified in this sandbox (no mjlab needed, since this is
pure `importlib`/`getattr`) against a synthetic fake robot package tree:
the success path resolves all four pieces correctly, and a deliberately
broken fake package (missing rl_cfg factory, no runner class) produces
the expected named-attribute error.

New: `scripts/robot_train.py --robot {go1,g1} --seeds ...` -- Gate 1 of
the multi-robot inflicted-failure predictor + protective "small-fall"
policy plan (see the approved plan; not reproduced here). Generalizes
`go1_train.py` via `get_robot_backend`, plus a direct port of
`scripts/train.py`'s already-validated bracketed-window auto-densify and
seed-perturbation retry mechanism (`_find_bracketed_windows`,
`MAX_SEED_PERTURBATION_ATTEMPTS`), adapted to key off
`sampled_action.mean_episode_length` (not `mean_action` -- carries
forward `go1_select_checkpoints.py`'s already-learned "do nothing is
falsely stable under deterministic action" lesson) and the 4-point
contiguous-partition windows already established there
(`[90,140]/[140,300]/[300,850]/[850,1000]`), rather than cartpole's own
windows/field.

Verified the ported bracketed-window logic directly (extracted, pure
dict-based, no mjlab needed) against the EXACT real curve this session
observed earlier that skipped `target`
(`(0,120.7),(10,136.1),(20,464.7),(30,822.6),(40,884.2),(50,919.0),(60,936.7)`)
-- correctly flags `['target']` as bracketed, matching the manual
diagnosis that led to widening `--eval.interval` by hand earlier in this
session. A clean curve that passes through every window correctly flags
nothing. This is now automatic in `robot_train.py` rather than requiring
a human to notice and re-run with a finer interval.

**Push perturbation is never touched by this script**, per the user's
explicit choice logged in the plan: `env_cfg` comes straight from
`backend.env_cfg_factory(play=False)` with no override applied anywhere
in this file. `--max-iterations` left unset runs to mjlab's own shipped
default -- live-confirmed this session as 10,000 for Go1, 30,000 for G1 (a
real difference, not assumed equal -- see the Gate 0 entry above).

Full existing test suite (`.venv/bin/python -m pytest tests/`, 33 tests)
still passes -- this gate touched no shared predictor code.

Not yet run on the GPU (needs the user). Rollout order per the plan: Go1
`--seeds 0` first, end to end through every later gate, before G1.

### Bug fix: get_robot_backend's runner discovery false-matched the base class itself

Real run on the GPU machine (`robot_train.py --robot go1`) failed at
`get_robot_backend`: `"exactly one *FrozenEvalRunner class in
cartpole_term.go1.runner (found [<class
'cartpole_term.runner.FrozenEvalRunner'>, <class
'cartpole_term.go1.runner.Go1FrozenEvalRunner'>])"`.

Cause: the original filter was `name.endswith("FrozenEvalRunner")` over
`vars(runner_mod)`. `cartpole_term/go1/runner.py` (and `g1/runner.py`)
both do `from cartpole_term.runner import FrozenEvalRunner` at module
level -- a completely ordinary import for using the base class -- which
binds the name `FrozenEvalRunner` into that module's own namespace too.
The literal string `"FrozenEvalRunner"` trivially ends with
`"FrozenEvalRunner"`, so the base class became a second, false candidate
alongside the real `Go1FrozenEvalRunner` subclass, tripping the
"exactly one" check that exists specifically to catch ambiguity.

Fixed: filter by actual subclassing (`issubclass(obj, FrozenEvalRunner)
and obj is not FrozenEvalRunner`) instead of a name-suffix guess -- this
is what the check was always trying to express, just implemented as a
string-matching approximation that happened to collide with normal
Python import behavior. Verified against an exact reproduction of the
real bug (a fake runner module that also imports a fake base class into
its own namespace, exactly like the real files do): the subclass filter
correctly returns only the real subclass. Full test suite still passes.

Not yet re-run on the GPU -- next step is re-running
`scripts/robot_train.py --robot go1 --seeds 0` with this fix.

## Plan revision: single checkpoint per robot, push-severity axis instead of competence axis

User reconsidered the earlier "keep the full 4-point competence sweep"
call (itself an override of the assistant's original one-checkpoint
recommendation) after walking through what training one great walker,
then sweeping push severity against just that one checkpoint, would look
like. Reversed: the plan's Gates 1-4 are now single-checkpoint,
push-severity-swept, not 4-competence-point-swept. Full reasoning and the
updated gate text live in the plan file
(`~/.claude/plans/snug-booping-gosling.md`), not reproduced here.

Practical consequence, genuinely useful: **Gate 1 needed no code
changes.** `scripts/robot_train.py` (already built) already supports
"train straight through, no dense eval" via existing flags --
`--eval.interval 1000` (not equal to `--max-iterations`, which would
leave zero intermediate checkpoints if a long unattended Colab run gets
interrupted -- the exact risk already identified earlier this session)
with no `--early-stop-patience`. Verified the ported bracketed-window
auto-densify logic stays inert with only 1-2 eval records (nothing to
bracket between fewer than 2 points), so it doesn't misfire under this
usage -- no need to strip it out for this simpler path.

New: `scripts/measure_push_failure_rate.py --robot {go1,g1}
--checkpoint-file <path>` (Gate 2, mostly as originally designed --
sweeps push-velocity-range scale factors against ONE real checkpoint,
stochastic actions, prints every point, doesn't auto-pick a winner).
Carries a new note (not yet resolved by a number, just flagged): a push
strong enough to reliably fail a good policy could do so almost
instantly, which would likely still give great AUPRC but a short,
less-useful lead time for the later protective-policy handoff to act on
-- Gate 3 should carry forward multiple severities (not just whichever
clears the failure-rate target band fastest) so Gate 4 can show this
tradeoff instead of hiding it.

## Sideways-walking root cause found: shipped command distribution, not a hyperparameter issue

User reported a FULL 10,000-iteration Go1 run also walked "smoothly
sideways at times" and asked for research into Go1-specific
hyperparameter tuning on mjlab. Read `mjlab/tasks/velocity/velocity_env_cfg.py`
and `mjlab/tasks/velocity/mdp/velocity_command.py` directly (GitHub,
`mujocolab/mjlab@main`) rather than guessing or retuning PPO blind.

Root cause, confirmed from real source, not the PPO recipe: the shipped
`"twist"` `UniformVelocityCommandCfg` samples `lin_vel_y` (lateral
velocity) from the exact same range as forward velocity
(`ranges.lin_vel_y=(-1.0,1.0)`, same magnitude as `lin_vel_x`), and only
`rel_forward_envs=0.2` (20%) of environments ever get a forward-only
command -- the rest get real lateral velocity, a heading-tracking command
(`rel_heading_envs=0.3`), or a stand-still command
(`rel_standing_envs=0.1`). The `command_vel` curriculum term
(`mdp.commands_vel`) only widens `lin_vel_x`/`ang_vel_z` over training
steps (0 -> 5000x24 -> 10000x24 env-steps); it never touches `lin_vel_y`
at all -- lateral commands are present at full magnitude from iteration 0
through the full 10,000-iteration run, unchanged by run length.

Conclusion: "smooth sideways walking" is the policy correctly tracking a
command that legitimately included lateral motion -- a general
velocity-tracking policy doing exactly what it's rewarded to do, not a
symptom of bad PPO hyperparameters. Did NOT retune any hyperparameter --
mjlab's shipped PPO recipe was already confirmed earlier this session to
match published precedent (IsaacLab's own values for the equivalent
task), and there is no evidence here pointing at PPO tuning as the cause.
Retuning against a non-hyperparameter problem would have been exactly the
kind of unfounded "I tried X and it worked better" guess-tuning this
project's own discipline (CLAUDE.md S5.3, voluntarily extended to phase-2
work) prohibits.

Fix applied: forward-only commands for DEMO/VIDEO PURPOSES ONLY, never
for training/push-tuning/collection/predictor-training. mjlab already
ships a purpose-built mechanism for exactly this
(`UniformVelocityCommandCfg.rel_forward_envs`, docstring: "important for
stair climbing") -- read `velocity_command.py`'s real resampling code
(`UniformVelocityCommand._resample_command`/`_update_command`) to confirm
`rel_forward_envs=1.0` alone is not sufficient on its own: `_update_command`
runs every step (not just at resample time) and still re-derives yaw from
heading error for any env flagged `is_heading_env`, and zeroes the whole
command for any env flagged `is_standing_env` -- both would silently undo
the forward-only override for a fraction of envs on later steps if left
at shipped values. New `src/cartpole_term/common/commands.py`
(`force_forward_only_commands(cfg)`) sets all three fields together
(`rel_forward_envs=1.0, rel_heading_envs=0.0, rel_standing_envs=0.0`),
raises loudly if the `"twist"` command term is missing/renamed. Verified
against a fake cfg object (both the override-applies and the
missing-term-raises paths).

New: `scripts/robot_watch_episodes.py --robot {go1,g1} --checkpoint-file
...` -- generalizes `scripts/go1_watch_episodes.py` via
`cartpole_term.common` (RobotProfile/backend) instead of Go1-only
literals, fixing the audit's flagged gap (the original script hardcoded
`BASE_LIN_VEL_SLICE`/`COMMAND_SLICE` as standalone constants). Forward-only
by default (`--natural-commands` opts back into the real distribution).
Also filled in `RobotProfile.{GO1,G1}.obs_layout` with the exact,
live-confirmed actor-obs term order for both robots (from
`inspect_robot_task.py`'s `ObservationManager` dump this session) --
Go1: `base_lin_vel(0:3)/base_ang_vel(3:6)/projected_gravity(6:9)/
joint_pos(9:21)/joint_vel(21:33)/actions(33:45)/command(45:48)`; G1: same
order, widths 3/3/3/29/29/29/3 -> `joint_pos(9:38)/joint_vel(38:67)/
actions(67:96)/command(96:99)`. This is the shared, robot-keyed
replacement for `go1_watch_episodes.py`'s own hardcoded slices.

Full test suite (33 tests) still passes.

## Gate 1: --resume-from -- train once, checkpoint, decide, continue (no retrain-from-scratch)

User wants to train Go1 exactly once: stop at iteration 5000, watch a
video, then decide whether to continue to 10,000 (resuming, not
restarting) or keep the 5000-iteration checkpoint as final -- and never
have to retrain Go1 again after that.

Read `rsl_rl`'s real source (`leggedrobotics/rsl_rl@main`,
`runners/on_policy_runner.py` + `algorithms/ppo.py`) directly rather than
guessing whether a genuine resume (not just an actor-weight reload) is
supported:
- `OnPolicyRunner.save(path)` writes `PPO.save()`'s dict (actor + critic +
  optimizer state) plus `"iter": self.current_learning_iteration`.
- `OnPolicyRunner.load(path, load_cfg=None, ...)`: if `load_cfg is None`,
  `PPO.load()` defaults to `{"actor": True, "critic": True, "optimizer":
  True, "iteration": True, "rnd": True}` -- loads everything and returns
  `True` for `load_iteration`, which makes `OnPolicyRunner.load()` set
  `self.current_learning_iteration = loaded_dict["iter"]`.
- `OnPolicyRunner.learn(num_learning_iterations, ...)` already computes
  `start_it = self.current_learning_iteration; total_it = start_it +
  num_learning_iterations` -- it was ALREADY resume-aware; nothing to add
  there.

This confirms a genuine resume (not the actor-only `load_cfg={"actor":
True}` `build_runner_from_checkpoint` already uses for inference-only
eval/collection) is fully supported by calling `runner.load(path,
load_cfg=None, ...)` before `learn()`.

**New:** `scripts/robot_train.py` gained `--resume-from <checkpoint>`.
`--max-iterations` now means a TOTAL target (works the same whether
resuming or not) -- when resuming, the script computes `remaining =
target - runner.current_learning_iteration` and calls
`learn(num_learning_iterations=remaining, ...)`. Continues writing into
the SAME run directory the checkpoint came from (derived from the
checkpoint's parent path) rather than starting a new timestamped one, so
tensorboard/`resolved_args.json` stay one continuous record.

**Real bug caught and fixed while implementing this:** `FrozenEvalRunner.
__init__` always starts `self._eval_history` as an empty list, and
`_write_eval_history()` does a full overwrite (`path.write_text(...)`),
not an append. Without intervention, the FIRST frozen eval during a
resumed segment would silently truncate `eval_history.json`, losing every
record from before the resume. Fixed: on `--resume-from`, if
`eval_history.json` already exists in the target log dir, its contents
are loaded into `runner._eval_history` before `learn()` runs, so appends
continue correctly across the resume boundary.

**Known, accepted quirk (not fixed, matches rsl_rl's own upstream
semantics exactly):** `current_learning_iteration` is set to the LAST
processed iteration index (not iteration+1), and `learn()`'s loop range
starts AT that index again on resume -- so the checkpoint's final
iteration gets reprocessed once more after a resume (one extra gradient
update, ~0.01% redundancy on a 10,000-iteration run). Not patched: this
is rsl_rl's own established resume convention, not a bug introduced
here, and "reuse the library's tested behavior verbatim" is this
project's standing rule everywhere else too.

Notebook (`part_a_gate0_gate1.ipynb`) updated to the actual flow: train
to 5000 -> forward-only demo video -> explicit decision point -> optional
resume-to-10000 cell (with its own video) -> an explicit
"FINAL_CHECKPOINT" download cell (the one artifact meant to make
retraining Go1 unnecessary going forward) -> Gate 2 push sweep against
whichever checkpoint was kept.

Full test suite (33 tests) still passes -- this gate touched no shared
predictor code, only the Go1/G1-generic trainer script.

## Incident: bracketed-window auto-densify misfired, causing an unwanted full retrain

Real run on the GPU machine: `robot_train.py --robot go1 --seeds 0
--max-iterations 5000 --eval.interval 1000` completed its 5000-iteration
run, then immediately started training AGAIN from scratch at
`eval_interval=1` (evaluating every single iteration) with no user
action. Caused by the bracketed-window auto-densify mechanism ported
from `scripts/train.py` in an earlier version of this file: Go1's
episode length jumps from ~110 (iteration 0) to ~900+ (iteration 1000) --
already observed at a finer grid earlier this session -- which, checked
against `DEFAULT_OPERATING_POINTS`, reads as `target` AND `competent`
(and possibly `fluent`) all being "skipped over" between those two eval
records, triggering the "retrain once at eval_interval=1" branch
automatically.

Root cause of the root cause: this mechanism was ported "for safety" when
this file was first written, on the reasoning that "with only 1-2 eval
records there's nothing to bracket" -- checked against a *smoke*-sized
history, never against the actual `--eval.interval 1000` / 5-record
history a real run produces. That reasoning was wrong for the real
invocation, and the mechanism has no purpose in this design regardless:
this plan's Gate 1-4 (REVISED) uses a single checkpoint, no
competence-window selection, so there is no consumer for
`DEFAULT_OPERATING_POINTS`/`SELECTION_FIELD` in this file at all.

**Fixed by deletion, not patching**: removed `_find_bracketed_windows`,
`_select_from_history`, `DEFAULT_OPERATING_POINTS`, `SELECTION_FIELD`,
`MAX_SEED_PERTURBATION_ATTEMPTS`, `SEED_PERTURBATION_STRIDE`, and the
`eval_interval_override`/`_perturbation_attempt` plumbing that only
existed to support the recursive retry calls -- the whole mechanism, not
a guard around it. `train_one_seed` is back to a plain, one-shot function
(`(args, seed) -> Path`), same shape `go1_train.py`'s always had.

**Practical consequence for the user's run in progress:** the original
5000-iteration checkpoint was NOT lost -- `learn()` for the first
(non-recursive) call fully completes, including its own final
unconditional save, before the retry logic (now removed) would ever have
run. The wasteful retrain was writing into a SECOND, separately
timestamped run directory. Safe to interrupt; the correct checkpoint is
in the FIRST (earlier-timestamped) `*_seed0` directory, not
`sorted(glob(...))[-1]` if a second, aborted directory now also exists.

Full test suite (33 tests) still passes.

## Process fix: download checkpoint immediately after training, before the video step

Direct consequence of the incident above: the user's Colab runtime
disconnected between the 5000-iteration training finishing and the video
step running, and the runtime was not recoverable -- the checkpoint was
lost even though training itself had succeeded, because nothing had been
downloaded yet.

`part_a_gate0_gate1.ipynb` reordered: the checkpoint is now downloaded
immediately after `robot_train.py` finishes, in its own cell, before the
video/decision-point cells run at all. Same fix applied after the
optional resume-to-10000 cell. Purely a notebook ordering change, no
script changes.

## robot_watch_episodes.py: --commanded-speed, pin the demo to a fixed forward speed

User wants to see the Go1 checkpoint walking at the highest forward speed
the task ever asks for, not the naturally-sampled forward-only range
(0.3-1.0 m/s, per `rel_forward_envs`'s own clamp).

Confirmed live (re-read `velocity_env_cfg.py`'s `command_vel` curriculum,
already cited earlier this session): the absolute highest forward speed
the shipped `command_vel` curriculum ever commands is **3.0 m/s** (final
`velocity_stages` entry, `{"step": 10000*24, "lin_vel_x": (-2.0, 3.0)}`).
Important nuance, not obvious from the checkpoint alone: this curriculum
tracks the ENV's own live step counter, not something a checkpoint's
weights carry -- a freshly-built demo env always restarts it from the
step-0 stage (`lin_vel_x=(-1.0,1.0)`) regardless of which checkpoint is
loaded. So reaching 3.0 m/s in a demo requires explicitly pinning
`ranges.lin_vel_x`, not just trusting natural resampling in a fresh env.
Also worth naming honestly: the policy has had very little actual
training time at exactly this speed (only from iteration 10,000 onward),
so struggling/falling at this pinned speed would be a real, informative
result, not a bug to chase.

**New:** `cartpole_term.common.commands.force_forward_only_commands`
gained an optional `fixed_speed` parameter -- pins
`cfg.commands["twist"].ranges.lin_vel_x = (fixed_speed, fixed_speed)` in
addition to the existing forward-only fields. Composes cleanly with
`UniformVelocityCommand`'s own forward-env override
(`vel_command_b[fwd_ids,0] = vel_command_b[fwd_ids,0].abs().clamp(min=0.3)`
is a no-op on an already-fixed positive value). New module constant
`MAX_FORWARD_COMMAND_LIN_VEL_X = 3.0`, cited from the curriculum source
above, not invented.

**New:** `scripts/robot_watch_episodes.py --commanded-speed <float>` --
ignored if `--natural-commands` is set; otherwise pins every episode's
forward command to that exact speed instead of the natural forward-only
sampled range.

Verified `force_forward_only_commands(cfg, fixed_speed=...)` against a
fake cfg object -- pins `ranges.lin_vel_x` correctly, and confirmed the
default (`fixed_speed=None`) path leaves `ranges` untouched (first
attempt at this check had a bug in the test's OWN fake class -- a
class-level dict shared across instances made two supposedly-independent
fake cfgs alias the same object; fixed the test, not the real code, and
re-verified both paths are independent as expected).

Full test suite (33 tests) still passes.

## Housekeeping: reorganized the top-level directory

Repo root had accumulated 5 loose `.ipynb` files, 2 loose trained
checkpoints, and two old ~250MB leftover download bundles
(`full_results (1).zip`/`(2).zip`, Colab's own browser-download collision
naming). Reorganized, no code changes:

- `notebooks/` -- all 5 `.ipynb` files (`colab_entry.ipynb` moved via
  `git mv` since it was tracked; the other 4 were untracked, moved with
  `mv`). `README.md`'s one reference to `colab_entry.ipynb`'s path
  updated to `notebooks/colab_entry.ipynb`.
- `checkpoints/` -- `model_5000.pt`, `model_9999.pt` (the Go1 checkpoints
  from this session's training runs).
- `archive/` -- the two old `full_results (*).zip` bundles, kept rather
  than deleted (not this session's call to make).
- `.gitignore` gained explicit `checkpoints/` and `archive/` entries for
  clarity (both were already covered by the existing `*.pt`/`*.zip`
  patterns, so this changes nothing about what's tracked -- confirmed via
  `git status`: both new directories don't appear at all, as expected).

`cartpole_term_code.zip` and the core project files (`CLAUDE.md`,
`DEVIATIONS.md`, `README.md`, `NEXT_STEPS_GO1.md`, `pyproject.toml`)
stayed at the top level -- all actively referenced/regenerated, not
clutter.

## Bug fix: --commanded-speed silently overwritten by the command_vel curriculum

Real run: `--commanded-speed 3.0` had no visible effect -- printed
commands were `+0.300`/`+0.999`/etc., exactly the natural forward-only
range, not the pinned 3.0.

Root cause, confirmed from `mjlab/tasks/velocity/mdp/curriculums.py`'s
real `commands_vel` function: the `command_vel` curriculum term
re-asserts `cfg.ranges.lin_vel_x`/`lin_vel_y`/`ang_vel_z` from
`env.common_step_counter` against `velocity_stages`, and it runs every
step (CurriculumManager's normal cadence) -- silently overwriting
whatever `force_forward_only_commands` had just set. A fresh demo env
always sits at `common_step_counter=0`, so it kept re-imposing the
shipped stage-0 range (-1.0,1.0), and the forward-only clamp then
produced exactly the 0.3-1.0 values observed -- looked like the pin was
ignored, but it was being actively overwritten every step, not ignored
at setup.

Fixed: `force_forward_only_commands(cfg, fixed_speed=...)` now also pops
the `command_vel` curriculum term (`cfg.curriculum`) when `fixed_speed`
is given, so nothing re-asserts the range afterward. Warns (doesn't
crash) if that term isn't found, naming the actual keys present, in case
mjlab's naming ever changes. Verified against a fake cfg: the pin
sticks with the curriculum removed, the curriculum is left untouched on
the natural (no fixed_speed) path, and a missing curriculum term warns
instead of crashing.

Full test suite (33 tests) still passes.

## New: scripts/go1_wild_push_predictor.py -- collect under a wild push + train the predictor, one script

User's next step, using the fully-trained 10,000-iteration Go1
checkpoint: inflict much stronger perturbations than mjlab ships, and
train a termination predictor offline on the resulting failures.
Explicitly requested as one new, standalone script rather than the
multi-robot plan's separate Gate 2 (push tuning)/Gate 3 (collection)/
Gate 4 (predictor) steps.

Reuses, unmodified: `cartpole_term.go1.runner.build_go1_runner_from_checkpoint`
(loads the frozen checkpoint), `cartpole_term.predictor.labeling.
TerminationLabeler` (same ring-buffer horizon-censoring labeller every
dataset in this project uses, horizon=16 -- CLAUDE.md S7.4's cartpole
default, kept unchanged, not the abandoned horizon-sweep from earlier in
this session), `cartpole_term.predictor.{model.TerminationPredictor,
train.train_predictor, split.split_episodes}` (same architecture/loss/
optimizer as every other predictor here), and
`scripts/eval_predictor.py`'s already-generic metric functions (same
ones `scripts/go1_eval_predictor.py` already imports).

New in this file only (nothing else needs to inject a push override
between building the collection env and starting rollout):
`collect_under_wild_push` -- adapted from
`scripts/go1_collect_frozen.py`'s `collect_one`, with one addition: right
after `runner._build_eval_env()`, scales the shipped `push_robot`
event's `velocity_range` by `--push-scale` (default 8.0x) before
rollout starts. 8.0 is not a fresh guess -- it's the same upper end of
the escalation ladder `scripts/measure_push_failure_rate.py` already
sweeps before considering a different perturbation mechanism
(`apply_body_impulse`). Prints the shipped vs. wild `velocity_range` and
a running base-rate every 200 steps during collection, so the actual
resulting failure rate is measured and visible, not assumed.

Training (`scripts/robot_train.py`) is untouched -- the checkpoint being
loaded was trained under mjlab's shipped, mild push like every checkpoint
in this project; only this script's OWN collection env sees the wild
version, same train/collect separation as the rest of the plan.

Collects the full gate-8-style targets by default (>=1M transitions AND
>=1000 terminated episodes, 4096 envs) since this is meant to produce
the real predictor dataset, not a probe -- `--smoke` available for a
quick sanity pass. Saves the frozen dataset (npz+json sidecar) to disk
before training the predictor, so the dataset stays a durable,
independently-reusable artifact even though this script trains on it
immediately afterward (CLAUDE.md S1's "retrain fifty times without
re-running collection" property is preserved, not lost, by combining the
steps into one invocation).

Verified `scaled_velocity_range` directly (pure function, no mjlab
needed): 8x the shipped Go1 `push_robot` config gives kicks up to ±4 m/s
linear, ±6.24 rad/s yaw. `py_compile` clean. Full test suite (33 tests)
still passes -- this script touches no shared predictor code, only adds
a new standalone caller.

Not yet run on the GPU (needs the user, and the 10,000-iteration
checkpoint).

## go1_wild_push_predictor.py: added the pre-collection verification it skipped, grounded in literature

User's real run (8x push scale) gave only 13 falls in 7 minutes -- far
too sparse to ever reach the 1000-terminated-episode target -- and
correctly called out that this script never checked the push severity
against THIS checkpoint before committing to the full, expensive
collection run (unlike `measure_push_failure_rate.py`, which exists
specifically for that check). User also asked, generally: for anything
new or uncertain, check published literature and follow it rather than
guessing.

**Applied to both the fix and the escalation ceiling:**

- New `verify_push_scale()`: runs a quick, bounded probe (300 episodes
  per candidate, not the full collection targets) via
  `measure_push_failure_rate.run_one_config` (imported directly -- same
  measurement that script's own sweep already uses, not a
  reimplementation) BEFORE the expensive collection starts. Escalates
  `--push-scale` (bumped default 8.0 -> 16.0, since 8x is now known
  insufficient) through `(1x, 1.5x, 2x, 3x)` until finding one that
  clears `MIN_ACCEPTABLE_FAILURE_RATE` (10%, this project's own
  established target band from `measure_push_failure_rate.py`, not a
  literature number). `--skip-verify` opts back into the old
  no-check behavior, discouraged.

- **Where the literature check actually changed the design**: re-read
  `mjlab/envs/mdp/events.py`'s real `push_by_setting_velocity` docstring
  -- it's an instantaneous, MASS-INDEPENDENT velocity teleport, not a
  physical force. Cross-checked against the three perturbation-based
  failure/fall papers already researched this session (SafeFall
  arXiv:2511.18509; the self-protective falling policy, arXiv:2512.01336;
  GYF, arXiv:2310.04828): every one of them induces falls with a
  sustained FORCE (Newtons, held over a duration) or an actual
  momentum-transfer collision -- none of them scale a velocity multiplier
  into extreme territory the way this script's escalation ladder was
  about to do by default. Capped `VERIFY_ESCALATION_MULTIPLIERS` at 3x
  (so at most 48x the shipped push, ~±8-24 m/s depending on how far up
  the ladder it needs to go) rather than continuing to escalate velocity
  indefinitely -- past that point it stops resembling any real
  disturbance and risks being simulation-nonsense rather than a
  meaningful "wind gust." If even 48x can't clear a bare safety floor
  (2%), `verify_push_scale` now raises loud, naming the literature-
  grounded next step explicitly: switch to `apply_body_impulse`
  (confirmed real signature this session: a stateful class,
  `force_range`/`torque_range` in Newtons/Nm, `duration_s`/`cooldown_s`,
  optional `body_point_offset`) with `force_range` up to ~850N -- the
  self-protective-falling paper's own real induced-fall ceiling -- rather
  than scaling velocity further. NOT implemented this turn: wiring up
  `apply_body_impulse` requires adding the event term to the env cfg
  BEFORE `ManagerBasedRlEnv` construction (confirmed by reading mjlab's
  EventManager pattern for stateful class-based terms, which are
  instantiated once at env-build time) -- mutating `raw_env.cfg.events`
  AFTER construction, the pattern this script already uses for
  `push_robot`'s params, only works for stateless FUNCTION-based terms
  like `push_by_setting_velocity`, not for adding a new class-based term
  after the fact. Flagged as a real follow-up, not silently glossed over.

Verified `verify_push_scale`'s control flow directly (mocked failure
rates, all three branches: stops at first candidate clearing target;
warns-and-proceeds with the best found if only the safety floor clears;
raises if nothing does). Full test suite (33 tests) still passes.

Not yet re-run on the GPU.

## MAJOR bug fix: push_robot scale override was silently never applying at all

Real run showed the smoking gun: `verify_push_scale`'s escalation ladder
(16x/24x/32x/48x) measured EXACTLY the same failure rate
(0.0033333333333333335 = 1/300) at every single scale -- bit-for-bit
identical across four supposedly different push strengths. That's not
"coincidentally similar," it's proof the scale was never actually
changing anything.

Root cause, confirmed directly from `mjlab/managers/event_manager.py`'s
real source:

```python
class EventManager(ManagerBase):
  def __init__(self, cfg: dict[str, EventTermCfg], env):
    self.cfg = deepcopy(cfg)
```

`EventManager` deep-copies the event config at construction time, and
every event dispatch (`apply()`) reads from that private copy, never
from `env.cfg.events` again. Both `measure_push_failure_rate.py`'s
`run_one_config` and `go1_wild_push_predictor.py`'s
`collect_under_wild_push` were mutating `raw_env.cfg.events["push_robot"]
.params["velocity_range"]` AFTER `runner._build_eval_env()` had already
constructed the env (and therefore its `EventManager`, which had already
deep-copied the ORIGINAL, unscaled params). The mutation landed on a
dead object -- the live simulation was running the shipped, unmodified
push the entire time, at every scale ever tried, including the original
"8x gives 13 falls in 7 minutes" result. That number describes the
SHIPPED push's natural failure rate against this checkpoint, not 8x
anything.

**This invalidates every push-scale measurement taken with these two
scripts up to this point.** Nothing about the checkpoint or the
escalation ceiling reasoning was wrong -- the measurement mechanism
itself was broken.

Note this is a DIFFERENT bug from the earlier `--commanded-speed`
curriculum issue (`force_forward_only_commands`'s fix) -- that one was
correctly structured (mutating the cfg BEFORE
`ManagerBasedRlEnv(cfg=...)` construction, inside
`robot_watch_episodes.py`'s own env-building function) and was genuinely
about a LIVE runtime curriculum re-asserting a value every step, not a
construction-time deepcopy. Confirmed via a standalone check that that
fix's structure was already sound and needs no further change.

**Fixed in both files** by moving the override into the `env_cfg_factory`
closure itself -- applied BEFORE `ManagerBasedRlEnv` is ever constructed,
so `EventManager`'s deepcopy captures the already-scaled values:
- `measure_push_failure_rate.py`'s `run_one_config` now builds a local
  `env_cfg_factory(play=False)` that wraps `backend.env_cfg_factory`,
  applying `config_params["velocity_range"]` to the cfg it returns,
  passed into `build_runner_from_checkpoint(..., env_cfg_factory=...)`
  instead of mutating `raw_env.cfg` afterward.
- `go1_wild_push_predictor.py`'s `collect_under_wild_push` now calls the
  GENERIC `cartpole_term.runner.build_runner_from_checkpoint` directly
  (with `runner_cls=Go1FrozenEvalRunner`, `task_id=TASK_ID`) instead of
  the Go1-specific `build_go1_runner_from_checkpoint` wrapper, which
  hardcodes its `env_cfg_factory` with no override hook -- needed the
  generic entry point specifically to inject the wrapped factory.

Verified the fix directly: simulated the exact closure pattern with a
fake cfg object, confirmed the scaled value survives a `copy.deepcopy`
applied afterward (mimicking `EventManager.__init__`'s real behavior) --
the wild-scaled range is what a live EventManager would actually see and
dispatch from now, not the shipped default.

Full test suite (33 tests) still passes. Not yet re-verified on the GPU
-- next step is re-running the verification ladder with this fix and
confirming the failure rate actually VARIES across scales this time.

## Bug fix: stale variable name (base_velocity_range) left over from the EventManager fix

Real run: the deepcopy fix worked correctly (16x measured ~9.2% failure
rate over the first 200 collection steps, 4288 terminated episodes --
real variation this time, not the flat 1/300 from before), but
`collect_under_wild_push` crashed building its sidecar dict:
`NameError: name 'base_velocity_range' is not defined`. Simple oversight
from the fix itself: the local variable holding the shipped range was
renamed to `shipped_velocity_range` (populated via a closure over
`wild_env_cfg_factory`), but one later reference at the sidecar
construction site was still using the old name. Fixed (one-line rename
at the call site); `verify_push_scale`'s own identically-named local
variable in a different function is unrelated and was already correct.
Full test suite (33 tests) still passes.

## New: scripts/go1_multiseverity_predictor.py -- pool multiple push severities, matching SafeFall's own approach

After the predictor's first real result (AUPRC 0.70 vs baseline 0.10,
AUROC 0.85, median lead 10/16 steps -- meaningfully weaker than this
project's natural-failure Go1 predictors, expected given how sudden a
16x push-induced fall is), user asked to improve it and pick whichever
of three options (gentler single push / history-stacking / pooled
multi-severity) matches the reference literature most closely.

Picked pooled multi-severity: SafeFall (arXiv:2511.18509) builds its own
training data from a COMBINATION of multiple failure factors (several
push magnitudes among them), not one fixed condition -- the same idea
applied here to the one factor already wired up. History-stacking was
also considered but is real, unimplemented work (`eval_predictor.py`
explicitly raises "not implemented" for `predictor.history > 1`, never
approximates it) -- left as a documented follow-up, not attempted this
turn.

New: `scripts/go1_multiseverity_predictor.py`. Collects full gate-8-sized
datasets (1M transitions/1000 terminated episodes each, same targets as
every other dataset here) at 3 push severities (default 4x/16x/32x,
spanning what's already measured this session), pools them with a
per-source `episode_id` offset (`10_000_000 * source_index` -- same fix
already applied once before in `scripts/train_unified_predictor.py`, so
combining datasets can't silently collide unrelated episodes under the
same integer id), trains ONE predictor on the pooled result, and reports
a PER-SEVERITY test-set breakdown alongside the pooled numbers -- same
"don't silently pool and dilute, show whether it actually helps every
regime" discipline as that same prior script.

Reuses `scripts/go1_wild_push_predictor.py`'s already-fixed
`collect_under_wild_push` (correct EventManager-deepcopy-safe push
override) and `build_features_generic`/`feature_screening_generic`
directly via import -- no duplicated collection logic.

Verified the episode_id-offset pooling logic directly (fake datasets,
confirmed no id collisions across sources and correct per-row severity
labeling). Full test suite (33 tests) still passes.

Not yet run on the GPU.

## Multi-severity pooling result: worse than the best single severity, not better

Real run (`go1_multiseverity_predictor.py`, 4x/16x/32x pooled):
POOLED test_base_rate=0.0735, AUPRC=0.6399 (baseline=0.0735), AUROC=0.8402,
median_lead=9.0 steps. Per-severity breakdown told the real story:

| severity | base rate | AUPRC | median lead |
|---|---|---|---|
| low (4x) | 1.49% | 0.8348 | 14.0 steps (0.28s) |
| medium (16x) | 10.04% | 0.6949 | 10.0 steps (0.20s) |
| high (32x) | 11.22% | 0.5990 | 6.0 steps (0.12s) |
| pooled | 7.35% | 0.6399 | 9.0 steps (0.18s) |

Clean, monotonic trend: gentler push -> better predictor AND longer lead
time, matching the physical intuition that a gentle push produces a more
gradual, previewable loss-of-balance (closer to the natural-fall
predictors' own ~0.98 AUPRC results) while a violent push is closer to an
instant, discontinuous event. Pooling did NOT help -- the pooled AUPRC
(0.64) is worse than the low-severity-alone result (0.83), dragged down
by the harsher tiers rather than benefiting from their diversity. This is
exactly the "does pooling help every regime or just dilute" question this
script was built to surface, answered honestly: for this specific
severity range, it dilutes.

User's follow-up concern, also valid: the low-severity (4x) base rate
(1.49%) may be too sparse to give a future recovery-policy training stage
(Gate 5, not yet built) enough failure exposure to work with, even though
it's the best PREDICTOR. Not yet resolved with a number -- an intermediate
severity (around 8x, between the measured 4x and 16x points) is the next
thing to actually measure, not guess, given the clean monotonic trend
above suggests it should land at a reasonable base rate while keeping
more gradualness than 16x.

**Session interruption note:** the Colab runtime running this experiment
disconnected before the per-severity npz files (`_low_`/`_medium_`/`_high_`)
or the multiseverity predictor/metrics were downloaded -- only the pooled
`.npz` survived (browser default download landed at the repo root again,
same pattern as the earlier model checkpoint downloads). The trained Go1
policy checkpoint (`checkpoints/model_9999.pt`) was already safe from a
prior download and did not need re-uploading. Re-collection at a fresh
severity is needed rather than reusing anything from that run except the
policy checkpoint itself.

## New: scripts/go1_wildpush_video.py -- predictor-overlay video for the wild-push pipeline

User asked for a video "of everything" -- walking, the wild push,
predictor firing, and the fall. Adapted from
`scripts/go1_record_video.py` (the natural-fall predictor-overlay video
for the original operating-point sweep), reusing its already-tested
`_composite_overlay_frame`/`_sanity_check_renderer` helpers via import
unchanged. Three real differences from that file, not just a copy:

1. The wild push is baked into the env cfg factory BEFORE
   `ManagerBasedRlEnv` construction -- same `EventManager`-deepcopy fix
   already applied in `go1_wild_push_predictor.py`/
   `measure_push_failure_rate.py`. A naive adaptation copying
   `go1_record_video.py`'s post-construction-mutation-free pattern would
   have been fine for THAT file (which never touches push_robot at all)
   but would have silently no-op'd the push here if mutated the same
   (wrong) way the earlier bug did.
2. Forces forward-only WALKING commands
   (`cartpole_term.common.commands.force_forward_only_commands`, natural
   sampled speed, not pinned to a fixed max) so the pre-push baseline
   gait reads cleanly on camera -- `--natural-commands` opts out. The push
   itself still introduces real, uncontrolled motion once it fires; only
   the baseline walking direction is made unambiguous.
3. Loads checkpoint/predictor/metrics by explicit file paths
   (`--checkpoint-file`/`--predictor-weights`/`--predictor-metrics`/
   `--push-scale`) instead of the operating-point/
   `selected_checkpoints.json` convention `go1_record_video.py` uses --
   this pipeline's files are named by push-severity, not competence
   operating point.

`py_compile` clean. Full test suite (33 tests) still passes -- new
standalone script, touches no shared code.

Not yet run on the GPU.

## MAJOR change: switched the default perturbation mechanism from velocity-teleport to force-impulse

User watched the 8x video and confirmed exactly the risk flagged when
this escalation ceiling was originally designed: the robot "goes flying"
and falls almost instantly. This isn't a "pick a better scale" problem --
`push_by_setting_velocity` (mjlab's shipped mechanism, confirmed via its
own real source this session) is an INSTANTANEOUS, MASS-INDEPENDENT
velocity teleport with no dynamics behind it. At any scale strong enough
to reliably fail a well-trained policy, it stops resembling a real
disturbance. Every perturbation-based failure paper checked this session
(SafeFall, arXiv:2511.18509; the self-protective falling policy,
arXiv:2512.01336; GYF, arXiv:2310.04828) uses a real FORCE applied over a
duration instead, specifically because it respects the robot's own
mass/inertia and cannot teleport its velocity.

**Implemented the literature-correct mechanism**: `apply_body_impulse`
(mjlab/envs/mdp/events.py, confirmed real signature this session -- a
stateful class: cooldown -> trigger -> sustain -> expire lifecycle,
`force_range`/`torque_range` in Newtons/Nm, `duration_s`/`cooldown_s`,
optional `body_point_offset` for off-center torque).

**Mass-scaled the force magnitude rather than copying the literature's
number verbatim**: the self-protective-falling paper's own induced-fall
force range (50-850N) was validated on a "full-scale" humanoid-class
robot; SafeFall (the closest architectural match to this whole project)
explicitly validates on "full-scale Unitree G1", whose published mass is
~35kg. Go1's actual simulated mass was confirmed directly by summing
every `<geom mass=...>` in mjlab's real `go1.xml` this session: **12.74
kg**. F=ma means copying 50-850N verbatim onto a robot roughly a third
the mass would be proportionally far more violent than what that paper
validated -- scaled by the mass ratio (12.74/35 = 0.364) instead, giving
a reference range of ~(18, 310)N for Go1. Default starting point: 150N
force, 0 torque (pure linear force at the trunk's own center of mass,
deliberately no added spin, given "sometimes it goes flying" may partly
have been a rotational effect too) -- biased toward the gentler half of
that range, consistent with the already-observed "gentler is better"
pattern from the multi-severity experiment.

mjlab's shipped `push_robot` event (velocity-teleport, unscaled) is left
completely untouched for the impulse mechanism -- it was never flagged
as visually problematic at its shipped magnitude, so the new impulse is
layered ON TOP of it as an additional stressor (matching the literature's
own "combine multiple factors" approach) rather than replacing it.

**Refactored `go1_wild_push_predictor.py`**: extracted the shared
rollout/labelling/dataset-assembly loop (`_run_collection_loop`) so both
mechanisms reuse identical collection logic; added
`collect_under_impulse_push` (new) alongside the renamed/kept
`collect_under_wild_push` (velocity, kept for comparison/debugging, now
explicitly documented as not recommended). `--mechanism {impulse,velocity}`
now defaults to `impulse`. No automated verification ladder for the
impulse mechanism yet (unlike `verify_push_scale` for velocity) --
starting from the mass-scaled reasoned default and watching the printed
running_base_rate, same real-time visibility as before verification was
added for the other mechanism; a proper impulse-force escalation ladder
is a reasonable follow-up if 150N doesn't land in a usable range.

**Updated `go1_wildpush_video.py`** identically -- same `--mechanism`
flag, imports the mass-scaled constants directly from
`go1_wild_push_predictor.py` rather than duplicating the reasoning/numbers.

Verified via `EventTermCfg`/`apply_body_impulse` param names matching
the real confirmed signature exactly (`force_range`, `torque_range`,
`duration_s`, `cooldown_s`, `asset_cfg`, no `body_point_offset` --
deliberately omitted for pure center-of-mass force). `py_compile` clean
on both files. Full test suite (33 tests) still passes.

Ran on the GPU (fresh runtime): `go1_wild_push_predictor.py --mechanism
impulse --impulse-force 150`, 4,096 envs, against the 10,000-iteration
Go1 checkpoint. Real results: 1,003,212 labelled transitions, 1,164
terminated episodes (all `fell_over`), collection base_rate=0.0186,
episode_length mean=164.9/median=161.0. Predictor: AUPRC=0.8489
(baseline=0.0180), AUROC=0.9906, precision=0.8215/recall=0.6976/
f1=0.7545, median_lead=12.0 steps (0.24s), 167/167 positive episodes
detected. Beats logistic regression (0.7355) and both constant baselines
(0.0180). Best single feature: `action_l2norm` (AUROC=0.9530) -- notably
different from the earlier velocity-mechanism run's best feature
(`obs_l2norm`), and more physically interpretable (large corrective
actions right before a fall). This is the best predictor result of any
push-induced-failure attempt this session, on both quality metrics
(AUPRC, AUROC) at once, and the demo video confirms the fall no longer
looks physically implausible (no "flying") the way the velocity
mechanism did at any scale strong enough to be useful.

**Checked whether the literature reports a target/expected base rate
for push-induced failure data**, per this project's own "read the
literature before guessing" rule, since 1.86% is on the sparse side and
the user flagged the same concern about the earlier velocity-mechanism
"low severity" result (1.49%). Fetched and searched the full text of
all four papers actually cited by this project's push/predictor design:
STATE-NAV (arXiv:2506.01046, this project's primary predictor-design
reference), SafeFall (arXiv:2511.18509), the self-protective-falling
policy (arXiv:2512.01336), and GYF (arXiv:2310.04828). **None of the
four reports a base rate, class-imbalance ratio, or positive/negative
label fraction for their fall data.** Each reports collection *volume*
only (STATE-NAV: 72,000 gait cycles / 8h sim; the self-protective paper:
81,920 trajectories, 65,536/16,384 train/val split) with no breakdown of
how many were positive. **Conclusion: there is no published number to
hold 1.86% against -- this is an unreported detail in the exact
literature this project has been following, not a documented target
this result falls short of.** Decision, made on that basis plus the
result's own quality (1,164 real failure episodes collected, comparable
in count to this project's own `≥1,000 terminated episodes` collection
floor used throughout, and a strong AUPRC/AUROC/lead-time already):
**locked in `mechanism=impulse`, `force=150N`, `torque=0` as the
accepted final operating point for the Go1 wild-push predictor.** Not
escalating force further to chase a higher base rate, since doing so
would trade against the exact failure mode already found and fixed
(implausible, near-instant, "flying" falls at high severity). If a
future recovery-policy stage needs more raw failure examples, the
right lever is collecting more transitions at 150N (more terminated
episodes at the same, already-validated severity), not a stronger push.

Final artifacts downloaded and organized under `outputs/`:
`go1_wildpush_predictor_seed0.pt`, `go1_wildpush_metrics_seed0.json`,
`go1_wildpush_frozen_seed0.npz`,
`go1_wildpush_overlay_impulse150.0N_seed0_take0.mp4`.

### New: live online lead-time measurement + `DeferredResetEnv` (Gate 5 prep)

At the user's request: measure the wild-push predictor's detection lead
time under a REAL live rollout (run the locked checkpoint + locked push
mechanism, score every step online, note the first step the predictor
crosses its decision threshold, keep rolling to the real fall, compute
`termination_step - detection_step`) rather than reading the existing
post-hoc `mlp_lead_time` figure out of `outputs/go1_wildpush_metrics_seed0.
json` (median=12 steps). That existing figure is computed from the frozen,
16-step-horizon-censored dataset via `scripts/eval_predictor.py`'s
`median_detection_lead_time` -- it cannot report a lead time longer than
the training horizon, because nothing before it was ever labelled
positive, whether or not the predictor was actually confident there. The
new script, `scripts/go1_collect_flagged_states.py`, has no such ceiling:
it flags an episode the instant its live probability first crosses
`outputs/go1_wildpush_metrics_seed0.json`'s val-selected threshold
(0.5467), keeps that episode running exactly as normal (no dataset
censoring involved -- this is a live rollout, not a labelling problem),
and records `lead_steps = termination_step - flag_step` whenever a real
fall (`terminated`) follows. Episodes that truncate after being flagged
are discarded from the stat (counterfactual unknown, same reasoning as
CLAUDE.md S7.4's censoring rule elsewhere in this project); episodes that
terminate without ever being flagged are counted separately as a live
recall diagnostic, not silently dropped. Reuses
`go1_wild_push_predictor.py`'s `build_impulse_env_cfg_factory` verbatim
(extracted to module level from what was a nested closure inside
`collect_under_impulse_push`, so both scripts share one push-mechanism
implementation instead of duplicating it) -- same locked 150N/0Nm impulse,
same trunk body target.

**New capability, added mid-implementation at the user's explicit request**
("apply some sort of wrapper or mechanic that allows the run to continue
after termination conditions are true, ... so that a recovery policy can
still be implemented"): `src/cartpole_term/common/deferred_reset.py`'s
`DeferredResetEnv`. mjlab's default behaviour resets a terminated env's
physics WITHIN the same `step()` call that detected the termination,
before returning -- the terminal/post-fall state is never observable,
which makes training or evaluating Gate 5's protective policy (whose job
is to act on states that only exist after the nominal policy has already
failed) structurally impossible without some mechanism to keep simulating
past that boundary.

Confirmed against the real installed source this session (mjlab 1.6.0,
fetched from `mujocolab/mjlab@main`,
`src/mjlab/envs/manager_based_rl_env.py`) -- this **supersedes** the
earlier "unconfirmed `ManagerBasedRlEnvCfg.auto_reset` field" note in this
file's S6.2.8 entry from the original cartpole phase. `auto_reset` is a
real, documented field (default `True`); with `auto_reset=False`, `step()`
skips the automatic per-env reset, returning the true terminal observation,
and physics keeps advancing normally for that env on every subsequent
`step()` call. The one obstacle: mjlab's own `step()` raises
`RuntimeError` if any env has a pending `_manual_reset_pending` flag
(set for every env that was done last call) when `step()` is called again
without an intervening `reset(env_ids=...)` -- a safety rail intended to
force prompt manual resets under mjlab's own documented `auto_reset=False`
usage pattern ("intended for users running their own training loop...
that handles the reset between steps"). `DeferredResetEnv` deliberately
bypasses that rail for envs the caller explicitly marks via `hold(env_ids)`,
by clearing their pending bit directly instead of resetting -- this IS a
deviation from mjlab's intended usage (the library expects a prompt reset,
not an indefinite hold), accepted here because it's the only mechanism
found this session that makes a post-termination continuation possible at
all.

**Known, accepted timing tradeoff:** for envs never held, `DeferredResetEnv`
still reproduces auto_reset=True's externally-observed behaviour (done ->
reset), but one wrapper-`step()` later rather than synchronously inside
the call that detected termination -- so an unheld env's action on the
step immediately after its own termination is chosen against its
terminal-state observation, not a fresh post-reset one, discarded once the
deferred reset lands one call later. This does not affect anything this
project currently measures with it: lead time, missed-detection count, and
censored count are all read directly from the termination-detection step
itself, before any reset (deferred or not) happens. It would matter for a
future use with many unheld envs whose very next action must be clean;
noted here rather than hidden.

Wired into `go1_collect_flagged_states.py` via `--hold-post-fall-steps`
(default 30, grounded by loose analogy to SafeFall's arXiv:2511.18509
~40-step recovery-episode length, not an exact citation match; 0 disables
the feature entirely with no change to prior behaviour beyond the same
auto_reset=False/one-step-later-reset tradeoff above, which does not
affect correctness): once a flagged episode's env really terminates, its
physics keeps running under the same nominal policy (no protective policy
exists yet -- Gate 5 is still unbuilt) for `hold_post_fall_steps` more
env steps, and that continuation trajectory (`post_fall_obs`,
`post_fall_action`) is saved into `outputs/go1_flagged_states_seed{N}.npz`
alongside the lead-time data. This is real, physically-grounded seed data
for Gate 5's "nominal policy's own uncontrolled continuation" ablation
baseline once a protective policy exists to compare it against -- not
itself a trained recovery policy, which remains future work per the plan.

**Update, first real GPU run:** env construction, predictor loading, and
the `DeferredResetEnv`/`auto_reset=False` plumbing all worked as designed
-- no bug there. The actual crash was elsewhere and unrelated to deferred
reset: `raw_env.reset()`/`.step()` return `obs` as a plain Python `dict`
(confirmed by the traceback: `AttributeError: 'dict' object has no
attribute 'to'`), not the `TensorDict`-wrapped object the WRAPPED env
(`RslRlVecEnvWrapper`) normally hands `runner.alg.actor()`. Fetched
`src/mjlab/rl/vecenv_wrapper.py` from the real source
(`mujocolab/mjlab@main`) to confirm exactly what that wrapper does:
`TensorDict(obs_dict, batch_size=[num_envs])`, from the `tensordict`
library (an `rsl_rl` dependency, already installed transitively). Fixed
by reproducing that exact wrapping around the raw dict in
`go1_collect_flagged_states.py` right after every `deferred.reset()`/
`deferred.step()` call, rather than either calling `.to()` on a plain
dict (crashes) or passing the plain dict straight to `alg.actor()`
untested (risked a second, less obvious failure deeper inside RSL-RL's
own TensorDict-specific handling). Re-upload the notebook's zip and
re-run; not yet confirmed past this point.

**Confirmed working end to end, second GPU run (post-fix):** 505 resolved
trials, `live_recall=1.0` (every real fall in this run had been flagged
beforehand -- no missed detections), `censored_after_flag=0`. Live
lead time (no 16-step ceiling): **mean=16.19 steps (0.324s),
median=14.00 steps (0.280s)** -- both above the offline, horizon-capped
`mlp_lead_time` figure (median=12 steps), consistent with this
measurement being free of that ceiling rather than a discrepancy. 297 of
505 trials also captured a full 30-step post-fall continuation
trajectory (`post_fall_obs`/`post_fall_action`); the other 208 had an
incomplete hold window when the run hit its 500-trial target and stopped
-- expected, per the script's own documented behavior. Artifacts:
`outputs/go1_flagged_states_seed0.npz`, `outputs/go1_flagged_states_seed0.
json`.

**Follow-up analysis (still within this run's data, no retraining):** the
worst-case tail of the 505-trial lead-time distribution is genuinely thin.
`min_lead_steps=5` (0.10s) sits exactly at SafeFall's (arXiv:2511.18509)
own stated "minimum reliable detection horizon before impact"; ~22%
(110/505) of trials give <=10 steps (0.20s), the same order of duration
GYF (arXiv:2310.04828, same robot) reports for an ENTIRE Go1 fall start to
finish. The median (14 steps/0.28s) and mean (16.19/0.324s) are fine by
the same papers' own numbers (SafeFall's real deployment achieves 410ms);
the problem is specifically the bottom of the distribution, not the
typical case.

### New: history-stacking (CLAUDE.md S8.1's `--predictor.history H`, implemented for the first time)

CLAUDE.md S8.1 lists `--predictor.history H` as an optional, off-by-default
input-stacking mechanism (STATE-NAV's own analogue: aggregating over a
gait cycle) but it was never actually built -- `scripts/eval_predictor.py`
raises `NotImplementedError` for `history != 1` explicitly to avoid
"silently approximating" a history-window reconstruction it hadn't
verified. Built now, at the user's request, specifically to attack the
worst-case-tail finding above: a single-frame snapshot can't distinguish
"falling fast" from "about to recover" until very late, because it has no
access to a trend; stacking recent frames gives the model that trend
directly.

New: `src/cartpole_term/predictor/history.py`,
`build_history_stacked_dataset(dataset, history)`. Generic over any flat
per-step dataset (obs/action width-agnostic -- works for cartpole's 6-wide
input or Go1's 60-wide one unchanged). Implementation note: relies on
CLAUDE.md S7.4's own labelling ring buffer guaranteeing that every step
actually present in a saved flat dataset for a given episode is preceded
by every smaller step in that same episode (terminated episodes emit the
full contiguous prefix; truncated episodes emit a contiguous prefix up to
the censored tail) -- verified this holds by direct re-derivation of the
labeler's emission logic, and the module asserts it against the real data
at runtime (`_assert_contiguous_prefix`) rather than trusting it silently.
Drops (does not zero-pad-and-mask, despite CLAUDE.md's literal prose) any
row that would need a frame from before its own episode's start --
functionally identical to zero-pad+mask from the loss's perspective (the
row never contributes gradient either way), simpler, and avoids ever
handing the network a real-but-disabled all-zero frame that could become
an artifact correlated with "near episode start" (itself adjacent to the
S8.1-forbidden `step_in_episode` leak).

Unit-tested in-sandbox before touching the GPU at all (CLAUDE.md S0.2.5's
"anything runnable without a GPU must actually be run", generalized to
this phase-2 addition): `tests/test_history_stacking.py`, 6/6 pass --
single-episode boundary-drop correctness (by value, not just shape),
multi-episode no-cross-contamination with interleaved/unsorted input,
history=1 exact no-op, and a deliberately-broken mid-episode-gap input
correctly raising rather than silently producing wrong windows.

New: `scripts/go1_retrain_predictor_history.py`. Reuses the ALREADY-
COLLECTED `outputs/go1_wildpush_frozen_seed0.npz` (1M+ transitions) --
no new GPU collection needed, since history-stacking is a post-hoc
reconstruction from the same flat per-step records. Reuses
`go1_wild_push_predictor.train_and_evaluate_predictor` unmodified except
for one added parameter, `tag: str = ""` (default preserves the existing
call site's exact output filenames), used here to write
`go1_wildpush_predictor_history{H}_seed{N}.pt` /
`go1_wildpush_metrics_history{H}_seed{N}.json` instead of overwriting the
locked H=1 predictor's files.

Default `history=8` (160ms at 50Hz): grounded in the locked impulse push's
own `duration_s=(0.1, 0.3)` range (5-15 steps), not guessed -- long enough
to see a push's onset developing, short of a full gait cycle (not the
goal here; the goal is catching a fast-developing fall early). One
grounded value tested, not a sweep (CLAUDE.md's "no hyperparameter
sweeps" spirit, extended to this phase-2 choice).

**Scope note, deliberate:** this only retrains and evaluates OFFLINE,
against the existing frozen dataset's own (16-step-horizon-capped)
`mlp_lead_time` metric -- it does NOT yet touch
`scripts/go1_collect_flagged_states.py`'s LIVE, uncapped worst-case
measurement (the actual thing we're trying to fix). That's a real,
separate follow-on (the online scorer needs a rolling per-env H-frame
buffer, not just a wider single-frame input) deliberately deferred until
this offline signal justifies the extra work -- no point wiring history
into the expensive live path before knowing it helps at all.

**Not yet run on the GPU** -- unexecuted, verified only by the in-sandbox
unit tests above and a syntax check.

### Gate 5: protective "small-fall" policy -- design pivot away from the plan's own "rollout-then-handoff"

At the user's direction ("we need to start training the small fall"),
began Gate 5 of the multi-robot plan
(~/.claude/plans/snug-booping-gosling.md). Before writing any code,
fetched and read the real mjlab source this session (mujocolab/mjlab@main:
src/mjlab/sensor/contact_sensor.py, src/mjlab/entity/{entity.py,data.py},
src/mjlab/envs/mdp/{events.py,rewards.py}, src/mjlab/tasks/velocity/mdp/
{rewards.py,terminations.py}, src/mjlab/tasks/velocity/config/go1/
env_cfgs.py) rather than implementing from the plan's text alone, per this
project's own "verify against the real install, never guess an API"
discipline.

**Major finding, changes the plan's own design:** the plan's Gate 5
section explicitly chose a "rollout-then-handoff" initial-state mechanism
(run the nominal policy under the push for a random prefix, then hand
control to the trainee policy, all within one continuous episode)
specifically because literal state injection was believed to need "an
unconfirmed mjlab 'set full state directly' API." Reading
src/mjlab/entity/entity.py directly this session confirms that API is
REAL and fully documented: `write_root_link_pose_to_sim(root_pose:
[N,7], env_ids)`, `write_root_link_velocity_to_sim(root_velocity: [N,6],
env_ids)`, `write_joint_state_to_sim(position, velocity, joint_ids,
env_ids)`. The exact reset-event calling convention (a plain
`func(env, env_ids, **params) -> None`, `env_ids=None` meaning "all
environments", re-adding `env.scene.env_origins[env_ids]` onto a stored
env-local root position) was copied directly from mjlab's own shipped
`reset_root_state_uniform` (src/mjlab/envs/mdp/events.py), not invented.

Given this, literal state injection is adopted INSTEAD of rollout-then-
handoff: simpler (an ordinary mjlab reset event, no custom PPO rollout
loop needed at all -- scripts/go1_train_protective.py is a completely
standard from-scratch training script, structurally identical to the
already-validated scripts/go1_train.py), and MORE faithful to SafeFall's
(arXiv:2511.18509) own real design, whose Stage II literally inits
episodes from predictor-flagged unsafe states -- exactly what this now
does.

**New: `scripts/go1_collect_flagged_states.py --capture-state-bank`.**
For every resolved lead-time trial (a real fall the live predictor
flagged in advance -- the exact "predictor-flagged unsafe state"
population SafeFall's Stage II uses), reservoir-samples ONE raw physics
state (root pose/velocity + joint pos/velocity, read via
`asset.data.root_link_{pos,quat,lin_vel,ang_vel}_w` / `joint_pos` /
`joint_vel`) uniformly at random from [flag_step, termination_step) --
standard reservoir sampling (replace the held candidate with probability
1/k at the k-th step since flagging), chosen because termination_step is
only known in hindsight, after the window has already passed; this
algorithm needs no lookahead and is provably uniform over the window.
Output: `outputs/go1_fall_state_bank_seed{N}.npz`. Root position is saved
relative to the COLLECTING env's own origin (`env.scene.env_origins`
subtracted at capture time) and must have the INJECTING env's own origin
added back at use time -- same reasoning `reset_root_state_uniform` itself
follows, noted explicitly in both the collection script and
`cartpole_term/protective/events.py` to avoid a silent, hard-to-spot bug
if ever "simplified" later.

**New: `src/cartpole_term/protective/` package** (`__init__.py` registers
`Mjlab-Go1-Protective-Fall`; `env_cfg.py`, `rewards.py`, `events.py`,
`runner.py`, `rl_cfg.py`). Built ON TOP of Go1's own already-registered
walking task cfg (same scene/action-space/48-dim observation layout, so a
future Gate 6 handoff needs no remapping), with:

- **Episode length: 10 steps = 0.2s at Go1's confirmed 50Hz.** SafeFall's
  own protective episode is a fixed 40 steps at their 200Hz sim rate =
  0.2s; TIME-scaled (not step-count-scaled -- the plan's own phrasing,
  "scaled... to each robot's confirmed Hz", already called for this) to
  Go1's 50Hz gives 10 steps. Cross-checked against this project's OWN
  measured post-fall dynamics (`outputs/go1_flagged_states_seed0.npz`'s
  `post_fall_obs`, captured under the UNCONTROLLED nominal policy): mean
  base angular speed peaks around step 4-5 then decays but is still
  clearly elevated at step 28 (the last captured step) -- confirming the
  CRITICAL high-impact window is concentrated in roughly the first 10
  steps, not contradicting the time-scaled literature precedent.
- **A new full-body ContactSensorCfg** (`mode="body"` primaries = every
  name in `RobotProfile.GO1.vulnerability_regions`, `secondary=
  ContactMatch(mode="body", pattern="terrain")`, `reduce="netforce"`,
  `history_length=4` -- matching Go1's own decimation and the rough
  variant's own identical choice for its per-body sensors). The flat
  variant this whole project trains on removes every per-body-group
  ground sensor the rough variant ships (confirmed via
  `unitree_go1_flat_env_cfg`'s own `remove_sensors` set) -- this had to
  be added fresh, it does not already exist on the base task.
- **`vulnerability_weighted_contact_force`** (new, `protective/
  rewards.py`): blends 0.3 peak + 0.7 average force per body (SafeFall's
  cited ratio) then applies tier multipliers 1000/1/0.5 for high/medium/
  low vulnerability (arXiv:2512.01336 Table I, cited as-is). The OUTER
  Newtons->reward scale (-1e-5, the RewardTermCfg weight) is NOT cited
  from either paper -- neither reports one in comparable units -- and
  instead reuses THIS codebase's own already-shipped, real-world-
  calibrated scale for the same category of quantity: Go1's own
  `soft_landing` term ships at exactly -1e-05 for an un-tiered force
  magnitude. Presented here transparently as a reasoned, NOT directly-
  cited choice, not fabricated-but-hidden.
- **`root_linear_velocity_penalty`** (new, weight -1.03): the weight is a
  literal number from arXiv:2512.01336 Table I, reused as-is -- adapted,
  not independently calibrated for Go1's own mass/units (same caveat
  every "adapted, not verified for this robot" number in this project
  carries).
- **`action_rate_l2` and `joint_pos_limits`**: NOT reimplemented -- reused
  directly from mjlab's generic, already-shipped `envs.mdp` module, at
  Go1's OWN already-shipped weights (-0.1, -1.0 respectively, from this
  session's live RewardManager dump) -- the strongest possible grounding
  available (this robot's own validated numbers, not adapted from
  elsewhere).
- **Explicitly NOT included, logged rather than silently omitted:**
  joint-torque / DoF-acceleration regularization. The plan's own summary
  of arXiv:2512.01336 mentions these as real terms that paper uses, but
  its exact cited weights weren't fetched this session and Go1's own
  shipped task doesn't include a torque-regularization term to borrow a
  number from either -- rather than inventing one, this was left out.
  Revisit once arXiv:2512.01336's own Table I is read directly, if the
  trained policy's torque behavior turns out to need it.
- **Terminations**: only `time_out` -- `fell_over` removed, since the
  robot starts ALREADY mid-fall/fallen by design; ending the episode
  early on orientation grounds would defeat the entire task.
- **`push_robot` removed**: no ongoing perturbation needed once the
  episode already starts unsafe.

**Update, first real GPU run (`--smoke`):** hit a confirmed real bug in
mjlab itself, not this project's code. `ContactSensorCfg(history_length=4)`
on the new `protective_body_ground_contact` sensor (used for SafeFall's
0.3-peak/0.7-average-over-substeps force blend) allocates a rolling
history buffer; mjlab's own `ContactSensor.reset()`
(src/mjlab/sensor/contact_sensor.py:365) does `buf[env_ids] = 0.0` on
that buffer, and under this project's `FrozenEvalRunner`'s eval-env
construction/rollout pattern, that buffer ends up marked a PyTorch
"inference tensor" -- permanently unwritable in-place outside
`torch.inference_mode()`. Exact error:
`RuntimeError: Inplace update to inference tensor outside InferenceMode
is not allowed.` No other ContactSensor anywhere in this project (Go1's
own shipped `feet_ground_contact` included) has ever set
`history_length>0`, so this code path had simply never been exercised
before. Not fixable here -- mjlab is site-packages, never edited
(CLAUDE.md S12#18) -- and not something worth spending further GPU cycles
diagnosing further given a clean, low-cost alternative exists.

**Fix:** dropped the peak/average-over-substeps blend entirely.
`vulnerability_weighted_contact_force` now uses the INSTANTANEOUS
per-policy-step `data.force` (still `reduce="netforce"`, no
`history_length`) instead of `data.force_history`. This is a real
simplification versus SafeFall's exact blend, but keeps the actual point
of the term -- differentiated per-body-vulnerability weighting
(1000/1/0.5, arXiv:2512.01336) -- fully intact, and completely avoids the
buggy code path. `CONTACT_HISTORY_LENGTH`/`PEAK_BLEND` constants removed
from `protective/env_cfg.py`; the sensor's `history_length` param is now
simply omitted (left at its default, disabled).

**Not yet confirmed past this point** -- the `--smoke` run got past env/
reward/sensor construction and failed specifically at the first frozen-
eval reset, which is exactly where this fix targets. Re-run
`--smoke` again after re-uploading the fixed code before trusting this
further.

**Second issue, caught by the user before any further GPU time was
spent:** `protective/rl_cfg.py` reused `go1_velocity_term_ppo_runner_cfg()`
verbatim, which inherits `max_iterations=10_000` (confirmed via
`mjlab/tasks/velocity/config/go1/rl_cfg.py`) -- Go1's OWN calibration for
the 1000-step-episode WALKING task, not a deliberate choice for this
10-step-episode task. At the shipped `num_steps_per_env=24`, one PPO
iteration already covers roughly 9,800 complete episodes at the default
4096 envs -- reusing 10,000 iterations unmodified would have been
un-thought-through inheritance, not a reasoned choice, and would have
wasted a large amount of GPU time. Fixed: `protective/rl_cfg.py` now
overrides `max_iterations=500` -- a bounded, reasoned STARTING POINT
(not independently validated against either cited paper's own training
budget, which wasn't fetched this session), to be raised via
`--max-iterations` if the frozen-eval curve hasn't converged by then, per
CLAUDE.md S5.3's own explicit protocol for this exact situation. Not
itself an algorithm hyperparameter (CLAUDE.md S5.3's "no tuning" rule
covers lr/clip_param/entropy_coef/etc., not this) -- CLAUDE.md S5.3
explicitly treats max_iterations as task-dependent.

### Per-term reward logging (diagnosing "reward stays flat around -1.5")

The completed 500-iteration run (checkpoint `model_499.pt`) showed no
visible improvement in the total reward across training. Rather than
guess further at which of the four terms dominates (root_linear_velocity
was the suspect, given its weight -1.03 vs vulnerability_contact_force's
-1e-5), added per-term diagnostic logging to all four reward terms in
`protective/rewards.py`, via `env.extras["log"]["Metrics/..."]` -- the
SAME mechanism mjlab's own reward functions already use for this exact
purpose (e.g. `soft_landing`'s `Metrics/landing_force_mean`,
`feet_slip`'s `Metrics/slip_velocity_mean`), which RSL-RL's
`OnPolicyRunner` already picks up and reports during real training, not
just frozen eval.

`vulnerability_weighted_contact_force` and `root_linear_velocity_penalty`
log their own raw (unweighted) per-step mean directly.
`action_rate_l2`/`joint_pos_limits` are mjlab generic functions (can't add
logging inside them -- site-packages, never edited) -- wrapped with thin
pass-through functions (`logged_action_rate_l2`, `logged_joint_pos_limits`
in `protective/rewards.py`) that call the real function unchanged and log
its mean, then wired into `env_cfg.py`'s `RewardTermCfg`s in place of the
unwrapped versions. New Metrics keys: `vuln_contact_force_raw_mean`,
`vuln_contact_force_raw_max`, `vuln_contact_force_penalty_mean`,
`root_lin_vel_sq_mean`, `action_rate_l2_mean`, `joint_pos_limits_mean`.

**Update, confirmed by the logged run's actual TensorBoard curves:**
`Metrics/root_lin_vel_sq_mean` went 4.5 -> 4.1 over the run (small but
real, directionally-correct learning -- weight -1.03 -> per-step
contribution ~-4.2 to -4.6). `Metrics/vuln_contact_force_raw_mean` rose
0 -> ~13 (plausible: a more-inert early policy naturally makes less
ground contact than one that's actively moving/bracing).
`Metrics/vuln_contact_force_penalty_mean` (the tier-weighted value the
reward actually uses) measured ~7000-12000, noisy and non-monotonic
(12000 -> 8000 -> 12000 -> 7000, then flat) -- at the original weight
-1e-05, that's a per-step contribution of only ~-0.07 to -0.12.

**Root cause confirmed: a ~40x weight-scale mismatch.**
`root_linear_velocity_penalty` was contributing roughly 40x more to the
total reward than `vulnerability_contact_force` -- the actual point of
this task was effectively invisible to what PPO was optimizing, and the
noisy, non-monotonic observed curve for it is consistent with incidental
drift under near-zero gradient pressure, not real optimization. This
directly explains "reward stays flat around -1.5": the ONE term with
enough scale to move the total (root velocity) only had a small
achievable improvement to begin with (§ higher up in this file), and the
other term that might have moved the sum further essentially couldn't.

**Fix:** `VULNERABILITY_CONTACT_WEIGHT` in `protective/env_cfg.py`
changed from -1e-05 to **-4e-4** (~40x increase) -- grounded directly in
this run's own measured per-term magnitudes (bringing its contribution
into the same order as `root_linear_velocity`'s ~-4.2 to -4.6), not
re-borrowed from `soft_landing`'s un-tiered scale, which evidently wasn't
the right calibration once the 1000x/1x/0.5x tier multiplier is applied
on top of it.

**Second round, after re-running with the -4e-4 fix:** the user reported
the total reward was STILL flat. RSL-RL's own console output (per-
iteration summary, including each term's real weighted contribution via
`Episode_Reward/*`) made the reason directly visible: fixing the first
imbalance exposed a SECOND one. Measured, iterations 456-499:

| Term | Episode_Reward/* (weighted contribution) |
|---|---|
| `root_linear_velocity` | ~-4.12 to -4.23 |
| `action_rate_l2` | ~-2.62 to -2.79 (!) |
| `vulnerability_contact_force` | ~-0.15 to -0.22 (confirms the -4e-4 fix DID apply -- raw ~450 * -4e-4 ~ -0.18, matches) |
| `joint_pos_limits` | ~-0.14 to -0.16 |

`action_rate_l2`'s weight (-0.1) was copied verbatim from Go1's own
walking-task config -- same class of mistake as the first fix, just
undiscovered until the first one was corrected. Go1's raw action-rate
magnitude (`Metrics/action_rate_l2_mean`, measured ~26-28) is much larger
in THIS task than during normal walking, because reacting to an already-
falling injected state involves far more abrupt action changes than a
steady gait -- the borrowed weight was never checked against this task's
own typical magnitude, and at -0.1 it became the second-most dominant
term in the whole reward, swamping the two terms that actually matter
almost as badly as `root_linear_velocity` was swamping
`vulnerability_contact_force` before.

**Fix, second round:**
- `VULNERABILITY_CONTACT_WEIGHT`: -4e-4 -> **-4e-3** (further 10x) --
  targeting a contribution (~450 * -4e-3 = -1.8) that's a meaningful
  fraction of `root_linear_velocity`'s own ~-4.15, appropriate given
  SafeFall/arXiv:2512.01336 present differentiated contact-force
  weighting as their core contribution, not a minor regularizer.
- `ACTION_RATE_WEIGHT`: -0.1 -> **-0.02** (5x reduction) -- targeting a
  contribution (~27 * -0.02 = -0.54) small enough to stop dominating,
  while still discouraging jerky actions (its actual purpose).
- `JOINT_POS_LIMITS_WEIGHT`: unchanged (-1.0) -- its measured
  contribution (~-0.16) was already small and appropriate.

**Update, after re-running with the second-round weights:** the total
reward was STILL flat, and per-term data revealed the fix hadn't worked
the way intended. `root_lin_vel_sq_mean` sat completely flat at ~4.0-4.5
across the ENTIRE 500-iteration run -- no decreasing trend at all,
matching the injected fall-states' own average starting velocity. The
policy is not learning to reduce fall velocity in any configuration
tried so far; this is the real, still-unresolved problem, independent of
which reward term nominally dominates.

Separately, `action_rate_l2`'s weighted contribution DIDN'T shrink
despite the 5x weight cut (-0.1 -> -0.02): `Metrics/action_rate_l2_mean`
(the raw quantity) grew ~5x in response (27 -> ~130), and `Mean action
std` nearly tripled (0.70 -> 1.85) between the two runs. Cutting the
smoothness penalty removed the main check on PPO's entropy bonus, which
pushed the policy toward much noisier actions -- the raw magnitude grew
by roughly the same factor the weight shrank, leaving the product (the
actual reward contribution) unchanged. A self-defeating adjustment.
`Mean value loss` also never stably converged (bounced between ~0.4 and
13+ the whole run), consistent with the value function struggling to
predict returns given how varied the injected fall states are --
plausible further explanation for why nothing is visibly improving,
independent of reward-shaping.

**Consulted the literature directly** (at the user's explicit
instruction) rather than continuing to guess at another weight
adjustment. Fetching arXiv:2512.01336's real Table I this time (not just
recalling it) surfaced two things:

1. **A citation error.** The "1000/1/0.5 per-body-vulnerability tier
   multiplier" attributed to arXiv:2512.01336 (in the original plan this
   project inherited, and in this file/rewards.py's earlier entries)
   does not appear in that paper's real Table I at all. That figure
   belongs to SafeFall (arXiv:2511.18509), a different paper -- the two
   papers' contact-force schemes were conflated somewhere upstream of
   this session and never re-verified until now.
2. **A complete, better-grounded weight set**, since arXiv:2512.01336
   reports real numbers for several terms this task already uses:
   root linear velocity -1.03125 (matches what was already in use),
   contact force on critical bodies -0.02, contact force on all bodies
   -0.001, action rate -0.00667. That last number is 15x smaller than
   the original Go1-borrowed weight (-0.1) and still 3x smaller than the
   already-once-reduced -0.02 -- directly explaining why the action-rate
   fix never actually took hold.

**Fix, third round:** replaced the entire SafeFall-tier-multiplier
design (`TIER_MULTIPLIERS` + a separately-tuned outer
`VULNERABILITY_CONTACT_WEIGHT`) with arXiv:2512.01336's own real 2-
category scheme: `CONTACT_FORCE_WEIGHTS = {"critical": -0.02,
"non_critical": -0.001}`, used as the FINAL per-Newton weights directly
(RewardTermCfg.weight=1.0, pass-through) rather than requiring a self-
reverse-engineered outer scale. `RobotProfile.GO1.vulnerability_regions`'
"high" tier (trunk) maps to "critical"; "medium"+"low" (hips/thighs/
calves) collapse into "non_critical". `ACTION_RATE_WEIGHT` corrected to
arXiv:2512.01336's own -0.00667. `joint_pos_limits` deliberately left
unchanged (-1.0, Go1's own weight) -- arXiv:2512.01336 also reports a
DoF-position-limit term (-29) but its raw quantity's exact definition
isn't confirmed comparable to mjlab's own `joint_pos_limits` formula, so
it is NOT blindly copied, same reasoning that caught the other two
mismatches in the first place.

**Explicitly flagged as still open, not resolved by this fix alone:**
(1) whether `entropy_coef` (0.01, inherited unchanged from Go1's walking
task, at a much larger reward scale) needs adjusting for this task --
a real PPO hyperparameter, not to be changed without the user's
go-ahead; (2) the value-function-instability finding above, which may
need addressing independently of reward-term weights; (3) per the user's
own explicit note, a "footprint" / contact-spread-area penalty (the
plan's own flagged-as-optional, not-sourced term) is still to be added
on top of this -- the current reward set is a working baseline, not
presented as final. Not yet re-run on the GPU with these third-round
weights.

## Gate 5 (Go1 protective fall): episodes were terminating mid-fall -- fixed episode length replaced with a settling-based termination

**Reported by the user** after watching the trained protective policy's
video: "some of the falls terminate while its still in the falling
motion." Asked to either fix so the episode only ends once the robot
stops moving, treating "eventually comes to a stop" as a requirement for
a fall, or to check the literature for how this is normally handled.

**Root cause, confirmed against data already in this repo:** the
protective task's episode length was fixed at 10 steps (0.2s),
reasoned (see env_cfg.py's prior docstring) by time-scaling SafeFall's
40-step/200Hz protective episode to Go1's 50Hz, and cross-checked at the
time against `outputs/go1_flagged_states_seed0.npz`'s `post_fall_obs`
array with the claim that the "critical high-impact window is
concentrated in roughly the first 10 steps." Re-reading that same data
while investigating this report showed that cross-check was wrong: mean
base angular speed under the UNCONTROLLED nominal policy is *still
elevated at step 28* of the captured 30-step post-fall window -- i.e.
NOT concentrated in the first 10 steps at all. The fixed 10-step episode
was therefore genuinely ending training rollouts (and the demo video)
mid-fall, before the robot had a chance to settle -- exactly the reported
symptom, not a misreading of a fine video.

**Literature check, done before changing anything (per the user's own
suggested alternative):** searched specifically for how fall-recovery RL
work defines "the robot has stopped/settled" for termination purposes.
General web search results (FRASA, arXiv:2410.08655; Stubborn,
arXiv:2606.12814; and others) show episode termination in this literature
is usually gated on reaching an *unsafe* state (excessive trunk pitch or
pitch-rate, joint/velocity limits) or a fixed max duration -- not on a
"has it stopped" check, since most of that literature is a stand-up
controller, not a protective-fall controller. FRASA is the one paper that
gives an explicit, directly relevant **task-completion** criterion,
confirmed by direct fetch of the paper: recovery is declared successful
only once "the error in the state vector components ... must remain
below 5 degrees for 0.5 seconds" -- a state-error threshold **sustained
for a fixed duration**, not touched once and done. FRASA does not give
any velocity number at all (confirmed by direct fetch) -- its criterion
is a position-tracking error, which this task has no analogue for (no
target pose to track, unlike a stand-up controller).

**Fix:** added a new sustained-duration `settled` termination
(`src/cartpole_term/protective/terminations.py`), reusing FRASA's
sustained-duration STRUCTURE (below-threshold, held for a duration, not
an instantaneous crossing) applied to root linear+angular speed instead
of FRASA's position error:
- `LINEAR_SETTLED_THRESHOLD_MPS = 0.2`, `ANGULAR_SETTLED_THRESHOLD_RAD_S
  = 0.5` -- **not from FRASA** (it gives no velocity number); this
  project's own calibration, common "quasi-static/at rest" cutoffs in
  legged-robot control, not yet tuned against this task's own measured
  data. Logged here as invented, not cited, per this project's own
  discipline.
- `SETTLED_HOLD_S = 0.5` -- FRASA's own literal hold duration, reused as
  the sustain window (time-scaled via `env.step_dt` at call time, so it
  carries over unmodified to G1 at its own Hz per the multi-robot plan).
- `PROTECTIVE_EPISODE_LENGTH_S` raised from 0.2s (10 steps) to 3.0s (150
  steps) -- now a safety-cap BUDGET for the rare episode that never
  settles, not the expected episode length. Most episodes should end
  well before this via `settled`.
- Registered `settled` with `time_out=False` (a genuine terminal state --
  once at rest under this reward there's nothing further to gain, no
  value bootstrap should occur), distinct from the base cfg's existing
  `time_out` term (unchanged, now serves only as the safety-cap
  truncation).
- **Made settling a rewarded requirement, not just an episode-ending
  condition** (per the user's explicit "make this a requirement for a
  fall"): added `settled_bonus` (`src/cartpole_term/protective/
  rewards.py`), a one-time reward that fires on the exact step `settled`
  triggers, reading the termination manager's own already-computed
  per-term result (`env.termination_manager.get_term("settled")`,
  confirmed via direct read of mjlab's `termination_manager.py` /
  `manager_based_rl_env.py` that termination is computed before reward
  every step and cached per-term) rather than recomputing the same
  threshold/hold logic a second time. `SETTLED_BONUS_WEIGHT = 5.0` is
  this project's own choice, not literature-sourced -- an episode that
  times out without ever settling collects no such bonus, which is the
  actual mechanism creating "must eventually come to a stop" pressure.
- Confirmed via direct fetch of mjlab source
  (`mjlab/managers/manager_base.py`, `termination_manager.py`) that
  `TerminationTermCfg` supports class-based terms identically to reward
  terms (`inspect.isclass(term_cfg.func)` -> instantiated once with
  `cfg=term_cfg, env=env`; a `.reset(env_ids)` method is called by the
  manager on every env reset) -- `settled`'s per-env consecutive-steps-
  below-threshold counter is real per-env state kept exactly this way,
  not a module global or a re-derived-every-step guess. Also confirmed
  `root_link_ang_vel_w` exists on the entity data class (`mjlab/entity/
  data.py`), parallel to the already-used `root_link_lin_vel_w`.
- Updated `scripts/go1_watch_protective_episodes.py`'s per-episode
  step-cap and printed termination-cause label (`"settled"` /
  `"time_out (never settled -- check this run)"`) to match -- episode
  length is now expected to vary noticeably run to run, which is the
  correct behavior, not a regression.

**Not yet done:** a fresh GPU training run with this change (needed to
confirm `settled_bonus` at weight 5.0 actually produces the intended
pressure rather than being drowned out or dominating, same "measure, then
adjust" discipline as every other weight in this task) and a fresh watch
of the resulting video.

## Gate 5 (Go1 protective fall), round 4: joint-limit reweighting + torque measurement, bundled with the settling change

Bundled into the same retrain as the `settled` termination above, so one
GPU run tests both.

**Why:** the trained policy's video looked uncontrolled and inconsistent
run to run. A re-read of arXiv:2512.01336's real Table I shows its two
largest weights by far are out-of-limit DoF position (-29) and
out-of-limit driving torque (-24.65), roughly 25-30x its root-linear-
velocity weight and 1000x+ its contact-force weights. Joint/torque-limit
avoidance therefore looks like a primary driver of that paper's
controlled falls, and it was nearly absent here: `joint_pos_limits` at
Go1's borrowed walking weight (-1.0, measured contribution ~-0.3/step vs
root_linear_velocity's ~-4.2/step) and no torque term at all. (A separate
hypothesis, a missing angular-velocity penalty, was checked against the
same table and rejected: the paper has no such term.)

**Changes:**
- `JOINT_POS_LIMITS_WEIGHT` -1.0 -> -10.0. Not the paper's -29 (raw
  quantity's definition not confirmed comparable to mjlab's soft-limit-
  overshoot `joint_pos_limits`); sized from our own round-3 measurement
  (raw mean ~0.33 late in training) so its contribution (~-3.3/step) is
  comparable to, not dominating, root_linear_velocity.
- Torque: mjlab has no out-of-limit-torque function (confirmed:
  `envs/mdp/rewards.py` only has `joint_torques_l2`, sum of squared
  `actuator_force`). Its raw scale on this task has never been measured,
  and three prior rounds showed that guessing a weight first just creates
  a new dominant term, so this round it is logged only
  (`Metrics/joint_torques_l2_mean`), NOT rewarded. Weight to be chosen
  from that measurement next round.
- `PYTORCH_JIT=0` now set via `os.environ.setdefault` at the top of
  `scripts/go1_train_protective.py` and `scripts/go1_watch_protective_episodes.py`
  (before torch is imported) instead of relying on a command-line prefix
  that had to be remembered on every invocation.

**Round 4, first GPU attempt:** training produced no checkpoint (the notebook's
checkpoint glob came back empty; Colab doesn't stop at a failed `!python`
cell, so the failure itself scrolled past). Suspected cause, fixed
pre-emptively pending the real traceback: `settled.reset()` wrote into its
per-env counter in place (`[env_ids] = 0` / `.zero_()`). That counter can be
an inference tensor when the frozen-eval env is built under
`torch.inference_mode`, which is the same bug class as the earlier
ContactSensor `history_length` crash. `reset()` now rebuilds the counter out of place.
The round-4 notebook now asserts `_exit_code == 0` after every script cell.

**Root cause confirmed locally (no GPU needed):** installed mjlab 1.6.0 +
CPU-only torch into a scratch venv on the dev machine (MuJoCo itself can't
load there: the Celeron N4120 has no AVX, and MuJoCo's wheels require it, so
env-level tests remain GPU-only). The crash path, read from the real source:
`FrozenEvalRunner._run_eval_rollout` (src/cartpole_term/runner.py) calls
`env.reset()` OUTSIDE inference mode, then steps inside
`torch.inference_mode()`. `settled.__call__` rebinds its counter each step, so
after the first eval that counter is an inference tensor; at the next eval,
mjlab 1.6.0's `TerminationManager.reset` passes `env_ids=slice(None)` to
`settled.reset`, and the old in-place `counter[slice(None)] = 0` raised. New
`tests/test_protective_settled.py` (7 tests, runs without mjlab/MuJoCo)
reproduces that exact sequence, asserts the in-place write raises, and
asserts the out-of-place reset doesn't; it also covers the hold/spike/
both-thresholds/per-env logic. Full suite: 46 passed. Every mjlab API and
config key the protective task uses (`inspect.isclass` class terms,
`get_term`, termination-before-reward order, `step_dt`,
`root_link_ang_vel_w`, `joint_torques_l2`, `time_out`/`fell_over`/
`push_robot`/`reset_base`/`reset_robot_joints`) re-checked against the
installed 1.6.0 source.

## Gate 5 (Go1 protective fall), round 5: settling becomes an eval pass criterion; torque penalty added

**Round 4 result (GPU, 500 iterations):**
- Training: `Episode_Termination/settled` was 0 throughout, `settled_bonus`
  was 0, and episode length stayed at the 150-step cap.
- Frozen eval at iteration 490, clean (mean) actions: mean episode length
  82.3 steps (1.6s), only 3.1% (8/256) never settled. Noisy (sampled)
  actions: 150.0 steps, 0 settled.
- Video (deterministic): 8/8 episodes ended `settled`, typically rolling onto
  the side.

**Cause:** PPO's exploration noise. The final action-rate contribution
(-0.1093 at weight -0.00667 -> raw 16.4 = 12 joints x 2*sigma^2) implies
a per-joint action noise of sigma ~0.83. At Go1's joint-position action
scale, that re-targets every joint by tenths of a radian every 20 ms, so
under sampled actions the base never stays below 0.5 rad/s for 0.5s. The
bonus therefore could never be earned and taught nothing. The clean policy
learned to settle from the root-velocity penalty alone.

**Round-4 final Episode_Reward values:** root_linear_velocity -0.3873,
vulnerability_contact_force -0.1648, action_rate_l2 -0.1093,
joint_pos_limits -0.0177 (the -10 weight largely eliminated limit
violations), settled_bonus 0. Measured raw `joint_torques_l2_mean` was about 200
and rising.

**Changes:**
- Removed `settled_bonus`. The `settled` termination stays: it never fires
  under training noise, so it's inert in training. In frozen eval with
  mean actions, `censored_fraction` = the share of falls that never come to
  rest, which is now Gate 5's "must come to a stop" pass criterion (FRASA's
  completion check is likewise an evaluation criterion). Alternative not
  taken: lowering exploration noise (`entropy_coef` / init noise std), a
  PPO hyperparameter that stays untouched without the user's approval, and
  unnecessary since the clean policy already settles 96.9% of the time.
- Added `joint_torques_l2` at -5e-4 (-> about -0.1/step at the measured raw
  ~200, on par with action_rate). Not arXiv:2512.01336's -2.9e-3, which would
  give about -0.58/step and become the dominant term, and whose raw-quantity
  definition isn't confirmed to match.
- Round-5 notebook prints the clean-policy settling pass rate from
  `eval_history.json`.

## Gate 5 (Go1 protective fall), round 6: log-only "space" metrics; round-5 results

**Round 5 result** (Colab runtime disconnected afterwards, so TensorBoard
logs were lost; only these survive):
- Frozen eval, clean actions, iterations 450-490: mean episode length
  73.0-75.9 steps, never settled 0.0-0.4%.
- Round 4 at iteration 490: 82.3 steps, 3.1%.
- Noisy actions: 150 steps.
- Video: 8/8 `settled`, after 72-142 steps.
- The user confirmed that ending on its side is acceptable (no orientation
  term).

**Normalization note:** mjlab's `Episode_Reward/*` = episode sum /
`max_episode_length_s` (mjlab/managers/reward_manager.py `reset()`), i.e. a
per-second rate over the MAX episode length. Values are comparable across
terms within a round, but not with rounds 1-3 (max length 0.2s vs 3.0s now).

**Space metrics** (`src/cartpole_term/protective/metrics.py`), registered
as mjlab `MetricsTermCfg` terms, which have no weight and no reward effect
(confirmed in the installed 1.6.0 source that a weight-0 reward term is
skipped entirely, so that couldn't be used for logging):
- `fall_envelope_radius` (reduce=max): farthest body from the fall-start
  root xy, i.e. the radius of the clear circle the fall needed.
- `root_travel_at_end` (reduce=last): start-to-end root travel.
- `body_spread_at_end` (last) and `body_spread_mean` (mean): RMS horizontal
  distance of the bodies from their centroid.

These are this project's own measures; no published fall reward penalizes
space directly (closest: target end-pose rewards, arXiv:2511.10635).

The fall-start point is captured on the first post-reset step, not in
`reset()`: `root_link_pos_w`/`body_link_pos_w` read derived `xpos`, which is
stale until the next forward() after the reset event injects the fall
state, so a read inside `reset()` would return the pre-reset terminal pose.
Cost: up to about 4 cm of bias at the bank's ~2 m/s median.

Baseline (local computation from `go1_flagged_states_seed0.npz`): uncontrolled
falls move the base 0.61 m median horizontally (p90 1.06, max 1.87) in the
first 0.58s, and 92% are still moving above 0.2 m/s at that point.

`tests/test_protective_metrics.py` (4 tests, no mjlab needed) covers the
origin-capture timing, the values, and per-env/inference-mode-safe reset.
The watch script prints these metrics per episode. The round-6 notebook zips
and downloads the run folder immediately after training.

**Next:** weight a space penalty from the measured values, after the user
picks the sense (A) ground covered, (B) sprawl, or both.

## Housekeeping (2026-09-25): filed loose Colab downloads, archived superseded notebooks

No code changes and nothing deleted. All moves are within gitignored folders
or of untracked notebooks. Round labels were inferred from download
timestamps vs when each round's results were reported. All five files had
distinct md5s, so there were no duplicates.
- `checkpoints/protective/model_499.pt` (2026-09-22) -> `round3_model_499.pt`
- `model_499 (2).pt` (09-25 17:41) -> `checkpoints/protective/round4_model_499.pt`
- `model_499 (3).pt` (09-25 19:17) -> `checkpoints/protective/round5_model_499.pt`
- `go1_protective_watch_seed0_ep8.mp4` (17:41) -> `outputs/videos/go1_protective_round4_watch_ep8.mp4`
- `go1_protective_watch_seed0_ep8 (1).mp4` (18:18) -> `outputs/videos/go1_protective_round5_watch_ep8.mp4`
- `go1_watch_seed0_ep3 (1).mp4` (09-23) -> `outputs/videos/go1_walker_fixed_command_3mps_ep3.mp4`
- `notebooks/{go1_train_protective,go1_train_protective_retrain,go1_protective_round4,go1_protective_round5}.ipynb`
  -> `notebooks/archive/` (superseded by `go1_protective_round6.ipynb`).

Top level is now only the project files (CLAUDE.md, DEVIATIONS.md,
README.md, pyproject.toml), `cartpole_term_code.zip` (the Colab upload
bundle every notebook asks for by this path), and folders.

## Gate 5 (Go1 protective fall), round 6 results (read locally from the logs zip)

The logs zip arrived intact (it was still transferring at first; `unzip -t`
clean). It was unpacked to `logs/rsl_rl/go1_protective_fall/round6_2026-09-25/`
and the archive moved to `archive/`. The checkpoint is filed as
`checkpoints/protective/round6_model_499.pt` (md5 identical to the zipped
`model_499.pt`). All numbers below were parsed from the tfevents file and
eval_history.json, not read off graphs.

**Training trend** (compared from iteration 6 onwards; iterations 0-5 are
biased short episodes from the runner's randomized initial episode ages;
iteration 0 mean episode length was 23 steps):

| metric | iter 6 | iter 200 | iter 499 |
|---|---|---|---|
| fall_envelope_radius (m) | 0.93 | 0.79 | 0.77 |
| root_travel_at_end (m) | 0.61 | 0.49 | 0.45 |
| body_spread_at_end (m) | 0.22 | 0.22 | 0.22 |
| Episode_Reward/vulnerability_contact_force | -1.26 | -0.17 | -0.16 |
| Episode_Reward/root_linear_velocity | -0.53 | -0.41 | -0.38 |
| Metrics/joint_torques_l2_mean | 155 | 80 | 96 |

- Value loss fell from 0.56 to ~0.001 (round 3 had it rising to 0.35-0.40), so
  that open item is resolved.
- Policy std went from 1.00 to 0.69.
- Final reward terms: root_linear_velocity -0.387, contact force -0.163,
  action_rate -0.075, joint_torques -0.046, joint_pos_limits -0.021. The
  torque term came in below the predicted -0.1 because the raw torque itself
  fell (230 -> ~90).

**Frozen eval (clean actions):**
- Mean return improved from -3.37 (iteration 0) to -1.60, and was flat from
  about iteration 200, so training converged around 200-300 of 500.
- Settling: 0-0.8% never settle in the last 5 evals, with one excursion to
  2.7% at iteration 400.
- Settle time drifted from 66 steps (iterations 50-100) to about 71 (iteration
  490), a mild late regression alongside torque creeping from 80 to 96.

**Key finding: the settling pass criterion does not discriminate.** At
iteration 0 the UNTRAINED policy (clean action ~0, i.e. PD-holding the
default joint pose) already settles 100% of eval falls, in 81 steps. So
"comes to a stop" is necessary but says nothing about whether the
protective policy helps. Gate 5's real pass test has to be the plan's
comparison against baselines (peak contact force / torque / envelope on
identical fall states).

**Space:**
- Sprawl (`body_spread`) is 0.20-0.23 m in every episode and every
  iteration, i.e. essentially the robot's own size, leaving nothing for a
  penalty to act on.
- Ground covered varies about 2x between episodes (video, clean: envelope
  0.55-1.12 m, root travel 0.30-0.77 m) and already fell 17%/26% from the
  root-velocity penalty alone.
- The video (`outputs/videos/go1_protective_round6_watch_ep8.mp4`) had 1/8 episodes never settle (150 steps), consistent with the
  eval rate.

## Gate 5 (Go1 protective fall), round 7: ground-covered penalty, 300 iterations, and the baseline-comparison pass test

Approved by the user after round 6.

- **Penalty:** `fall_envelope_growth` (`src/cartpole_term/protective/space.py`,
  renamed from `metrics.py` since it now holds a reward term too). Per step,
  it charges the increase in the farthest body's horizontal distance from the
  fall-start root position. Increments telescope, so an episode's raw sum is
  the final envelope minus the robot's own extent on the first post-reset
  step (which is not penalized). Weight -30, sized so Episode_Reward is
  about -0.1 (mjlab's Episode_Reward = weight x mean raw per step over the
  150-step episode; round 6's envelope was ~0.77 m under noise, and assuming
  ~0.3 m own extent, growth ~0.47 m -> ~0.0031/step). The own-extent figure
  is an estimate; `Metrics/fall_envelope_growth_mean` logs the real value.
  Sprawl is not penalized (constant ~0.22 m in round 6). This is this
  project's own term, not from the literature.
- **Iterations:** `PROTECTIVE_MAX_ITERATIONS` 500 -> 300 (round-6 eval return
  was flat from ~200).
- **Deterministic injection:** `inject_fall_state(..., sequential=False)`.
  When True, env i gets bank state i mod size on every reset. Training leaves
  it False (unchanged behaviour).
- **Gate 5 pass test:** `scripts/go1_protective_compare.py`, with
  bookkeeping in `src/cartpole_term/protective/compare.py`. It replays every
  bank state once under 3 arms:
  - the trained policy (deterministic)
  - the walking policy continuing (`checkpoints/model_9999.pt`, same 48-dim
    obs contract; its velocity command is whatever the protective env
    samples)
  - zero action (PD-hold the default pose, i.e. what round 6 showed an
    untrained policy already does, settling 100%)

  Details:
  - `auto_reset=False`, so post-step readings are each fall's true final
    state; only each env's first episode is recorded, and done envs are reset
    explicitly.
  - Per fall it records: peak ground-contact force (critical trunk and any
    body, from the protective contact sensor's per-body net force at each
    policy step), peak |actuator force|, mean squared torque, envelope and
    envelope growth, root travel, steps, settled, and return.
  - PASS = the protective policy's median beats every baseline's median on
    peak critical force, peak any-body force, peak torque, and envelope
    growth.
  - Asserts: all arms start from identical injected root poses, and actor
    weights are unchanged.
  - Output: `outputs/go1_protective_compare_seed0.json` plus `.npz`.
- **Tests:** `tests/test_protective_space.py` (+3 for the penalty: the
  telescoping total, per-env reset, inference-mode safety) and
  `tests/test_protective_compare.py` (3: first-fall-only recording, peaks,
  envelope, verdict).
- **Notebook:** `notebooks/go1_protective_round7.ipynb` (all uploads up
  front including the walker checkpoint; logs + comparison saved to Drive
  and downloaded).

## Gate 5 (Go1 protective fall), round 7 results: comparison verdict FALSE (3 of 4 metrics)

Run 2026-09-30, Colab, seed 0, 502 bank states (these are the training states,
so this is a train-set evaluation). Medians:

| arm | peak crit force N | peak any force N | peak torque Nm | envelope growth m | root travel m | steps | settled (mean) | return |
|---|---|---|---|---|---|---|---|---|
| protective | 0.0 | 1072 | 16.4 | 0.275 | 0.395 | 75 | 0.922 | -1.25 |
| walker | 501 | 1197 | 35.6 | 0.615 | 0.656 | 117 | 0.741 | -7.35 |
| zero_action | 574 | 1179 | 13.5 | 0.445 | 0.528 | 77 | 0.998 | -3.45 |

- The protective policy beats **both** baselines on critical-body force (the
  median fall never loads the trunk/hips), any-body force, and ground covered
  (envelope growth 0.275 m, down 55% vs the walker and 38% vs zero action).
- It beats the walker on all four metrics. That is the plan's original Gate 5
  criterion, which compares against the nominal policy's own continuation.
- It **loses on peak torque to zero_action** (16.4 vs 13.5 Nm). The
  zero_action arm (a stiff PD hold of the default pose) was added in round 7
  as a stricter baseline, so under the stricter rule the result is PASS=False.
  The pass rule is not being redefined after the fact; which rule governs is
  the user's decision.
- The protective policy settled in only 92.2% of falls, so about 39 falls hit
  the 3 s cap still moving. Zero action settled in 99.8%. This conflicts with
  the user's requirement that every fall comes to rest.
- **Per-fall and eval-history analysis** (files filed: `checkpoints/protective/round7_model_299.pt`,
  `logs/rsl_rl/go1_protective_fall/round7_2026-09-30/`,
  `outputs/go1_protective_compare_round7_seed0.json`,
  `outputs/videos/go1_protective_round7_watch_ep8.mp4`, zip in `archive/`;
  the per-fall `.npz` was not downloaded):
  - **Peak torque saturates.** The maximum of 35.55 Nm and the p90 of 23.7 Nm
    are Go1's knee and hip/thigh effort limits, so the median largely measures
    whether any joint hits its limit. Mean squared torque tells the same
    story: protective 38 vs zero_action 30 (medians); the walker is 391.
  - **The unsettled falls grew late in training.** Frozen-eval
    `censored_fraction` (mean action) was 0 through iteration 120, rose from
    iteration 130, peaked at 24% at iteration 240, and was 4% at iteration 290.
    Mean return barely moved after about iteration 100 (-1.95 to -1.88).
    Checkpoints around iterations 100-120 had essentially 100% settling at
    near-final return.
  - The protective policy's p90 critical-body force is 0 N. Its p90 any-body
    force is higher than the baselines (2825 vs 2609/2346 N): median better,
    tail worse.

## Gate 5 (Go1 protective fall): held-out evaluation of round 7 (notebook prepared 2026-09-30)

- **Why:** the round-7 comparison replayed the 502 training bank states, so it
  was a train-set evaluation.
- **Held-out bank:** `scripts/go1_collect_flagged_states.py --seed 1
  --capture-state-bank` is the identical collection (same walker, same 150 N
  push, same predictor and threshold) with collection seed 401 instead of 400.
  Output: `outputs/go1_fall_state_bank_seed1.npz`. The notebook asserts that no
  held-out state duplicates a training state (nearest-neighbour distance over
  root pose, quaternion, joint pos, and joint vel).
- The collector runs with a `PYTORCH_JIT=0` prefix (same Colab JIT crash as
  before). This affects speed only.
- **Checkpoints compared:** round-7 iterations 100, 120, and 299. Iterations
  100 and 120 were chosen from the frozen-eval history (0% unsettled up to
  iteration 120). Caveat: picking the best of three checkpoints on this
  held-out bank makes it partly a selection set; with three candidates that
  bias is small but real. A final paper number should come from a third bank
  (seed 2), or the held-out bank should be split.
- The notebook also records the package versions and GPU used
  (`outputs/versions_heldout_round7.json`), which is a start on the
  unpinned-versions gap.
- Notebook: `notebooks/go1_protective_heldout_eval_round7.ipynb`. All uploads
  are staged in `colab_uploads/heldout_eval/` (gitignored).

## Gate 5 (Go1 protective fall): held-out round-7 results (2026-09-30)

Files: `outputs/heldout_round7/` (held-out bank `go1_fall_state_bank_seed1.npz`,
three comparison JSON + per-fall npz files, versions); zip in `archive/`.

- **Versions (first record for the Go1 runs):** mjlab 1.6.0, mujoco and
  mujoco-warp 3.11.0, rsl-rl-lib 5.4.2, torch 2.11.0+cu128, warp-lang 1.17.0,
  CUDA 12.8, A100-SXM4-40GB. Same versions as the cartpole phase.
- **Held-out bank:** 505 states, live recall 1.0, median lead 14 steps (same
  as the training bank). Nearest-training-state distance is at least 4.2
  (median 12.2), so there is no overlap. Collection took only 5.7 s of
  rollout (152 steps x 4096 envs), so building more banks is cheap.

Medians on the held-out bank (baselines from the iteration-100 run; they are
recomputed in each run and agree):

| | crit force N | any force N | peak torque Nm | mean torque^2 | envelope growth m | steps | settled | pass |
|---|---|---|---|---|---|---|---|---|
| protective it100 | 0 | 1107 | 12.92 | 30.7 | 0.347 | 62 | 1.000 | True |
| protective it120 | 0 | 1126 | 12.78 | 30.4 | 0.342 | 63 | 1.000 | True |
| protective it299 | 0 | 1126 | 16.33 | 38.8 | 0.294 | 76 | 0.931 | False |
| walker | 507 | 1173 | 35.55 | 377.8 | 0.619 | 115 | 0.733 | |
| zero_action | 635 | 1235 | 13.38 | 28.4 | 0.470 | 77 | 0.998 | |

- Iterations 100 and 120 pass the verdict and settle 100% of the held-out
  falls. Iteration 299 reproduces its training-set result (generalizes, but
  fails on torque and settling).
- **The torque "win" is a tie, not a win.** Paired per fall at iteration 120,
  protective peak torque is lower than zero_action in 49.3% of falls and
  higher in 48.7%. Mean squared torque is slightly worse than zero_action
  (median 30.4 vs 28.4; worse in 60% of falls). What should be claimed is
  "torque comparable to a passive stiff hold, about 12x lower mean squared
  torque than the walker", not "lower torque than every baseline".
- The strong effects are trunk impact (critical force lower than zero_action
  in 89.5% of falls; p90 107 vs 1775 N) and ground covered (lower in 82%).
  Any-body force is close to a tie (lower in 52%; p90 slightly worse).
- **Selection caveat:** iteration 120 was chosen after seeing these held-out
  numbers, so they are selection-set numbers. A confirmatory run on a fresh
  bank (seed 2) with iteration 120 fixed in advance is needed for a reportable
  result.

## Gate 5 (Go1 protective fall): confirmation run, pre-registered 2026-09-30 (before running)

Fixed before any seed-2 data exists:
- **Candidate:** round 7, iteration 120 (`checkpoints/protective/round7_model_120.pt`),
  and only that checkpoint. No other checkpoint will be evaluated on this
  bank, and the bank will not be reused for selection.
- **Bank:** a fresh seed-2 bank (`go1_fall_state_bank_seed2.npz`, collection
  seed 402), same collection as seeds 0 and 1. It must share no state with
  either earlier bank (asserted in the notebook).
- **Pass rule:** unchanged `compare.py` verdict. The protective median must be
  strictly lower than both the walker and zero_action medians on peak critical
  force, peak any-body force, peak torque, and envelope growth. Settling rate
  is reported alongside; the user requirement is that every fall comes to
  rest.
- **Reporting:** the result is reported whatever it is. The torque and
  any-body-force comparisons are expected to be near-ties (seed-1 per-fall
  analysis), so a narrow FAIL on either is plausible and will be reported as
  such, not re-run.
- Notebook: `notebooks/go1_protective_confirm_seed2.ipynb`. Uploads are staged
  in `colab_uploads/confirm_seed2/`.

## Gate 5 (Go1 protective fall): confirmation result (seed 2): PASS = False

Pre-registered run as specified above. Round 7 iteration 120 on the fresh
seed-2 bank, medians:

| | crit force N | any force N | peak torque Nm | envelope growth m | steps | settled |
|---|---|---|---|---|---|---|
| protective it120 | 0 | 1206 | 12.74 | 0.322 | 63 | 1.000 |
| walker | 482 | 1155 | 35.55 | 0.607 | 116 | 0.747 |
| zero_action | 569 | 1193 | 13.08 | 0.414 | 77 | 1.000 |

- **FAIL on peak any-body force.** The protective median of 1206 N is higher
  than the walker (1155, +4%) and zero_action (1193, +1%). This was one of the
  two metrics flagged in advance as a near-tie. Per the pre-registration it
  is reported as a fail and not re-run.
- The other three metrics pass: critical force is 0 vs 482/569 N, peak torque
  beats both (12.74 vs 13.08; a near-tie again), and ground covered is 0.322
  vs 0.607/0.414 m. Settling is 100%.
- Reportable Gate 5 result: the protective policy eliminates trunk/hip impact
  in the median fall, reduces ground covered, torque is comparable to a
  passive stiff hold, and peak force on any body (in practice the legs/feet)
  is not reduced (within a few percent of the baselines either way across
  three banks).
- Likely mechanism, not yet verified: the reward weights contact force by
  vulnerability (high 1000 / medium 1 / low 0.5), so the policy protects the
  trunk by taking the impact on the legs. The any-body metric counts those
  leg/foot contacts at full weight.
- **Per-fall paired analysis** (files in `outputs/confirm_seed2/`; zip in
  `archive/`; same versions, mjlab 1.6.0). It covers seed 1 and seed 2 for
  iteration 120.
  - **Mechanism confirmed:** in 98.4% of protective falls the peak force is on
    a non-critical body (legs/feet). For zero_action the peak lands on the
    trunk/hips in 37-39% of falls.
  - **Any-body peak force is a coin flip** against both baselines (protective
    lower in 49-52% of falls on both banks). The upper tail is worse: p90 is
    2882-2947 N vs 2308-2822 N.
  - Critical force is lower in 86-91% of falls, envelope growth in 82-96%, and
    peak torque in 98% of falls vs the walker. Against zero_action, torque is
    a tie (median paired diff 0).
- **Decision (user, 2026-09-30):** accept Gate 5 as it stands (option A). The
  protective policy for Gate 6 is round 7, iteration 120. The any-body-force
  tie and the heavier leg-impact tail are stated limitations. No round 8.

## Gates 6 + 7 (Go1): live hand-off controller and ablation, pre-registered 2026-09-30 (before running)

**What:** `scripts/go1_handoff_eval.py`, notebook `notebooks/go1_handoff_eval.ipynb`,
uploads staged in `colab_uploads/handoff_eval/`. The plan had separate
Gate 6 (hand-off) and Gate 7 (ablation) scripts; they are one script
because the ablation is exactly the hand-off arm vs the same setup without
it.

- **Live system:**
  - The walker (`model_9999.pt`, stochastic actions, as in every predictor
    collection) runs under the locked 150 N impulse push, plus the shipped
    `push_robot`.
  - The predictor scores the walker's raw (obs, sampled action) each step,
    exactly as `go1_collect_flagged_states.py` does, at the val-selected
    threshold.
  - On the first crossing, a per-env latch switches control to the
    protective policy (round 7, iteration 120, deterministic) from that same
    step onwards, irreversibly (hard binary trigger, the SafeFall
    precedent).
  - Pushes keep firing after the switch in both arms. The protective policy
    was trained without pushes, which is a known train/deploy difference.
    Its obs "last action" slot holds the walker's last action at the switch
    (zeros in training).
- **Arms:** handoff vs nominal (latch never wired). Each arm uses a fresh env
  with identical seeds (env seed + 500, torch seed). The walker is sampled
  for every env in both arms, so the RNG streams match.
- **One trial per env:** 1024 envs, nothing ever reset (DeferredResetEnv
  holds every done env). Walking budget 1000 steps (the walker's 20 s
  episode); env time limit extended by the 150-step window.
- **Measurement window:** opens at the flag, or at an unflagged real fall.
  It closes on settle (0.2 m/s and 0.5 rad/s held 0.5 s, same as the
  protective task), after 150 steps, or at the time limit. Metrics are the
  same as `compare.py`, with the envelope origin at the window start.
- **Pairing:** envs pair if their event, flag step, and root pose at the
  window start are identical across arms. This depends on GPU sim
  determinism, which is measured, not assumed.
- **Gate 6 pass:**
  - Asserted: all three networks' weights unchanged; no env switches back;
    the handoff arm switches exactly its flagged envs.
  - Required: live recall >= 0.95 in the nominal arm (earlier live
    measurement: 1.0, median lead 14 steps). Precision and lead are
    reported.
- **Gate 7 pass:**
  - Requires >= 95% of event trials to pair. Otherwise the paired test is
    reported as not evaluable, and only unpaired all-event summaries are
    given.
  - Test set: the paired fall trials, i.e. events whose nominal-arm
    counterpart really fell in the window (true alarms + missed falls).
  - Pass if the handoff median is lower than the nominal median on peak
    critical force, peak any-body force, peak torque, and envelope growth
    (the Gate 5 metrics, vs the walker only, as the plan's ablation
    specifies).
  - Given Gate 5, any-body force vs the walker is a plausible narrow fail
    (seed 2: 1206 vs 1155 N). It will be reported as-is.
  - False alarms (flag, but the nominal walker did not fall) are reported
    separately; they are what the hand-off costs when it is wrong.
- **Code:**
  - `protective/compare.py` gains `FirstFallTracker.begin()` and
    `HandoffTrials` (7 new tests, `tests/test_protective_handoff.py`; 63/63
    pass locally).
  - `protective/env_cfg.py`'s contact sensor was factored into
    `protective_contact_sensor_cfg()` (shared, so identical).
  - `cartpole_term_code.zip` rebuilt; the old one is in `archive/`.
- **Found while reading, not changed:** the Go1 predictor takes raw,
  un-normalized inputs in training and live scoring alike (consistent, so
  not a bug here). CLAUDE.md S8.1 asks for a train-split-fit input
  normalization; it was never implemented for the Go1 predictor. Record it
  as a deviation for the paper.
- **Demo video (rest of Gate 7):** `scripts/go1_handoff_demo_video.py`,
  added as step 4b of the same notebook.
  - It shows the same rollout seed side by side: the walker alone vs the
    hand-off. It uses the same env factory, `HandoffTrials`, and models as
    the eval.
  - It keeps only true-alarm takes (flag, then the walker really falls),
    3 by default.
  - Overlay: predictor bar and threshold, controller in charge, a
    flag/hand-off banner, and the end state.
  - Forward-only commands are used for a clean pre-push gait (as in
    `go1_wildpush_video.py`); this is demo only, the eval uses the real
    command distribution.
  - It reports whether the two halves' flag steps matched (determinism).
    The overlay composition was dry-run locally on synthetic frames.
    Output: `outputs/videos/go1_handoff_demo_seed0.mp4` plus a `.json`.

## Gates 6 + 7 (Go1): live run results (2026-09-30)

Files: `outputs/handoff_eval/` (JSON, npz, versions); zip in `archive/`. The
demo video was not in the zip (not run).

- **GATE 6 PASS.** Nominal arm, 1024 trials:
  - 915 flagged, 1 missed fall, 108 no event.
  - 763 true alarms, 152 false alarms (flag, but no fall within the 3 s
    window).
  - Live recall 0.9987, precision 0.834, median lead 15 steps (0.30 s;
    earlier live measurement 14).
  - The handoff arm switched exactly its 909 flagged envs, none switched
    back, and all weights were unchanged.
- **Gate 7 (live, paired): not evaluable.** Pairing was 0.0.
  - Per-env diagnosis from the npz: poses already differed by 0.007-7 m at
    the very first flags (step 62), and flag steps matched in only 83 of 846
    envs.
  - Cause (a script bug): each arm built its own env. mjlab's startup events
    (per-env physics randomization) run at construction, before any seeding
    here, so the two arms simulated different robots.
  - Fixed: one env is reused for both arms, reseeded and reset, with
    `common_step_counter` zeroed. The fix also records whether the initial
    states are bit-identical.
  - That alone is not enough: mjlab's own `seed_rng` docstring says "MuJoCo
    Warp is not fully deterministic yet" (mujoco_warp#562). Live pairing
    across hundreds of pre-flag steps cannot be relied on.
- No Gate 7 numbers were looked at. The unpaired summaries in the JSON were
  not analysed.

## Gate 7 (Go1), amended pre-registration, 2026-09-30 (before running)

- **Design change (a methods fix, not result-driven):** pair by
  construction instead of by determinism.
  - `go1_collect_flagged_states.py --capture-at-flag` saves each robot's exact
    state at the flag step: the hand-off moment, rather than the Gate 5 banks'
    uniform draw from [flag, fall).
  - `go1_protective_compare.py` replays each state under the protective policy
    (the hand-off), the walker (no hand-off), and zero action. Identical start
    states are asserted.
- **Bank:** fresh seed 3 (collection seed 403), same live collection. The
  notebook asserts captured_step == flag_step and no overlap with the seed
  0-2 banks.
- **Pass (unchanged metrics, vs the walker only, as pre-registered):** the
  hand-off median is lower than the walker median on peak critical force,
  peak any-body force, peak torque, and envelope growth. Zero action and
  settling are reported.
- **Known differences from the live system:**
  - No pushes after the flag in the replay. The protective policy was
    trained without pushes, and the walker arm gets none either.
  - A push in progress at the flag is cut off.
  - Obs "last action" starts at zero, as in training.
- **Scope of the bank:** it holds only flags that were followed by a real
  fall (the collector's resolved trials). False alarms (17% of flags live)
  are covered by the live Gate 6 numbers, not by this test.
- Notebook: `notebooks/go1_gate7_flagstate.ipynb` (also runs the demo video).
  Uploads are staged in `colab_uploads/gate7_flagstate/`.
- **Demo video:** it prefers takes whose two halves flag on the same step,
  and falls back to unmatched takes (labelled in its JSON) if too few match.

## Housekeeping (2026-10-03)

- `checkpoints/protective/round3_model_499.pt` renamed to
  `round1_model_499.pt`. Its timestamp (2026-09-22 23:41) predates the round 2
  and 3 reward changes of 2026-09-23, so it is the first full run. The
  2026-09-25 housekeeping entry's "round3" label was wrong.
- All notebooks under `notebooks/` (not the archive) now install
  `mjlab==1.6.0 rsl-rl-lib==5.4.2`, the versions recorded on 2026-09-30.
  mujoco, mujoco-warp, and warp-lang follow from mjlab's own pins. torch is
  whatever the Colab image ships (2.11.0+cu128 when recorded) and is not
  pinned, because replacing Colab's CUDA build is slow and fragile.

## Gate 8 (Go1): consolidated notebook (2026-10-03, written, not run end to end)

- `notebooks/go1_end_to_end.ipynb`: Parts A-E cover Gates 0-7 for Go1 in one
  file. Every code cell is a `scripts/*.py` call plus path glue.
  `SKIP_WALKER_TRAINING` lets a run start from an existing `model_9999.pt`.
- **Not executed as a whole.** Each script in it has been run on Colab
  individually, except the Gate 7 pieces (pending at the time of writing). A
  full run repeats the 10,000-iteration walker training.
- **Difference from the plan:** the plan's notebook was parameterized by
  `ROBOT` and used generalized `robot_*` scripts for every part. Only Part A
  has those; Parts B-E call the `go1_*` scripts, because the G1 port of the
  predictor and protective stages was never built. The notebook is Go1-only.
- **Difference from how the results were produced:** it trains the protective
  policy for 121 iterations and takes `model_120.pt`. The reported checkpoint
  is iteration 120 of a 300-iteration run. Same seed and schedule up to that
  point, but MuJoCo Warp is not bit-deterministic, so a re-run will not
  reproduce the weights exactly.
- It builds the held-out Gate 5 bank with seed 2 and the Gate 7 bank with
  seed 3, as in the reported results. It skips the seed-1 selection step,
  since the iteration is now fixed.

## Gate 7 (Go1): flag-moment ablation result (2026-10-03): PASS = False

Files: `outputs/gate7_flagstate/` (flag bank, comparison JSON + npz,
versions); video and its JSON in `outputs/videos/`; zip in `archive/`.
Versions: mjlab 1.6.0, mujoco and mujoco-warp 3.11.0, rsl-rl-lib 5.4.2,
warp-lang 1.17.0, on an A100. The Colab image's torch build changed to
**2.11.0+cu130 (CUDA 13.0)**; it was cu128 on 2026-09-30.

- **Bank (seed 3, collection seed 403):** 500 states, every one captured at
  the flag step (asserted), no overlap with the seed 0-2 banks. Live recall
  1.0; flag-to-fall median 14 steps (range 3-74).
- **Medians, 500 flag moments, round 7 iteration 120:**

| | crit force N | any force N | peak torque Nm | envelope growth m | steps | settled |
|---|---|---|---|---|---|---|
| hand-off (protective) | 0 | 1159 | 12.86 | 0.370 | 67 | 1.000 |
| walker continues | 380 | 1026 | 35.55 | 0.871 | 128.5 | 0.620 |
| zero action | 611 | 1323 | 13.68 | 0.547 | 82 | 0.996 |

- **Pre-registered verdict (hand-off vs walker on all four): FAIL**, on peak
  any-body force (1159 vs 1026 N, +13%). The other three pass. Reported
  as-is, as pre-registered.
- **Per fall (paired, identical start states), hand-off vs walker:**
  - critical force lower in 71.8% of falls, higher in 4.0% (median paired
    difference -350 N);
  - any-body force lower in 43.2%, higher in 56.8% (median +106 N);
  - peak torque lower in 98.2% (median -20.9 Nm);
  - envelope growth lower in 96.2% (median -0.44 m).
  - Hand-off p90 any-body force is 2858 N vs 2258 N for the walker.
- **Against zero action:** critical force is lower in 89.0% of falls,
  envelope growth in 91.0%, peak torque in 58.0%, any-body force in 52.0%.
- The hand-off's peak force is on a non-critical body in 97.4% of falls. This
  is the same trunk-to-legs trade as Gate 5, and from the true hand-off moment
  it is a clear loss on any-body force against the walker, not a tie.
- **Demo video:** 3 takes (rollout seeds 701-703), 17.6 s. All three takes'
  halves flagged on the same step (135, 486, 331), so single-env rollouts
  did pair.
  - Defect: it rendered at mjlab's default 320x240 per half, so the overlay
    text (laid out for 640x480) is cut off.
  - Fixed in `scripts/go1_handoff_demo_video.py` (`cfg.viewer.width/height`
    = 640x480, plus an upscale fallback).
  - Re-render with `notebooks/go1_handoff_demo_video.ipynb` (uploads in
    `colab_uploads/demo_video/`). The 320x240 file is kept until replaced.

## Gate 5/7 redo: round 8, pre-registered 2026-10-03 (before any round-8 data)

The user asked for a redo after Gate 7 failed. Re-running the same test would
not be a redo, so this is a new training round aimed at the failed metric,
tested once on fresh data.

- **Reward change (both variants):** one added term,
  `peak_contact_force_growth` (rewards.py), weight
  `PEAK_CONTACT_FORCE_WEIGHT = -0.02`.
  - It returns the per-step increase in the episode's running peak contact
    force on any single body, so its episode sum is exactly
    `peak_force_any_n`, the metric rounds 7's checkpoint failed on.
  - Why: the existing per-Newton force term charges non-critical bodies only
    -0.001/N per step, so a single hard leg impact was nearly free.
  - Weight calibration (ours, not literature): a 1100 N peak costs
    1100 x 0.02 x dt = -0.44 of return, the same order as the envelope term
    (-0.28).
  - All other terms and weights are unchanged from round 7. PPO config
    unchanged, 300 iterations.
  - Limitation: like every force reading here it is the instantaneous force
    at 50 Hz policy steps, not a substep peak. The evaluation metric is
    sampled the same way.
- **Two variants, trained in the same notebook:**
  - **8a:** trained on the seed-0 fall bank (states drawn uniformly from
    [flag, fall)), as in round 7. Isolates the reward change.
  - **8b:** trained on a new flag-moment bank (seed 5, `--capture-at-flag`).
    The hand-off always starts at the flag, so this matches deployment; it is
    SafeFall's Stage II idea (init from predictor-flagged states). From the
    flag moment the best policy may avoid the fall altogether; that would
    count as a valid protective outcome.
- **Banks and their roles (no bank has two roles):**
  - train: fall seed 0 (8a), flag seed 5 (8b);
  - selection: flag seed 3 (the bank the failed Gate 7 used; already seen,
    so selection only);
  - test: flag seed 4 (new, used once).
- **Selection rule (mechanical, `compare.select_checkpoint`, unit-tested):**
  - Candidates: iterations 80, 100, 120, 150, 200, 299 of each variant (12
    checkpoints), each compared on the selection bank.
  - Eligible = settled rate >= 0.99 and median below the walker's on all four
    pass metrics.
  - Pick the eligible checkpoint with the lowest median any-body peak force.
  - If none is eligible, pick the lowest any-body force among those with
    settled >= 0.99 and report that it was not eligible.
- **Test (run once, on the selected checkpoint only):** same rule as the
  failed Gate 7. The hand-off median must be lower than the walker's on peak
  critical force, peak any-body force, peak torque, and envelope growth, on
  the seed-4 flag bank. Zero action and settling are reported. The result is
  reported whatever it is; it replaces nothing. The round-7 Gate 5 and Gate 7
  fails stay on record.
- **Code:** `peak_contact_force_growth` (+4 tests), `select_checkpoint` (+2
  tests), `scripts/go1_select_protective_checkpoint.py`; 69/69 tests pass
  locally. Notebook: `notebooks/go1_round8_redo.ipynb`; uploads staged in
  `colab_uploads/round8/`.

## Round 8 pre-registration, amended 2026-10-03 (still before any round-8 data)

- **User requirement, stated 2026-10-03:** "its ok if the force on the robot
  is still a lot, it should be reduced a bit but the main goal is to decrease
  the amount of space taken up." The pass condition and the selection rule
  now follow that. This supersedes the selection rule and test in the entry
  above.
- **By this standard the existing round-7 result already meets the main
  goal.** On the seed-3 flag bank the hand-off covers 0.370 m vs 0.871 m for
  the walker (58% less; lower in 96% of falls) and 0.547 m for zero action
  (32% less), and settles 100%. Applying the amended verdict to that file
  gives space-goal PASS = True, any-body force below the walker = False. That
  file was the earlier test, so this is a re-reading of seen data, not a
  fresh result.
- **Round 8 still runs,** aimed at "force reduced a bit". The reward term and
  both variants are as in the entry above.
- **Round 7's iteration-120 checkpoint is now a 13th candidate** in the
  selection, so round 8 only replaces it if it is better under the rule.
- **Selection rule (`compare.select_checkpoint`, 3 tests):**
  - eligible = settled >= 0.99 and median below the walker's on ground
    covered, trunk/hip force, and peak torque;
  - preferred = eligible and also below the walker on any-body peak force;
  - pick the lowest median ground covered among the preferred checkpoints, or
    among the eligible ones if none is preferred.
- **Test (once, seed-4 flag bank, the selected checkpoint):**
  - PASS (space goal) = median ground covered below both the walker's and
    zero action's, settled >= 0.99, and trunk/hip force and peak torque below
    the walker's.
  - Reported beside it, not part of the pass: whether median any-body peak
    force is below the walker's.
- The earlier four-metric verdicts (the Gate 5 fail on seed 2 and the Gate 7
  fail on seed 3) stay on record as the results under the rule in force when
  they were run.
- 70/70 tests pass locally. Uploads: 7 files in `colab_uploads/round8/`.

## Changes made while round 8 was running (2026-10-03; none are in that run's uploaded code)

- **Perturbations stop once a robot is on the ground** (user request).
  - Scope: the live hand-off env only (`go1_handoff_eval.py`,
    `go1_handoff_demo_video.py`). The protective training env and the
    bank-replay comparison never had pushes. Collection envs reset on a fall,
    so they are unaffected.
  - "Down" (`HandoffTrials.down`, latched for the trial) = the `fell_over`
    termination has fired, or the root has dropped below 0.5 x its own height
    at reset. The 0.5 is our definition, needed because a robot lowered by the
    protective policy can be on the ground without tilting 70 degrees.
  - Mechanism (`src/cartpole_term/common/perturbation.py`):
    - `apply_body_impulse_until_down` subclasses mjlab's `apply_body_impulse`.
      For envs in `env.perturbations_off` it zeroes an impulse in progress and
      sets the cooldown so no new one starts.
    - `push_by_setting_velocity_until_down` wraps mjlab's function and drops
      those env ids.
    - Both were confirmed against mjlab 1.6.0's `envs/mdp/events.py`,
      including that the event's `reset()` re-samples the cooldown, so a full
      reset clears the switch.
    - `--keep-pushing-when-down` restores the old behaviour.
  - Effect on recorded results: none. Gate 6's detection numbers are measured
    before the fall; Gate 7 used the replay. A re-run of the live eval would
    show different post-fall window metrics (fewer unsettled falls).
- **Demo video text:** every overlay line is now drawn at the largest font
  that fits the frame width (`_draw_fitted`), on top of the 640x480 render fix.
  Dry-run locally at both 640- and 320-pixel inputs.
- **Reward switch:** `go1_protective_fall_env_cfg(peak_force_weight=...)` and
  `go1_train_protective.py --peak-force-weight`. 0 drops the round-8 term and
  reproduces the round-7 reward. `go1_end_to_end.ipynb` passes 0, because the
  reported policy is still round 7.
- **Bug found and fixed:** with a non-default `--state-bank-file`, the
  training env used that bank but the in-training frozen eval used the
  default seed-0 bank (`ProtectiveFrozenEvalRunner._default_env_cfg_factory`).
  The train script now passes one factory to both.
  - Consequence for the running round 8: variant 8b's `eval_history.json` is
    measured on seed-0 fall states, not on its own flag-moment bank. Its
    training, the selection, and the test are unaffected (they use separate
    comparison runs).
- **Housekeeping:**
  - `CLAUDE.md` is now a short pointer file. The cartpole spec moved unchanged
    to `docs/CARTPOLE_SPEC.md`; "CLAUDE.md S<n>" references mean that file.
    The spec's own layout section says CLAUDE.md is "this file, unmodified";
    the user approved the move.
  - 13 stage notebooks moved to `notebooks/archive/`; `notebooks/README.md`
    indexes all of them. Paths like `notebooks/go1_gate7_flagstate.ipynb` in
    earlier entries now live under `notebooks/archive/`.
  - Removed `colab_uploads/{heldout_eval,confirm_seed2,handoff_eval,gate7_flagstate}`
    after checking by hash that every file in them is a copy of a kept file.
  - `tools/build_code_zip.py` rebuilds the code zip and refreshes the copies in
    `colab_uploads/`. `tools/notebook_builders/` holds the generators for the
    three current Go1 notebooks; regenerating reproduces them byte for byte.
    The video notebook now takes whichever `round*_model_*.pt` is uploaded.
- Tests: 75/75 pass locally (new: 3 perturbation, 2 down-latch).

## Round 8 results (2026-10-03)

Files: `outputs/round8/` (selection JSONs under `select/`, both new banks,
the test comparison JSON + npz, versions); logs without checkpoints in
`logs/rsl_rl/go1_protective_fall/round8{a,b}_2026-10-03/`; selected checkpoint
`checkpoints/protective/round8b_model_200.pt`; video
`outputs/videos/go1_handoff_demo_round8b_it200.mp4`; zips in `archive/`. The
full run folders (all checkpoints) are only on Drive
(`MyDrive/cartpole_term_runs/round8/`). Versions as on the Gate 7 run
(mjlab 1.6.0, torch 2.11.0+cu130, A100).

- **Which notebook ran:** the first round-8 notebook, i.e. before the
  space-first amendment: 12 candidates (round 7 was not re-compared), the
  original selection rule (lowest any-body force among checkpoints beating the
  walker on all four), and the four-metric verdict cell.
  - Applying the amended rule to the same selection files, with round 7's
    existing seed-3 result added as the 13th candidate, selects the **same**
    checkpoint. So the test bank was spent on the checkpoint both rules pick.
- **Selection (flag bank seed 3), medians.** Walker 1073 N / 0.911 m, zero
  action 1323 N / 0.530 m:

| candidate | any force N | peak torque Nm | ground m | settled |
|---|---|---|---|---|
| 7 it120 (earlier run) | 1159 | 12.9 | 0.370 | 1.000 |
| 8a it100 | 1071 | 13.6 | 0.444 | 1.000 |
| 8a it299 | 1043 | 17.5 | 0.379 | 0.998 |
| 8b it150 | 1038 | 15.6 | 0.392 | 0.990 |
| **8b it200** | **974** | 18.6 | **0.363** | 0.998 |
| 8b it299 | 942 | 21.5 | 0.345 | 0.980 (not eligible) |

  - Every candidate has median critical force 0.
  - Under the amended rule the preferred set is 8a it100, 8a it299, 8b it150,
    8b it200; the lowest ground covered among them is 8b it200.
  - Run-to-run noise: the walker's median any-body force on this same bank was
    1026 N in the Gate 7 run and 1073 N here (MuJoCo Warp non-determinism),
    so differences of a few percent between medians are within noise.
- **Test (fresh flag bank seed 4, 500 flag moments, run once on 8b it200).**
  Live recall 0.996; no overlap with the seed 0, 3, 5 banks. Medians:

| | crit force N | any force N | peak torque Nm | ground m | steps | settled |
|---|---|---|---|---|---|---|
| hand-off (8b it200) | 0 | 994 | 18.4 | 0.365 | 70 | 0.998 |
| walker continues | 347 | 991 | 35.6 | 0.940 | 134 | 0.582 |
| zero action | 609 | 1240 | 13.4 | 0.531 | 83 | 0.994 |

- **Verdict under the rule this run printed (four metrics vs walker): FAIL,**
  on any-body force, by 3 N (994 vs 991): a tie, not below.
- **Verdict under the amended, space-first rule (written before any round-8
  data): PASS.**
  - Ground covered is 61% below the walker and 31% below zero action.
  - Settled 99.8% (1 unsettled fall of 500).
  - Critical force and peak torque are below the walker.
  - Reported beside it: any-body force is not below the walker.
- **Per fall (paired, identical start states):**
  - vs the walker: ground covered lower in 97.4% of falls (median -0.52 m);
    critical force lower in 69.0%, higher in 3.2%; peak torque lower in
    95.6%; any-body force lower in 46.4%, higher in 53.6% (median +57 N).
  - vs zero action: ground covered lower in 88.6%; critical force lower in
    90.0%; any-body force lower in 58.8% (median -232 N); peak torque
    **higher** in 71.6% (median +4.0 Nm).
  - The trunk never touches the ground in 89.8% of falls.
- **What round 8 changed against round 7** (different test banks, so
  indicative only):
  - any-body force relative to the walker went from +13% (seed 3) to about
    0%;
  - ground covered is unchanged (0.37 m);
  - peak torque rose from 12.9 to 18.4 Nm, and its tail is higher (p90 27.6
    vs 22.1 Nm);
  - the any-body force tail is still above the walker's (p90 2652 vs 2332 N).
- **Variant 8b (trained from flag moments) beat 8a** on both any-body force
  and ground covered at the same iteration, on the selection bank.
- **Demo video:** 3 takes (seeds 701, 702, 705), halves 640x480, text fits,
  all flag steps matched. It was rendered with the uploaded code, so pushes
  still continue after the fall in it.
- **After round 8:**
  - README section 14 now reports round 8b iteration 200 as the protective
    policy; the round-7 tables are kept and labelled.
  - `go1_end_to_end.ipynb` (regenerated from `tools/notebook_builders/`) now
    trains round 8b: flag-moment bank seed 5, round-8 reward, 201 iterations.
    It tests on flag bank seed 4 with the space-first verdict. The seed-2
    Gate 5 cell was dropped, because no such test exists for round 8. Still
    not run end to end.
  - `colab_uploads/demo_video/` now carries `round8b_model_200.pt`, for a
    re-render with pushes stopping once the robot is down.
- Not done for round 8: the live hand-off check (Gate 6) with the new
  checkpoint, and any multi-seed repeat.

## Final demo video (2026-10-03)

- `outputs/videos/go1_handoff_demo_round8b_it200.mp4` (15.5 s, 3 takes,
  rollout seeds 700, 701, 705; halves 640x480; all three takes' halves flagged
  on the same step). Rendered with the current code, so **pushes stop once the
  robot is down**. This was the first GPU run of
  `common/perturbation.py`; it ran without error, and the user reviewed the
  video ("they look great").
- Per take, walker alone -> hand-off: ground covered 1.11 -> 0.28 m,
  0.57 -> 0.46 m, 1.42 -> 1.14 m; trunk/hip peak force 553 -> 0 N,
  799 -> 0 N, 180 -> 0 N. All hand-off halves came to rest; the first
  walker half did not within 3 s.
- The earlier render of the same checkpoint (pushes continuing after the
  fall) is kept as `..._pushes_continue.mp4`; the round-7 320-pixel render as
  `go1_handoff_demo_round7_it120_320px.mp4`.

## Limp baseline, pre-registered 2026-10-03 (before running)

- **User request:** also compare against a limp policy: once the failure is
  detected, no torque is applied at any joint.
- **Why a new arm:** the existing `zero_action` arm is not limp. Action 0 is a
  position target at the default pose, so the PD motors hold the joints
  stiffly (median peak torque about 13 Nm).
- **Implementation (`go1_protective_compare.py`, arm `limp`):**
  - The env gets one startup event, mjlab's own `dr.pd_gains` with
    `kp_range=(0, 0)`, `kd_range=(0, 0)`, `operation="scale"`.
  - Go1's actuators are `BuiltinPositionActuator` (MuJoCo `<position>`,
    force = kp*(ctrl - q) - kd*qd). Confirmed in mjlab 1.6.0
    `envs/mdp/dr/actuator.py`: for that actuator type the function writes
    `actuator_gainprm[..., 0]`, `actuator_biasprm[..., 1]`, and
    `actuator_biasprm[..., 2]`, so the actuator force is identically zero
    whatever the action.
  - The script raises if the limp arm's measured peak actuator torque exceeds
    1e-4 Nm.
  - What still acts on a limp robot: gravity, contact, joint limits, and the
    motors' reflected inertia (armature). No passive joint damping or friction
    is added. A real unpowered gearbox would add some, so this is the most
    free-falling version of limp.
- **In the replay, the hand-off moment is step 0** (the bank states are flag
  moments), so "limp after detection" is motors off from the first step.
- **Test:** the reported policy (round 8b iteration 200) on the **seed-4 flag
  bank**, the same 500 falls as the reported round-8 table, now with four
  arms.
  - Re-using seed 4 is legitimate here: nothing is being selected or tuned,
    only a baseline is added.
  - Because the simulator is not deterministic, the three existing arms'
    medians will differ a little from the first run; the new run's four
    numbers are the ones to compare with each other.
- **Pass rule vs limp (space-first, as for the other baselines):** the
  hand-off's median ground covered is below limp's, and it settles >= 99%.
  - Reported beside it: trunk/hip force and any-body force against limp.
  - Torque is not compared (limp is zero by construction).
- **No expectation is assumed.** A limp robot may crumple close to where it
  is and cover little ground; if it covers less than the protective policy,
  that is reported as a fail against this baseline.
- The script's original four-metric verdict is unchanged and still uses its
  three original arms; the old "GATE 5 PASS" print line is renamed to say so.
- Notebook: `notebooks/go1_limp_baseline.ipynb`; uploads in
  `colab_uploads/limp_baseline/`.

## Limp baseline result (2026-10-03): space PASS against limp

Files: `outputs/limp_baseline/` (JSON, npz, versions); zip in `archive/`.
Same versions as the round-8 run (mjlab 1.6.0, torch 2.11.0+cu130, A100).
Motors-off confirmed: the limp arm's peak actuator torque and mean squared
torque are exactly 0 over all 500 falls.

Medians, seed-4 flag bank (500 flag moments), round 8b iteration 200:

| | ground m | crit force N | any force N | peak torque Nm | steps | settled |
|---|---|---|---|---|---|---|
| hand-off (protective) | 0.364 | 0 | 1001 | 18.4 | 70 | 0.998 |
| walker continues | 0.949 | 363 | 995 | 35.6 | 136 | 0.586 |
| zero action (stiff hold) | 0.522 | 617 | 1292 | 13.2 | 83 | 1.000 |
| limp (motors off) | 0.459 | 281 | 1099 | 0 | 77 | 0.998 |

- **Pre-registered rule vs limp: PASS.** Ground covered is 21% below limp's
  (0.364 vs 0.459 m), and the hand-off settles 99.8%.
- **Per fall (paired) vs limp:**
  - ground covered lower in 77.0% of falls, higher in 22.2% (median -0.11 m);
  - critical force lower in 72.0%, higher in 4.8% (median -249 N);
  - any-body force lower in 55.4%, higher in 44.6% (median -107 N).
- The trunk never touches the ground in 89.8% of protective falls vs 24.8% of
  limp falls.
- **Limp is the strongest naive baseline.** It beats the stiff hold on ground
  covered (0.459 vs 0.522 m), critical force (281 vs 617 N), and any-body
  force (1099 vs 1292 N), and beats the walker on ground covered and critical
  force, at zero torque. The protective policy's margin over it on space
  (21%) is smaller than over the walker (62%) or the stiff hold (30%).
- **Tails:** the protective p90 any-body force (2652 N) is still above limp's
  (2530 N). p90 ground covered: 0.89 m vs limp 1.13 m.
- **Reproducibility of the earlier run:** the three original arms came out
  within about 1% of the first seed-4 run (protective 0.364 vs 0.365 m,
  994 vs 1001 N; walker 0.949 vs 0.940 m). The original four-metric verdict
  on those arms is again False (any-body force 1001 vs walker 995 N, a tie).

## Stiff-hold baseline (`zero_action`) removed, 2026-10-03 (user decision)

- **Why:** on the four-arm seed-4 run, limp was better than the stiff hold on
  ground covered (0.459 vs 0.522 m), critical force (281 vs 617 N), and
  any-body force (1099 vs 1292 N), with zero torque (vs 13.2 Nm). The one
  thing the stiff hold did marginally better was settling (500 of 500 falls
  vs 499). The user asked for it to be dropped.
- **What changed:**
  - `go1_protective_compare.py` runs three arms: protective, walker, limp.
    Its printed four-metric verdict is now protective vs the walker only;
    limp is printed beside it.
  - The space-first pass rule in the notebooks (`go1_end_to_end`,
    `go1_round8_redo`, `go1_limp_baseline`) is now: ground covered below both
    the walker's and limp's, settled >= 99%, and trunk force and torque below
    the walker's.
  - The README's round-8 table has three rows.
- **What did not change:**
  - Every result file and entry written before this still contains
    `zero_action`, and the verdicts recorded then (which required beating it)
    stand as recorded.
  - The README's round-7 tables (Gate 5 on seed 2, Gate 7 on seed 3) keep
    their zero-action rows, because no limp arm was run for round 7.
  - `compare.verdict()` is generic over whichever arms it is given; its unit
    test still uses the name `zero_action` as an arbitrary baseline label.
- The round-8 result under the new rule is unchanged: PASS (0.364 m vs walker
  0.949 and limp 0.459; settled 99.8%; trunk force 0 vs 363; torque 18.4 vs
  35.6).

## Demo video: limp added as a third panel (2026-10-03; not yet rendered)

- `scripts/go1_handoff_demo_video.py` now records three arms per take, in this
  order: walker alone | limp | hand-off.
- **Limp in the live setting:** the robot walks under normal motor gains; at
  the step the predictor fires, `dr.pd_gains(..., scale 0)` switches every
  joint motor off for the rest of the take, and gains are restored to scale 1
  before the next take.
  - The video env registers an identity (`scale 1`) `dr.pd_gains` startup
    event. It changes nothing, but makes mjlab allocate the gain fields per
    env so they can be written mid-episode.
  - The script raises if any actuator torque is measured after motors-off.
  - This differs from the comparison's limp arm only in timing mechanics:
    there the motors are off from step 0 of a replayed flag state; here they
    go off at the live flag step.
- A take is kept when the walker really falls after the flag and all three
  panels flag; takes whose three panels flag on the same step are preferred.
- **Memory:** frames from before the flag are kept only for the lead-in
  window (a bounded deque), each take is composed frame by frame (a
  generator) and written straight to the mp4. Found because the three-panel
  dry run was killed on the local 2 GB machine.
- The layout was dry-run locally on synthetic frames (1932x480, all text
  fits). The three-arm recording itself has not run on a GPU yet.

## Statistical evaluation: analysis plan, pre-registered 2026-10-03

User request: an extensive evaluation of the three recovery policies and some
evaluation of the predictor, each with statistical significance tests.

**Tools (pure numpy/scipy, unit-tested, no GPU):**
`src/cartpole_term/common/stats.py` (6 tests), `scripts/go1_recovery_stats.py`,
`scripts/go1_predictor_stats.py`.

### A. Recovery policies (confirmatory; data not yet collected)

- **Arms:** hand-off to the protective policy (round 8b iteration 200), walker
  continues, limp (motors off).
- **Data:** three fresh flag-moment banks, seeds 6, 7, 8 (about 1,500 flag
  moments, each followed by a real fall), pooled. Every state is replayed
  under all three arms from an identical start, so the unit is the fall and
  every test is paired.
  - The banks must not overlap each other, the training bank (flag seed 5),
    the selection bank (flag seed 3), or the earlier test bank (flag seed 4).
- **Primary outcome:** ground covered (`envelope_growth_m`).
  - H1: hand-off vs limp. H2: hand-off vs walker.
- **Secondary outcomes:** trunk/hip peak force, any-body peak force, peak
  torque (vs the walker only; limp is zero by construction), steps until at
  rest, came to rest (binary).
- **Tests:**
  - Wilcoxon signed-rank, two-sided, for each measure and each pair of arms
    (zero differences dropped), with the matched-pairs rank-biserial effect
    size;
  - exact McNemar for came-to-rest;
  - Friedman omnibus across the three arms per measure;
  - Holm adjustment over all pairwise tests as one family (16 tests);
    alpha = 0.05 on the adjusted p-values.
- **Reported with every test:** median paired difference with a bootstrap 95%
  CI (10,000 resamples), share of falls in which the first arm is lower (ties
  excluded) with a Wilson 95% CI, per-arm medians and 90th percentiles with
  bootstrap CIs, and per-bank medians to show whether the three banks agree.
- **Robustness, reported separately and not pooled:**
  - one fresh mid-fall bank (seed 9; states drawn between the flag and the
    fall), i.e. a late hand-off;
  - a second replay of the seed-6 bank, to show how much the non-deterministic
    simulator moves the numbers.
- **Exploratory, already run:** the same script on the 500 seed-4 falls
  (`outputs/stats/recovery_exploratory_seed4/`). Those medians had been seen
  before the tests were chosen, so they are labelled exploratory.
- **Scope of the inference:** one walker, one predictor, one protective
  policy. The tests quantify uncertainty over falls for those fixed policies,
  not over training seeds, and the replay has no pushes after the flag. Only
  flags followed by a real fall are in the banks; false alarms are not
  covered.

### B. Predictor (on the existing, locked model and data)

- The script first re-creates the test-split predictions from the saved
  weights and the recorded split seed, and stops unless AUPRC and AUROC match
  the recorded values to 1e-6.
- **Offline (test split):** 95% CIs from a cluster bootstrap that resamples
  whole episodes (1,000 resamples), because steps within an episode are
  correlated.
  - AUPRC, AUROC, Brier, and precision, recall, F1 at the validation-chosen
    threshold;
  - MLP minus logistic regression (same inputs) in AUPRC and AUROC, with the
    bootstrap p-value for a zero difference;
  - MLP AUPRC minus the base rate (chance).
- **Per falling episode:** share flagged before the fall (exact binomial CI)
  and median lead (bootstrap CI over episodes).
- **Live (Gate 6 data, 1,024 trials):** recall and precision with exact
  binomial CIs, median lead with a bootstrap CI.
- The test split was already used once for the reported metrics. This adds
  intervals and tests to those same numbers; nothing is tuned against it.

## Exploratory statistics on the seed-4 falls (2026-10-03; labelled exploratory)

`outputs/stats/recovery_exploratory_seed4/` (JSON, markdown, two figures):
500 paired falls, round 8b iteration 200 vs walker vs limp, 4,000 bootstrap
resamples, Holm over 16 tests. Not confirmatory, because these medians were
seen before the tests were chosen.

- **Ground covered (primary):**
  - hand-off vs limp: median paired difference -0.113 m [95% CI -0.134,
    -0.094]; lower in 77.6% of untied falls [73.7, 81.1]; rank-biserial
    -0.66; Holm p < 0.0001;
  - hand-off vs walker: -0.516 m [-0.566, -0.471]; lower in 97.6%; r = -0.99;
    p < 0.0001.
- **Trunk/hip force:** hand-off vs limp -249 N [-297, -216], lower in 93.8% of
  untied falls (116 ties, both zero), p < 0.0001; vs walker -322 N, p < 0.0001.
  Walker vs limp: no difference (p = 0.68).
- **Any-body peak force:** hand-off vs walker +48 N [-23, 115], p = 0.68;
  hand-off vs limp -107 N [-200, -25], p = 0.26 after Holm. Neither is
  significant. Walker vs limp: -144 N, p = 0.005.
- **Steps until at rest:** hand-off vs limp -6.5 steps [-9, -4], p < 0.0001;
  vs walker -59, p < 0.0001.
- **Came to rest:** hand-off 99.8% vs walker 58.6% (McNemar p < 0.0001);
  hand-off vs limp identical (p = 1).
- **Friedman across the three arms:** significant for every measure (any-body
  force p = 0.002, the rest p < 0.0001).

## Three-panel demo video rendered; predictor statistics run (2026-10-03)

- **Video:** `outputs/videos/go1_handoff_demo_round8b_it200.mp4` (14.8 s,
  1936x480, 3 takes, rollout seeds 701, 705, 708; all three panels flagged on
  the same step in every take: 135, 91, 230).
  - First GPU run of the three-arm recording and the mid-episode motors-off
    switch; the zero-torque check passed (limp peak torque 0 in all takes).
  - Per take, walker / limp / hand-off:
    - ground covered 0.57 / 0.28 / 0.46 m, 1.42 / 1.20 / 1.14 m,
      1.06 / 0.93 / 0.94 m;
    - trunk/hip peak force 556 / 0 / 0 N, 180 / 3762 / 0 N, 799 / 89 / 0 N.
  - Limp covers less ground than the hand-off in take 1 and ties it in
    take 3, consistent with limp winning on space in about 22% of falls.
  - The two-panel render is kept as `..._two_panel.mp4`.
- **Predictor statistics** (`outputs/stats/predictor/`; 1,000 episode
  resamples; test split 148,831 transitions from 772 episodes).
  - The reproduction check passed: test AUPRC and AUROC match the recorded
    values exactly.
  - The logistic-regression baseline refit locally gives AUPRC 0.7345 vs the
    recorded 0.7355 (a different scikit-learn version than Colab's); the
    intervals use the local refit.
  - MLP AUPRC 0.849 [0.825, 0.870]; AUROC 0.991 [0.988, 0.994]; Brier 0.0062
    [0.0052, 0.0073]; precision 0.822 [0.781, 0.858]; recall 0.698 [0.670,
    0.727]; F1 0.755 [0.731, 0.778].
  - MLP minus logistic regression: AUPRC +0.114 [0.090, 0.141], AUROC +0.026
    [0.019, 0.035]; bootstrap p < 0.001 for both.
  - MLP AUPRC minus base rate: +0.831 [0.808, 0.852], p < 0.001.
  - Per falling test episode: 167 of 167 flagged before the fall (exact 95% CI
    97.8-100%); median lead 12 steps [12, 13], capped at the 16-step horizon.
  - Live (Gate 6): recall 763/764 = 99.87% [99.27, 100.00]; precision 763/915
    = 83.4% [80.8, 85.7]; median lead 15 steps [14, 15].

## Predictor overlay video now shows only the falls (2026-10-03, user request)

- **Problem:** `scripts/go1_wildpush_video.py` wrote one continuous recording
  until five falls had happened. In the end-to-end run that was 61 s and six
  episodes, one of which (ending at step 1739) ran its full 1,000 steps without
  falling, so the video showed a trial with no failure.
- **Change:** the video is now one clip per fall: the last `--clip-seconds`
  (default 4 s) of that episode up to the fall, never reaching back into the
  previous episode, captioned "Fall k of N" with a countdown, then a 0.75 s
  freeze. Episodes that end on the time limit are still simulated (the rollout
  is continuous) but not shown. `--full-rollout` restores the old output.
- The rollout, the predictor scoring, and the printed fall log are unchanged.
  Only which frames are written changed.
- Dry-run locally on the reported fall steps (303, 739, 2142, 2463, 3070):
  five clips, 23.8 s. Not yet re-rendered on a GPU.
- Also: the end-to-end notebook's walker video (A3) now pins the command to
  2 m/s straight ahead (`--commanded-speed`), at the top of the trained
  forward range (-1.5 to 2.0 m/s from iteration 5,000).

## End-to-end notebook: first full-run failure and audit (2026-10-03)

- **Failure (Part D, `go1_handoff_eval.py`):** `RuntimeError: Inplace update to
  inference tensor outside InferenceMode` in `apply_body_impulse.reset()` ->
  `write_external_wrench_to_sim` -> `xfrc_applied[...] = force`, raised by the
  env reset at the start of the second arm.
  - Cause: since the 2026-09-30 pairing fix both arms share one env. Arm 1
    steps inside `torch.inference_mode()`; the reset for arm 2 ran outside it.
    mjlab's impulse event clears any impulse still active on reset by writing
    into the sim's `xfrc_applied` buffer, which by then is an inference tensor.
  - It only triggers if some env has an impulse in progress when arm 1 ends,
    which is why the demo video script (same pattern between takes) had not
    hit it.
  - Same bug class as the round-4 `settled.reset()` crash.
  - This was the code path flagged as changed since its last GPU run.
- **Fix:** all env calls after construction stay inside inference mode. The
  arm loop in `go1_handoff_eval.py` and each take in
  `go1_handoff_demo_video.py` are now wrapped in `torch.inference_mode()`,
  including their resets and the limp panel's motor-gain writes.
  - Precedent that mid-run resets inside inference mode work on GPU:
    `go1_collect_flagged_states.py` (`deferred.release`) and
    `go1_protective_compare.py` (`env.reset(env_ids=...)`).
  - Not yet re-run on GPU.
- **Audit of `notebooks/go1_end_to_end.ipynb` (static; no GPU here):**
  - every flag passed to every script exists in that script's argparse (18
    command lines);
  - every project-internal import used by those scripts resolves (40 files,
    178 names);
  - every file a step reads is written by an earlier step under the same
    name;
  - every `{variable}` on a shell line is defined;
  - every other GPU script resets exactly once, on a fresh env, before any
    inference-mode stepping.
- **Notebook changes:**
  - `RUN_NAME` gives each run its own Drive folder, so the final zip no longer
    mixes in earlier attempts.
  - A walker already saved under that `RUN_NAME` is reused after a disconnect.
  - The protective iteration count is a named variable.
- **Still not verified on a GPU after their latest edits:**
  `go1_train_protective.py` (one env-cfg factory for training and eval, the
  `--peak-force-weight` flag), `go1_handoff_eval.py` (this fix),
  `go1_protective_compare.py` with three arms, `go1_wildpush_video.py` (clips
  only), and the two statistics scripts under Colab's library versions.

## Recovery-statistics figures replaced (2026-10-03, user request)

- The user found the two figures hard to read: the cumulative-distribution
  curves ("share of falls at or below") and the per-fall win-rate chart both
  had percentage axes, and neither showed the measures in their own units.
- `scripts/go1_recovery_stats.py` now writes:
  - `fig_boxplots.png`: one panel per measure (ground covered in m, trunk/hip
    force in N, any-body force in N, peak torque in Nm); one box per policy
    (box = middle 50% of falls, line and number = median, whiskers = 5th to
    95th percentile); brackets carry the Holm-adjusted Wilcoxon result of the
    hand-off against each baseline ("p < 0.001", "p = ...", or "n.s.").
  - `fig_histograms.png`: number of falls (y) against the measure in its own
    unit (x), one outline per policy, round bin widths, values above the 99th
    percentile counted in the last bin; the trunk-force panel states how many
    falls had no trunk contact at all.
- The tables, the tests, and the JSON are unchanged. Only the figures changed.
- `go1_end_to_end.ipynb` (Part F1) and `go1_recovery_eval.ipynb` display the
  new files. Rendered locally on the exploratory seed-4 data.
- **Later the same day (user request):** the histogram figure was dropped as
  not useful. `go1_recovery_stats.py` writes only `fig_boxplots.png` (four
  panels) and gained `--quiet` (tables are written to `recovery_stats.md` but
  not printed). In `go1_end_to_end.ipynb` Part F1 the statistics cell now
  displays only that figure, saves it to Drive, and downloads it;
  `go1_recovery_eval.ipynb` does the same. The first full end-to-end run
  completed on Colab on 2026-10-03 (user report) before this change.

## G1 port: plan written (2026-10-03; nothing implemented)

`docs/G1_PORT_PLAN.md`. Facts it rests on, read from mjlab 1.6.0 source and
the live-confirmed `RobotProfile.G1`: G1 mass 33.34 kg (sum of the 30
`<inertial>` masses in `g1.xml`; Go1 12.74 kg by the same sum), standing pelvis
height 0.76 m, every G1 actuator a `BuiltinPositionActuatorCfg` (so the limp
arm's `dr.pd_gains` mechanism applies), effort limits up to 139 Nm, mjlab's G1
recipe 30,000 iterations, same `fell_over` termination (70 degrees). The plan
keeps the Go1 procedure and re-derives every Go1-calibrated number.

## Multi-seed back half: design and analysis plan, pre-registered 2026-10-03 (before running)

**User request:** show the process is robust without training five walkers:
keep one walker and run everything after it five times, then analyse all the
data together.

- **What is repeated per seed s (default 0-4), with one fixed walker:**
  - the push dataset and predictor (collection seed and training seed s);
  - the flag-moment training bank and the protective policy (seed s, round-8
    reward, checkpoint fixed at iteration 200);
  - the live hand-off check;
  - a fresh test bank and three evaluation banks, each replayed under hand-off,
    walker, and limp.
- **No sharing between seeds:** every bank is collected with seed s's own
  predictor, under bank seed 1000 x s + role (5 trains, 4 tests, 6-8 evaluate).
  Seed 0's bank seeds are the ones used so far.
- **Fixed in advance for all seeds:** iteration 200 (chosen on seed 0's
  selection bank), the 150 N push, the reward, and the space-first pass rule.
  No per-seed selection or tuning. A seed whose iteration-200 policy fails to
  settle or to pass is reported as a failing seed.
- **Notebook:** `notebooks/go1_end_to_end.ipynb` now loops Parts B-F over
  `SEEDS`. Finished steps are mirrored to Drive under their relative paths and
  restored on a re-run. Videos are made for the first seed only.
  `SEEDS = (0,)` reproduces the single run.
- **How the data are combined (`scripts/go1_multiseed_stats.py`):** the data
  are nested (about 1,500 falls inside each seed, 5 seeds), and the two levels
  are kept apart.
  - *Within a seed:* paired Wilcoxon on that seed's falls, hand-off against
    each baseline for every measure, Holm-adjusted within the seed. This asks
    whether that seed's trained policy is better across falls.
  - *Across seeds (the robustness claim):* each seed is reduced to one number
    per measure and baseline, its median paired difference (hand-off minus
    baseline). The five numbers are the sample: mean, SD, t-based 95% CI (4
    degrees of freedom), two-sided one-sample t-test against zero, the number
    of seeds with the hand-off lower, and the number of seeds where it is both
    lower and significant within the seed.
  - *Not done:* pooling all falls from all seeds into one test. Falls from the
    same trained policy are not independent replications of the procedure;
    pooling would treat about 7,500 falls as the sample size when the
    procedure was run 5 times.
  - *Primary outcome:* ground covered, hand-off vs limp and hand-off vs walker.
    The robustness claim is supported if the hand-off is lower in all 5 seeds
    and the across-seed 95% CI excludes zero.
  - *Predictor:* per-seed test AUPRC, AUROC, the MLP-minus-logistic-regression
    difference, precision and recall, live recall, precision and lead,
    reported as the five values with mean, SD, and range, plus the number of
    seeds in which the MLP's within-seed interval over logistic regression
    excludes zero.
  - *Figure:* the four box plots over all falls (descriptive), each seed's
    median as a dot, brackets carrying the across-seed result.
- **Limits:**
  - With 5 seeds the across-seed test has 4 degrees of freedom, so only large,
    consistent effects will be significant. An exact sign test cannot reach
    p < 0.05 with 5 seeds (5 of 5 gives p = 0.0625), which is why the t-test
    and the count of agreeing seeds are both reported.
  - The walker is the same in all seeds, so this is robustness to predictor
    and protective-policy training and to the sampled falls, not to walker
    training.
- **Code:** `stats.seed_level_summary` (+1 test; 82/82 pass locally),
  `go1_multiseed_stats.py` (exercised locally on five stand-in "seeds" resampled
  from the seed-4 falls; no real multi-seed data exists yet),
  `go1_predictor_stats.py --quiet`. Not yet run on a GPU.

## Multi-seed analysis changed at the user's direction (2026-10-03, before any multi-seed data)

This supersedes the "how the data are combined" part of the multi-seed entry
above. No multi-seed run existed when it was changed.

- **Each back-half run does its own significance tests** on its own falls
  (`go1_recovery_stats.py`, unchanged: paired Wilcoxon per measure and pair,
  McNemar for came-to-rest, Friedman, Holm within the run) and saves them to
  `outputs/stats/recovery_seed{s}/recovery_stats.json`. Nothing is combined
  during the runs.
- **At the very end,** `go1_multiseed_stats.py` reads the saved results and
  reports the mean and standard deviation across the runs of every statistic:
  per-policy medians and 90th percentiles, settle rates, and for every pair of
  policies the median paired difference, the share of falls lower, the
  rank-biserial effect size, and the p-value, plus the number of runs
  significant at 0.05 (and in which direction). Predictor metrics and its two
  bootstrap p-values are averaged the same way.
- **Dropped:** the across-seed one-sample t-test and t-interval, and the
  per-seed dots on the figure. `stats.seed_level_summary` remains in the
  module (tested) but is no longer used.
- **Figure:** four box plots, each box drawn from all falls of all runs for
  that policy; under each policy name the mean and SD of the per-run medians;
  brackets carry the mean and SD of the per-run p-values for the hand-off
  against that baseline and the number of significant runs.
- **Caveat recorded with the output:** a mean of p-values summarises how the
  runs came out. It is not the p-value of a combined test, and when the
  per-run p-values span many orders of magnitude the mean is set by the
  largest one. The count of significant runs and the per-run values are
  reported beside it.
- Exercised locally end to end on five stand-in "runs" resampled from the
  seed-4 falls (per-run tests, then the averaging and the figure). 82/82 unit
  tests pass. Not yet run on a GPU.

## Multi-seed run "multiseed1": results (2026-10-03)

Files: `outputs/multiseed1/` (the run's Drive folder, unpacked with its own
`outputs/` and `checkpoints/` tree so nothing in the project was overwritten);
zip in `archive/`. One walker (`model_9999.pt`), seeds 0-4, A100, mjlab 1.6.0,
torch 2.11.0+cu130, scipy 1.16.3, scikit-learn 1.6.1. First complete GPU run
of the multi-seed notebook and of every script changed since its last GPU run.
1,507-1,515 paired falls per seed on the evaluation banks, 7,550 in all.

Each run tested its own falls; the figures below are the mean and SD across
the five runs (full tables: `outputs/multiseed1/outputs/stats/multiseed/`).

- **Ground covered (primary): consistent in all five runs.**
  - Medians: hand-off 0.368 +/- 0.012 m, walker 0.998 +/- 0.070, limp
    0.479 +/- 0.013.
  - Hand-off vs walker: median paired difference -0.52 +/- 0.05 m, lower in
    97.1% of falls, significant in 5 of 5 runs.
  - Hand-off vs limp: -0.126 +/- 0.007 m, lower in 78.4% +/- 2.0% of falls,
    significant in 5 of 5.
- **Trunk/hip force: consistent in all five runs.** Median 0 for the hand-off
  in every run (walker 365 +/- 44 N, limp 288 +/- 5 N); significantly lower
  than both baselines in 5 of 5.
- **Any-body peak force.**
  - vs the walker: the hand-off is significantly **higher** in 4 of 5 runs
    (median paired difference +68 +/- 31 N; per-run p = 0.75, 4e-5, 6e-4,
    0.003, 5e-4). Medians are level (974 +/- 59 vs 971 +/- 79 N).
  - vs limp: significantly lower in 5 of 5 (-142 +/- 26 N).
  - 90th percentile: hand-off 2674 +/- 90 N, walker 2323 +/- 121, limp
    2757 +/- 114.
- **Peak torque:** 19.0 +/- 1.2 Nm, significantly below the walker in 5 of 5.
- **Coming to rest is not robust.**
  - Hand-off settle rate per run on the evaluation banks: 0.911, 0.999,
    0.981, 0.963, 0.918 (mean 0.954 +/- 0.039). Limp: 0.996 +/- 0.002.
  - The hand-off settles significantly less often than limp in 4 of 5 runs.
  - On each run's test bank the rates were 0.896, 0.998, 0.974, 0.964, 0.918.
  - Each run's own in-training frozen eval already showed it at iteration 200:
    never-settled share 8.2%, 0.0%, 1.2%, 5.9%, 10.9% for seeds 0-4.
- **Pre-registered space-first rule (ground covered below walker and limp,
  settled >= 99%, trunk force and torque below the walker), per run on its
  test bank: 1 of 5 pass** (seed 1). The four fails are all on the settle
  requirement; ground covered, trunk force, and torque pass in every run.
- **Seed 0 here differs from the earlier seed-0 result** (settled 0.998 on
  the earlier seed-4 bank). The earlier policy was iteration 200 of a
  300-iteration run trained on a bank made with the earlier predictor
  (threshold 0.547). This seed 0 retrained the predictor (threshold 0.445),
  rebuilt its bank, and trained 201 iterations. So "iteration 200" does not
  reliably give a settling policy; the round-8 selection found it on one
  training run.
- **Predictor: consistent.**
  - AUPRC 0.850 +/- 0.009, AUROC 0.990 +/- 0.002; MLP above logistic
    regression by 0.123 +/- 0.011 AUPRC, significant in 5 of 5.
  - Live recall 1.000 in every run; median lead 17.0 +/- 1.6 steps.
  - The validation-chosen threshold varies (0.26-0.45), and with it live
    precision: 0.897, 0.698, 0.841, 0.767, 0.734 (0.787 +/- 0.081). That is
    92-273 false alarms per 1,024 trials.
- **Figure fix after the run:** the bracket text said "significant in 4/5" for
  any-body force vs the walker without the direction. It now reads "lower in
  a/5" and/or "HIGHER in b/5". The figure and tables in the run folder were
  regenerated locally from the run's own saved per-run results and per-fall
  files (no re-simulation).

## Settle requirement relaxed by the user after the multi-seed run (2026-10-03)

- **User decision:** the "comes to rest in at least 99% of falls" part of the
  pass rule is treated as largely insignificant. From watching the falls, the
  unsettled ones show barely any movement, usually the feet wiggling. No
  per-seed checkpoint selection will be added.
- **This is a change made after seeing the results.** Under the rule as
  pre-registered, 1 of 5 runs passes; that stays on record above. With the
  settle requirement set aside, all 5 runs pass the remaining parts (ground
  covered below both baselines, trunk force and torque below the walker).
- **What the per-fall data says about the unsettled hand-off falls**
  (evaluation banks, all five runs; 344 of 7,550 falls, 4.6%):
  - Trunk/hip peak force: median 0 N in every run, same as the settled falls.
  - Ground covered (median): 0.47 m (seed 0), 0.37 (seed 2), 0.17 (seed 3),
    0.21 (seed 4), against 0.34-0.39 m for the settled falls. Seed 1 has a
    single unsettled fall (1.29 m). So in three of the four runs with many
    unsettled falls they cover no more ground than settled ones; in seed 0
    they cover somewhat more.
  - On those same falls the hand-off still covers less ground than the walker
    in 96-100% of cases and less than limp in 69-77% (seed 1's single fall
    aside).
  - This is consistent with the user's description (the robot stays where it
    is but keeps some joints moving above the 0.2 m/s / 0.5 rad/s thresholds).
    The data cannot show what the motion looks like.
- **Not addressed by this decision:** any-body peak force significantly higher
  than the walker's in 4 of 5 runs, and live precision varying 0.70-0.90.
  Both remain as reported.

## Multi-seed run report written (2026-10-03)

`docs/GO1_MULTISEED_REPORT.md`, with figures under `docs/figures/multiseed1/`,
generated by `tools/build_multiseed_report.py --run-dir outputs/multiseed1`
from the run's saved result files (no re-simulation; every number is read
from those files or computed from the per-fall arrays). At the user's
direction the report leads with the consistent results (ground covered, trunk
force, torque, the predictor) and gives the settle rate, any-body force
against the walker, and the false-alarm spread a short "secondary
observations" section. All per-run p-values, including those, are tabulated
in full in its section 3.3.

## G1 humanoid port begins: walker notebook, 10,000 iterations (2026-11-08)

- **User decision:** the G1 walker is trained for 10,000 iterations, not the
  30,000 mjlab ships for the G1 (`docs/G1_PORT_PLAN.md` section 8, decision 2).
  This matches the Go1 walker's budget. It overrides `scripts/robot_train.py`'s
  "run to mjlab's shipped default" rule, which is why it is logged here; the
  script itself is unchanged and is given `--max-iterations 10000`.
- **Risk:** a walker short of mjlab's full budget may walk less robustly, and
  every later stage depends on it. The notebook's frozen-evaluation curve (A4)
  and video (A5) are the check, as for the Go1: near-full 1,000-step episodes
  with almost no falls, and steady forward walking in the video. If it fails,
  training resumes from `model_9999.pt` towards a larger total.
- **Notebook:** `notebooks/g1_walker.ipynb` (built by
  `tools/notebook_builders/build_g1_walker_nb.py`). Checkpoint and frozen
  evaluation every 500 iterations (Go1 used 1,000), so a Colab disconnect
  loses at most 500 iterations. The run directory lives on Google Drive
  through a link, and a re-run resumes from the newest checkpoint that still
  loads. Code comes from cloning the public repo instead of uploading
  `cartpole_term_code.zip`; the commit is recorded in
  `outputs/versions_g1_walker.json`.
- Later stages (push sweep, predictor, protective policy, evaluation) still
  need the refactor in `docs/G1_PORT_PLAN.md` section 5.
- **First Colab run, smoke test (2026-11-08):** crashed at the first frozen
  evaluation with `RuntimeError: Inplace update to inference tensor outside
  InferenceMode` in mjlab's `ContactSensor.reset()` (`buf[env_ids] = 0.0`).
  The G1 task's foot-contact sensor keeps a history buffer (the Go1's never
  did). `_run_eval_rollout` reset the eval env outside inference mode but
  stepped it inside, so after the "mean" pass the buffer was an inference
  tensor and the "sampled" pass's reset failed. Same bug class as the
  protective `settled` and `apply_body_impulse` crashes above. **Fix:** the
  reset in `FrozenEvalRunner._run_eval_rollout` now runs inside
  `torch.inference_mode()` (obs from it only ever feed the frozen actor).
  New `tests/test_frozen_eval_inference_reset.py` reproduces the sequence
  without mjlab; it fails on the old code and passes on the new. Go1 eval
  behaviour is unchanged apart from that wrapper.

## G1 walker attempt 1 sat down instead of walking; air_time override removed (2026-11-12)

- **Observed (user, from the A5 video of the 10,000-iteration walker,
  `RUN_NAME = walker1`, L4):** the robot drops into a split, sits on its
  pelvis, and stays there for the whole episode.
- **Why the frozen evaluation did not flag it:** its only measure is episode
  length, and the G1's only failure ending is `fell_over` (torso tilted more
  than 70 degrees). A seated robot with an upright torso never ends early, so
  it scores the full 1,000 steps, the same as a walker.
- **Likely cause (from mjlab 1.6.0's source, read locally from the wheel):**
  `tasks/velocity/config/g1/env_cfgs.py` sets `cfg.rewards["air_time"].weight
  = 0.0` explicitly for the G1. `RobotProfile.G1.reward_overrides` set it to
  0.25, copied from the Go1, where 0.25 is IsaacLab's Go1 value
  (`cartpole_term/go1/env_cfg.py`). There was no G1 source for it.
  `mdp.feet_air_time` pays +1 per foot whose current air time is between
  0.05 and 0.5 s whenever the command is above 0.5, with no requirement
  that the robot moves. Sitting frees both feet to be lifted repeatedly
  without any balance risk. Not confirmed from this run's per-term reward
  logs, which stayed on the Colab runtime.
- **Fix:** `G1.reward_overrides = {}`, so the G1 trains with mjlab's G1 recipe
  exactly as shipped. The Go1 keeps its override (its walker is the reported
  one and is unaffected).
- **Notebook changes:** default `RUN_NAME = "walker2"` (fresh run, never
  resumes the seated policy); the finished walker's file name now carries
  `RUN_NAME`, so an old walker left on a runtime is not mistaken for the new
  one; A4 now also plots the training velocity-tracking, air-time and upright
  rewards from TensorBoard, because episode length alone cannot tell walking
  from sitting.
- **Check for attempt 2, fixed before it runs:** the tracking reward rises
  over training, and the A5 video shows the robot walking forward on its
  feet. If it sits again with mjlab's own recipe, the 10,000-iteration budget
  (against mjlab's 30,000) becomes the next suspect, and training continues
  from the final checkpoint.
- **Walker2 safeguards (2026-11-12, before it runs):**
  - **Reference search:** no published learning curve for mjlab's stock G1
    flat walker was found. mjlab's paper and docs give only a
    `--agent.max-iterations 10_000` usage example and qualitative videos.
    Unitree's own mjlab G1 walker (github.com/unitreerobotics/unitree_rl_mjlab)
    trains for 10,001 iterations, so the 10,000 budget is normal. It does use
    a different reward set: a phase-based `feet_gait` term, `stand_still`,
    `is_terminated` at -200, and a phase observation. Those functions are not
    in mjlab 1.6.0, so porting them would be its own untested change. Not done.
  - **Chunked training:** A3 now trains in 2,000-iteration chunks (resuming
    each time). After each chunk it prints the training speed-tracking reward
    (mean of the last 50 logged values) and renders a one-episode video.
  - **Guard (invented, loose):** from 4,000 iterations on, training stops
    before the next chunk if that reward is below 0.8. Walker1 ended at 0.434,
    which is weight 2 times a mean tracking score of 0.22.
  - **Checked locally** by running the A3 cell against fake training:
    resuming each chunk, skipping on a rerun, skipping a corrupt newest
    checkpoint, the guard firing, and `None` overriding it.

## G1 port, stages 2-3: push and predictor (2026-11-13, before any G1 data)

- **Walker accepted (user, from the walker2 videos):** walks, falls no more
  than the frozen evaluation shows, and tracks speed (training
  `track_linear_velocity` 1.52 at the full command range). The gait looks slow
  and stiff. The slow part is at least partly the video: a fresh video env
  starts mjlab's command curriculum at its initial range. The stiffness
  matches mjlab's tight G1 posture reward. Kept as the walker for the fall
  study.
- **Push, fixed now:** impulse ±390 N per axis on `torso_link`, with the same
  duration (0.1-0.3 s), cooldown (1-3 s), zero torque and mild shipped push
  as the Go1. The rule is the Go1's: scale arXiv:2512.01336's 50-850 N
  humanoid range by mass. Equivalently, Go1's 150 N times 33.3411 / 12.7434
  kg (masses summed from the `<inertial>` tags of mjlab 1.6.0's g1.xml and
  go1.xml) is 392 N, the same acceleration, rounded to 390. The body is
  `torso_link`, the upper body above the waist joints, as in
  `docs/G1_PORT_PLAN.md`; the G1's root body (pelvis) sits at hip height.
- **Check, fixed now:** falls per 1,000 robot-steps at 390 N, measured like
  the Go1's collections (4,096 robots, 262 steps from reset; the Go1 at
  150 N scored 1.03-1.09 over its five collections, mean 1.07). It must lie
  within 0.25x-4x of the Go1's. If it does not, the notebook stops and the
  force is decided with the user and logged here, not picked from the sweep
  in the notebook. The 200/300/500/600 N rows are context only. A video under
  the push (B2) is the plausibility check, as for the Go1.
- **Predictor:** unchanged from the Go1. 16-step label horizon, 70/15/15
  episode split, MLP 128→128→128→1 (input 99 obs + 29 actions), logistic
  regression baseline, F1 threshold on validation, ≥1M labelled steps and
  ≥1,000 falls, episode-bootstrap statistics. One seed first.
- **Code:**
  - The impulse push moved to `src/cartpole_term/common/push.py`. For the
    Go1 the definition is identical: same event, params and body.
  - `RobotProfile` gained `push_body`, `mass_kg` and `impulse_force_n`.
  - `go1_wild_push_predictor.py` takes `--robot` (default `go1`, file names
    unchanged for it; G1 files are prefixed `g1_`).
  - `measure_push_failure_rate.py --mechanism impulse` runs the sweep.
  - `robot_watch_episodes.py --impulse-force` records under the push.
  - New `tests/test_push.py`.
  - Notebook: `notebooks/g1_predictor.ipynb`.
- **Sweep result and decision (2026-11-13):** falls per 1,000 robot-steps,
  `outputs/g1_impulse_sweep_seed0.json` (walker2, L4):

  | Push | Falls per 1,000 steps |
  |---|---|
  | 200 N | 1.17 |
  | 300 N | 3.11 |
  | 390 N | 4.38 |
  | 500 N | 5.36 |
  | 600 N | 5.87 |

  The Go1 at 150 N scored 1.07. The mass-scaled 390 N came out at 4.10x the
  Go1's rate, just outside the pre-set 0.25x-4x check, so the notebook
  stopped as designed.
  - **User decision: 200 N**, which matches the Go1's fall rate (1.09x). The
    rule for the G1 is therefore "same fall frequency as the Go1", not "same
    acceleration".
  - **This was chosen after seeing the sweep**, which the pre-registration was
    meant to avoid; it is recorded as such. The reasons, given before the
    choice:
    - The predictor's chance level (the positive fraction) and AUPRC are then
      directly comparable with the Go1's.
    - The robot walks for a similar time between falls (about 17 s, against
      about 4.5 s at 390 N).
    - 200 N is inside arXiv:2512.01336's 50-850 N humanoid range.
  - **Likely reason the equal-acceleration force overshoots** (not tested):
    the G1's push lands on the upper torso, high above its feet, while the
    Go1's lands on its trunk, so the same acceleration tips a tall, narrow
    biped over far more readily.
- `RobotProfile.G1.impulse_force_n = 200.0`. The notebook now computes the
  check for `PUSH_N` from the saved sweep rows, so the saved sweep is reused.

## G1 port, stages 4-8: SmallFall for the humanoid (2026-11-13, before any G1 protective data)

- **Predictor result that this builds on** (`g1_predictor.ipynb`, seed 0,
  200 N): test AUPRC 0.944 [0.934-0.953], logistic regression 0.894, chance
  0.022. Every test falling episode (204/204) was flagged; median lead 0.30 s
  (at the 16-step cap). The push video was judged plausible by the user.
- **Procedure:** the Go1's final run (round 8b), with only robot-specific
  numbers re-derived by the Go1's rules:
  - **Banks:** flag-moment states captured live (walker + 200 N + predictor
    seed 0). One bank seed per role: training 5, selection 3, test 4. The Go1
    multi-seed run used 1000·s + role for training (5) and test (4); 3 is new,
    for selection.
  - **Episode budget:** 5 s. The Go1's 3 s is scaled by √(start height)
    (0.76 m G1 pelvis against the 0.278 m Go1 trunk, mjlab init_state): fall
    time ∝ √h, ×1.65, giving 4.96 s. It is a safety cap; `settled` ends
    episodes.
  - **Reward:** the Go1 round-8 reward. The contact-force categories come
    from the G1 profile: critical = pelvis, torso and waist links; all other
    G1 bodies non-critical.
    - arXiv:2512.01336 weights (contact force, root velocity, action rate)
      are kept.
    - The four Go1-calibrated weights (joint torques, joint limits, ground
      covered, peak force) are re-sized by
      `protective/calibration.py`. Train 11 iterations with the Go1 weights;
      then weight × (Go1 contribution / G1 contribution), where contribution
      is mean `Episode_Reward/<term>` over iterations 0-10. The Go1
      reference is the same window of round 8b: torques −0.0580, joint
      limits −0.0016, ground covered −0.1190, peak force −0.1787. Factor
      clamped to 0.02-50.
    - Checked locally: run on round 8b's own log, the script returns the
      Go1 weights (factors 0.99-1.00).
  - **Training:** 301 iterations, 4,096 envs, the Go1's PPO settings.
  - **Selection:** candidates at iterations 100/150/200/250/300 on the
    selection bank, chosen by the unchanged `select_checkpoint` rule.
  - **Test:** once, on the test bank: SmallFall against the walker
    continuing and limp, with the same verdict code as the Go1.
- **Not yet:** the live hand-off check (Go1 Gate 6), several seeds, and the
  multi-seed statistics. These come after this notebook's result.
- **Code:**
  - `protective_fall_env_cfg(robot, ..., weight_overrides)`.
    `go1_protective_fall_env_cfg` is now a wrapper giving the identical Go1
    task (new test pins its weights, task id and budget).
  - G1 protective task registered (`Mjlab-G1-Protective-Fall`, experiment
    `g1_protective_fall`).
  - `--robot` on `go1_collect_flagged_states.py`, `go1_train_protective.py`
    (plus `--weights-json`), `go1_protective_compare.py` and
    `go1_watch_protective_episodes.py`. Their Go1 defaults and file names are
    unchanged. The watch script's per-episode step cap is now the longest
    robot budget (251); Go1 episodes still end at their own 150-step
    time-out.
  - New `scripts/robot_calibrate_protective.py`.
  - Notebook: `notebooks/g1_smallfall.ipynb`.

## 2026-11-13 -- Fall-size measure ablation (G1): which policy falls smallest overall

- **Why:** "how small a fall is" can be measured many ways, and different
  measures could rank policies differently. User request: implement several
  measures and test them against each other ("way 3": combine them by
  stated rules).
- **Fixed before any data.**
- **Ten measures** (`src/cartpole_term/protective/fall_size.py`). All are
  horizontal, from the fall's start; lower = smaller:
  - ground covered (the existing primary, `envelope_growth_m`);
  - max reach;
  - root travel at the end;
  - root's largest travel;
  - root path length;
  - centroid travel;
  - swept convex-hull area (32-direction support polygon);
  - hull growth;
  - contact reach;
  - floor area touched (10 cm cells).

  Body positions are body-frame origins, as before. Contact means a net
  ground force above 1 N on a vulnerability body.
- **Three combining methods** (`protective/fall_size_ablation.py`):
  - **A. Dominance:** on per-policy medians.
  - **B. Mean rank:** per fall and measure; ties averaged.
  - **C. Geometric-mean ratio to limp:** per fall and measure, with a floor
    of 1% of the measure's pooled median.
- **Overall rule:** if B and C name the same policy and it is undominated
  in A, that policy is the answer. Otherwise the answer is A's undominated
  set, reported as a disagreement.
- **Robustness checks:**
  - 1,000 paired bootstrap resamples of the falls, giving each policy's
    win share;
  - Kendall tau between the measures' rankings;
  - per-fall Spearman between measures;
  - per-measure paired win shares and Wilcoxon p.
- **Policies:**
  - chosen SmallFall checkpoint;
  - the other SmallFall candidates (iterations 100-300);
  - walker;
  - limp.
- **Bank:** a fresh bank (seed 6). The SmallFall test bank (seed 4) stays
  used once.
- **Code:**
  - `scripts/go1_protective_compare.py` now also records the ten measures
    (`size__<metric>` in the npz). It asserts that its ground covered
    equals the existing tracker's.
  - New `--extra-policy LABEL=CHECKPOINT` (repeatable). The existing
    verdicts are unchanged.
  - New `scripts/robot_fall_size_ablation.py` (CPU).
  - Notebook: `notebooks/g1_fall_size_ablation.ipynb`.

## 2026-11-13 -- Reward ablation (G1): one SmallFall policy per fall-size measure

- **Why (user):** each way of measuring fall size should be its own reward,
  training its own policy; then compare the policies.
- **Fixed before any data.**
- **Reward:** the G1 SmallFall reward, with the fall-size term swapped for
  one of the ten measures from the entry above
  (`protective/fall_size_reward.py`). Each step pays the measure's increase,
  so an episode's raw total equals the measure for that fall. For "at the
  end" measures the increase can be negative.
  - Ground covered keeps the original `fall_envelope_growth` term
    unchanged.
  - Everything else is unchanged: the calibrated weights, training bank 5,
    selection bank 3, 301 iterations, seed 0.
- **Weights** (`calibration.fall_size_weights`):
  - one 11-iteration run of SmallFall's own reward logs all ten measures as
    episode totals (`--log-fall-sizes`, no reward effect);
  - weight_m = ground-covered weight × (ground-covered total / measure-m
    total), averaged over iterations 0-10;
  - factor clamped to 0.02-50;
  - ground covered gets SmallFall's weight exactly.
- **Checkpoint:** each policy picks among iterations 200/250/300 on the
  selection bank by its own measure: lowest median among those that come to
  rest in at least 99% of falls (`select_by_measure`). If none does, the
  lowest overall, flagged.
- **Scoring:** a fresh bank (seed 6), used for scoring once. Policies:
  - the ten policies;
  - the original SmallFall;
  - walker;
  - limp.

  Analysis:
  - the ten-measure ablation with the three combining methods;
  - a trained-on × measured matrix (median relative to limp);
  - how many policies are smallest on their own measure.
- **Known pairs:**
  - max reach = ground covered + starting size;
  - hull area = hull growth + starting hull.

  Within each pair the reward differs only by a per-episode constant the
  policy can't change, but the weights differ. So the pair shows the effect
  of weight scale alone. `r_ground` is a second run of SmallFall's own
  reward, which shows run-to-run variation.
- **Code:**
  - `fall_size_reward.py`;
  - `protective_fall_env_cfg(fall_size_measure=, fall_size_weight=,
    log_fall_sizes=)`;
  - `go1_train_protective.py` flags `--fall-size-measure`,
    `--fall-size-weights-json`, `--log-fall-sizes` and `--run-tag`;
  - new scripts `robot_calibrate_fall_size.py` and
    `robot_select_by_measure.py`;
  - notebook `notebooks/g1_reward_ablation.ipynb`.

## 2026-11-14 -- Reward ablation (G1): results

Run `rewardablation1`. Results are unpacked under
`outputs/g1_reward_ablation1/`; the zip is in `archive/`. Scored on bank 6
(501 falls). A local re-run of the ablation reproduces the Colab report.

- **Pre-registered overall answer: r_centroid.** Mean rank, ratio to limp
  and dominance agree, in 100% of 1,000 resamples.
  - **It fails the project's settle rule:** it comes to rest in 57% of
    falls (r_rootend 54%, r_rootpath 69%).
  - All three "where it ends up / how far it moves" rewards learned not to
    settle within the 5 s cap.
  - Their unsettled falls are not smaller on ground covered than their
    settled ones, so the effect is not "unfinished falls look small".
  - The behaviour still needs a video.
- **Checkpoint choice:** in 9 of 10 policies no candidate reached the 99%
  settle bar on the selection bank (r_ground: 0.97-0.98), so the
  flagged fallback was used. The 99% bar is too strict for one 301-iteration
  run. Every policy's frozen-eval return was still rising at iteration 300.
- **The measures fall into families that trade off against each other**
  (mean Kendall tau 0.23):
  - **Ground covered / max reach / contact reach:** every trained policy
    gets 0.72-0.81 of limp. The spread is about the run-to-run noise.
  - **Root / centroid travel:** the root and centroid rewards reach
    0.62-0.72 of limp. The hull rewards reach about 1.0, no better than limp.
  - **Hull area:** the hull rewards reach 0.72-0.82 of limp. Every other
    trained policy is 1.06-1.25, worse than limp.
  - **Touched area:** only r_touched improves it (0.70). The others are
    0.93-1.11.
- **Run-to-run noise (r_ground vs the original SmallFall, same reward):**
  - ground covered: no difference, p=0.23;
  - centroid travel: 0.04 heights apart, p=1e-7.

  With one seed per reward, gaps below about 0.05 heights are not evidence
  of a reward effect.
- **r_reach vs r_ground on ground covered:** 0.017 heights smaller,
  p=0.11. Not shown.
- **Settling 95% or more** (limp, SmallFall, r_ground, r_contact, r_hull,
  r_hullgrow):
  - **mean rank:** r_contact, r_ground and r_hullgrow are within 0.1 of
    each other;
  - **ratio to limp:** r_hullgrow 0.889, in 100% of resamples;
  - **dominance:** the methods disagree.
- **One measure per family** (ground covered, centroid, hull growth,
  touched): r_hullgrow wins under both methods, with all policies or only
  those that settle.
- **Forces:** every trained policy has median trunk force 0 N and any-body
  force 1,412-1,612 N (limp 2,552; walker 2,315). No reward made impacts
  harder.

## 2026-11-14 -- G1 round 2: the smallest fall that is still safe (plan, fixed before data)

- **Goal (user):** "a policy that falls the smallest and keeps the robot
  fairly safe", keeping SmallFall's ideas: separate train / selection /
  test banks, rules fixed first, the test used once.
- **Rewards (3):** everything else is the G1 SmallFall reward with the
  calibrated weights.
  - **`ground`:** SmallFall's own reward.
  - **`hullgrow`:** hull growth in place of ground covered. Best among the
    settling policies in the ablation.
  - **`mix`:** 0.5 × each of their sized weights
    (`--fall-size-mix`; weights from `rewardablation1`'s
    `g1_fall_size_weights.json`).
- **Training:**
  - 3 runs each (seeds 0, 1, 2), so reward differences can be told apart
    from run-to-run noise. The ablation showed 0.04 heights between two runs
    of one reward.
  - 601 iterations: returns were still rising at 300.
  - Training bank 5.
- **Size score** (`fall_size_ablation.FAMILY_MEASURES`): one measure per
  family the ablation found:
  - ground covered;
  - centroid travel;
  - hull growth;
  - floor touched.

  Each is relative to limp on the same fall. Score = geometric mean over
  the four measures and the falls. A family-balanced score is used because
  the families trade off, and the ablation's all-ten score over-weighted the
  four root/centroid measures.
- **Safe:** median trunk/pelvis peak force at or below the walker's, and
  median any-body peak force below limp's.
- **Comes to rest:** at least 97% of falls. The 99% bar was unreachable in
  the ablation (r_ground 0.97-0.98 on selection), and this keeps out the
  restless policies (54-69%).
- **Checkpoint per run:**
  - choose among iterations 400 / 500 / 600 on selection bank 3;
  - take the lowest size score among the safe candidates that come to rest;
  - fallbacks are flagged.

  Rule: `select_safe_small`.
- **Reward and policy:**
  - a reward whose every run has an eligible checkpoint ranks first;
  - then the lowest mean size score over its 3 runs wins;
  - the policy is that reward's best run.

  Rule: `scripts/robot_round2_choose.py`. Chosen before the test.
- **Test (fresh bank 7, once):** every run's choice, the original
  SmallFall, walker and limp. Pass requires all of:
  - size score below the original SmallFall's (paired Wilcoxon on per-fall
    scores, p < 0.05) and below limp's (< 1);
  - median ground covered below both limp's and the walker's;
  - comes to rest in at least 97% of falls;
  - safe.

  The result is reported whatever it is (`robot_round2_summary.py`). The
  ten-measure table is printed alongside.
- **Code:**
  - `fall_size_mix` in `protective_fall_env_cfg`;
  - `--fall-size-mix`;
  - new scripts `robot_select_safe_small.py`, `robot_round2_choose.py` and
    `robot_round2_summary.py`;
  - notebook `notebooks/g1_round2.ipynb`.
