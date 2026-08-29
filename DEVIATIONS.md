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
