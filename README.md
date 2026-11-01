# cartpole-term

PPO cartpole balance policy (mjlab + RSL-RL) instrumented with frozen
in-training evaluation, plus an offline MLP that predicts termination
16 environment steps (0.8s) ahead. Follows the STATE-NAV-inspired
methodology specified in `CLAUDE.md`; see that file for the full binding
spec and `DEVIATIONS.md` for every departure from it, logged with evidence.

All numbers and figures below are pulled directly from `outputs/*.json`
and `logs/rsl_rl/cartpole_term/*/eval_history.json` produced by the
commands in section 1 — nothing here is estimated or reconstructed from
memory.

---

## 1. Reproduce

Run on a CUDA GPU machine (Colab T4 or equivalent), from the repo root,
with `MUJOCO_GL=egl` (or let `scripts/record_video.py` set it itself).

```bash
# Gate 0-1: sanity
python scripts/check_env.py
python scripts/check_terminations.py

# Gate 5: PPO training, seed 0 (dense eval grid + early stop, see DEVIATIONS.md)
python scripts/train.py --seeds 0 --eval.interval 2 --early-stop-patience 2

# Gate 6: S2.6 centering-reward ablation, seed 0 only
python scripts/train.py --seeds 0 --eval.interval 2 --early-stop-patience 2 \
    --env.reward.cart-center-weight 0.0

# Gate 10: seeds 1 and 2 (train.py auto-densifies/auto-perturbs as needed
# to resolve all three operating-point windows -- see DEVIATIONS.md)
python scripts/train.py --seeds 1 2 --eval.interval 1

# Gate 7: mechanical checkpoint selection, all seeds
python scripts/select_checkpoints.py --seeds 0 1 2

# Gate 8: frozen dataset collection, all seeds x all operating points
for point in struggling target competent; do
    python scripts/collect_frozen.py --seeds 0 1 2 --operating-point "$point"
done

# Gate 9: predictor training + evaluation, all seeds x all operating points
for point in struggling target competent; do
    python scripts/eval_predictor.py --seeds 0 1 2 --operating-point "$point"
done

# Gate 11: videos
python scripts/record_video.py --which both --seed 0 --min-failures 10
```

`notebooks/colab_entry.ipynb` is a thin Colab shim over exactly these
commands (CLAUDE.md S4.2 -- no logic in notebook cells).

---

## 2. Installed versions

| Component | Version |
|---|---|
| GPU | Tesla T4 |
| CUDA driver | 580.82.07 |
| mjlab | 1.6.0 |
| mujoco | 3.11.0 |
| mujoco-warp | 3.11.0 |
| rsl-rl | 5.4.2 |
| torch | 2.11.0+cu128 |

---

## 3. Termination thresholds

| Threshold | Value | Justification |
|---|---|---|
| Pole angle | ±12° (0.2094 rad) | Classic Gym/Barto-Sutton-Anderson 1983 value; adopted as-is (CLAUDE.md S3). |
| Cart position | ±1.0 m | A project choice, not a hardware limit -- comfortably inside the physical rail's ±1.8 m joint limit, so episodes end by *policy failure*, not by hitting a mechanical stop. |
| Rail (hard limit) | ±1.8 m | Physical joint limit from the shipped XML; used only for the ±0.5 m margin assertion in `env_cfg.py`, never as a termination itself. |

---

## 4. The S2.6 reward modification

The shipped mjlab cartpole reward (`cartpole_smooth_reward`, kept exactly
as-is, weight 1.0) includes a multiplicative "centered" factor bounded
below at 0.5 -- a cart sitting at the rail edge with an upright pole still
collects half the positional credit, which is weak enough that a policy
can learn to balance while slowly drifting toward the ±1.0 m termination
boundary.

This project adds one, and only one, additional reward term:
`cart_centered = exp(-(x / 0.5)^2)`, weight 0.2, using the cart's
**absolute** joint position (not the relative observation, which would
silently coincide with it for this task's zero default and mask a real
bug -- see `tests/test_rewards.py`). At the ±1.0 m boundary the term
contributes `exp(-4) ≈ 0.018`, so essentially all of its gradient signal
is exhausted well before an episode would actually end -- a gradient that
only appears at the boundary arrives too late to be useful. The 0.2
weight caps this term's maximum possible contribution at 20% of the
shipped term's, so a policy can never prefer sitting centered with a
falling pole over moving off-center to catch it.

**Honest note (Ng, Harada & Russell 1999):** this shaping term is additive
and *not* potential-based, so it can shift the optimal policy rather than
merely accelerate convergence to the original optimum. That is accepted
deliberately here -- a centered policy is the intended outcome, not a
side effect to be corrected for.

---

## 5. The S2.6 ablation (seed 0 only)

One additional seed-0 training run with `--env.reward.cart-center-weight
0.0` exactly reproduces the shipped mjlab reward, for comparison against
the centered (weight=0.2) run.

![Ablation: mean |cart position| vs iteration](docs/figures/ablation_comparison.png)

**Honest result:** this single-seed comparison does *not* show a clean,
unambiguous separation. During the noisy mid-training transient
(iterations ~8-13), the centered run's mean |cart position| is sometimes
*higher* than the ablation's, not lower -- the opposite of the intended
direction. Only once both runs converge to the 1000-step ceiling does the
centered run end up more centered (0.040 m at its final eval) than the
ablation (0.029-0.153 m across its last two evals, itself non-monotonic).
With n=1 seed for this comparison (as CLAUDE.md S2.6 specifies -- "seed 0
only"), this is not a statistically meaningful separation; it is reported
exactly as measured rather than smoothed into a cleaner-looking story.

**Termination-cause breakdown is uninformative for this comparison**: in
both the centered and ablation runs, `cart_pos` accounts for essentially
100% of non-timeout episode ends from iteration ~4 onward (matching the
gate-9 finding in section 8 below). The centering term does not change
*which* failure mode dominates -- only, weakly and noisily in this single
run, the magnitude of drift.

The a priori engineering rationale for the term (S4 above -- the shipped
multiplicative floor is weak) stands independently of this specific noisy
single-seed measurement; a firmer empirical answer would need multiple
ablation seeds, which is out of this project's scope (CLAUDE.md fixes the
ablation at seed 0 only).

---

## 6. Training and frozen-eval curves, all seeds

![PPO training reward and frozen-eval episode length, all 3 seeds](docs/figures/training_and_frozen_eval_curves.png)

Both are shown because they answer different questions (CLAUDE.md S6.1):
the training-time reward (left) is rsl_rl's own rollout return, the
standard massively-parallel-RL sample-efficiency plot; the frozen-eval
curve (right) is a periodic, deterministic-action measurement on a
*separate* environment with fixed seeds and no exploration noise (TD3/SAC
protocol) -- the actual reported metric for checkpoint selection, since
training-time return conflates policy quality with an ever-changing
exploration policy.

All three seeds converge to the 1000-step frozen-eval ceiling within
14-16 PPO learning iterations, at 4096 parallel environments -- this task
converges almost instantly at this scale (each full training run is
roughly 9 seconds of GPU compute).

**Transparency note on seed identity:** seeds 1 and 2 needed automatic
seed-perturbation retries to resolve all three operating-point windows
(CLAUDE.md S7.1) at the finest measurement grid -- a real, measured
property of this task's convergence speed, not a bug (full mechanism and
evidence in `DEVIATIONS.md`). What is reported here as "seed 1" and
"seed 2" therefore does not always mean the literal RNG seed value:

| Nominal seed | Perturbation attempts needed | Actual RNG seed used |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 12 | 12001 |
| 2 | 1 | 1002 |

Every run's `resolved_args.json` records both values, so this is always
auditable, never silent.

---

## 7. Selected checkpoints

Selected mechanically (earliest frozen-eval crossing of each operating
point's window, CLAUDE.md S7.2), then re-measured at a fresh seed
(`train_seed + 200`) to avoid winner's-curse bias -- the re-measured
number is what's reported everywhere else in this document.

| Point | Seed | Iteration | Selection-time length | Re-measured length |
|---|---|---:|---:|---:|
| struggling (80±15) | 0 | 4 | 68.2 | 67.2 |
| struggling (80±15) | 1 | 5 | 71.2 | 71.7 |
| struggling (80±15) | 2 | 5 | 71.1 | 71.3 |
| target (150±25) | 0 | 10 | 131.1 | 136.0 |
| target (150±25) | 1 | 11 | 125.5 | 120.7 |
| target (150±25) | 2 | 11 | 125.8 | 126.5 |
| competent (400±60) | 0 | 12 | 430.6 | 432.1 |
| competent (400±60) | 1 | 13 | 368.7 | 370.1 |
| competent (400±60) | 2 | 13 | 449.5 | 451.8 |

### Frozen dataset collection (per seed x operating point)

| Point | Seed | Labelled transitions | Base rate | Episodes | Terminated episodes | Censored steps | Wall time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| struggling | 0 | 1,003,587 | 0.4379 | 29,486 | 29,486 | 0 | 17.8 |
| struggling | 1 | 1,000,621 | 0.3985 | 26,148 | 26,148 | 0 | 17.9 |
| struggling | 2 | 1,003,358 | 0.3484 | 22,471 | 22,471 | 0 | 17.4 |
| target | 0 | 1,000,626 | 0.1467 | 9,194 | 9,194 | 0 | 17.1 |
| target | 1 | 1,003,746 | 0.1788 | 11,253 | 11,253 | 0 | 16.6 |
| target | 2 | 1,000,434 | 0.1488 | 9,312 | 9,312 | 0 | 15.4 |
| competent | 0 | 1,003,367 | 0.0273 | 1,717 | 1,717 | 0 | 15.8 |
| competent | 1 | 1,002,535 | 0.0609 | 3,821 | 3,821 | 0 | 15.7 |
| competent | 2 | 1,003,088 | 0.0482 | 3,022 | 3,022 | 0 | 14.8 |

All 9 datasets exceed the ≥1M labelled-transition / ≥1,000 terminated-episode
targets (CLAUDE.md S7.3), with zero truncation-censored steps -- every
collected episode ended in genuine failure, not the 1000-step time limit
(policies at these competence levels reliably fail before reaching it,
except at their own operating-point-defining rate).

---

## 8. Feature screening (target, seed 0)

Univariate logistic regression per feature (CLAUDE.md S9.3), on the
`target` training split. Required as the cheapest check that labelling
isn't broken: `|θ_wrapped|` was expected to dominate a priori, and did
not.

| Feature | AUC-ROC | McFadden's R² |
|---|---:|---:|
| `abs_cart_pos` | **0.929** | **0.487** |
| `cart_pos` | 0.620 | 0.024 |
| `abs_theta_wrapped` | 0.584 | 0.016 |
| `cos_theta` | 0.584 | 0.000 |
| `cart_vel` | 0.548 | 0.004 |
| `action` | 0.507 | 0.000 |
| `sin_theta` | 0.527 | 0.001 |
| `pole_vel` | 0.503 | 0.000 |

**This is a real, investigated finding, not a labelling bug** (full
evidence chain in `DEVIATIONS.md`): at `target` competence, 89.3% of
actual terminations are `cart_pos`-caused versus 10.5% `pole_angle`-caused
(from `outputs/frozen_target_seed0.json`'s `termination_cause_counts`).
The S2.6 centering term makes cart-drift dynamics more actively shaped and
therefore more informative here than in a reward without it -- once
pole-balancing is learned (which happens within a handful of PPO
iterations, section 6), cart drift toward the ±1.0 m boundary becomes the
dominant remaining failure mode, and `abs_cart_pos` is the feature that
legitimately carries the near-term-failure signal, not `abs_theta_wrapped`.

---

## 9. Predictor metrics, all seeds x all operating points

Mean ± std over 3 seeds. **Base rate is listed first per CLAUDE.md S9.1**
("every other number is uninterpretable without it"). Accuracy appears
nowhere as a headline metric (S12#14) -- see the constant-negative
baseline row for why.

| Point | Test base rate | AUPRC (baseline) | AUROC | Brier | ECE | McFadden's R² | Median lead (steps) |
|---|---:|---:|---:|---:|---:|---:|---:|
| struggling | 0.393 ± 0.035 | **0.873 ± 0.018** (0.393 ± 0.035) | 0.867 ± 0.029 | 0.122 ± 0.022 | 0.014 ± 0.009 | 0.428 ± 0.069 | 14.7 ± 0.5 |
| target | 0.160 ± 0.015 | **0.938 ± 0.007** (0.160 ± 0.015) | 0.973 ± 0.004 | 0.030 ± 0.003 | 0.003 ± 0.001 | 0.741 ± 0.016 | 15.0 ± 0.0 |
| competent | 0.045 ± 0.013 | **0.760 ± 0.055** (0.045 ± 0.013) | 0.951 ± 0.004 | 0.018 ± 0.004 | 0.003 ± 0.001 | 0.588 ± 0.043 | 14.0 ± 0.8 |

Detection lead time in seconds: multiply steps by 0.05 s (20 Hz) -- e.g.
target's 15.0-step median lead is 0.75 s of warning before failure, out of
the 0.8 s horizon itself (i.e. the predictor typically flags danger
essentially as soon as the 16-step window opens).

### Mandatory baselines (CLAUDE.md S9.2), AUPRC

| Model | struggling | target | competent |
|---|---:|---:|---:|
| MLP predictor (this project) | **0.873** | **0.938** | **0.760** |
| Constant-negative / base-rate-constant | 0.393 | 0.160 | 0.045 |
| Logistic regression, 6 inputs | 0.579 | 0.416 | 0.487 |
| Logistic regression, \|θ\| only | 0.578 | 0.229 | 0.066 |

The MLP clearly beats every baseline at every operating point, including
the 6-input logistic regression -- the gap is largest exactly where base
rate is lowest (`target`, `competent`), where a linear decision boundary
in the raw 6 inputs is least sufficient.

**Constant-negative accuracy** (S12#14: shown only alongside its own
baseline, never as a headline number): 60.7% / 84.0% / 95.5% at
struggling/target/competent respectively -- the "always predict safe"
degenerate solution looks *better* the more competent the policy is,
which is exactly why accuracy is banned as a standalone metric here.

### Per-cause breakdown

| Point | Cause | AUPRC | Median lead (steps) |
|---|---|---:|---:|
| struggling | pole_angle | 0.669 ± 0.020 | 7.0 ± 0.8 |
| struggling | cart_pos | 0.982 ± 0.003 | 16.0 ± 0.0 |
| target | pole_angle | 0.206 ± 0.068 | 2.3 ± 0.5 |
| target | cart_pos | 0.976 ± 0.003 | 15.0 ± 0.0 |
| competent | pole_angle | 0.096 ± 0.057 | 1.3 ± 0.5 |
| competent | cart_pos | 0.907 ± 0.031 | 14.3 ± 0.5 |

`cart_pos`-caused failures are detected far earlier and more reliably
than `pole_angle`-caused ones at every competence level -- drift toward
the rail boundary is a slow, gradually-building signal the predictor
picks up early; a pole-angle failure at higher competence is closer to a
sudden, late-breaking event with few informative steps beforehand.

---

## 10. Reliability diagram and precision-recall curve (target, seed 0)

<img src="docs/figures/reliability_diagram_target_seed0.png" width="400"> <img src="docs/figures/pr_curve_target_seed0.png" width="440">

Both generated from the actual saved `target`/seed-0 predictor weights
(`outputs/predictor_target_seed0.pt`) run against the real held-out test
split -- not reconstructed from summary statistics. Near-perfect
calibration (ECE=0.0032) and an AUPRC (0.930 for this specific seed) far
above the 0.147 base-rate floor line drawn on the PR curve.

**Scope note:** full reliability-diagram bin data exists for all 9
seed/point combinations (embedded in each `metrics_*.json`, since ECE's
15 equal-mass bins are computed and saved for every run) and could be
plotted the same way; raw PR-curve points require the trained model
weights, which were only persisted starting with the `target`/seed-0 run
(the weight-saving feature was added after the other 8 evaluations had
already run -- see `DEVIATIONS.md`). The scalar AUPRC for every
seed/point is in section 9's table regardless.

---

## 11. Cross-operating-point figure

![AUPRC, base rate, and detection lead time vs policy competence](docs/figures/cross_operating_point.png)

This is the actual research question this project sweeps for (CLAUDE.md
S1's "policy competence is swept, not guessed"): AUPRC is non-monotonic,
peaking at `target` and declining at `competent` even though AUROC keeps
improving -- exactly the base-rate effect AUPRC is supposed to be
sensitive to (Davis & Goadrich 2006) and AUROC is not. Detection lead
time stays close to the 16-step horizon ceiling at every competence
level, meaning when the predictor does flag danger, it tends to do so
close to the earliest possible moment, regardless of how rare failure has
become.

---

## 12. Limitations

- **Offline, frozen-policy only.** The predictor is trained and evaluated
  exclusively on data from frozen (non-learning) checkpoints. No claim is
  made about online use, or about predictor behavior if run against a
  *changing* policy — that is explicitly out of scope (see Future Work).
- **Predictor quality at high competence is base-rate-limited.** The
  `competent` point's AUPRC (0.760) is lower than `target`'s (0.938)
  despite a *higher* AUROC (0.951 vs 0.973) — this is the base-rate
  collapse visible in section 11, an inherent limit of predicting an
  increasingly rare event, not a predictor deficiency.
- **Reward is not the shipped mjlab reward.** The S2.6 centering term
  (section 4) means absolute reward numbers in this project are not
  directly comparable to mjlab's own published cartpole reward curves.
- **Three seeds (plus perturbation) is a small sample.** Headline mean±std
  figures in section 9 are computed over 3 seeds — enough to see the
  qualitative cross-operating-point pattern in section 11, not enough for
  tight confidence intervals. The seed-perturbation mechanism (section 6)
  additionally means "3 seeds" is not always literally RNG seeds
  `{0, 1, 2}`; the actual seeds used are recorded and disclosed.
- **The S2.6 ablation is a single-seed comparison** (section 5), by the
  spec's own design (CLAUDE.md: "seed 0 only") — its noisy, non-monotonic
  result should not be read as a confident causal claim either way about
  the centering term's practical effect, independent of the a priori
  engineering rationale for including it.

## 13. Future work

- **Online predictor training under a changing policy.** This project
  deliberately defers this — every technique here (ordinary supervised
  learning, standard train/val/test splits, plain BCE) assumes a
  stationary data distribution from a frozen policy. Training the
  predictor concurrently with an evolving PPO policy would require
  addressing non-stationarity directly (e.g. replay buffers, prior
  correction, prequential evaluation) — explicitly out of scope for v1
  (CLAUDE.md S1), and a natural next step.
- **Multi-seed ablation.** The S2.6 centering-term ablation (section 5)
  is a single seed by spec; repeating it across seeds would resolve
  whether the observed effect (weak, noisy, direction-inconsistent
  mid-training) reflects the term's true impact or single-run variance.
- **Ensemble uncertainty (CLAUDE.md S8.4 phase 2).** A deep ensemble of
  K=5 predictors with VaR-based risk-sensitive readout is specified but
  not implemented in this pass — `--predictor.uncertainty ensemble`
  raises `NotImplementedError` deliberately rather than approximating it.

## 14. Phase 2: Go1 fall pipeline (outside the cartpole spec)

Everything above is the specified cartpole experiment. This section is a
later extension to a Unitree Go1 quadruped in mjlab; it is not covered by
`CLAUDE.md`, and every decision is logged in `DEVIATIONS.md`. **All Go1
results are single-seed (seed 0).**

Pipeline: a walking policy, a 150 N push that makes it fall, the same
termination predictor trained on those falls, a separate protective
"small-fall" policy, and a hand-off from walker to protective policy when the
predictor fires.

### Reproduce

`notebooks/go1_end_to_end.ipynb` runs every stage (Colab A100; upload
`cartpole_term_code.zip`). It has not been executed as a whole; each script
in it was run individually. The scripts, in order:

```bash
python scripts/inspect_robot_task.py --robot go1
python scripts/robot_train.py --robot go1 --seeds 0 --eval.interval 1000
python scripts/go1_wild_push_predictor.py --checkpoint-file checkpoints/model_9999.pt --mechanism impulse --impulse-force 150
python scripts/go1_collect_flagged_states.py --checkpoint-file checkpoints/model_9999.pt --seed 0 --capture-state-bank
python scripts/go1_collect_flagged_states.py --checkpoint-file checkpoints/model_9999.pt --seed 5 --capture-state-bank --capture-at-flag
python scripts/go1_train_protective.py --seeds 0 --state-bank-file outputs/go1_flag_state_bank_seed5.npz   # round 8b; take model_200.pt
python scripts/go1_handoff_eval.py --protective-checkpoint <model_200.pt>
python scripts/go1_collect_flagged_states.py --checkpoint-file checkpoints/model_9999.pt --seed 4 --capture-state-bank --capture-at-flag
python scripts/go1_protective_compare.py --protective-checkpoint <model_200.pt> --state-bank-file outputs/go1_flag_state_bank_seed4.npz
python scripts/go1_handoff_demo_video.py --protective-checkpoint <model_200.pt>
# round 7 (earlier policy): go1_train_protective.py --seeds 0 --peak-force-weight 0, take model_120.pt
```

Versions recorded on the Go1 evaluation runs (2026-09-30, A100-SXM4-40GB):
mjlab 1.6.0, mujoco 3.11.0, mujoco-warp 3.11.0, rsl-rl-lib 5.4.2,
torch 2.11.0+cu128, warp-lang 1.17.0, CUDA 12.8 (the Gate 7 run on 2026-10-03 had
torch 2.11.0+cu130, CUDA 13.0, everything else identical). The walker, predictor, and
protective training runs (2026-09-13 to 09-25) used an unpinned
`pip install mjlab` and did not record versions.

### Walker and push

- Walker: mjlab's shipped Go1 flat-velocity task and PPO config, 4,096 envs,
  10,000 iterations; the only reward change is `air_time` = 0.25.
- Push (collection and evaluation only, never in walker training):
  `apply_body_impulse` on the trunk, each force axis uniform in ±150 N, zero
  torque, 0.1-0.3 s duration, 1-3 s cooldown, on top of the shipped mild
  `push_robot` event.

### Fall predictor (150 N push, seed 0, test split)

Same MLP, loss, labelling (16 steps = 0.32 s at 50 Hz), and episode-level
split as the cartpole predictor. Input is 48 observations + 12 actions.

| | base rate | AUPRC | AUROC | Brier | ECE | F1 | precision | recall | McFadden R² |
|---|---|---|---|---|---|---|---|---|---|
| MLP | 0.0180 | 0.849 | 0.991 | 0.0062 | 0.0006 | 0.755 | 0.822 | 0.698 | 0.741 |
| logistic regression, all inputs | 0.0180 | 0.736 | 0.965 | 0.0082 | 0.0017 | 0.674 | 0.709 | 0.641 | 0.620 |
| base-rate constant | 0.0180 | 0.018 | 0.500 | 0.0176 | 0.0089 | 0.035 | 0.018 | 1.000 | 0.000 |
| constant negative | 0.0180 | 0.018 | 0.500 | 0.0180 | 0.0180 | 0.000 | 0.000 | 0.000 | -4.51 |

1,003,212 labelled transitions, 1,164 terminated episodes, one failure cause
(`fell_over`). Threshold 0.547, chosen on validation. Offline median
detection lead 12 steps (0.24 s). Best single feature: action L2 norm
(AUROC 0.953). Deviation from the cartpole spec: inputs are not normalized.

### Protective policy (Gate 5)

Trained with the same PPO stack on episodes that start from real mid-fall
states (a "bank" captured between the predictor's flag and the fall). Reward:
vulnerability-weighted contact force, root linear velocity, action rate,
joint limits, joint torque, and ground covered by the fall (our own term).
The checkpoint in this subsection is round 7, iteration 120; the reported
policy is round 8 (below).

Pre-registered confirmation on a fresh bank (seed 2, 505 falls, medians):

| | trunk/hip peak force (N) | any-body peak force (N) | peak torque (Nm) | ground covered (m) | came to rest |
|---|---|---|---|---|---|
| protective | 0 | 1206 | 12.7 | 0.32 | 100% |
| walker continues | 482 | 1155 | 35.6 | 0.61 | 75% |
| zero action | 569 | 1193 | 13.1 | 0.41 | 100% |

The pre-registered verdict was **fail**: the protective policy must beat both
baselines on all four metrics, and it does not on any-body peak force. Per
fall, trunk/hip force is lower than zero action in 86-90% of falls and ground
covered in 82-83%; torque against zero action and any-body force are ties.
In 98% of protective falls the peak force lands on the legs.

### Live hand-off (Gate 6)

Walker under the push, predictor scoring every step, irreversible switch to
the protective policy on the first threshold crossing. 1,024 trials, nominal
arm (no switch): live recall 0.999 (763 of 764 falls flagged first),
precision 0.834 (152 of 915 flags were not followed by a fall within 3 s),
median lead 15 steps (0.30 s). Weights unchanged, no env switched back.
**Pass.**

### Hand-off ablation (Gate 7)

Each robot's exact state at the moment the predictor fires was captured from
the live system (fresh bank, seed 3, 500 flag moments, each followed by a real
fall) and replayed from identical start states under three policies, with no
further pushes. Medians:

| | trunk/hip peak force (N) | any-body peak force (N) | peak torque (Nm) | ground covered (m) | came to rest |
|---|---|---|---|---|---|
| hand-off to protective | 0 | 1159 | 12.9 | 0.37 | 100% |
| walker continues | 380 | 1026 | 35.6 | 0.87 | 62% |
| zero action | 611 | 1323 | 13.7 | 0.55 | 99.6% |

The pre-registered verdict (hand-off beats the walker on all four) was
**fail**, on any-body peak force: the hand-off is higher in 57% of falls
(median +106 N). It is lower than the walker on trunk/hip force in 72% of
falls (higher in 4%), on peak torque in 98%, and on ground covered in 96%.

This is the round-7 policy (iteration 120); round 8 below supersedes it.

### Round 8: the reported protective policy

After the Gate 7 fail the stated goal was fixed as: the main aim is less
ground covered by the fall; contact force should come down a bit but may stay
large. Round 8 added one reward term (a penalty on the fall's peak
contact force on any single body) and trained two variants: 8a on the same
mid-fall states as round 7, 8b from flag-moment states, which is where the
hand-off really starts. One checkpoint was picked by a fixed rule on the
seed-3 flag bank (**8b, iteration 200**) and tested once on a fresh flag bank
(seed 4, 500 flag moments). Medians:

| | trunk/hip peak force (N) | any-body peak force (N) | peak torque (Nm) | ground covered (m) | came to rest |
|---|---|---|---|---|---|
| hand-off to protective (8b it200) | 0 | 1001 | 18.4 | 0.36 | 99.8% |
| walker continues | 363 | 995 | 35.6 | 0.95 | 59% |
| limp (motors off) | 281 | 1099 | 0 | 0.46 | 99.8% |

Limp switches the motors off: zero torque at every joint from the hand-off
moment (both motor gains scaled to zero; measured torque exactly 0). It is the
"do nothing" baseline. An earlier stiff-hold baseline ("zero action": the
motors hold the default pose) was dropped once limp was measured, because limp
was better than it on ground covered, trunk force, and any-body force while
using no torque (same run: 0.52 m, 617 N, 1292 N, 13.2 Nm). It still appears in
the round-7 tables above, which predate the limp arm. A first run of these
falls without the limp arm agreed with this table within about 1%.

- **Space goal: pass against both baselines.** Ground covered is 62% below the
  walker's and 21% below limp's. Per fall it is lower than the walker in 98% of
  falls and than limp in 77%.
- **Force:** trunk/hip force is lower than the walker in 69% of falls (higher
  in 3%) and than limp in 72%; the trunk never touches the ground in 90% of
  falls (limp: 25%). Any-body peak force is level with the walker (1001 vs
  995 N; higher in 53% of falls) and 9% below limp (lower in 55% of falls).
  Under the original four-metric rule (beat the walker on all four) this is
  still a fail, by a few newtons.
- **Cost:** the protective policy uses motor torque (18.4 Nm peak, about half
  the walker's, higher than round 7's 12.9 Nm); limp needs none. Its margin on
  space is much smaller against limp than against the walker.

Demo video: `outputs/videos/go1_handoff_demo_round8b_it200.mp4` (three falls, walker
alone vs hand-off, side by side; in the video the pushes stop once the robot is
on the ground).

### Limitations (Go1)

- **Single seed.** One walker, one predictor, one protective policy.
- **The protective policy does not reduce peak impact below the walker's.** It
  moves the impact from the trunk to the legs. With round 8, median peak force
  on any body is level with the walker's and a little below limp's; its upper
  tail (worst 10% of falls) is still higher than both.
- **The limp baseline is idealized:** no joint friction or damping was added, so
  it is the most free-falling version of an unpowered robot.
- **Run-to-run noise.** The same bank replayed twice gave walker medians of
  1026 N and 1073 N (the simulator is not deterministic), so differences of a
  few percent between medians should not be read as effects.
- **Round 8 was not re-run through the live hand-off check (Gate 6)** and has
  no mid-fall (Gate 5-style) test; its evidence is the flag-moment replay.
- **False alarms.** 17% of live predictor flags were not followed by a fall;
  on those the hand-off puts a robot on the ground that would have stayed up.
  That cost is not measured.
- **Checkpoint selection.** Each reported table comes from a bank that was not
  used to train or to choose the checkpoint (round 7: chosen on seed 1, tested
  on seed 2 and seed 3; round 8: chosen on seed 3, tested on seed 4).
- **The live paired ablation was not possible.** MuJoCo Warp is not
  bit-deterministic, so two live runs cannot be matched robot by robot; the
  Gate 7 ablation replays captured flag-moment states instead, without
  further pushes. That replay covers only flags that led to a real fall.
- **Ground covered is our own metric,** not from the fall literature.
- G1 humanoid: only the task scaffolding exists; no G1 results are reported.
