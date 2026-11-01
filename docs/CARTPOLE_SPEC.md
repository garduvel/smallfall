# CLAUDE.md — mjlab Cartpole: PPO + Offline 16-Step Termination Predictor

**Audience:** Claude Code. **Status:** binding specification. Every numbered
requirement is a constraint, not a suggestion.

---

## 0. Prime directive

You are implementing a **specified** experiment, not designing one. Every
architectural, algorithmic, and hyperparameter decision has already been made and
is written below.

1. **Do not innovate.** If this document specifies a value, use that value. If it
   specifies a method, use that method. Do not "improve" the reward, the
   architecture, the optimizer, the schedule, the metrics, or the evaluation
   protocol. Every technique in this document is standard published practice; that
   is why it is here.
2. **Do not invent APIs.** This cites mjlab docs as of v1.5–1.6. If an API named
   here doesn't exist in the installed version: **stop**, read the actual source in
   `site-packages/mjlab/`, adapt *minimally*, record it in `DEVIATIONS.md`. Never
   guess a signature; never paper over an `ImportError` with a reimplementation.
3. **Log every deviation.** `DEVIATIONS.md` must exist with one entry per
   departure, including the reason and the evidence (error text, source snippet).

If a requirement is ambiguous or self-contradictory, **stop and ask.** Do not
resolve ambiguity by choosing.

### 0.1 The goal, stated once

Produce a working, reproducible pipeline in which:

1. PPO trains a cartpole balance policy that can fail (angle and position limits).
2. Training is instrumented with an **accurate frozen evaluation** so that a
   specific checkpoint can be selected by a measured property of that policy.
3. Frozen checkpoints generate **static labelled datasets**.
4. A small MLP is trained **offline** on those datasets to predict termination
   within 16 environment steps, and is evaluated with proper splits and metrics.
5. Two videos demonstrate the result.

Everything in this document serves that. Nothing else is in scope.

---

### 0.2 Execution environment — two machines

The machine where you write code is **not** the machine that runs it. Your sandbox
is CPU-only and has no mjlab, no torch, and no GPU. Training, evaluation, dataset
collection, and video all require an NVIDIA GPU and run elsewhere (Colab T4 or
equivalent).

**Consequences, all binding:**

1. **Everything is a real file in the repo.** No notebook cells, no `!command`
   one-liners, no asking the user to paste output back for you to interpret. Each
   gate's work is a script under `scripts/` or a module under `src/` that the user
   can run unattended on the GPU machine.
2. **Every script writes its own results to disk** — `DEVIATIONS.md`,
   `outputs/*.json`, `logs/` — rather than printing something a human must
   transcribe. Assume nobody is watching the terminal.
3. **Scripts must fail loudly and specifically.** You will often not see the
   failure yourself, so an error message is the entire debugging channel. No bare
   `except:`, no silent fallbacks.
4. **A gate you cannot execute is passed when the artifact is correct and the user
   confirms the run succeeded.** Write the script, state exactly what command the
   user should run and what output means pass, then stop and wait. Do not mark a
   gate passed on your own say-so.
5. **Pure-Python work you *can* run, you must run** — the labelling tests (§7.4),
   the split test (§7.5), and the reward maths tests (§11) need neither GPU nor
   mjlab if written against plain tensors. Write them so they run in your sandbox
   and run them there. Gate 2 is fully executable by you; treat it as such.

If a repo remote is available, prefer it over manual file shuttling: you push, the
user pulls and runs on the GPU machine, results come back in the repo. Ask once,
then work with whatever the user says.

---

## 1. Design decisions already made, and why

These were settled before you started. Do not relitigate them.

**The predictor is trained offline, not online.** The reference paper (§1.2)
collects 8 hours of simulation from a *fixed* controller, parses it into 72,000
gait cycles, and trains offline with a train/val split. A frozen policy means a
stationary data distribution, which means ordinary supervised learning applies and
every metric means what it normally means. It also means the dataset is a file you
can retrain against fifty times without re-running PPO.

**Online predictor training is explicitly out of scope for v1.** It is recorded as
future work in the README and nowhere else. Do not implement replay buffers,
prior correction, prequential evaluation, or base-rate tripwires — those exist only
to survive a non-stationary stream, and there is no stream here.

**Policy competence is swept, not guessed.** Rather than stopping training early to
get a mediocre policy, PPO trains to convergence and datasets are collected at
**three frozen operating points** (§7). This yields predictor performance as a
function of policy competence — a stationary, properly-split measurement at each
point.

**mjlab + RSL-RL is the stack.** PPO is not reimplemented. Mass parallelism is fine
here because the only per-episode stateful logic (labelling) lives in a standalone
offline collection script, not interleaved with training.

### 1.2 Reference paper mapping

Methodology mirrors **STATE-NAV** (Yoon, Zhu, Lu, Gan, Zhao, arXiv:2506.01046,
RA-L):

| STATE-NAV | This project |
|---|---|
| Digit biped in MuJoCo on random rough terrain | mjlab cartpole (balance) |
| Fixed ALIP controller, random velocity commands | PPO policy **frozen at a selected checkpoint** |
| Gait cycle as prediction unit | One env step (20 Hz) |
| Label: 1 if a fall occurs within **two subsequent steps** | Label: 1 if failure within **16 subsequent env steps** |
| Inputs: elevation patch `m_ego` + command `a=[v,ω]` | Inputs: observation `o_t` + action `a_t` |
| 8 h of sim → 72k labelled cycles, trained offline | Frozen rollouts → ≥1M labelled steps, trained offline |
| TravFormer (ResNet-18 + attention) | MLP — the terrain input motivating the CNN does not exist here |
| Two-phase: point estimate, then uncertainty head | Two-phase: BCE, then ensemble (§6.5) |
| Risk-sensitive readout, VaR at α = 0.97 | Same α on the ensemble predictive distribution |
| Feature screening by logistic regression, reported with McFadden's R² and AUC-ROC | Same two statistics (§8.3) |

**Deference rule: where this document is silent and a choice must be made, choose
what STATE-NAV chose.** Where it has no analogue (no terrain, no planner, no MPC),
choose the plainest standard option and log it.

TravRRT\*, MPC, and stability-aware command velocity are **out of scope.**

### 1.3 PPO provenance

mjlab ships RSL-RL (Schwarke et al., arXiv:2509.10771) and a cartpole PPO config;
both are used verbatim. Underlying methodology: Schulman et al. 2017; Rudin et al.
CoRL 2022 (adaptive-KL schedule, massively parallel on-policy). The
implementation-fidelity literature (Engstrom et al. ICLR 2020; Andrychowicz et al.
ICLR 2021; Huang et al. ICLR Blog Track 2022) exists to make one point: **use the
library's tested implementation and don't touch it.**

**You will not write a PPO implementation.** If you find yourself writing a GAE
loop, a ratio clip, or advantage normalization, you have gone wrong.

---

## 2. Verified environment facts

Checked against mjlab docs (Cartpole tutorial, Terminations, RSL-RL training).
**Re-verify against the installed version; log mismatches.**

### 2.1 Shipped model (`src/mjlab/tasks/cartpole/cartpole.xml`)

- `slider` joint: `type="slide"`, `axis="1 0 0"`, **`limited="true"`,
  `range="-1.8 1.8"`**, `damping="5e-4"`
- Cart geom: box, `size="0.2 0.15 0.1"`, `mass="1"`
- Pole body `pole_1`, hinge joint `hinge_1` — **unlimited**; the angle does not
  wrap and grows without bound as the pole spins
- Actuator: `<motor name="slide" joint="slider" gear="10" ctrllimited="true"
  ctrlrange="-1 1"/>` → max force **10 N**, clamped internally by MuJoCo

### 2.2 Timing

`timestep=0.01`, `disableflags=("contact",)`, `decimation=5` → **one env step =
0.05 s = 20 Hz.** `episode_length_s=50.0` → **max 1000 env steps.**

**The 16-step horizon is therefore 0.8 seconds.** Assert `timestep` and
`decimation` at startup; fail loudly if either differs.

### 2.3 Observations (shipped, 5-dim, do not change)

| Term | Dim | Source |
|---|---|---|
| `cart_pos` | 1 | `joint_pos_rel` on `slider` |
| `pole_angle` | 2 | custom `pole_angle_cos_sin` on `hinge_1` → `[cos θ, sin θ]` |
| `cart_vel` | 1 | `joint_vel_rel` on `slider` |
| `pole_vel` | 1 | `joint_vel_rel` on `hinge_1` |

Groups `"actor"` and `"critic"` share terms. **No observation noise** in the
shipped config — leave it that way.

### 2.4 Reset events (shipped, do not change)

`reset_joints_by_offset`, relative to the entity's initial state:
- `slider`: `position_range=(-0.1, 0.1)`, `velocity_range=(-0.01, 0.01)`
- `hinge_1`: `position_range=(-0.034, 0.034)` (≈ ±1.95°), `velocity_range=(-0.01, 0.01)`

Balance init: `{"slider": 0.0, "hinge_1": 0.0}`.

### 2.5 Reward — shipped term (keep) plus one added term (§2.6)

The shipped reward is a single multiplicative term `cartpole_smooth_reward`
reproducing dm_control's shaped reward:

```
r_smooth = (cosθ+1)/2 × (1+g(x))/2 × (4+q(u))/5 × (1+g(θ̇))/2
           upright      centered     small ctrl    small velocity
```

each factor in [0, 1]. **Keep this term exactly as shipped, weight 1.0.** Do not
replace it with a +1-per-step alive bonus, do not rescale it, do not alter any
factor.

### 2.6 Added reward term: cart centering (REQUIRED)

The shipped `centered` factor is multiplicative and bounded below at 0.5, so a cart
at the rail edge still collects half the positional credit while the pole is
upright. That is weak enough that a policy can balance while slowly drifting into
the ±1.0 m termination boundary. This project adds an explicit additive centering
term. **It is the only reward change permitted.**

`src/cartpole_term/rewards.py`:

```python
def cart_centered(env, asset_cfg, sigma: float) -> torch.Tensor:
    """Gaussian centering kernel on absolute cart position. Returns [num_envs] in (0, 1]."""
    asset = env.scene[asset_cfg.name]
    x = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)   # absolute, not _rel
    return torch.exp(-(x / sigma) ** 2)
```

```python
rewards = {
    "smooth_reward": RewardTermCfg(              # shipped, unchanged
        func=cartpole_smooth_reward, weight=1.0,
        params={"cart_cfg": cart_cfg, "hinge_cfg": hinge_cfg},
    ),
    "cart_centered": RewardTermCfg(              # added
        func=cart_centered, weight=CART_CENTER_WEIGHT,
        params={"asset_cfg": cart_cfg, "sigma": CART_CENTER_SIGMA},
    ),
}
```

**Binding parameters and their rationale:**

- `CART_CENTER_SIGMA = 0.5` — half the ±1.0 m termination limit. The term falls to
  `exp(-1) ≈ 0.37` at ±0.5 m and `exp(-4) ≈ 0.018` at the boundary, so nearly all
  centering credit is gone well before the episode ends. A gradient that only
  appears at the boundary arrives too late to be useful.
- `CART_CENTER_WEIGHT = 0.2` — the shipped term contributes at most 1.0 per step,
  capping this at 20%. **The policy must never prefer sitting centred with a falling
  pole to moving off-centre to catch it**; a weight well under 1.0 guarantees that.
- **Use absolute `joint_pos`, not `joint_pos_rel`**, per §3.2.2.

**Honest note, to be reproduced in the README:** this term is additive and *not*
potential-based, so by Ng, Harada & Russell (1999) it can shift the optimal policy
rather than merely accelerate convergence to the original optimum. That is accepted
deliberately — a centred policy is the intent, not a side effect. Do not "fix" this
by converting it to a potential-based form.

**Required ablation (seed 0 only):** one additional training run with
`--env.reward.cart-center-weight 0.0`, which exactly reproduces the shipped reward.
Report both mean-|x| curves and both termination-cause breakdowns.

### 2.7 Terminations (shipped)

Only `time_out` (registered `time_out=True`, a truncation). We add two failure
terminations (§3). Together with §2.6 these are the only environment modifications.

### 2.8 Task variant

Use **`Mjlab-Cartpole-Balance`** (pole starts upright). **Not Swingup** — it starts
at `hinge_1 = π`, instantly outside ±12°, making the task impossible.

### 2.9 Verified API surface

The following were confirmed working against a real mjlab install (prior notebook,
Colab T4). Prefer these over guessing; still re-verify and log mismatches per §0.2.

```python
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import time_out
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.cartpole.cartpole_env_cfg import (
    cartpole_balance_env_cfg, cartpole_ppo_runner_cfg,
)
from mjlab.tasks.registry import (
    list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls, register_mjlab_task,
)
from mjlab.utils.os import get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import check_nan
```

**Both** `cartpole_balance_env_cfg` and `cartpole_ppo_runner_cfg` live in
`cartpole_env_cfg.py`. There is no `cartpole/rl_cfg.py`.

`register_mjlab_task(task_id=..., env_cfg=..., play_env_cfg=..., rl_cfg=...,
runner_cls=...)` — `runner_cls` is confirmed available and is the §6 insertion point.

**Step signature (critical for §7.4).** After `RslRlVecEnvWrapper`, `env.step()`
returns a **4-tuple**, not a 5-tuple, and truncation arrives via `extras`:

```python
obs, rewards, dones, extras = env.step(actions)
time_outs = extras.get("time_outs")
truncated = time_outs.bool() if time_outs is not None else torch.zeros_like(dones, dtype=torch.bool)
terminated = dones.bool() & ~truncated
```

Runner internals used by §6: `self.alg.act(obs)`, `self.alg.transition.values`,
`self.alg.process_env_step(...)`, `self.alg.compute_returns(obs)`,
`self.alg.update()`, `self.alg.train_mode()` / `eval_mode()`;
`self.logger.lenbuffer` (rolling window over the last ≤100 completed episodes),
`self.logger.writer`, `self.logger.log(...)`;
`runner.save(path, infos)` / `runner.load(path, load_cfg={"actor": True}, strict=True,
map_location=...)` / `runner.get_inference_policy(device=...)`.

**Known gotchas:**
- `agent_cfg.logger` defaults to **`wandb`** (needs an account). Override to
  `"tensorboard"` explicitly — see §5.3.
- `get_checkpoint_path`'s default checkpoint regex `.*` also matches tensorboard and
  params files. Pass `checkpoint=r"model_.*\.pt"`.
- A task registered in-process is invisible to the `train` CLI (it launches a
  subprocess). In-process training must construct the runner directly.
- `init_at_random_ep_len=True` on `learn()` pre-ages envs to random offsets — the
  IsaacGym/IsaacLab idiom. Without it, envs started together never resync and all hit
  the time limit on the same step once the policy stops failing, producing bursty
  completions instead of a steady trickle. **Use it.**

---

## 3. Termination thresholds

The shipped mjlab cartpole **has no position termination at all.** What the sources
say:

- **mjlab:** the rail is a *physical* joint limit at **±1.8 m**. Hitting it is a
  mechanical stop, not an episode end.
- **Gym/Gymnasium `CartPole-v1`:** terminates at **|x| > 2.4** units and **|θ| >
  12° (0.2095 rad)**. The 12° figure is field-standard, from Barto–Sutton–Anderson
  1983.
- **dm_control `cartpole`:** no failure termination; shaped reward over a
  fixed-length episode — what mjlab inherited.

**±12° is standard and adopted as-is. ±2.4 m is impossible here** (it exceeds the
rail). **±1.0 m is a sound choice:** comfortably inside ±1.8 m, so episodes end by
*policy failure* rather than by hitting a joint limit — exactly what a termination
predictor needs.

### 3.1 Binding values

```python
POLE_ANGLE_LIMIT_RAD = 0.20943951023931953   # 12 degrees
CART_POS_LIMIT_M     = 1.0
RAIL_LIMIT_M         = 1.8                   # hard XML limit, assertions only
CART_CENTER_SIGMA    = 0.5
CART_CENTER_WEIGHT   = 0.2
PREDICTION_HORIZON   = 16                    # env steps (= 0.8 s at 20 Hz)
```

Expose as config fields with these defaults. Do not change the defaults. Assert
`CART_POS_LIMIT_M < RAIL_LIMIT_M` with ≥ 0.5 m margin and
`0 < CART_CENTER_WEIGHT < 1.0`; fail loudly otherwise.

### 3.2 Termination implementation

`src/cartpole_term/terminations.py`, following mjlab's custom-termination pattern
(plain function taking `env` + params, returning a bool tensor `[num_envs]`):

```python
def pole_angle_out_of_bounds(env, asset_cfg, limit_rad: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    theta = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    # hinge_1 is UNLIMITED: the raw angle does not wrap. Wrap before comparing.
    theta_wrapped = torch.atan2(torch.sin(theta), torch.cos(theta))
    return theta_wrapped.abs() > limit_rad


def cart_pos_out_of_bounds(env, asset_cfg, limit_m: float) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    x = asset.data.joint_pos[:, asset_cfg.joint_ids].squeeze(-1)
    return x.abs() > limit_m
```

**Mandatory correctness notes:**

1. **Wrap the angle.** `hinge_1` is unlimited. An unwrapped comparison is a real bug
   that silently passes on balance and fails on anything else.
2. **Use absolute `joint_pos`, not the relative observation.** `joint_pos_rel`
   subtracts the default joint position. For balance the defaults are 0.0 so the two
   coincide — which is exactly why this bug goes unnoticed. Use
   `asset.data.joint_pos`, and write a unit test that fails if someone swaps in the
   relative term. Applies equally to §2.6's reward.
3. **Register both with `time_out=False`** (the default). They must map to
   `terminated`, never `truncated`. Keep `time_out` registered `time_out=True`. If
   failures leak into the truncation channel, both value bootstrapping and the
   entire labelling scheme are wrong.

```python
terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "pole_angle": TerminationTermCfg(
        func=pole_angle_out_of_bounds,
        params={"asset_cfg": hinge_cfg, "limit_rad": POLE_ANGLE_LIMIT_RAD},
    ),
    "cart_pos": TerminationTermCfg(
        func=cart_pos_out_of_bounds,
        params={"asset_cfg": cart_cfg, "limit_m": CART_POS_LIMIT_M},
    ),
}
```

**Log the termination-cause breakdown** per iteration: fraction of episode ends that
were `pole_angle`, `cart_pos`, `time_out`. Needed for the §2.6 ablation, and a cheap
check that both terminations fire.

### 3.3 Register new tasks; do not mutate the shipped one

Register **`Mjlab-Cartpole-Balance-Term`** via `register_mjlab_task` with its own
`env_cfg`, `play_env_cfg`, `rl_cfg`, and `runner_cls` (§6). Build the env cfg by
calling mjlab's shipped balance config factory, then adding the two termination
terms and the one reward term. **Never edit files inside `site-packages/mjlab/`.**

---

## 4. Repository layout

```
.
├── CLAUDE.md                       # this file, unmodified
├── DEVIATIONS.md                   # required, §0.3
├── README.md                       # generated last
├── pyproject.toml                  # uv-managed
├── src/cartpole_term/
│   ├── __init__.py                 # register_mjlab_task
│   ├── env_cfg.py                  # shipped balance cfg + terminations + reward
│   ├── terminations.py             # §3.2
│   ├── rewards.py                  # §2.6
│   ├── rl_cfg.py                   # §5
│   ├── runner.py                   # §6 — frozen-eval runner subclass
│   └── predictor/
│       ├── labeling.py             # ring buffer + censoring  ← highest-risk file
│       ├── model.py                # MLP, optional ensemble
│       └── train.py                # offline supervised training
├── scripts/
│   ├── train.py                    # PPO with frozen eval
│   ├── select_checkpoints.py       # §7.2
│   ├── collect_frozen.py           # §7.3
│   ├── eval_predictor.py           # §8
│   └── record_video.py             # §9
├── tests/
│   ├── test_terminations.py
│   ├── test_rewards.py
│   ├── test_labeling.py            # MANDATORY, §7.4
│   └── test_split.py
├── reference/                      # READ-ONLY prior work, see §4.1
│   └── old_notebook_setup.md
├── colab_entry.ipynb               # optional thin shim, see §4.2
├── logs/                           # rsl_rl checkpoints, tensorboard
└── outputs/                        # datasets, metrics, figures, mp4s
```

### 4.1 The `reference/` folder — scoped read-only

`reference/` contains extracted notes from a **superseded** prior design (a fixed
controller, no PPO, a different predictor architecture). It exists for exactly one
reason: the environment setup in it was debugged against real hardware and is still
valid.

**It may be consulted only for:** dependency versions and pins, install commands,
`MUJOCO_GL` / EGL configuration, asset and `cartpole.xml` paths, and known platform
quirks.

**It must not be consulted for:** architecture, hyperparameters, network design,
training loop structure, evaluation protocol, labelling, metrics, or methodology.
All of those have changed and **this document supersedes them.**

- Do not copy code from `reference/` into `src/` or `scripts/`.
- Do not edit anything in `reference/`.
- If something in `reference/` appears to contradict this document, **this document
  wins.** Note the conflict in `DEVIATIONS.md` and continue.

If `reference/` is absent, proceed normally — it is a convenience, not a dependency.

### 4.2 Colab (optional)

If the work runs on Colab, `colab_entry.ipynb` is a **thin shim only**: pip install,
clone or `cd`, set `MUJOCO_GL`, then shell out to `scripts/*.py`. **No logic in
notebook cells.** Everything that needs unit tests, fixed seeds, or reproducible
invocation lives in `src/` and `scripts/`. Do not port the pipeline into notebook
cells, and do not create this file unless Colab is actually being used.

---

## 5. PPO configuration — copy, do not choose

### 5.1 Primary instruction

**Read the installed `src/mjlab/tasks/cartpole/cartpole_env_cfg.py` and reuse
`cartpole_ppo_runner_cfg()` verbatim** (confirmed location, §2.9 — there is no
`cartpole/rl_cfg.py`), changing only the fields in §5.3. That file
is authoritative. Do not retype values from memory.

### 5.2 Fallback (only if the installed file can't be read; log in DEVIATIONS.md)

```python
RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu",
                        obs_normalization=True, stochastic=True, init_noise_std=1.0),
    critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu",
                         obs_normalization=True, stochastic=False, init_noise_std=1.0),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.01, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="cartpole_term",
    logger="tensorboard",
    save_interval=10,
    num_steps_per_env=24,
    max_iterations=300,
)
```

### 5.3 The only permitted changes

1. `experiment_name="cartpole_term"` — so logs don't collide with the shipped task.
1b. `logger="tensorboard"` — the shipped default is **`wandb`**, which requires an
   account (§2.9). Override it explicitly.
2. **`save_interval=10`** (not 50). Required: §7 selects checkpoints by measured
   episode length, and a coarse interval skips past the target windows.

`max_iterations` may be raised if the policy hasn't converged, but only after
reporting the training curve. Nothing else is tunable. **No hyperparameter sweeps.
No "I tried 3e-4 and it worked better."**

### 5.4 Runtime

- `--env.scene.num-envs 4096` default. Reduce only if the GPU can't hold it; log it.
- Seeds: **3 seeds** default (`0, 1, 2`), via `--seeds`. Headline numbers are mean ±
  std across seeds.
- Logger: **tensorboard**, local. W&B optional, off by default.
- mjlab needs an NVIDIA GPU for training. Set `MUJOCO_GL=egl` for headless rendering.

### 5.5 Required CLI surface

Nothing in this document is implicit, including the interface. Every flag below must
exist, with the stated default, on the script indicated. Use a single shared argument
parser module so defaults cannot drift between scripts.

| Flag | Default | Scripts | Purpose |
|---|---|---|---|
| `--smoke` | off | all | §11 gate 4 tiny-run mode: 64 envs, 5 iterations, 32 eval episodes, 5000 transitions |
| `--seeds` | `0 1 2` | train, select, collect, eval | §5.4 |
| `--env.scene.num-envs` | 4096 | train | §5.4 |
| `--eval.num-envs` | 256 | train | §6.2.2 |
| `--eval.interval` | 10 | train | §6.2.1 |
| `--env.reward.cart-center-weight` | 0.2 | train | §2.6 ablation |
| `--env.reward.cart-center-sigma` | 0.5 | train | §2.6 |
| `--env.termination.angle-limit-rad` | 0.20943951023931953 | train, collect | §3.1 |
| `--env.termination.cart-limit-m` | 1.0 | train, collect | §3.1 |
| `--operating-point` | `target` | collect, eval | one of `struggling` / `target` / `competent` (§7.1) |
| `--predictor.horizon` | 16 | collect, eval | §7.4 |
| `--predictor.history` | 1 | eval | §8.1 |
| `--predictor.uncertainty` | `none` | eval | `none` or `ensemble` (§8.4) |
| `--checkpoint-file` | — | collect, video | explicit path |

Every run must write its full resolved config (all flags, including defaults) to a
JSON file alongside its outputs. A result whose exact invocation cannot be recovered
is not reproducible.

---

## 6. Frozen in-training evaluation

This is the mechanism that makes checkpoint selection trustworthy. Without it, the
only available signal is RSL-RL's training-time mean episode length, which is a
property of a *policy trajectory* (a 1000-step episode spans ~40 PPO updates), is
length-biased toward completed episodes, and lags during improvement.

### 6.1 Provenance — this is standard practice, not innovation

Periodic frozen evaluation on a separate environment with fixed seeds and no
exploration is the established protocol in continuous control. TD3 (Fujimoto et al.
2018) evaluates every 5000 steps over 10 episodes with no exploration noise, and its
published learning curves are those evaluations rather than training returns. SAC
and DDPG follow the same pattern. Stable-Baselines3 makes it the default via
`EvalCallback` (separate `eval_env`, `n_eval_episodes`, `deterministic=True`), and
OpenAI Spinning Up's `test_agent()` is the same recipe. Fixed eval seeds so that
every evaluation sees identical initial states is likewise published variance
reduction.

**Report training curves too.** The massively-parallel robotics lineage (Rudin et
al. 2022, Isaac Lab, mjlab) plots training-time returns, and mjlab's own published
cartpole figure is training reward over 5 seeds. Both go in the README: the training
curve is the sample-efficiency plot; the frozen eval is the measurement of a
specific policy.

### 6.2 Implementation

`register_mjlab_task` accepts `runner_cls`. Subclass `MjlabOnPolicyRunner` in
`src/cartpole_term/runner.py` and add an eval hook. **This is the documented
insertion point — do not monkey-patch RSL-RL.**

Requirements, all binding:

1. **Frequency:** every **10 iterations**, matching `save_interval`, so every saved
   checkpoint has a measurement.
2. **Separate env instance:** a second `ManagerBasedRlEnv` with **256 envs**, built
   from a dedicated `eval_env_cfg`. **It must have identical dynamics, terminations,
   and reward to the training config.** Do **not** use `play_env_cfg` — that
   disables randomization and sets episode length to infinity. If eval dynamics
   differ from training dynamics, the measurement doesn't describe the trained
   policy. Do not carve the eval pool out of the training envs.
3. **Fixed eval seed** = `train_seed + 100`, reset identically before every
   evaluation, for the whole run. Same initial states every time.
4. **Freeze everything:** policy weights and the observation normalizer. Assert no
   parameter tensor changed across the eval, and that no optimizer step occurred.
5. **Run until all 256 episodes complete**, not a fixed step budget. A fixed step
   budget reintroduces length bias.
6. **Eval rollouts never enter the PPO buffer.**
7. **Log both action modes:** episode length under the distribution **mean** action
   and under **sampled** actions. One extra line; a divergence immediately reveals a
   collapsed or exploded `log_std`.
8. **Record per evaluation:** mean and median episode length, **standard error over
   episodes** (not over steps — episodes are the independent unit), censored
   fraction (episodes hitting the 1000-step cap), mean return, termination-cause
   breakdown.
9. Write `logs/.../eval_history.json`, one record per evaluation.

Cartpole is 2 DOF with contacts disabled, so a second Warp instance should be cheap.
**Verify memory and kernel-compile cost on the first run and log it** — this is the
one claim here that hasn't been checked against your hardware.

---

## 7. Frozen datasets at three operating points

### 7.1 Operating points

Rather than stopping training early to obtain a mediocre policy, train to
convergence and collect at three competence levels, selected by **frozen-eval mean
episode length** (§6):

| Name | Target | Tolerance |
|---|---|---|
| `struggling` | 80 steps | ±15 |
| `target` (primary) | 150 steps | ±25 |
| `competent` | 400 steps | ±60 |

`target` produces the headline numbers. The three together give predictor
performance as a function of policy competence — which is the actual research
question, obtained from three stationary datasets rather than one non-stationary
stream.

### 7.2 Selection must be mechanical, not eyeballed

`scripts/select_checkpoints.py`:

1. Read `eval_history.json`.
2. For each operating point, choose the **earliest** checkpoint whose frozen-eval
   mean episode length is inside the window. Earliest, not closest — if the metric
   is non-monotonic, the early crossing is on the improving limb and is the
   reproducible choice.
3. If no checkpoint qualifies for a point, **fail with a clear message** listing
   available checkpoints and their measured lengths. Do not silently substitute.
4. **Re-measure each selected checkpoint** with a *fresh* eval seed
   (`train_seed + 200`) and record that number as the reported one. Selecting and
   reporting on the same measurement is winner's-curse biased — you selected on
   favourable noise.
5. Write `outputs/selected_checkpoints.json`: per point, path, iteration, selection
   measurement, re-measurement, seed.

Run per seed. Three seeds × three points = nine datasets.

### 7.3 Collection

`scripts/collect_frozen.py`, a standalone script — no PPO, no interleaving:

- Load the checkpoint. Eval mode, but **keep sampling stochastically.** The
  deterministic mean action collapses the state distribution and strips out the
  near-failure diversity the predictor needs. The label is defined under the
  behaviour policy, so the behaviour policy is what runs.
- Freeze weights and the observation normalizer; assert no parameter changes.
- Derive `terminated` / `truncated` from the 4-tuple step return exactly as in §2.9.
  Do not assume a 5-tuple and do not reach for `auto_reset` unless §2.9's pattern
  proves insufficient; if it does, log why in `DEVIATIONS.md`.
- Collect until **≥ 1,000,000 labelled (non-censored) transitions** AND **≥ 1,000
  terminated episodes**. If the second bound isn't met at `competent`, keep
  collecting and report the wall-clock cost.
- Record per transition: `obs (5)`, `action (1)`, `label`, `env_id`, `episode_id`,
  `step_in_episode`, `steps_to_termination` (sentinel for negatives),
  `termination_cause`.
- Save `outputs/frozen_{point}_seed{N}.npz` with a JSON sidecar: base rate, episode
  count, terminated-episode count, censored-step count, termination-cause
  breakdown, episode-length mean/median/histogram.

### 7.4 Labelling — the highest-risk component

For env `i` at step `t`:

```
y_t^i = 1   iff  ∃ k ∈ {1,…,16} such that env i emits terminated=True at step
                 t+k within the same episode
y_t^i = 0   iff  env i survives all of steps t+1 … t+16 within the same episode
y_t^i = ⊥   (censored, DISCARD) otherwise
```

- `terminated` = the **failure** channel only. `time_out` is `truncated` and is
  **never** a positive.
- Horizon strictly future: `k ≥ 1`. The terminal step itself is not an example.
- **The censored case is the one everyone gets wrong.** If an episode is *truncated*
  at step `T`, steps `T-15 … T` have unknown futures — the episode would have
  continued but for the time limit. They are **not negatives.** Discard them.
  Labelling them 0 injects systematic bias exactly into the region where the policy
  performs best.

`predictor/labeling.py`, a per-environment ring buffer of depth 17:

```
For each env step, for each env i:
  1. Push x_t^i into env i's pending buffer with age 0.
  2. Increment the age of every pending entry.
  3. If terminated_i:
        every pending entry with age in [1, 16]  ->  emit with label 1
        flush the entire pending buffer for env i
  4. elif truncated_i:
        flush the entire pending buffer for env i, emitting NOTHING   # censored
  5. else:
        any pending entry whose age reaches 17 -> emit with label 0, pop it
```

Treat simultaneous `terminated AND truncated` as **terminated** (failure wins);
assert it is rare.

**Mandatory tests in `tests/test_labeling.py`.** Do not collect any dataset until
all pass:

- A synthetic episode of length 40 ending in `terminated` emits exactly 16 positives
  (steps 24–39, zero-indexed) and 23 negatives (steps 0–22). Step 23 is the last
  positive; terminal step 39 emits nothing.
- A length-40 episode ending in `truncated` emits 23 negatives (steps 0–22) and
  **zero** positives; steps 23–39 are discarded.
- A length-10 episode ending in `terminated` emits 9 positives and 0 negatives.
- Interleaved envs terminating on different steps do not cross-contaminate buffers.
- Total emitted count is invariant to `num_envs`.

### 7.5 Splitting — episode-level, no exceptions

Consecutive steps within an episode are strongly correlated, and steps inside one
16-step window are near-duplicates. **A random split over transitions leaks test
into train and will inflate AUROC to a meaningless number.**

Split **by `episode_id`**: 70% train / 15% val / 15% test, fixed seed. Every step of
an episode lands in exactly one split. `tests/test_split.py` must assert the
intersection of episode-id sets across splits is empty. Report each split's base
rate; they should agree within about a percentage point.

---

## 8. The predictor

### 8.1 Inputs

```
x_t = concat(o_t, a_t)     # 5 + 1 = 6 dimensions
```

`o_t` is the raw actor observation (before normalization); `a_t` is the action
actually sent at step `t` (sampled, not the mean). The predictor applies its own
normalization, fit on the **training split only** and applied unchanged to val and
test.

This mirrors STATE-NAV's `p_θ(m_ego, a)`: environment state plus commanded action.

**Optional, off by default:** `--predictor.history H` stacks the last `H` frames
into a `6H` input (STATE-NAV aggregates over a gait cycle). Default `H=1`. If `H>1`,
zero-pad at episode start and mask those steps out of the loss.

**Forbidden inputs:** the value function, policy hidden activations, the reward,
`step_in_episode`, or anything else unavailable at deployment. Episode-elapsed-time
in particular is a leak — it correlates with termination through the time limit.

### 8.2 Architecture

```python
class TerminationPredictor(nn.Module):
    # input_dim = 6 * H
    # hidden = (128, 128), activation ELU   (matches mjlab's actor/critic choice)
    # output = 1 logit
```

That is the whole model. **No RNN, LSTM, GRU, transformer, attention, residual
blocks, layer norm, or dropout.** STATE-NAV needed a CNN+attention backbone because
its input was a 2.5D elevation map patch; a 6-dimensional state vector needs an MLP.

### 8.3 Training

- Loss: **plain `BCEWithLogitsLoss`. No class reweighting, no `pos_weight`, no focal
  loss.** Base rates here run roughly 20% / 11% / 4% across the three operating
  points — well inside the range where plain BCE is fine, and reweighting would
  distort calibration, which is a reported metric.
- Optimizer: Adam, `lr = 1e-3`, `weight_decay = 0.0`, `max_grad_norm = 1.0`.
- Batch size 1024, up to **50 epochs**, with **early stopping on validation AUPRC**,
  patience 5 epochs. Restore the best-validation weights before testing.
- Fixed torch seed per run; report it.
- **The test split is touched exactly once, at the end.** No tuning against it.

### 8.4 Two-phase training (STATE-NAV structure, adapted)

STATE-NAV trains a point estimate first (MSE, 10 epochs), then extends the network
with a second head under Gaussian NLL, because optimizing prediction and uncertainty
jointly from scratch is unstable. Gaussian NLL is a regression loss and does not
transfer to a binary target, so the *structure* carries over and the *loss* does not.

- **Phase 1 (default, required):** single MLP, BCE. Headline results.
- **Phase 2 (implemented, `--predictor.uncertainty ensemble`, off by default):** a
  deep ensemble of **K = 5** MLPs (Lakshminarayanan et al. 2017), each initialized
  from the Phase-1 weights and trained on independently shuffled data. Report the
  ensemble mean probability and, mirroring STATE-NAV's risk-sensitive readout, the
  **VaR at α = 0.97** over member probabilities.

Do not implement evidential deep learning, Bayesian layers, or conformal prediction.

---

## 9. Evaluation

### 9.1 Required metrics, per operating point per seed

In `outputs/metrics_{point}_seed{N}.json` plus a combined markdown table:

| Metric | Why required |
|---|---|
| **Base rate of the test split** | Every other number is uninterpretable without it. Print it first. |
| **AUPRC**, with the base-rate baseline beside it | Correct primary metric under imbalance (Davis & Goadrich, ICML 2006). A no-skill classifier scores exactly the base rate. |
| **AUROC** | Comparability with STATE-NAV Table I. |
| **Brier score** | Calibration + sharpness in one number. |
| **ECE** (15 equal-mass bins) + reliability diagram | STATE-NAV's \|ΔPICP\| analogue for a binary target. |
| **F1, precision, recall** at a threshold chosen **on validation** | The threshold is a fitted quantity; choosing it on test is leakage. |
| **Confusion matrix** at that threshold | Makes the "always no" degenerate solution visible at a glance. |
| **Median detection lead time** | For each terminated episode, the earliest step in its 16-step window where `p > threshold`; report in steps and seconds. The metric that says whether the predictor is useful. |
| **Per-cause breakdown** of AUPRC and lead time (`pole_angle` vs `cart_pos`) | The two failure modes look different in state space. |
| **McFadden's R²** | Direct comparability with STATE-NAV Table I. |

### 9.2 Mandatory baselines, same table

1. **Constant-negative** (always 0) — include so the accuracy trap is visible.
2. **Base-rate constant** (always `π`) — calibration floor.
3. **Logistic regression** on the same 6 inputs. If the MLP doesn't clearly beat it,
   that is a finding to report, not a problem to fix by enlarging the MLP.
4. **Single-feature logistic regression on `|θ|` alone.** Same reason.

**Accuracy is banned as a headline metric.** It may appear only on the same line as
the constant-negative baseline's accuracy.

### 9.3 Feature screening (STATE-NAV §III-A analogue)

Fit a univariate logistic regression per feature (`x`, `cos θ`, `sin θ`, `ẋ`, `θ̇`,
`a`, plus derived `|θ_wrapped|` and `|x|`) on the `target` training split, and
report **McFadden's R² and AUC-ROC per feature** in a table shaped like STATE-NAV's
Table I. Required — it's the cheapest check that labelling isn't broken, because
`|θ_wrapped|` must come out strongly predictive. **If it doesn't, the labels are
wrong; stop and debug.**

### 9.4 Aggregation

All headline numbers as **mean ± std over 3 seeds.** Never report a single seed as
the result. The cross-operating-point comparison (AUPRC and lead time versus policy
competence) is a required figure.

---

## 10. Video deliverables

Two mp4s in `outputs/`. Set `MUJOCO_GL=egl` (headless).

1. **`outputs/policy_final.mp4`** — the fully trained policy balancing, ≥ 20 s, one
   environment, single camera. **Do not use `uv run play`** — it opens an interactive
   viewer, not a batch renderer, and does not work headless (§2.9). Build
   `ManagerBasedRlEnv(cfg=play_env_cfg, render_mode="rgb_array")`, wrap it in
   `mjlab.utils.wrappers.VideoRecorder`, then drive it with
   `runner.get_inference_policy()`. **Sanity-check the renderer first** — call
   `env.render()` once and assert a non-None array of the expected shape; EGL failures
   are silent (see `reference/`). The cart should stay visibly near centre — this is
   also the visual check on §2.6.

2. **`outputs/predictor_overlay.mp4`** — the **`target` checkpoint** policy with the
   predictor's output overlaid per frame: probability as text, a horizontal bar, the
   decision threshold marked, and the ground-truth label for that frame once known.
   Run long enough to capture **at least three failure episodes.** This is where a
   reader sees whether the probability actually rises before the pole goes.

For the overlay, render with `mujoco.Renderer` and composite text/bars with PIL or
OpenCV, then write with imageio-ffmpeg. **No animation frameworks, no interactive
dashboards, no web viewers.**

---

## 11. Execution order and gates

**Each gate must pass before the next step.**

| # | Step | Gate |
|---|---|---|
| 0 | Setup; print installed `mjlab`, `mujoco`, `mujoco-warp`, `rsl-rl`, `torch` versions into `DEVIATIONS.md` | GPU visible; `uv run demo` or equivalent runs |
| 1 | Terminations + centering reward; register `Mjlab-Cartpole-Balance-Term` | A short scripted rollout with random actions shows early termination, not 1000 steps; both causes fire |
| 2 | `test_terminations.py`, `test_rewards.py`, `test_labeling.py`, `test_split.py` | **All pass.** Labelling tests are non-negotiable. |
| 3 | Frozen-eval runner (§6) | Eval runs, weights provably unchanged, `eval_history.json` written, second env memory cost logged |
| 4 | **Smoke test:** `--smoke` — 64 envs, 5 iterations, 32 eval episodes, 5000 collected transitions, all scripts end-to-end | Completes in **under 5 minutes**, produces every artifact type |
| 5 | PPO training, seed 0 | Reaches ≥ 400-step frozen-eval mean; all three operating-point windows crossed |
| 6 | §2.6 ablation, seed 0, `--env.reward.cart-center-weight 0.0` | Completes; both mean-\|x\| and cause-breakdown curves saved |
| 7 | `select_checkpoints.py` | Three checkpoints found and re-measured; JSON written |
| 8 | `collect_frozen.py` × 3 | Base rates roughly 20% / 11% / 4%; ≥1M labelled transitions and ≥1000 terminated episodes each |
| 9 | `eval_predictor.py` | `\|θ_wrapped\|` strongly predictive (§9.3); MLP beats constant-negative on AUPRC at `target` |
| 10 | Seeds 1 and 2 | — |
| 11 | Videos | Both play; overlay contains ≥ 3 failures |
| 12 | `README.md` + tables + `DEVIATIONS.md` finalized | — |

**Do not start step 5 before step 4 passes.** A full run that dies at labelling costs
an hour for a bug a 5-minute smoke test catches.

**`tests/test_rewards.py` must assert:** `cart_centered` returns exactly 1.0 at
x = 0; is strictly decreasing in `|x|`; is below 0.02 at the ±1.0 m boundary; uses
absolute `joint_pos` (fails if swapped for `joint_pos_rel` with a non-zero default);
and total per-step reward stays within `[0, 1.2]`.

---

## 12. Hard prohibitions

Violating any of these means the deliverable is wrong even if it runs:

1. Writing a PPO implementation, or modifying RSL-RL's PPO in any way. Monkey-patching
   RSL-RL instead of using `runner_cls`.
2. Changing any PPO hyperparameter in §5.2 beyond those permitted in §5.3.
3. Changing the reward beyond exactly what §2.6 specifies. No third term, no
   retuning, no potential-based rewrite.
4. Training the predictor online, during PPO. Out of scope for v1 (§1).
5. Implementing replay buffers, prior correction, prequential evaluation, or
   base-rate tripwires. They solve a problem that no longer exists here.
6. Letting the predictor influence the policy in any way — reward, action, advantage,
   termination, or auxiliary loss.
7. Using `play_env_cfg` for the frozen evaluation, or carving the eval pool out of the
   training envs (§6.2.2).
8. Evaluating with unfrozen weights or an unfrozen observation normalizer.
9. Reporting the selection measurement rather than the re-measurement (§7.2.4).
10. Labelling truncated-episode tail steps as negatives (§7.4). Top-priority bug.
11. Random-splitting transitions instead of episodes (§7.5).
12. Using the unwrapped `hinge_1` angle, or `joint_pos_rel` in place of `joint_pos`,
    in the terminations or in `cart_centered`.
13. Adding observation noise, domain randomization, or perturbation forces.
14. Reporting accuracy as a headline metric, or any metric without the base rate
    beside it.
15. Reporting a single seed as the result.
16. RNNs, transformers, attention, or any architecture other than the specified MLP.
17. Tuning the predictor by trying variants and keeping the best on the test split.
18. Editing anything inside `site-packages/` or `reference/`; copying code out of
    `reference/` into `src/` or `scripts/`; consulting `reference/` for anything
    beyond environment setup (§4.1); putting pipeline logic in notebook cells (§4.2).
19. Inventing an mjlab API that doesn't exist in the installed version.
20. Committing checkpoints, datasets, or mp4s to git. `.gitignore` `logs/` and
    `outputs/`.

---

## 13. Reporting

`README.md`, written last, in this order:

1. Exact commands to reproduce every artifact.
2. Installed versions of mjlab, mujoco, mujoco-warp, rsl-rl, torch, CUDA driver.
3. Termination thresholds and justification (§3), including that ±1.0 m is a project
   choice inside mjlab's ±1.8 m rail and ±12° is the classic
   Gym/Barto–Sutton–Anderson value.
4. The §2.6 reward modification: the term, its parameters, why the shipped
   multiplicative `centered` factor was judged too weak, and the Ng–Harada–Russell
   note that this shaping is not potential-based and can shift the optimum by design.
5. The §2.6 ablation: mean cart |x| and termination-cause breakdown, with and without
   the centering term.
6. PPO training curves **and** frozen-eval curves (mean and median episode length with
   standard error over episodes, censored fraction, termination causes) for all seeds,
   with the §6.1 note on why both are shown.
7. The selected-checkpoint table: point, seed, iteration, selection measurement,
   re-measurement, dataset size, base rate, terminated-episode count, censored
   fraction.
8. The §9.3 feature-screening table.
9. The §9.1 metrics table with all §9.2 baselines, mean ± std over seeds.
10. Reliability diagram and precision–recall curve with the base-rate line drawn on it.
11. The cross-operating-point figure: AUPRC, base rate, and detection lead time versus
    policy competence.
12. A short **Limitations** section stating plainly: the predictor is trained and
    evaluated offline on frozen policies, so no claim is made about online use or about
    behaviour under a changing policy; predictor quality at high competence is limited
    by base rate and this is visible in the cross-point figure; the reward differs from
    the shipped mjlab cartpole by the §2.6 term, so absolute reward numbers are not
    comparable to the mjlab reference curve; and three seeds is a small sample for
    mean ± std.
13. A **Future work** section noting that online predictor training under a changing
    policy is the natural next step and was deliberately deferred.

Items 4 and 12 are not optional throat-clearing. They are the honest framing of what
this experiment does and does not show.

---

## 14. References

- Yoon, Zhu, Lu, Gan, Zhao. *STATE-NAV: Stability-Aware Traversability Estimation for
  Bipedal Navigation on Rough Terrain.* arXiv:2506.01046 (RA-L). — reference methodology.
- mjlab documentation: Cartpole tutorial, Terminations, Training with RSL-RL —
  `mujocolab.github.io/mjlab`.
- Schwarke, Mittal, Rudin, Hoeller, Hutter. *RSL-RL: A Learning Library for Robotics
  Research.* arXiv:2509.10771.
- Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization
  Algorithms.* arXiv:1707.06347.
- Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel
  Deep RL.* CoRL 2022.
- Engstrom et al. *Implementation Matters in Deep Policy Gradients.* ICLR 2020.
- Andrychowicz et al. *What Matters for On-Policy Deep Actor-Critic Methods?* ICLR 2021.
- Huang et al. *The 37 Implementation Details of PPO.* ICLR Blog Track 2022.
- Fujimoto, van Hoof, Meger. *Addressing Function Approximation Error in Actor-Critic
  Methods (TD3).* ICML 2018. — the periodic frozen-evaluation protocol of §6.
- Haarnoja et al. *Soft Actor-Critic.* ICML 2018. — same protocol.
- Raffin et al. *Stable-Baselines3.* JMLR 2021. — `EvalCallback` as the tooling default.
- Ng, Harada, Russell. *Policy Invariance Under Reward Transformations.* ICML 1999. —
  the potential-based shaping result behind the §2.6 caveat.
- Tassa et al. *dm_control: Software and Tasks for Continuous Control.* — origin of the
  shipped tolerance-style smooth reward.
- Thananjeyan, Balakrishna et al. *Recovery RL: Safe RL with Learned Recovery Zones.*
  RA-L 2021. — safety-critic framing of "probability of constraint violation in the
  near future."
- Srinivasan, Eysenbach, Ha, Tan, Finn. *Learning to be Safe: Deep RL with a Safety
  Critic.* arXiv:2010.14603.
- Davis, Goadrich. *The Relationship Between Precision-Recall and ROC Curves.* ICML 2006.
- Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty
  Estimation Using Deep Ensembles.* NeurIPS 2017.
- Barto, Sutton, Anderson. *Neuronlike Adaptive Elements That Can Solve Difficult
  Learning Control Problems.* IEEE SMC, 1983. — origin of the ±12° threshold.
