# reference/old_notebook_setup.md

Extracted from `mjlab_cartpole_ppo_predictor.ipynb` (superseded design: 512 envs,
online predictor, `V(s)` as an input feature, 1.6 m cart limit, 32-step horizon).

**Scope: environment setup and platform quirks only.** Per CLAUDE.md §4.1, nothing
here about architecture, hyperparameters, labelling, or methodology is authoritative.
The empirical findings in §5 are recorded because they are observations about *this
environment*, not design choices — treat them as evidence, not instructions.

---

## 1. Runtime

- Colab **T4 GPU** runtime. Verified working.
- `!pip install -q mjlab` — no version pin was used, no extra deps needed.
- The install may pull a newer `torch` and prompt a runtime restart. If so, restart
  and rerun from the top.

## 2. MUJOCO_GL — ordering matters

**`MUJOCO_GL` only takes effect on the first `mujoco` import per process.** Setting
it after import is silently ignored. If the notebook has already been run in the
session, the runtime must be restarted before changing it.

Set both variables before importing `mujoco`:

```python
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
```

CPU fallback (no CUDA) used `osmesa`, after:

```bash
apt-get -qq update && apt-get -qq install -y libosmesa6-dev
```

## 3. The EGL vendor ICD workaround (important)

Colab sometimes lacks the NVIDIA EGL vendor ICD file. The failure mode is **silent**
— you get a non-functional GL context rather than a clear error. Write it explicitly
before importing `mujoco` (same workaround as DeepMind's MuJoCo Colab tutorial):

```python
icd_path = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if not os.path.exists(icd_path):
    os.makedirs(os.path.dirname(icd_path), exist_ok=True)
    with open(icd_path, "w") as f:
        f.write('{"file_format_version": "1.0.0", '
                '"ICD": {"library_path": "libEGL_nvidia.so.0"}}')
```

## 4. Render sanity check

Because EGL failures are silent, the old notebook checked the renderer explicitly
before recording anything:

```python
env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
env.reset()
frame = env.render()
print("render() ->", type(frame), np.asarray(frame).shape if frame is not None else "")
```

Worth keeping as a gate before any video work.

## 5. Empirical findings about this environment

Observations from actual runs. Evidence, not instructions.

- **Cart drift is real and needs a direct gradient.** Evaluation showed the cart
  drifting off-track at a roughly constant rate across training. An upright-only
  reward gave no local gradient to recentre — only an indirect one via termination
  cutting off future reward — and that was not enough in practice. Adding an explicit
  centering term fixed it. This corroborates CLAUDE.md §2.6.
- **`num_envs=512`** was used, matching IsaacGymEnvs' bundled Cartpole config.
- **`max_iterations=80`** at 512 envs (~1.3M transitions) was enough to cross a
  ~100-step mean episode length on the way up.
- **The rail's hard stop is ±1.8 m**, confirmed against the model. The old build used
  a 1.6 m termination limit; the current spec uses 1.0 m.
- **Control timestep is 0.05 s** (decimation 5 × sim timestep 0.01), confirmed from
  `cartpole_balance_env_cfg`.

## 6. Miscellaneous gotchas

- **`agent_cfg.logger` defaults to `wandb`**, which requires an account. Override to
  `"tensorboard"` explicitly.
- **`get_checkpoint_path`'s default checkpoint regex is `.*`**, which also matches
  tensorboard event files and params files in the run directory. Pass
  `checkpoint=r"model_.*\.pt"`.
- **mjlab's `play` script opens an interactive viewer**, not a batch renderer. It does
  not fit a headless or notebook workflow. Videos were produced by driving
  `ManagerBasedRlEnv` with `render_mode="rgb_array"` wrapped in
  `mjlab.utils.wrappers.VideoRecorder`.
- **A task registered in-process is invisible to the `train` CLI**, which launches a
  subprocess. In-notebook training must construct the runner directly.
- **`init_at_random_ep_len=True`** (IsaacGym/IsaacLab idiom) pre-ages envs to random
  offsets. Without it, envs started together never resync and all hit the time limit
  on the same step once the policy stops failing, producing bursty completions
  instead of a steady trickle.
- Inline video preview used `mediapy`; Colab download used
  `shutil.make_archive` + `google.colab.files.download`.
