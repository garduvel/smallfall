# CLAUDE.md

Two pieces of work live in this repo. Read the part that applies before
changing anything.

## 1. Cartpole: PPO + offline 16-step termination predictor (complete)

The binding specification is **`docs/CARTPOLE_SPEC.md`** (this file's former
full contents, moved unchanged on 2026-10-03). It is a specified experiment,
not a design space: before touching cartpole code (`src/cartpole_term/` outside
`go1/`, `g1/`, `protective/`, `common/`; `scripts/` without a `go1_`/`robot_`
prefix), read that file and follow it exactly. Section references written as
"CLAUDE.md S<n>" in code comments, `README.md`, and `DEVIATIONS.md` mean
section `<n>` of `docs/CARTPOLE_SPEC.md`.

Rules from it that apply to all work here:

- **Do not invent APIs.** If an mjlab or rsl-rl name is not confirmed, read the
  installed source, adapt minimally, and log it. Never guess a signature.
- **Log every deviation** in `DEVIATIONS.md`, with the reason and the evidence.
- **No PPO reimplementation,** no monkey-patching RSL-RL; use `runner_cls`.
- **Never edit `site-packages/` or `reference/`.**
- **Never commit checkpoints, datasets, mp4s, or zips** (`.gitignore` covers
  `logs/`, `outputs/`, `checkpoints/`, `archive/`, `colab_uploads/`).
- Scripts fail loudly, write their results and resolved config to disk, and
  carry no pipeline logic in notebook cells.
- Labelling, episode-level splits, and "the test split is touched once" are
  non-negotiable (spec sections 7.4, 7.5, 8.3).

## 2. Go1 fall pipeline (phase 2, outside the cartpole spec)

Walker -> 150 N push -> fall predictor -> protective "small-fall" policy ->
hand-off. Not covered by the spec; every decision and result is in
`DEVIATIONS.md` (append-only, newest at the bottom) and summarized in
`docs/PROJECT_NOTES.md` section 14; the five-run results are in
`docs/GO1_MULTISEED_REPORT.md`. `README.md` is a short public overview.
Older references to "README.md S<n>" (in `DEVIATIONS.md` and code comments)
mean section `<n>` of `docs/PROJECT_NOTES.md`, the former README.

- Code: `src/cartpole_term/{go1,protective,common}/`, `scripts/go1_*.py`,
  `scripts/robot_*.py`. G1 (`g1/`) is scaffolding only.
- One notebook for the whole pipeline: `notebooks/go1_end_to_end.ipynb`.
  `notebooks/README.md` says what every other notebook is for.
- Evaluation rule used throughout: fix the pass rule in `DEVIATIONS.md` before
  the data exists, keep train / selection / test fall banks separate (one seed
  per role), and report the result whatever it is.
- The main goal of the protective policy is less ground covered by the fall;
  contact force should come down a bit but may stay large (user, 2026-10-03).

## 3. Two machines

- **Here (Chromebook Linux container):** no GPU, no AVX, so MuJoCo cannot run.
  Pure-torch logic is unit-tested with mjlab stubbed:
  `.venv/bin/python -m pytest -q tests`. Run everything that can run here.
- **Colab (A100):** training, evaluation, rendering. Work is handed over as a
  notebook under `notebooks/` plus one upload folder under `colab_uploads/`;
  results come back as one zip dropped in the repo root, which is then
  unpacked under `outputs/` and archived under `archive/`.
- `cartpole_term_code.zip` (repo root) is what notebooks upload; rebuild it
  after any change under `src/` or `scripts/`.
