# Notebooks

Every notebook is a thin Colab shim: install, upload, then call `scripts/*.py`.
Upload `cartpole_term_code.zip` (rebuild with `python3 tools/build_code_zip.py`).

## Current

| Notebook | What it is for |
|---|---|
| `colab_entry.ipynb`, `colab_entry_2.ipynb` | The cartpole experiment (spec: `docs/CARTPOLE_SPEC.md`). |
| `go1_end_to_end.ipynb` | The whole Go1 pipeline in one file. One walker, then everything after it repeated per seed in `SEEDS` (default five), combined by `scripts/go1_multiseed_stats.py` into one set of box plots. Resumes from Drive after a disconnect. `SEEDS = (0,)` is the original single run. |
| `go1_round8_redo.ipynb` | Round 8 of the protective policy (run 2026-10-03 with its first version; produced the reported policy, 8b iteration 200): trains two variants, picks one checkpoint by a fixed rule, tests it once on a fresh bank, renders the demo video. Upload folder: `colab_uploads/round8/`. |
| `go1_recovery_eval.ipynb` | Confirmatory statistical evaluation of the three recovery policies: three fresh flag banks (about 1,500 paired falls), a mid-fall bank, a repeat replay, then the paired tests and figures. Upload folder: `colab_uploads/recovery_eval/`. |
| `go1_limp_baseline.ipynb` | Adds a limp (motors off) baseline to the reported round-8 comparison, on the same 500 flag moments. Upload folder: `colab_uploads/limp_baseline/`. |
| `go1_handoff_demo_video.ipynb` | Re-renders only the side-by-side hand-off demo video. Upload folder: `colab_uploads/demo_video/`. |
| `g1_walker.ipynb` | Stage 1 of the G1 humanoid port (`docs/G1_PORT_PLAN.md`): trains the G1 walker for 10,000 iterations, resuming from Drive after a disconnect; then its evaluation curve and a video. Clones the public repo, so no upload. Built by `tools/notebook_builders/build_g1_walker_nb.py`. |

## `archive/`: the runs that produced the recorded Go1 results

(Local only: `notebooks/archive/` is not published in the repository.)

Kept as the record of how each result in `DEVIATIONS.md` was produced. Their
steps are all in `go1_end_to_end.ipynb`. Their upload folders were removed
(every file was a copy); the inputs are under `checkpoints/` and `outputs/`.

| Notebook | Gate | What it ran |
|---|---|---|
| `part_a_gate0_gate1.ipynb` | 0-2 | Task inspection, first walker training, push failure-rate sweep. |
| `go1_train_full.ipynb` | 1 | The 10,000-iteration walker (`checkpoints/model_9999.pt`). |
| `go1_watch_fixed_command.ipynb` | 1 | Walker video at a fixed velocity command. |
| `go1_multiseverity_predictor.ipynb` | 2-4 | Predictor pooled over several push severities (not the final choice). |
| `go1_wildpush_video.ipynb` | 2-4 | The final 150 N push dataset, predictor, and overlay video. |
| `go1_horizon_sweep.ipynb` | 4 | Predictor horizon sweep. |
| `go1_collect_flagged_states.ipynb` | 5 prep | Live lead-time measurement and post-fall hold. |
| `go1_train_protective.ipynb`, `go1_train_protective_retrain.ipynb` | 5 | Fall-state bank and protective rounds 1-3. |
| `go1_protective_round4.ipynb` ... `round7.ipynb` | 5 | Protective rounds 4-7. |
| `go1_watch_protective_episodes.ipynb` | 5 | Video of the protective policy. |
| `go1_protective_heldout_eval_round7.ipynb` | 5 | Held-out bank (seed 1), checkpoint choice (iteration 120). |
| `go1_protective_confirm_seed2.ipynb` | 5 | Pre-registered confirmation on seed 2. |
| `go1_handoff_eval.ipynb` | 6 | Live hand-off check (Gate 6 pass; live pairing not possible). |
| `go1_gate7_flagstate.ipynb` | 7 | Flag-moment ablation on seed 3 and the first demo video. |
