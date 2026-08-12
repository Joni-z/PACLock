# paclock-bench

PACLock — a phase-amplitude-coupling tokeniser for EEG — and the nine-corpus
benchmark it is measured on, against five foundation models and five light
supervised baselines under each model's own published recipe.

## Start here on a new cluster

```bash
export PACLOCK_DATA=/path/to/raw          # raw corpora, read-only
export PACLOCK_PROC=/path/to/processed    # parent of processed*/

sbatch slurm/bootstrap.slurm              # clone upstream repos, audit checkpoints
sbatch slurm/run.slurm scripts.verify_frontend   # frontend invariants
sbatch slurm/run.slurm scripts.smoke_paths       # real config -> loader -> model
```

Both variables default to the paths this was built on, so nothing needs setting
to keep working there. `docs/MIGRATION.md` is the full checklist; `docs/DATASETS.md`
covers where each corpus comes from.

Nothing heavier than an editor belongs on the login node — that includes
anything importing torch or touching the preprocessed arrays. `slurm/run.slurm`
takes any module: `sbatch slurm/run.slurm scripts.collect_waves`.

## Running an experiment

```bash
sbatch slurm/preprocess.slurm frozen tuev              # once per corpus+protocol
sbatch slurm/seeds_packed.slurm configs/experiments/tuev_paclock_full.yaml
sbatch slurm/run.slurm scripts.fill_xlsx --xlsx results/_in.xlsx
```

`seeds_packed` runs three seeds of one config on one node, one GPU each;
`configs_packed` runs up to four different configs. Both exist because this
cluster exposes no GPU GRES, so `--exclusive` is the only way to get a GPU and
it hands over all four. With proper GRES, ask for one GPU per job and use
`train.slurm`.

## Layout

```
paclock_bench/     the package
  paths.py           every filesystem location, resolved once
  training/          loop, metrics (hard rule 3), losses, LaBraM layer decay
  data/              three loaders: frozen/PAC, BIOT+TFM, LaBraM
  models/
    paclock/         frontend/triaxial.py is the core: sinc -> Hilbert -> PAC
    foundation/      five upstream adapters
    baselines/       the light supervised group
configs/           datasets, experiments (the matrix), deliverable, _cand, _diag
preprocessing/     four protocols, one module per corpus
scripts/           collection, verification, config generation
slurm/             seven scripts; run.slurm is the generic one
docs/              read PERF, ARCH_SEARCH, DELIVERABLE, PROTOCOLS
runs/              every result.json — config, val curve, verdict
archive/           superseded results, with why each was retired
```

## Four rules the results depend on

1. **Reproduction gate.** Group A must reproduce published numbers before any
   group B/C number is believed. A mismatch is the pipeline, not the model.
2. **Each model runs its own repo's recipe** — preprocessing, normalisation and
   finetuning. Feeding a foundation model our pipeline instead of its own turned
   0.6772 into 0.4436 once. Hence four preprocessing protocols, not one.
3. **A mis-configured cell is not written.** `training/metrics.py` refuses cells
   whose validation curve is flat, or that never clear chance.
4. **Fewer than three seeds is withheld.**

And one learned the hard way: **seed spread is measured per corpus, never
assumed.** ISRUC sd 0.0021, TUEV sd 0.0235 — eleven times apart. A delta has to
clear roughly twice its own corpus's sd to mean anything.

## Two results worth knowing before changing code

**Training was 9.4x slower than it needed to be** — `set_seed()` forces
`cudnn.deterministic=True`, which on ROCm made the `in_channels=1` patch
convolutions select an atomics-free backward-weights kernel. Replaced with a
GEMM. Neither factor costs anything alone; together they cost 3.76x. On CUDA
this may not reproduce — check with the 2x2 in `docs/PERF.md`
(`scripts/bench_ab.py`), do not assume.

**A mathematically identical frontend change moved an ISRUC result by 0.0276 —
13.4 seed standard deviations.** After any numerical change to the frontend,
re-run rather than reuse. That is why `archive/runs_conv_tokenizer/` exists.
