# Moving this repo to another cluster

Written for the full pretraining run. Everything the repo needs is either in
git, regenerable, or listed here as something to copy.

## 1. What to copy, and what not to

| | size | how |
|---|---|---|
| the git repo | ~30 MB | `git clone` / `git bundle`. Includes `runs/` and `archive/` — every stored result, config and val curve |
| `vendor/` | **1.6 GB** | **copy, do not clone.** Five upstream repos *plus their pretrained checkpoints*. `slurm/vendor_clone.slurm` re-clones the code but not the weights |
| `$PACLOCK_PROC/processed*` | **~490 GB** | copy if the bandwidth is there, otherwise regenerate — see §4 |
| `$PACLOCK_DATA` (raw corpora) | ~510 GB | only needed to regenerate the above |

`logs/` is ignored and not worth moving.

## 2. The two environment variables

Nothing else is configured. `REPO` is derived from `paclock_bench/paths.py`'s
own location, so the checkout finds `vendor/` wherever it lands.

```bash
export PACLOCK_DATA=/new/path/raw          # raw corpora, read-only
export PACLOCK_PROC=/new/path              # parent of processed*/
```

Both default to the paths they had on the AMD HPC Fund, so the repo keeps
working there with nothing set. Configs say `$PACLOCK_PROC/processed/tuev` and
are expanded by `paths.expand()` as they are read; an absolute path passes
through untouched, so old configs still work.

Check the move with:

```bash
python3 -m scripts.smoke_paths      # builds real loaders + models from 3 configs
```

## 3. Cluster assumptions baked into `slurm/`

These are AMD HPC Fund specifics and will need editing:

* `--partition=mi2104x`, and **no GPU GRES** — `scontrol show node` reports
  `Gres=(null)`, so `--gpus=1` is rejected and the only way to get a GPU is
  `--exclusive`, which hands over all four MI210s. Both packing scripts exist
  because of that: `seeds_packed.slurm` runs three seeds of one config on one
  node, `configs_packed.slurm` runs up to four different configs. On a cluster
  with proper GRES, ask for one GPU per job and drop both.
* `module load pytorch/2.7.1` (ROCm 6.3.1).
* **24-hour wall limit**, enforced by a submit filter. `max_hours` in a config
  makes a run stop cleanly and still write its `result.json` rather than being
  SIGKILLed with nothing.
* `MIOPEN_USER_DB_PATH` / `MIOPEN_CUSTOM_CACHE_DIR` are set per process to
  node-local `/tmp`. Concurrent processes sharing one MIOpen SQLite cache on a
  shared filesystem corrupt it ("database disk image is malformed"). Keep this
  on any ROCm cluster; harmless on CUDA.

## 4. Regenerating the preprocessed corpora

Four protocols, because hard rule 2 says each model runs its own repo's
preprocessing (`docs/PROTOCOLS.md`):

```bash
sbatch slurm/preprocess.slurm         <dataset>   # frozen protocol -> processed/
sbatch slurm/preprocess_pac.slurm     <dataset>   # PAC protocol    -> processed_pac/
sbatch slurm/preprocess_biot.slurm    <dataset>   # BIOT + TFM      -> processed_biot/
sbatch slurm/preprocess_labram.slurm  <tuab|tuev> # LaBraM          -> processed_labram/
```

Two are currently missing on the source cluster and will have to be rebuilt
rather than copied, both lost to disk pressure: `processed_biot/tuab` and
`processed_labram/tuab`. `processed_tfm/` never existed; the TFM cells read
`processed_biot` through the same loader.

## 5. Read these before changing anything

* `docs/PERF.md` — the 9.4x speed fix. `set_seed()` forces
  `cudnn.deterministic=True`, which on ROCm made the `in_channels=1` patch
  convolutions pick an atomics-free backward-weights kernel. The fix replaced
  them with a GEMM. **If the new cluster is CUDA this may not reproduce** — the
  2x2 in that document is the way to check, not assumption.
* `docs/ARCH_SEARCH.md` — the PAC estimation window result, and the two
  predictions it falsified.
* `docs/DELIVERABLE.md` — the configuration to pretrain and the evidence for
  each choice.
* `docs/RECIPE_AUDIT.md` — every baseline recipe deviation and its source file.

## 6. The one number that governs how results are compared

Seed spread is **measured per corpus, never assumed**: ISRUC sd 0.0021, TUEV sd
0.0235 — eleven times apart, because 57% of TUEV cells peak at their first
evaluation and 0% of ISRUC cells do. A delta has to clear roughly twice its own
corpus's sd before it means anything.

And the reason `archive/runs_conv_tokenizer/` exists: a change that was
mathematically identical and verified to rel 8.9e-7 moved an ISRUC result by
0.0276, **13.4 seed standard deviations**. After any numerical change to the
frontend, re-run rather than reuse.
