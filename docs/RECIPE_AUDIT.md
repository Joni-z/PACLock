# Recipe fidelity audit

Hard rule 2 says every model runs the recipe from its own repository. This
records what was checked against the vendored source, what was found off it, and
what was deliberately left alone. Each row cites the file that settles it, so a
reader can disagree with a decision without having to re-derive it.

Reproduce the machine-checkable part with:

    python -m scripts.fix_recipe_fidelity     # dry run, exits clean when aligned

---

## Corrected

| Model | Setting | Was | Now | Source |
|---|---|---|---|---|
| CBraMod | `grad_clip` | `null` | `1.0` | `finetune_main.py` `--clip_value default=1`; `finetune_trainer.py` calls `clip_grad_norm_` every step |
| CBraMod | `label_smoothing` (TUEV) | `0.0` | `0.1` | `finetune_main.py` `--label_smoothing default=0.1` |
| CBraMod | `patience` | `10` | `0` (off) | `finetune_trainer.py` runs all epochs and keeps the best state; it has no early-stopping branch |
| CBraMod | `select_metric` (binary corpora) | primary metric | `auroc` | `finetune_trainer.py` selects on `roc_auc > roc_auc_best` |
| CBraMod | `eval_every_steps` | `100` | `0` (per epoch) | its trainer validates once per epoch |
| LaBraM | `label_smoothing` (6 multi-class corpora) | absent / `0.0` | `0.1` | `run_class_finetuning.py` `--smoothing default=0.1`, used whenever `nb_classes != 1` |
| PACLock | `lr` | `1e-3` | `1e-4` | `Joni-z/PACLock` `configs/pacint_tuev_measured.yaml` |
| PACLock | `dropout` | `0.1` | `0.2` | same |
| PACLock | `band_pe` / `spatial_pe` | absent (silently `hz` / learned index) | `index` / `xyz` | AGENT.md:2974 names these the architecture of record |
| PACLock | `epochs` (3 small corpora) | `20` | step-budget floor | see below |

### Measured effect on group C

| Cell | Off-recipe | On-recipe | Δ |
|---|---|---|---|
| TUEV / CBraMod-scratch | 0.5367 | 0.5638 | +0.027 |
| PhysioNet-MI / CBraMod-scratch | 0.5212 | (re-running) | |
| TUAB / CBraMod-scratch | 0.8734 | (re-running) | |
| TUSZ / CBraMod-scratch | 0.4903 | (re-running) | |

Every one of these strengthens a baseline. That is the point: a win over a
mis-configured opponent is not a win.

Intermediate figures measured while the corrections were being landed one at a
time are **not** reportable and are recorded here only so they are not mistaken
for results. TUEV/CBraMod-scratch read 0.6098 with clipping and smoothing but
early stopping still on, and 0.6233 under a `patience: 51` that was a units bug
on my part -- patience counts evaluations, so with `eval_every_steps: 100` that
was one epoch of patience, and the run stopped at epoch 2 having sampled a dense
curve and picked a lucky checkpoint. The recipe CBraMod actually publishes --
fifty full epochs, validated once per epoch, best kept by AUROC -- gives 0.5638.
Training longer scores lower here, and the published recipe is still what gets
reported.

---

## Checked and left alone

**BIOT, EEGPT — no label smoothing.** Both construct a plain
`CrossEntropyLoss()` (`run_multiclass_supervised.py`, `downstream/finetune_*.py`),
so `0.0` is faithful and raising it would be *our* invention.

**LaBraM `layer_decay: 0.65`.** The argparse default is `0.9`, which looks like a
discrepancy, but `0.65` is what LaBraM's own downstream script passes. A script
the authors ship beats a default they never use.

**LaBraM label smoothing on TUAB / TUSZ / CHB-MIT.** Absent is correct: those are
binary, and `run_class_finetuning.py` takes the `nb_classes == 1 ->
BCEWithLogitsLoss` branch, which never consults `smoothing`.

**CHB-MIT focal loss (α=0.25, γ=2).** Protocol-level, not per-model: PROTOCOLS.md
appendix A fixes the loss per corpus and every model gets the same one. Probed
α=0.75 anyway on the one cell that fails there; it did not help.

**FACED normalisation (`div100`).** Matches CBraMod's own pipeline exactly —
`preprocessing_faced.py` resamples to 200 Hz and `faced_dataset.py` returns
`data/100`. FACED does arrive ~30x hotter than the other corpora (training-signal
std 27.88 against 0.10-1.02 elsewhere), but that is upstream's scale, not our
error, so the data is left as published.

---

## Two harness bugs found while doing this

**`patience` counts evaluations, not epochs.** With `eval_every_steps: 100`,
CHB-MIT runs ~49 evaluations per epoch (4941 steps), so a patience of 51 is one
epoch of patience rather than fifty. Setting `patience: 0` is what actually
disables early stopping — `train.py` reads `if patience and since_best >=
patience`.

**Checkpoint selection and reporting used the same metric.** They answer
different questions: the protocol fixes what is *reported*, each repo fixes what
it *selects on*. They diverge most where variance does. CHB-MIT's validation
split holds 150 positives in 21,184 windows, so PR-AUC swings between 0.007 and
0.5 between consecutive evaluations while AUROC barely moves; selecting and
early-stopping on that noise ended CBraMod's from-scratch runs after one epoch,
at chance. `select_metric` now separates them and defaults to the primary metric,
so every config that does not set it is bit-identical to before.

---

## PACLock's step budget

The reference recipe is "batch 32, 20 epochs", but the quantity that was
validated is the number of optimiser steps, not the number of sweeps. On TUEV at
batch 32 that is `68445/32*20 = 42,760` steps. Copying the epoch count to a
smaller corpus copies a fraction of the training: BCI-IV-2a has 2,160 windows, so
20 epochs is 1,340 steps.

The consequence was visible in the training loss, not just the test score. On
FACED it went `2.2207 -> 2.1858` across a whole run against a `ln(9) = 2.1972`
floor — the model never fit the training set. On BCI-IV-2a under the corrected
budget the loss sits on a plateau near `ln(4) = 1.386` for about twenty epochs and
then breaks through, reaching 0.66 by epoch 35. Twenty epochs ends the run inside
the plateau.

`gen_configs_d.py` therefore derives epochs from a step floor. A floor, not a
target: matching the budget exactly would cut TUAB, TUSZ and CHB-MIT from 20
epochs to 4, and this must not shorten training anywhere. Only the three small
corpora move.

| Corpus | Epochs | Steps |
|---|---|---|
| PhysioNet-MI | 20 -> 218 | 42,728 |
| BCI-IV-2a | 20 -> 638 | 42,746 |
| FACED | 20 -> 120 | 25,200 (24h partition cap) |
| all others | unchanged at 20 | already above the floor |

This is the fair reading rather than a generous one. Early stopping still decides
where training actually ends, and the large corpora early-stop long before their
twentieth epoch, so raising a ceiling they never reach changes nothing for them.

### What it did not fix

A scale hypothesis was tested and rejected: FACED arrives ~30x hotter than the
other corpora, so an input standardisation was tried on the theory that it
distorted the PAC product term. It changes the input (PhysioNet-MI's epoch-0
loss moves from 1.4074 to 1.4073) but not the loss trajectory, because the
encoder's LayerNorms already absorb the scale. The code was removed rather than
left behind as an unused option.
