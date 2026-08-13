# Superseded runs

Each directory holds `result.json` files that were produced under a
configuration or a code version that later turned out to be wrong. They are
kept, not deleted, because the numbers in them were quoted at some point and
the record of *why* they were withdrawn is part of the provenance.

| directory | why it was retired |
|---|---|
| `runs_conv_tokenizer` | the Conv1d patch tokeniser, before it became a GEMM. A mathematically identical change, verified to rel 8.9e-7, moved an ISRUC result by 0.0276 = **13.4 seed standard deviations** (docs/PERF.md), so nothing here is comparable with anything after it |
| `runs_wrong_recipe` | 15 group-B cells run off their own repo recipe -- CBraMod without gradient clipping and with early stopping on, LaBraM without label smoothing. Every deviation understated a baseline, i.e. flattered us (docs/RECIPE_AUDIT.md) |
| `runs_stale`, `runs_superseded`, `runs_invalidated` | earlier config generations |
| `runs_starved_batch` | batch size below the recipe |
| `runs_double_norm` | normalisation applied twice |
| `runs_wrong_head` | wrong classifier head for the corpus |
| `runs_no_layerdecay` | LaBraM without its layer-wise LR decay |
| `runs_mixed_evalcfg` | inconsistent evaluation cadence |
| `runs_coarse_val` | validation too sparse to answer hard rule 3 |
| `runs_eegpt_recipe` | pre-correction EEGPT recipe |
| `runs_wrong_tfm_batch` | wrong TFM batch size |
| `runs_diag`, `runs_rerun_samples` | diagnostics, not cells |

Nothing here should be read into a table. `scripts/fill_xlsx.py` only ever
looks at `runs/`.

| `runs_paclock_p200` | `patch_len=200` full-scratch PACLock runs, the config before the wave 1-7 architecture search. Superseded by `paclock_v2` (`patch_len=50`, `docs/DELIVERABLE.md`), which wins on both TUEV (0.7076 vs the old config, 3 seeds, significant) and every other measured axis. Kept so a from-scratch run under the old default is still on record, not shown in the main table so the PACLock row never mixes two different configurations under one label |
| `runs_patch50_collapsed` | FACED/PhysioNet-MI `paclock_v2` at the original `patch_len=50` deliverable value. 2/3 seeds each collapse to chance at this token count (32/64 channels, the largest in the matrix) -- see docs/ARCH_SEARCH.md's collapse diagnosis. Superseded by `patch_len=200` reruns; deliverable configs updated accordingly |
