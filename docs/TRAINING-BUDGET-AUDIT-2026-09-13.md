# Training budget termination audit

An epoch interrupted by `max_hours` received a final validation before the
trainer checked patience. If that validation exhausted patience, the trainer
overwrote `stopped_by=time_budget` with `patience`. This could admit a censored
result to a completed-run comparison. Budget interruption now takes priority
after the existing operator-stop check. Patience is still checked at epoch
boundaries; this change does not change the stopping schedule of live processes.

The actual GPU trainer regression uses a tiny synthetic task with validation
after each step, an exhausted first-step time budget, and patience of one.
It reproduced `AssertionError: patience` before the fix and passed after it.
The existing monitored/unmonitored equivalence, stop-signal/file, checkpoint,
worker-signal and overwrite-protection checks also passed. Tests used free GPU
1 in existing allocation 416043, without requesting another node. Receipts are
retained in the local monitor as `budget-stop-before.log` and
`budget-stop-after.log`.

Eight historical results labelled `patience` exceeded their configured budget
in the recorded total wall time. Total time includes final evaluation and is
insufficient to prove a training interruption. Full logs confirmed the bug for
only `tuab-cf2_v1d192` seeds 0 and 1:

- Seed 0: `logs/CF2_v1d192_b-407990-tuab_cf2_v1d192-s0.out`, budget reached
  at epoch 2, step 9000, followed by a patience message.
- Seed 1: `logs/SEED_03-413806-tuab_cf2_v1d192-s1.out`, budget reached
  at epoch 2, step 8400, followed by a patience message.

Their completion metadata is corrected to `time_budget`, with original JSON
files and evidence hashes retained under `results/audits/budget-stop-20260913/`.
Validation/test metrics and all other original fields are preserved. Seed 2
already recorded `time_budget` correctly. Consequently all three TUAB CF2
results are budget-limited. The table generator already marks any such seed
with a budget flag and excludes that cell from bold best-value ranking.

The other six inspected logs contain no explicit mid-epoch budget interruption;
their metadata is preserved. Existing generated tables and historical metrics
are not edited manually. Future table generation reads the corrected records.
