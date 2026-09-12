# Narrow the concurrent AMD experiment scope

Filling idle GPUs admitted every pending configuration before reconciling the
old search grids with the new tokenizer work. The resulting 68 training processes
covered six overlapping workstreams. Capacity/regularization combinations were
premature while the core self-inclusive and factorized comparisons were unresolved.

## Deferred experiments

Stop 15 early-stage extensions on CHB-MIT, TUSZ, and IIIC:

- crofremo_n3/n4/n5: 24 bands, wider model, stacked regularization (nine runs).
- crofremo_s3/s4: self-inclusive coupling plus 16 bands or regularization (six runs).

This is a scope decision, not a rejection based on one seed's metric. Core n1 and
s1/s2 comparisons remain across multiple corpora. TUEV n3/n4 were at epoch 16 of
20 at inspection, and TUEV is the main event classification target, so its three
remaining n variants and four s variants continue. TUEV-only extensions cannot
establish cross-corpus factor effects.

The stopped design runs had used 0.9–1.7 hours; most large-corpus runs had not
completed their first validation. The six stopped backfill runs had just started.
PID identity, config filename, seed, owner, and allocation cgroup were verified
before terminating only those process trees. Existing logs/results were preserved.

## Retained work

| Workstream | Active configs | Purpose |
| --- | ---: | --- |
| Factorized tokenizer | 24 | Four arms on six corpora |
| Self-inclusive coupling | 10 | s1/s2 on four corpora, plus two TUEV extensions |
| Original design grid | 6 | n1 on three corpora, TUEV n3/n4/n5 |
| Existing CF2 regularization | 12 | Finish ISRUC/IIIC/CAUEEG comparisons, mostly at epoch 11 or 16 of 20 |
| Pure-coupling missing seeds | 3 | CHB-MIT seed1 and TUAB seeds1/2 |
| Existing pretrained-checkpoint fine-tuning | 3 | Complete TUSZ/CHB-MIT/TUAB diagnostics |
| Total | 58 | 68 minus 15 deferred plus five previously waiting |

SleepEDF f0–f3 and TUAR f3 filled freed slots. The backfill pending queue is empty.
No new combinations or pretraining were submitted. Existing regularization has
substantial progress and some validation signal; it will not be expanded before
the tokenizer decision. The old pretraining's mixed-rate issue remains a
limitation of its diagnostic fine-tunes. Incomplete/budget-stopped runs cannot
support final model comparisons.

## Receipts

Queue: /work1/chenyuyou/yifanwang/Zhizhe/PACLock-backfill-20260913.

priority-audit.json records the pre-stop progress inventory. priority-stop-plan.json
and priority-stop-receipt.json record intentional terminations. Six backfill
records moved to deferred/ with an explicit reason, so planned stops are not
reported as training failures. Nine original packed-script children appear in
the receipt rather than the backfill queue. Deferred work must not auto-retry.

priority-live-verification.json verifies surviving root training processes.
The status tool now checks Slurm allocation state after a controller drains:
released means its original batch shell resumed, not that original training ended.
Otherwise it would omit the four TUEV SELF runs on 416537 and incorrectly report
54 instead of 58 live training processes.
