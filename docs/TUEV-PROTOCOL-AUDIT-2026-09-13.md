# TUEV comparison protocols — 2026-09-13

**Historical native-loader validation scores do not share one validation
cohort.** Keep them separate from internally matched PACLock comparisons.

| Pipeline | Train windows | Validation windows | Test windows | Validation subjects |
|---|---:|---:|---:|---:|
| PACLock frozen | 68,445 | 15,487 | 29,421 | 58 |
| LaBraM native | 64,450 | 19,037 | 26,067 | 58 |
| BIOT native, historical result | 67,515 | 16,417 | 29,421 | Current manifest unavailable |

The first two validation subject sets intersect in **nine** subjects. Their
training sets have 232/231 subjects, with 182 shared; their test sets have
80/79 subjects, with 79 shared. Both use the official eval directory, but
different test windows prevent treating the scores as measurements on
identical examples. Equal BIOT/frozen test class counts alone do not establish
window identity.

The frozen implementation sorts train subjects for an 80/20 division;
LaBraM shuffles with seed 4523 and BIOT with 12345. Native conditioning also
differs. These are intentional implementation choices requiring disclosure.
The current LaBraM manifest records 19 exclusions for channel-order/count
mismatch after its drop list: 17 recordings with 28 remaining channels, one
with 25 and one with 24. The frozen manifest has none. These messages do not
establish which required/extra electrodes were present; this audit does not
alter channel selection or regenerate existing arrays.

## Metadata repair

The LaBraM and BIOT loaders previously returned an empty manifest even though
their preprocessing scripts emit one. They now read the actual manifest
before loading arrays, matching the common loader's existing requirement.
A missing manifest fails explicitly. Tests in existing allocation **416043**
passed for both loaders: metadata propagation, labels/counts, bit-exact
preservation of native normalization, and early failure for missing metadata.
The tests used CPU tensors inside the allocation; no GPU was required.

Historical result JSON is preserved. A current manifest is contextual evidence,
not retroactive proof of the data consumed by an old run. The historical BIOT
AMD data-root directory was absent at inspection; its current manifest has
not been recovered. Do not invent timestamps for it.

## Candidate and paper consequences

The running pretraining pair and lane-scale pairs retain their matched
internal questions. For external comparison, specify the target protocol and
verify subjects/events, conditioning, checkpoint selection and seed coverage.
Native-protocol results can support reproduction checks within their own
protocol; they cannot alone establish a common-cohort SOTA claim. Preserve
held-out evaluation and do not change cohorts in response to test scores.
No additional baseline training was launched for this audit.

Evidence: `/Users/mr.z/PACLock-monitor/tuev-protocol-manifests.tar` and
`/Users/mr.z/PACLock-monitor/tuev-protocol-comparison.json` contain the current
AMD manifests and their locally computed comparison. BIOT counts above come
from the historical `tuev-biot_prest16/seed0/result.json`.

## CBraMod adapter is not yet a faithful native-task reproduction

Upstream TUEV source confirms the 16-bipolar montage, 200 Hz, five-second
context, /100 scaling and sorted-subject 80/20 split conventions. Shared
conventions alone do not establish identical arrays: filtering implementation
and order, event indexing, irregular durations and exclusions require checking.

A separate concrete model discrepancy was found: the historical adapter uses
a hidden width of 800, whereas upstream TUEV all_patch_reps uses 1000, adding
3,240,200 parameters. That corrected head now passes full-state strict loading
and bit-identical synthetic-forward comparison against upstream on an allocated
CPU. The distinctly named 800/1000-wide seed-1 confirmations both completed
all 50 epochs. The 800-wide run exactly reproduced the historical 50-epoch
validation-kappa curve. The 1000-wide run changes selected validation kappa
from .443026 to .450285, but balanced accuracy declines from .464050 to
.408304. This head correction does not establish a large performance gain
or certify full numeric preprocessing parity. Preserve historical results as
adapter measurements, not native-task/SOTA reproduction proof. See
`results/audits/cbramod-native-head-inventory-20260913.json`,
`results/audits/cbramod-tuev-head-parity-20260913.json`, and
`results/audits/cbramod-head-completion-20260913.json`.
