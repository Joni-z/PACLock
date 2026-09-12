# Candidate review — 2026-09-13

## Decision

Treat factorized **f3 as the provisional primary candidate**, with **f2 as the alternate**. This is a prioritization decision, not a final architecture freeze or a claim of improvement over the historical best model. Keep f0/f1 as matched controls through the already-running pilot; do not discard a design scientifically from seed 0 alone. Do not add architecture sweeps or full-scale pretraining on the basis of this snapshot.

The candidate decision is based on validation logs only. Historical test scores motivate the research question but must not be used to swap a different winner into each dataset's main-paper row.

## Comparable early evidence

All four arms use seed 0. Values below are the best validation score among checkpoints shared by all four arms, not final test scores. Step checkpoints must be parsed as well as epoch-end checkpoints.

| Corpus | Shared progress | f0 | f1 | f2 | f3 |
|---|---|---:|---:|---:|---:|
| TUEV, kappa | epochs 0–5 | .5920 | .6294 | .6006 | .6474 |
| CHB-MIT, AUC-PR | epoch 0, steps 200–4000 | .0204 | .0571 | .0792 | .1807 |
| TUSZ, AUC-PR | epoch 0, steps 200–4000 | .2632 | .2911 | .2819 | .2822 |
| Sleep-EDF, kappa | epochs 0–5 | .6403 | .6416 | .6729 | .6629 |
| ISRUC, kappa | epochs 0–1 | .7018 | .6855 | .7359 | .7019 |
| TUAR, kappa | epochs 0–15 | .6049 | .6014 | .6231 | .6239 |

f3 = per-band local waveform coordinates plus lower effective sinc cutoff initialization; f2 = broadband local waveform coordinates. f0 allocates the same coordinates but zeros the local signal; f1 provides per-band waveform without the f3 cutoff change. Thus f3 versus f1 tests the initialization change, while f1/f2 versus f0 examine the local information path under the stated implementation.

TUEV has a concerning late decline: f3's epoch-5 kappa is .4961 versus its best .6474. All arms have fallen from earlier maxima. The early peak is insufficient evidence of stability, generalization, or a new best model. CHB-MIT remains in its first epoch and balanced accuracy is .5 at the latest checkpoint in all arms; ranking metrics have started to separate but the task is not solved. TUSZ has no clear winning arm. f2's advantage appears on both sleep corpora, making it a useful alternative for broad coverage.

## Freeze gate

1. Finish the existing pilot curves and inspect failures, budget stops, class behavior and validation stability. Compare equal progress; do not label truncated runs as converged.
2. If f3 remains promising on TUEV and does not collapse across the other diagnostic corpora, replicate the shortlist with additional seeds before a final choice. Keep the same candidate across datasets. f2 is the alternative if its cross-corpus behavior is more consistent.
3. Do not combine incompatible metrics into an arbitrary average. Use TUEV as the primary task while examining regression on seizure and sleep tasks separately. Candidate selection must use validation results, not repeated best-test selection.
4. Freeze architecture/config/source revision before expanding to the full >=10-dataset evaluation and before substantial new pretraining. Any pretraining pilot must then use the frozen architecture, correct sampling rates and a small explicit budget.

The historical coupling-only versus duplex tradeoff remains unresolved. A strong f3 early validation peak alone does not resolve it.
