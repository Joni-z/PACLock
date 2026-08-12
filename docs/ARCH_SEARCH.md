# PACLock architecture search — what the waves established

All numbers are TEST, on the GEMM tokeniser path (docs/PERF.md). Nothing from
`runs_conv_tokenizer/` is comparable and none of it is reused.

Seed spread is measured per dataset, not assumed. It differs by 11x:

| dataset | control sd (3 seeds) | a delta must clear |
|---|---|---|
| ISRUC | 0.0021 | 0.0041 |
| TUEV | 0.0235 | 0.0470 |

The difference is checkpoint selection, not the model: 57% of TUEV cells peak at
their first evaluation, 0% of ISRUC cells do.

## Finding 1 — the gain is the PAC estimation window, not the token count

`patch_len` moves two things at once: how many tokens the grid has, and how long
a window the PAC statistic is estimated over. `pac_patch_len` separates them.

TUEV, 3 seeds:

| arm | patch_len | PAC window | tokens P | test kappa | vs base |
|---|---|---|---|---|---|
| `base` | 200 | 200 (1.0 s) | 5 | 0.6266 ±0.0235 | — |
| `pacwin200` | 100 | 200 (1.0 s) | 10 | 0.6332 ±0.0296 | **+0.0066** |
| `patch100` | 100 | 100 (0.5 s) | 10 | 0.6836 ±0.0211 | **+0.0571** |

`pacwin200` and `patch100` have the **same token count**. The only difference is
the PAC window. Doubling the tokens at a fixed window buys +0.0066 — nothing.
Halving the window at a fixed token count buys +0.0504.

**The whole effect is the PAC estimation window.** An earlier read of the
single-seed conv-path sweep concluded the opposite ("the gain is temporal
resolution, not the PAC window, which is bad for the paper's narrative"). That
was wrong: it compared arms that moved both factors together, and it had one
seed against an assumed noise floor.

## Finding 2 — the sign flips with the timescale of the phenomenon

| dataset | window | phenomenon | shorter PAC window |
|---|---|---|---|
| TUEV | 5 s | epileptiform transients (SPSW/GPED/PLED) | **+0.0571** |
| ISRUC | 30 s | sleep stages, stationary | **−0.0055** |

Both significant against their own measured spread. `patch_len=100` therefore
fails the pre-registered rule (same sign on both datasets) and is **not** a
general improvement — it is the right PAC window for TUEV and the wrong one for
ISRUC.

## Pre-registered predictions for wave 5

Recorded before the runs finish. Wave 5 sweeps the PAC window at a **fixed**
token count, in both directions:

* TUEV at `patch_len=50` (P=20): windows 50 / 100 / 200 (0.25 / 0.5 / 1.0 s).
  **Predicted: monotone decreasing in window length.**
* ISRUC at `patch_len=200` (P=30): windows 200 / 400 / 600 / 1200 (1 / 2 / 3 / 6 s).
  **Predicted: monotone increasing in window length**, i.e. ISRUC prefers longer
  PAC estimation than it currently gets.

If TUEV is not monotone decreasing, or ISRUC not monotone increasing, the
timescale-matching story is dead and the effect is something else.

## Finding 3 — capacity, five negative tests and one positive

| test | budget | result |
|---|---|---|
| upstream §13.46 (1.64M -> 8.6M) | 20 ep | loss |
| ISRUC `size_base` / `size_large` | 20 ep | -0.036 / +0.031 (signs disagree) |
| TUEV `size_base` / `size_large` | 20 ep | +0.028 / -0.025 (signs disagree) |
| TUEV `e60_size_base` / `e60_size_large` | 60 ep | +0.002 / +0.041 (below 0.047) |
| ISRUC `e60_size_base` | 60 ep | -0.0175 |
| **ISRUC `e60_size_large`** | **60 ep** | **+0.0181 over `e60_base`, 8.6 sd** |

The one positive is ISRUC at 60 epochs, where 8.5M beats 1.6M by 0.0181. Every
earlier negative ran at 20 epochs. `p100_large` (wave 4) tests whether capacity
also pays once the PAC window is right, which is the combination none of the
five negatives ever tried.

## Finding 4 — denser checkpoint selection does not fix TUEV

57% of TUEV cells peak at their first evaluation, so evaluating 10x more often
(`eval_every_steps: 200`) should have found a better peak. It did not:

| arm | test kappa |
|---|---|
| `base` (1 eval/epoch) | 0.6266 ±0.0235 |
| `dense` (10 evals/epoch) | 0.5819 |
| `dense_lr3e5` | 0.6173 |
| `dense_lr1e5` | 0.5947 |
| `dense_patch100` | 0.6886 |

Denser sampling makes it worse, and lowering the LR does not help either. The
early peak is real, not an artefact of sparse sampling — TUEV's supervised
signal is exhausted in one epoch. `dense_patch100` reproducing `patch100`
(0.6886 vs 0.6836) is the useful part: the PAC-window effect survives a
completely different selection schedule.

For a model headed into **pretraining** this is the encouraging reading: the
supervised task saturates almost immediately, which is exactly the regime where
a pretrained initialisation should pay.

## Wave 5 outcome — half the pre-registered prediction held, half did not

TUEV, `patch_len=50` so the token count is 20 throughout and only the PAC
window moves, 3 seeds each:

| PAC window | test kappa |
|---|---|
| 50 (0.25 s) | 0.7076 ±0.0311 |
| 100 (0.50 s) | 0.6489 ±0.0345 |
| 200 (1.00 s) | 0.6105 ±0.0458 |

**Monotone decreasing, as predicted.** The span is 0.097 kappa, the largest
single-factor effect measured anywhere in this search, and the seed spread
widens with the window as well.

ISRUC, `patch_len=200` so the token count is 30 throughout:

| PAC window | test kappa | seeds |
|---|---|---|
| 200 (1 s) | 0.6909 ±0.0021 | 3 |
| 400 (2 s) | 0.6966 | 1 |
| 600 (3 s) | 0.6819 | 1 |
| 1200 (6 s) | 0.6945 | 1 |

**Not monotone increasing. The prediction failed.** The response is flat: the
whole 1-6 s range spans 0.0147 against TUEV's 0.097, with no systematic
ordering.

The error was extrapolating a trend from a single point. `patch100` (0.5 s) is
worse than `base` (1.0 s) by 0.0055 on ISRUC, and that one comparison was read
as "ISRUC prefers longer windows". Two points define a direction only when
there is already a reason to expect monotonicity, and there was not.

### Revised hypothesis — post-hoc, and being tested rather than asserted

Performance collapses when the PAC window is comparable to or longer than the
phenomenon, and is flat while the window stays comfortably shorter:

| | phenomenon | 1 s window vs phenomenon | observed response |
|---|---|---|---|
| TUEV | epileptiform transients, 0.1–0.5 s | **longer than the event** | steep, monotone |
| ISRUC | sleep stages, 30 s | far shorter than the stage | flat |

This accounts for both sides including the flatness, but it was written after
seeing the data, so it only counts once it survives its own test. It predicts
that pushing ISRUC's window up to the scale of a sleep epoch — 15 s and 30 s —
degrades performance. Those runs (`isruc-cand_p200_win3000`,
`isruc-cand_p200_win6000`, 3 seeds each) are that test. If they do not degrade,
the timescale account is wrong and TUEV's effect is something else.

### What is safe to carry into the deliverable either way

The TUEV ladder is 3 seeds at every point and monotone, so "the PAC estimation
window is the dominant architectural knob, and too long is expensive" stands on
its own. What is *not* established is a rule for choosing it from the corpus, so
the deliverable cannot ship a single fixed `pac_patch_len` and call it tuned.
That is the argument for the multi-scale frontend, independent of how the ISRUC
test resolves.

## Wave 7 outcome — the mechanism does not survive; the measurement does

The post-hoc claim was that performance collapses once the PAC window
approaches the duration of the phenomenon. ISRUC epochs are 30 s, so 15 s and
30 s windows are the test. 3 seeds each, token count fixed at 30:

| PAC window | test kappa | vs the 1.0 s control | Welch t |
|---|---|---|---|
| 1 s | 0.6909 ±0.0021 | — | — |
| 15 s | 0.6860 ±0.0129 | −0.0049 | −0.65 |
| 30 s | 0.6826 ±0.0097 | −0.0083 | −1.46 |

**Neither is significant.** The direction is right and the size is negligible.

Both versions of the timescale account have now failed:

* pre-registered: ISRUC should improve monotonically with window length,
  because sustained coupling makes SNR ∝ √L. It was flat over 1–6 s.
* post-hoc: ISRUC should collapse once the window reaches the 30 s stage
  duration. It does not, at 30 s or at 15 s.

So the SNR ∝ τ/√L dilution derivation explains the transient half (TUEV, where
the events are 0.1–0.5 s and the window ladder is monotone over 0.097 kappa) and
does **not** explain the sustained half. The paper should report this as a
measurement and name the sustained case as an open question, rather than fit a
third story to it.

### What did replicate: window length drives seed variance

Not the mean on ISRUC, but the spread, on both corpora and every setting tested:

| TUEV (20 tokens) | sd | ISRUC (30 tokens) | sd |
|---|---|---|---|
| 0.25 s | 0.0311 | 1 s | **0.0021** |
| 0.50 s | 0.0345 | 15 s | 0.0129 |
| 1.00 s | **0.0458** | 30 s | 0.0097 |

Six times on ISRUC, 1.5x on TUEV, monotone on TUEV. This was not predicted and
is not explained either, but it is consistent across two corpora with different
phenomena, different channel counts and different window lengths.

### Consequence for the deliverable: none, and the case is stronger

`pac_patch_len` short now rests on three independent legs instead of two:

1. large gain where the phenomenon is transient (TUEV, +0.097, monotone, 3 seeds);
2. no cost where it is not (ISRUC, nothing significant anywhere from 1 to 30 s);
3. lowest run-to-run variance on **both** corpora.

Short is the only setting that does not lose on any measured axis. The shipped
config is unchanged.

## Wave 7 outcome — the mechanism does not survive; the measurement does

The post-hoc claim was that performance collapses once the PAC window
approaches the duration of the phenomenon. ISRUC epochs are 30 s, so 15 s and
30 s windows are the test. 3 seeds each, token count fixed at 30:

| PAC window | test kappa | vs the 1.0 s control | Welch t |
|---|---|---|---|
| 1 s | 0.6909 ±0.0021 | — | — |
| 15 s | 0.6860 ±0.0129 | −0.0049 | −0.65 |
| 30 s | 0.6826 ±0.0097 | −0.0083 | −1.46 |

**Neither is significant.** The direction is right and the size is negligible.

Both versions of the timescale account have now failed:

* pre-registered: ISRUC should improve monotonically with window length,
  because sustained coupling makes SNR ∝ √L. It was flat over 1–6 s.
* post-hoc: ISRUC should collapse once the window reaches the 30 s stage
  duration. It does not, at 30 s or at 15 s.

So the SNR ∝ τ/√L dilution derivation explains the transient half (TUEV, where
the events are 0.1–0.5 s and the window ladder is monotone over 0.097 kappa) and
does **not** explain the sustained half. The paper should report this as a
measurement and name the sustained case as an open question, rather than fit a
third story to it.

### What did replicate: window length drives seed variance

Not the mean on ISRUC, but the spread, on both corpora and every setting tested:

| TUEV (20 tokens) | sd | ISRUC (30 tokens) | sd |
|---|---|---|---|
| 0.25 s | 0.0311 | 1 s | **0.0021** |
| 0.50 s | 0.0345 | 15 s | 0.0129 |
| 1.00 s | **0.0458** | 30 s | 0.0097 |

Six times on ISRUC, 1.5x on TUEV, monotone on TUEV. This was not predicted and
is not explained either, but it is consistent across two corpora with different
phenomena, different channel counts and different window lengths.

### Consequence for the deliverable: none, and the case is stronger

`pac_patch_len` short now rests on three independent legs instead of two:

1. large gain where the phenomenon is transient (TUEV, +0.097, monotone, 3 seeds);
2. no cost where it is not (ISRUC, nothing significant anywhere from 1 to 30 s);
3. lowest run-to-run variance on **both** corpora.

Short is the only setting that does not lose on any measured axis. The shipped
config is unchanged.

## Collapse investigation — FACED / PhysioNet-MI, patch_len=50 deliverable config

Both corpora have the highest channel counts in the matrix (FACED 32, PhysioNet-MI
64, vs 16/6/2/22 everywhere else). Under `patch_len=50` this multiplies to the
largest token grids in the sweep (FACED: 32 x 8 bands x 40 patches = 10240
tokens/sample). 2 of 3 seeds fail on each corpus; this is not one failure mode.

### Failure mode A — never leaves the initialization plateau

FACED, both the failing seeds and (initially) the succeeding one:

```
seed1 (fails):  epoch 0  train_loss 2.2148  val balanced_acc 0.1111
                epoch 20 train_loss 2.1967  val balanced_acc 0.1111   (unchanged)
seed0 (works):  epoch 0  train_loss 2.2171  val balanced_acc 0.1111
                epoch 9  train_loss 2.1987  val balanced_acc 0.1111   (still stuck)
                ...eventually escapes; final result at epoch 77
```

`ln(9) = 2.197` — FACED is 9-class, so this is exactly the uniform-guess
cross-entropy. Loss is not decreasing, it is drifting by rounding error around
the chance floor. Even the seed that eventually works spends its first ~9+
epochs indistinguishable from the seeds that never escape — escaping this
plateau is slow and seed-dependent for every run on this corpus, not something
2 of 3 seeds uniquely suffer from.

### Failure mode B — trains, but does not generalize at all

PhysioNet-MI seed1: train loss drops normally (1.4012 at epoch 0 -> 0.9280 by
epoch 62 — real, continuous descent) while validation is dead flat the entire
run:

```
epoch  0  val balanced_acc=0.2500  (chance is 0.25, 4-class)
epoch 24  val balanced_acc=0.2494
```

`weighted_f1` moves within noise (0.099-0.165) but kappa never clears ~0.004.
This is not the same mechanism as A — the optimizer found *some* direction
that reduces train CE, but it carries zero signal to held-out data. With 1.6M
params and O(10^4) tokens per sample, the model has enough capacity to fit
idiosyncrasies of individual training windows without learning anything about
the class boundary.

Both are downstream of the same design fact, not two independent bugs: at
these channel counts, `patch_len=50` produces token grids ~2x larger than
anywhere else validated, and the recipe (lr=1e-4, cosine, patience=20 evals)
was never re-tuned for that regime.

### Pre-registered test (before results land)

2x2 per corpus, single seed each, packed on one node (`configs_packed.slurm`):

|                  | lr=1e-4 (current) | lr=3e-5 |
|---|---|---|
| patch_len=50 (current) | existing collapsed runs | tests LR alone |
| patch_len=200 (pre-search default) | tests token count alone | tests both |

Predictions:
* if patch_len=200 alone fixes it (regardless of lr) -> token count is the
  driver, and patch_len needs to be corpus-conditional on channel count, not a
  single global deliverable value.
* if lr=3e-5 alone fixes it (regardless of patch_len) -> the optimizer step
  size is the driver, independent of resolution; a smaller lr (or a warmup,
  not yet tested) should generalize as the fix across all high-channel corpora.
* if only the combination fixes it -> the two interact, matching the general
  lesson this project already learned once for a different reason
  (docs/PERF.md's 2x2) -- neither single-factor probe would have found it.
* if nothing in the 2x2 fixes it -> the mechanism is something else entirely
  (init scale, a channel-count-dependent numerical issue in the frontend, or
  the classifier head's flatten-then-MLP blowing up parameter count at 32/64
  channels) and needs a different diagnostic, not a recipe tweak.

Configs: `configs/_diag/{faced,physionet_mi}_{patch200,lr3e5,patch200_lr3e5}.yaml`.
Jobs: `D1_faced` (370764), `D1_pmi` (370765) -- queued, mi2104x fully allocated
(21/21) across all users sharing the partition; no alternate partition has a
working PyTorch install (checked mi2508x, mi3258x, mi3008x, devel -- module
loads without error on all four but `import torch` fails with
`ModuleNotFoundError` on every one of them; only mi2104x has the package
actually installed).

### Mechanical diagnostic run in parallel (no training needed)

`scripts/diag_gradients.py` -- one forward+backward at init, comparing
gradient-norm-by-module and logit statistics between `patch_len=50` and
`patch_len=200` on FACED/PhysioNet-MI vs two healthy controls (TUEV, ISRUC).
Distinguishes "vanishing/exploding gradient at the frontend, mechanically,
right at step 0" from "the training dynamics only go wrong over many steps" --
the former would be visible in a single backward pass and would not need
waiting for the queued training jobs to say anything.

### Result — the fix is patch_len, not learning rate

Single seed, both corpora, mid-run reads (jobs still progressing, not yet
converged, but the qualitative question -- does the run escape the ln(K)
plateau at all -- is already answered):

| variant | FACED kappa | PhysioNet-MI kappa |
|---|---|---|
| `patch50` (current deliverable) | 0.0000 (dead, both collapsed seeds) | 0.0000 / at-chance-on-test (both collapsed seeds) |
| `lr3e5` (patch=50, lr lowered) | 0.0000 through epoch 10 | 0.0000 through epoch 14 |
| **`patch200`** (lr unchanged) | **0.0430 by epoch 33, still rising** | **0.0357-0.0449, noisier but off zero** |
| `patch200_lr3e5` | **ran to completion, early-stopped dead at 0.0000** | oscillating near zero, not clearly better than patch200 alone |

`patch200_lr3e5` finishing at exactly chance is the decisive negative result:
adding the lower learning rate on top of the resolution fix did not help, and
lr3e5 alone never escapes on either corpus through the epochs observed. Lower
learning rate is not the mechanism, and if anything a smaller step size makes
escaping a flat plateau *harder*, not easier -- consistent with the pattern
observed (patch200 alone escapes faster than patch200_lr3e5).

**patch_len=50 is not safe as a single global deliverable value.** At FACED's
32 channels and PhysioNet-MI's 64, it produces token grids (~10240 and ~6144)
roughly 2x larger than anywhere else in the matrix, and 2 of 3 seeds do not
survive optimization at that resolution regardless of learning rate. The
mechanical check (`scripts/diag_gradients.py`) had already ruled out a dead
gradient at initialization -- healthy, non-degenerate gradients reach the
frontend on every corpus and every patch_len tested, including the failing
ones. So the problem is not the model being broken; it is that patch_len=50's
token count makes the *loss landscape* someone has to search 20 epochs of
"stuck at chance" through, on some seeds, before finding the exit -- and lr3e5
does not shrink that search, it lengthens it.

**Recommendation:** patch_len should be conditioned on channel count for the
deliverable, not fixed at 50 everywhere. FACED and PhysioNet-MI should ship at
patch_len=200 (their pre-search default and the one config that reliably
escapes chance); the corpora patch_len=50 was validated on (TUEV, ISRUC,
Sleep-EDF, BCI-IV-2a, TUSZ -- all <=22 channels) keep it.

Still open, not yet run: 3-seed confirmation at patch_len=200 for these two
corpora, and a check of what channel count the transition actually sits at
(only two data points -- 32 and 64 -- separate "works" from "collapses"; the
16-channel corpora all work fine at patch_len=50, so the boundary is somewhere
between 16 and 32, untested).
## Ablation A — CBraMod's architecture, PACLock's tokenizer (docs/ARCH_SEARCH.md companion)

Tests the claim TFM-Tokenizer makes about itself -- that tokenization carries
the result, not model size or architecture -- against PACLock's own frontend
instead of TFM's, using a single-variable swap: CBraMod's official encoder,
positional encoding and classifier head (vendor code, byte-identical) with only
its `PatchEmbedding` (the raw-200-samples-to-200-dim-vector step) replaced by
PACLock's TriAxialFrontend at CBraMod's own resolution (`patch_len=200`,
matching CBraMod's native patch, so this does not also test the separate
patch-length finding above). See
`paclock_bench/models/foundation/cbramod_paclockfe_adapter.py` for exactly what
is and is not swapped, and `scripts/verify_cbramod_paclockfe.py` for the
shape/gradient checks run before any training compute was spent (backbone
param count 4.884M -> 4.898M, a 14K difference, not a capacity confound).

### Result -- TUEV, single seed

| | backbone params | val peak | **test kappa** |
|---|---|---|---|
| CBraMod (native tokenizer, scratch, 3-seed) | 4.884M | -- | 0.5638 ±0.0193 |
| **CBraMod + PACLock tokenizer (1 seed)** | 4.898M | 0.4358 (epoch 21) | **0.6280** |

+0.064 over the native-tokenizer baseline, on one seed -- roughly 3x TUEV's
own 3-seed spread for this architecture. The val curve is noisy throughout
training (0.28-0.44, no clean trend) and test comes in *above* the best val
checkpoint, which looks like an anomaly but matches a pattern already on
record for TUEV specifically: it is the corpus where checkpoint selection is
least reliable in this whole benchmark (docs/ARCH_SEARCH.md notes 57% of TUEV
cells peak at their very first evaluation). A noisy val signal that
underestimates test quality is the established failure mode here, not a new
one.

**Reading it:** a single seed is not proof, but it clears the bar the
project's own rule sets (a delta has to clear roughly the corpus's own seed
spread to be worth reading) by a wide margin, and it directly answers the
question this ablation was built to ask: PACLock's tokenizer, dropped under a
foreign, fixed architecture with the swap verified to change nothing but 14K
parameters, beats that architecture's own tokenizer. That is exactly the
TFM-Tokenizer-style claim, made about our tokenizer instead of theirs, with
the same discipline (single-variable swap, params checked, verified before
spending training compute) the rest of this benchmark holds itself to.

**Not yet done:** 3-seed confirmation (this was explicitly out of scope for
the pilot per the current goal); BIOT as the second architecture (the other
half of the originally proposed pair); the frozen-vs-native-preprocessing
question for BIOT specifically, since BIOT's own preprocessing differs from
ours and CBraMod's does not (see the original experiment-design discussion).
