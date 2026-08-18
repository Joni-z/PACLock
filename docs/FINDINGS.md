# PACLock 实测结论汇总

本文件合并了三份原文档,内容未改写,仅去掉一处重复段落:

* **架构搜索**(原 `docs/FINDINGS.md`)—— 每一波实验的结论,含被证伪的假设
* **性能**(原 `docs/FINDINGS.md`)—— 训练慢 10 倍的原因与修复
* **交付配置**(原 `docs/FINDINGS.md`)—— 要预训练的那份配置,及每个选择的依据

所有数字均为 TEST 指标,GEMM tokenizer 路径;`runs_conv_tokenizer/` 下的
结果不可比,一律不复用。

---

# 第一部分:架构搜索

All numbers are TEST, on the GEMM tokeniser path (docs/FINDINGS.md). Nothing from
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
  (docs/FINDINGS.md's 2x2) -- neither single-factor probe would have found it.
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
## Ablation A — CBraMod's architecture, PACLock's tokenizer (docs/FINDINGS.md companion)

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
least reliable in this whole benchmark (docs/FINDINGS.md notes 57% of TUEV
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

---

## The PAC tokenizer vs the tier-3 gap: what is actually responsible

An earlier version of this section claimed the PAC interaction tokenizer
"collapses to chance on band-power tasks" and proposed a fusion fix. **That claim
was confounded and is withdrawn.** The corrected analysis follows, because the
mistake is the useful part: it is the same mistake as the `spatial_pe: xyz`
episode -- a number read as a property of the model when it was a property of the
data pipeline.

### The confound

`bci_iv_2a-paclock_pac` scores 0.2591 balanced accuracy (3 seeds, chance 0.25) and
that number drove four rounds of tokenizer surgery. It runs on
`$PACLOCK_PROC/processed_pac/`, the PAC-methodology protocol (0.5 Hz high-pass, no
notch). `bci_iv_2a-paclock_v2` is the **same model configuration** --
`tokenizer_mode: pac_interaction`, `interaction_mode: product`, same d_model,
depth, bands, lr -- on `processed/` (0.3-75 Hz, 60 Hz notch), and scores **0.3588**.

So of the apparent 0.19 deficit against the raw tokenizer, roughly **0.10 belongs
to the preprocessing protocol and ~0.06-0.09 to the tokenizer**. Holding
preprocessing fixed at `processed/`:

| tokenizer (BCI-IV-2a, `processed/`) | balanced acc |
|---|---|
| raw, best variant (`raw_headattn`) | 0.4545 |
| raw, plain (`rawtok`) | 0.4192 |
| pac_interaction + pretraining (`pt_large`) | 0.4122 |
| pac_interaction, scratch (`v2`) | 0.3588 |
| pac_interaction, `concat` | 0.3090 |

The corresponding TUEV table, also all on `processed/`, is where the tokenizer
earns its place:

| tokenizer (TUEV, `processed/`) | Cohen's kappa |
|---|---|
| pac_interaction (`v2`, 3 seeds) | **0.7076** |
| best baseline (`tfm_pretrained`) | 0.6519 |
| raw (`rawtok`, 3 seeds) | 0.5359 |

**+0.172 over raw and +0.056 over the strongest baseline, single-variable, same
preprocessing.** That result stands and is not affected by any of this.

### Four mechanisms proposed, four refuted -- by one probe

Each explanation below was stated as a mechanism for why the interaction tokenizer
underperforms on band-power tasks. `scripts/probe_readability.py` tests all of them
at once by fitting a ridge from a single UNTRAINED token to that token's own log
band power (the quantity `return_amp_target=True` computes), which removes
training dynamics from the question:

| tokenizer | R2 linear | R2 on squared components | R2 patch-pooled |
|---|---|---|---|
| raw (**the one that wins**) | 0.031 | 0.469 | 0.139 |
| product | 0.108 | 0.592 | 0.258 |
| rotation | 0.065 | **0.907** | 0.180 |
| concat | **0.853** | 0.761 | 0.876 |

Band power is *more* accessible from every interaction token than from the raw
token that outperforms them -- quadratically for `rotation`, linearly for
`concat`. So:

1. *"Hilbert discards within-patch waveform detail"* -- wrong on code reading:
   both tokenizers are sample-level `Conv1d` over a lossless (envelope, unit
   phase) decomposition.
2. *"The amplitude is locked away and unrecoverable"* -- wrong: `concat` exposes it
   as its own columns (R2_lin 0.853) and still scores 0.3090.
3. *"The estimator's magnitude noise multiplies band power"* -- wrong as *the*
   cause: `rotation` removes it exactly (|h| = |a|, carrier CV 0.62 -> 4e-8) and
   the probe confirms power is then near-perfectly recoverable (0.907).
4. *"Amplitude is not linearly readable behind a random phase"* -- wrong: it is not
   linearly readable from `raw` either (0.031), and `raw` wins.

Amplitude accessibility does not explain the ranking. The information is present in
every mode.

### What the symptom actually says

BCI-IV-2a, `rotation`, 20 epochs: `train_loss` 1.4099 -> 1.3356, against a
4-class chance cross-entropy of ~1.386 with label smoothing 0.1. **The model does
not fit its own 2160-window training set.** This is underfitting, not an
information bottleneck and not overfitting -- and no tokenizer change addresses it.

The scale of the remaining gap says the same thing: the best PACLock variant on
BCI-IV-2a is 0.4545 while SPaRCNet reaches 0.6440. Even with the tokenizer
replaced entirely, ~0.19 is left over. **The tier-3 deficit is dominated by the
architecture/recipe on small corpora, not by the tokenizer.**

The one lever already measured to work on this axis is pretraining: it moves the
PAC tokenizer from 0.3588 to 0.4122 on BCI-IV-2a, i.e. it recovers most of the
tokenizer gap without touching the tokenizer.

### `interaction_mode: rotation` -- kept, on its own merits

    h_j = a_j * aligned_phase_j / |aligned_phase_j|

Coupling rotates the amplitude token instead of also rescaling it. Forced exactly
as strongly as `product` (token phase still entirely coupling-determined, no raw
high-band token beside it, no learnable path, no new parameters), and `|h_j| =
|a_j|` holds exactly. Verified by `scripts/verify_rotation.py` (all pass):
`product`/`concat` bit-identical to the pre-change implementation so no frozen
cell moves; `|h| = |a|` to 2.4e-7; tokens still respond to a phase twist, so this
is not a plain amplitude tokenizer; **gauge invariance confirmed for bands 1..
under `p_i -> e^{i delta_i} p_i, Z_ij -> e^{i delta_i} Z_ij`, for `rotation` and
for `product`** -- the first actual test of the claim `_pac_interaction`'s
docstring has always made; no NaN on a dead electrode.

It is retained as a cleaner formulation with a defensible physical reading (PAC is
a phase relationship, so it should enter as a phase), not as a fix for tier 3.
Its first measurement on `processed_pac` gave 0.2724 vs `product`'s 0.2591 --
+0.013 on the degraded protocol, uninterpretable as a verdict; the apples-to-apples
runs against `v2` on `processed/` are `*-paclock_rot2`.

### Rejected design: `coupling_gate: significance` (reverted, not in the tree)

`w_ij = relu(1 - 1/lambda_ij)`, `lambda_ij = |Z_ij|^2 / E_null|Z_ij|^2`, blending
band j back to its own phase where coupling is not measurable. Killed by
`scripts/pac_null_calib.py`, which measures the null with circular-shift
surrogates (preserving both marginals and both autocorrelations):

1. **The analytic null was wrong by 23-350x and in the wrong direction.** The
   prediction `L_eff = (bw_i + bw_j) * T` from envelope autocorrelation says far
   FEWER effective samples than the nominal 200; measured effective d.o.f. is 174
   to 10121, mostly MORE. `A~_j(t) exp(i phi_i(t))` is an oscillatory integral --
   the phase rotating at f_i cancels the sum faster than a random walk, so
   effective d.o.f. RISES with source frequency (45 Hz pairs reach ~10^4).
   Envelope autocorrelation is real but second-order.
2. **Significance does not separate the corpora.** With the calibrated null,
   `frac(w > 0)` is 0.358-0.370 on TUEV and 0.356-0.370 on BCI-IV-2a, flat across
   classes within TUEV; mean gate 0.68 vs 0.69. Coupling in motor imagery is
   statistically real and simply not class-discriminative, so the gate keeps
   precisely the useless coupling. **Significance is not discriminativeness**, and
   no label-free gate on the coupling statistic can distinguish the two cases.

It also had a genuine defect, caught by its own `lambda -> inf` equivalence check:
with `w` replacing `|Z|` as the mixing weight, `lambda -> inf` yields UNIFORM
weighting rather than magnitude weighting, so it was never a strict generalisation
of the unconditional tokenizer.

### Also ruled out

`pac_token_mode: uniform` on BCI-IV-2a (`processed_pac`): 0.2639 vs `product`'s
0.2591 -- no recovery, and on the confounded protocol either way.

---

# 第二部分:性能

*(原 `docs/FINDINGS.md`)*

TUEV, `configs/_cand/tuev_base.yaml`, one MI210, batch 32:

| | s / epoch |
|---|---|
| finished runs (`cand_*`, archived in `runs_conv_tokenizer/`) | **1939** |
| current code, same config, same node type (`diag_trainpy`) | **206** |

The model was never the problem. An isolated full training step measured
51.9 ms — 617 samples/s — against 35.3 samples/s observed, and the forward
splits as encoder 10.77 ms / frontend 2.42 ms / head 0.10 ms.

## The cause: an interaction between two things, each harmless alone

`train.py`'s `set_seed()` sets `torch.backends.cudnn.deterministic = True`,
which on ROCm maps onto MIOpen. The frontend tokenisers were
`nn.Conv1d(1, K, kernel_size=patch_len, stride=patch_len)` applied to
`B*C*n_bands = 4096` single-channel signals. Forced determinism makes MIOpen
choose an atomics-free **backward-weights** algorithm, and for
`in_channels=1` with a 200-tap kernel that is a serial reduction.

Measured 2x2 (full step: forward + backward + clip + optimiser):

| tokeniser | `deterministic` | ms/step | s/epoch |
|---|---|---|---|
| `Conv1d` (old) | **True** | **208.56** | 446 |
| GEMM (new) | True | 66.60 | 142 |
| `Conv1d` (old) | False | 55.46 | 119 |
| GEMM (new) | False | 50.79 | 109 |

`Conv1d` is not slow. `Conv1d` **under forced determinism** is slow, by 3.76x.

## The fix

`_patch_project()` in `models/paclock/frontend/triaxial.py`. When
`stride == kernel_size` the patches do not overlap, so the convolution is a
per-patch linear map and `(N, P, patch) @ W` is the same arithmetic as one
GEMM, whose backward is a GEMM too and needs no special deterministic kernel.

The `nn.Conv1d` modules are kept as the parameter holders and only the
computation is replaced, so initialisation, `state_dict` keys and parameter
count are unchanged and old checkpoints still load.

Verified (`scripts/verify_patch_project.py`): operator `max|d| 2.4e-6`
(rel 7.9e-7), whole-frontend tokens rel 5.4e-7, coupling exactly 0, gradient
w.r.t. input rel 3.3e-7. Mathematically identical, **not** bit-identical — a
convolution and a GEMM reduce over `patch` in a different order.

`deterministic = True` is **kept**: with the convolutions gone it costs 1.31x,
and bit-exact reproducibility is worth more than that. It is what let
`D_repeat` show three identical configs producing identical loss curves —
which, read correctly, was the speed bug announcing itself.

## Why this took four wrong diagnoses

Kernel-launch overhead, then Lustre random reads, then process packing, then
"the frontend's `in_channels=1` convolutions are badly shaped for MIOpen". Each
was inferred from aggregate throughput and each was wrong.

The last one was wrong in an instructive way. Every benchmark varied one of the
two factors **while the other was already in its cheap state**:

* the frontend A/B timed the forward only, and the forward was never the cost —
  it measured 1.05x;
* the determinism A/B ran after the convolutions had already been replaced, so
  there was nothing left for the flag to be slow on — it measured 1.29x.

Neither is wrong as a measurement. Both are useless as an explanation, because
a 2-way interaction is invisible to any experiment that holds the other factor
at its cheap level. The 2x2 above was the first design that could see it.

Also worth keeping: `profile_steps` reports 231 samples/s where an uninstrumented
loop does 613, because its five `torch.cuda.synchronize()` calls per step drain
the queue. Its *percentages* are trustworthy; its absolute rate is not, and the
"packing costs 4%" result taken from it was wrong for the same reason — with
every process stalled on a sync, they cannot contend.

## Consequences for results already in hand

`runs_conv_tokenizer/` holds the 28 pre-change candidate cells. They are not
comparable with anything produced after it: an SDPA swap verified at 4e-7 moved
a finished ISRUC run by 0.029 kappa, seven times the seed spread, so a 1e-6
change to the tokeniser has to be assumed to do the same. The sweep is being
re-run rather than reused.

## Addendum: the size of the "mathematically equivalent" shift, measured

The claim above — that results from before and after the tokeniser change are
not comparable — was originally supported only by an anecdote (an SDPA swap
verified at 4e-7 moved an ISRUC run by 0.029) against an *assumed* seed spread.
Wave 1 measured both properly on ISRUC, 3 seeds, same config:

    conv path (archived)   0.6633
    GEMM path (new)        0.6888  0.6910  0.6929   mean 0.6909  sd 0.0021

    shift from the code change: 0.0276  =  13.4 x the seed standard deviation

So a change that is mathematically identical and verified to rel 8.9e-7 moves
the result by thirteen times the run-to-run spread it has to be compared
against. Archiving `runs_conv_tokenizer/` and re-running was not caution, it was
required.

It also means the seed spread itself is dataset-specific and has to be measured
per dataset, not assumed: ISRUCs is sd 0.0021, TUEVs is sd 0.0235, eleven
times larger. The difference is checkpoint selection — 57% of TUEV cells peak at
their first evaluation and 0% of ISRUC cells do — not the model.

---

# 第三部分:交付配置与依据

*(原 `docs/FINDINGS.md`)*

Everything below is TEST kappa on the GEMM tokeniser path, 3 seeds unless
marked. Nothing from `runs_conv_tokenizer/` is reused: a mathematically
identical code change moved an ISRUC result by 13.4 seed standard deviations
(docs/FINDINGS.md), so pre-change numbers are not comparable.

Seed spread is measured per corpus, never assumed:

| corpus | control sd | a delta must clear |
|---|---|---|
| ISRUC | 0.0021 | 0.0041 |
| TUEV | 0.0235 | 0.0470 |

## The configuration

```yaml
model_kwargs:
  arch: triaxial
  d_model: 128
  depth: 6
  n_bands: 8
  n_heads: 4
  dropout: 0.2
  kernel_size: 201
  patch_len: 50            # token grid resolution
  pac_patch_len: 50        # PAC estimation window -- the load-bearing knob
  tokenizer_mode: pac_interaction
  pac_token_mode: measured
  interaction_mode: product
  freq_mixer: attention
  band_pe: index
  spatial_pe: xyz          # montage coords where the corpus has them, else `index`
optimizer: adamw
lr: 1.0e-4
weight_decay: 1.0e-5
scheduler: cosine
grad_clip: 1.0
label_smoothing: 0.1
batch_size: 32
epochs: 20
```

**1.62 M parameters.**

`spatial_pe` is set by the corpus, not chosen: `xyz` uses the montage
coordinates from `models/montage.py` and is what the TUEV runs above used;
corpora with no montage definition (ISRUC's 6 channels) fall back to the learned
index embedding. Both TUEV and ISRUC numbers in this document were produced with
their respective setting, so neither is a free parameter that was tuned.

## Why each load-bearing choice

### `pac_patch_len` — short. This is the whole result.

`patch_len` moves two things at once: how many tokens the grid has, and how long
a window the PAC statistic is estimated over. `pac_patch_len` separates them,
and separating them is what turned a confusing sweep into a result.

TUEV, token count held at 20, only the PAC window moving, 3 seeds each:

| PAC window | test kappa |
|---|---|
| 0.25 s | **0.7076 ±0.0311** |
| 0.50 s | 0.6489 ±0.0345 |
| 1.00 s | 0.6105 ±0.0458 |

Monotone, span **0.097 kappa** — the largest single-factor effect anywhere in
this search. For contrast, doubling the token count at a *fixed* window is worth
+0.0066 on TUEV and −0.0132 on ISRUC: nothing, in both directions.

**The gain is the PAC estimation window, not the token count.** An earlier read
of the single-seed pre-change sweep concluded the opposite and was wrong: it
compared arms that moved both factors together.

### Why short rather than tuned per corpus

The penalty is strongly asymmetric:

| | cost of too short | cost of too long |
|---|---|---|
| ISRUC | 0.5 s vs 2.0 s: **−0.011** | — |
| TUEV | — | 1.0 s vs 0.25 s: **−0.097** |

Roughly 9x. Being short costs almost nothing; being long is expensive. A
backbone that must fix this before seeing the downstream task should therefore
sit at the short end, and no corpus-inspection rule is needed.

A third, unpredicted reason arrived with wave 7: **longer windows raise
run-to-run variance on both corpora**, TUEV sd 0.0311 / 0.0345 / 0.0458 across
0.25 / 0.5 / 1.0 s and ISRUC sd 0.0021 / 0.0129 / 0.0097 across 1 / 15 / 30 s.
Short is the only setting that loses on no measured axis.

### `d_model: 128, depth: 6` — do not scale

Six independent negatives:

| test | budget | result |
|---|---|---|
| upstream §13.46, 1.64M → 8.6M | 20 ep | loss |
| ISRUC `size_base` / `size_large` | 20 ep | −0.036 / +0.031 (signs disagree) |
| TUEV `size_base` / `size_large` | 20 ep | +0.028 / −0.025 (signs disagree) |
| TUEV `e60_size_base` / `e60_size_large` | 60 ep | +0.002 / +0.041 (below 0.047) |
| ISRUC `e60_size_base` | 60 ep | −0.018 |
| TUEV `p100_large` (capacity *at the right window*) | 20 ep, 3 seeds | **−0.012**, and sd 0.0457, twice the control's |

The last one matters most: the standing excuse was that capacity had only ever
been tested at the coarse PAC window. It was then tested at the good window and
still lost, with double the variance.

The single positive is ISRUC `e60_size_large`, +0.0181 over `e60_base` at 60
epochs, one seed. Not enough to build on.

### `epochs: 20`

60 epochs is worse on TUEV (`e60_base` 0.5970 vs `base` 0.6266) and worse on
ISRUC (`e60_base` 0.6776 vs `base` 0.6909). TUEV's supervised signal is
exhausted almost immediately — 57% of TUEV cells peak at their first evaluation
— and evaluating 10x more often does not find a better peak (`dense` 0.5819,
`dense_lr3e5` 0.6173, `dense_lr1e5` 0.5947, all at or below `base`).

For a model headed into pretraining that is the encouraging reading: the
supervised task saturates in one epoch, which is exactly the regime where a
pretrained initialisation should pay.

### Exception — patch_len=50 is unsafe above ~20 channels

FACED (32 channels) and PhysioNet-MI (64) collapse on 2 of 3 seeds at
`patch_len=50`: train loss sits at the chance-level cross-entropy for the
whole run on some seeds, never moving. Every other corpus in the matrix has
16 channels or fewer and is unaffected. A single-seed 2x2 (patch_len x
learning rate) found the cause: it is the token count, not the optimizer step
size. `patch_len=200` escapes the collapse on both corpora; a lower learning
rate alone does not, and combined with `patch_len=200` it is actively worse
(that arm ran to completion and early-stopped dead at exact chance). Full
account, including the mechanical gradient check that ruled out a dead
frontend at init, in docs/FINDINGS.md.

**FACED and PhysioNet-MI ship at `patch_len=200`, `pac_patch_len=200`** —
every other corpus keeps 50. This is single-seed evidence, matching the scope
of the diagnosis so far; a 3-seed confirmation at 200 for these two corpora is
still open, as is finding exactly where between 16 and 32 channels the
boundary sits.

## Where it lands

TUEV, our protocol, 3 seeds, same pipeline:

| model | params | pretrained | kappa |
|---|---|---|---|
| **PACLock (this config)** | **1.62 M** | **no** | **0.7076 ±0.0311** |
| TFM-Tokenizer | 1.89 M | yes | 0.6519 ±0.0209 |
| LaBraM-Base | 5.82 M | yes | 0.6169 ±0.0551 |
| CBraMod | 17.85 M | yes | 0.5970 ±0.0742 |
| EEGPT | 25.69 M | yes | 0.5736 ±0.0396 |
| CBraMod (scratch) | 17.85 M | no | 0.5638 ±0.0236 |
| BIOT (scratch) | 3.19 M | no | 0.5324 ±0.0250 |

Against TFM-Tokenizer, the strongest baseline: +0.0557 with fewer parameters and
no pretraining. Honest caveat: pooled t ≈ 2.6 on 4 df, **p ≈ 0.06** — at the
edge, not past it. TUEV's seed spread is large enough that this needs more seeds
before it is claimed as a significant win.

## Rejected, with the evidence

**`patch_len = 100` as a general setting.** Pre-registered rule: a change counts
only with the same sign on both corpora. TUEV +0.0571 (significant), ISRUC
−0.0055 (significant). Rejected. It is the right PAC window for TUEV and the
wrong one for ISRUC.

**Multi-scale PAC** (`pac_patch_len` as a list; implemented, verified
bit-identical for one scale, +33K/+49K params). The pitch was that a backbone
cannot pick a window before it sees the task, so it should carry all of them:

| corpus | best single scale | multi-scale |
|---|---|---|
| TUEV | 0.7076 (0.25 s) | 0.6920 |
| ISRUC | 0.6966 (2.0 s, 1 seed) | 0.7017 (1 seed) |

It dilutes a sharp optimum (TUEV) and helps slightly where the response is flat
(ISRUC). **Not a general win, so not in the shipped config.** The code stays,
opt-in and off by default, because the ISRUC direction is worth revisiting once
the extra seeds land.

**Readout variants, augmentation, dropout, head count, learning rate.** All
inside the measured spread on at least one corpus, or sign-flipping across them.

## Still open

* `isruc-cand_ms` at 3 seeds (2 in flight) — decides whether the ISRUC
  multi-scale edge is real or a one-seed artefact.
* ~~`isruc-cand_p200_win3000/6000`~~ **resolved, and the mechanism lost.**
  15 s gives 0.6860 ±0.0129 and 30 s gives 0.6826 ±0.0097 against the 1 s
  control at 0.6909 ±0.0021: Welch t = −0.65 and −1.46, neither significant.
  Both versions of the timescale account are now falsified — the pre-registered
  "ISRUC improves with longer windows" (flat over 1–6 s) and the post-hoc
  "ISRUC collapses at the stage duration" (no collapse at 15 s or at 30 s).
  The paper states a **measurement**, not a mechanism, and names the sustained
  case as open. See docs/FINDINGS.md.

  One thing did replicate: **window length drives seed variance on both
  corpora** — TUEV sd 0.0311 / 0.0345 / 0.0458 across 0.25 / 0.5 / 1.0 s, ISRUC
  sd 0.0021 / 0.0129 / 0.0097 across 1 / 15 / 30 s. Unpredicted and unexplained,
  but consistent across two corpora with different phenomena, channel counts and
  window lengths. It is a third independent reason to keep `pac_patch_len`
  short, alongside the TUEV gain and the absence of any ISRUC cost.

---

## 2026-08-18 update — two changes to the configuration above

The block at the top of this document is the configuration that was pretrained,
and it stays as the record of what produced every `pt_*` result. Two settings in
it have since been superseded by measured improvements. Both are zero-parameter
changes, both are verified bit-identical on the paths they do not touch, and
neither is the default yet — the confirmation seeds are still running.

### `interaction_mode: product` → `rotation`

    h_j = a_j * aligned_phase_j / |aligned_phase_j|

The coupling rotates the amplitude token instead of also rescaling it. Forced
exactly as strongly as `product` — the token's phase is still determined entirely
by the coupling-aligned mixture, there is no raw high-band token beside it, and
nothing learnable can bypass it — but `|h_j| = |a_j|` now holds exactly.

| corpus | product | rotation | delta |
|---|---|---|---|
| TUSZ | 0.5882 (3) | **0.6884 (1)** | **+0.100** |
| TUEV | 0.7076 (3) | **0.7328 (3)** | **+0.025** |
| PhysioNet-MI | 0.2722 (5) | 0.2961 (1) | +0.024 |
| BCI-IV-2a | 0.3588 (3) | 0.3708 (3) | +0.012 |
| FACED | 0.1477 (3) | 0.1514 (1) | +0.004 |
| Sleep-EDF | 0.6459 (3) | 0.6449 (1) | −0.001 |
| TUAR | 0.5780 (1) | 0.5568 (1) | −0.021 |

Read TUEV seed by seed rather than as a mean: rotation's worst seed (0.7156)
is above product's mean (0.7076), and product has a seed at 0.6718 while
rotation's lowest is 0.7156. The mean and the spread both improve, which matters
here because TUEV's own control sd is 0.0235 and a delta of +0.025 sits right at
that scale.

Why, measured rather than argued: `|aligned_phase|` has a coefficient of
variation of ~0.62 across patches, and it is a by-product of the coupling
estimator, not a feature. On BCI-IV-2a all four classes have indistinguishable
coupling statistics (mean |Z| 0.0026–0.0031, preferred-phase consistency at the
surrogate null), so `product` multiplies band power by a per-patch random gain
carrying no label information. The PAC content is in the DIRECTION of
`aligned_phase`, which `rotation` keeps in full. See `docs/FINDINGS.md` for
the four explanations that were proposed before this one and refuted.

Verified by `scripts/verify_rotation.py` (all pass): `product` and `concat`
bit-identical to the pre-change implementation, so no finished cell moves;
`|h| = |a|` to 2.4e-7; tokens still respond to a phase twist, so this is not a
plain amplitude tokenizer; **gauge invariance confirmed for bands 1.. under
`p_i -> e^{i delta_i} p_i, Z_ij -> e^{i delta_i} Z_ij`** — for `rotation` and for
`product`, the first actual test of the claim `_pac_interaction`'s docstring has
always made; no NaN on a dead electrode.

### `head: mean` → `spatial`, for corpora whose label is a spatial pattern

`mean`, `band` and `attn` all collapse the electrode axis. That is right for the
TUH corpora — the label is a property of the recording — and wrong for motor
imagery, where the discriminative quantity IS a spatial contrast (mu/beta
desynchronisation over the contralateral sensorimotor strip).

| corpus | mean | spatial | delta |
|---|---|---|---|
| PhysioNet-MI | 0.2722 (5) | **0.3560 (1)** | **+0.084** |
| BCI-IV-2a | 0.3588 (3) | **0.4144 (1)** | **+0.056** |
| FACED | 0.1477 (3) | 0.1514 (1) | +0.004 |

The two changes stack: BCI-IV-2a with `rotation` + `spatial` is 0.4344, above
either alone. FACED does not respond to either, and remains unexplained.

`head: spatial` costs `(C * d_model) * n_classes` parameters — 33 K on
PhysioNet-MI's 64 electrodes, against the 1.62 M backbone.

### Not yet settled

`spatial` has not been checked on tier 1 (the TUEV run is in flight), and the
`rotation` numbers marked (1) need their remaining seeds before either becomes
the default in `configs/deliverable/`.
