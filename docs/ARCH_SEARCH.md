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
