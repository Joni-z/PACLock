# PACLock — the configuration to pretrain, and the evidence for it

Everything below is TEST kappa on the GEMM tokeniser path, 3 seeds unless
marked. Nothing from `runs_conv_tokenizer/` is reused: a mathematically
identical code change moved an ISRUC result by 13.4 seed standard deviations
(docs/PERF.md), so pre-change numbers are not comparable.

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
frontend at init, in docs/ARCH_SEARCH.md.

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
  case as open. See docs/ARCH_SEARCH.md.

  One thing did replicate: **window length drives seed variance on both
  corpora** — TUEV sd 0.0311 / 0.0345 / 0.0458 across 0.25 / 0.5 / 1.0 s, ISRUC
  sd 0.0021 / 0.0129 / 0.0097 across 1 / 15 / 30 s. Unpredicted and unexplained,
  but consistent across two corpora with different phenomena, channel counts and
  window lengths. It is a third independent reason to keep `pac_patch_len`
  short, alongside the TUEV gain and the absence of any ISRUC cost.
