# PACLock pretraining plan

Goal: decide, with experiments rather than a priori judgment, whether and how
to pretrain PACLock before the ICLR push, including the model-size question
(base/large only, no huge).

## 1. What we already knew going in

Every capacity-scaling result we had before this plan (6 negative data points)
was measured **from-scratch, supervised, on a single downstream corpus**.
None of them tell us anything about scaling under LaBraM/CBraMod's own
regime -- pretrain on a pooled corpus, then finetune -- because we had never
built that regime. This plan exists to close that gap with real runs, not to
re-interpret the from-scratch negatives as if they generalized.

## 2. Research grounding

**LaBraM** (Jiang et al., ICLR 2024, arxiv 2405.18765; read directly from the
PDF, table + Fig. 3 on p.6-7):

| tier  | params | how it's bigger |
|-------|-------:|------------------|
| Base  | 5.8M   | reference config |
| Large | 46M    | deeper Transformer encoder + wider hidden size |
| Huge  | 369M   | deeper + wider again |

Pretrained on **2,500+ hours** pooled from public + self-collected EEG (8x
A800). Fig. 3's own pretraining-loss/accuracy curves show Huge's loss "has an
obvious downward trend... if trained longer" -- LaBraM's own authors flag
Huge as *undertrained even at 2,500 hours*, not as a clean win. Base->Large
is the tier that shows unambiguous, converged benefit in their own plots;
Large->Huge is the tier they themselves caveat.

LaBraM's token-budget design (directly relevant to our pretraining-pool
architecture): patch window w=200 samples (1s @ 200Hz) is fixed, but total
sequence length is capped at 256 tokens by **scaling the time window with
channel count** -- 4s for 64ch, 8s for 32ch. This is a principled way to keep
one model's compute/memory bounded across corpora with very different
channel counts, and it directly informed the design choice below (section 4).

**Our own data scale for comparison**: summing (samples x window length)
across the training splits of all 9 corpora we can pretrain on today gives
roughly **3,800 hours** of windowed EEG (tuab 825h, tusz 907h, chbmit 878h,
sleepedf 510h, isruc ~579h, tuev 95h, faced 19h, physionet_mi 7h, bci_iv_2a
2h). This is *not* apples-to-apples with LaBraM's 2,500h figure -- several of
our corpora (tusz, chbmit especially) use overlapping sliding windows, so the
same seconds of raw recording are counted multiple times, whereas LaBraM
reports unique recording hours. The true unique-hour figure is smaller than
3,800h. But even discounted for overlap, our pool is the same order of
magnitude as LaBraM's Base/Large pretraining set, not a small fraction of it
-- enough to justify attempting Base and Large, not enough to justify Huge
when even LaBraM's own Huge (at their full 2,500h) is flagged as
undertrained by its authors. This is the concrete, evidence-based version of
"base/large, not huge," rather than taking the instruction on faith.

**CBraMod** (Wang et al., ICLR 2025, arxiv 2412.07236; read earlier this
project) ships a single model size, not a tiered family -- it isn't a source
of scaling evidence one way or the other, only a preprocessing/protocol
reference (see docs/FINDINGS.md's CBraMod-tokenizer-swap ablation).

## 3. The architectural constraint that shapes everything else

`patch_len` sets the `kernel_size` of the tokenizer's `Conv1d` -- a weight
**shape**, not a runtime argument. One model instance cannot serve two
different `patch_len` values, so a single pretraining run pooling multiple
corpora must commit to **one global `patch_len`** across the entire pool.
This is why `paclock_bench/training/pretrain.py` reads one corpus per step
(never mixes corpora in a batch) but still requires every corpus's data to
be resampled through the same fixed `patch_len`.

Corollary for finetuning: a downstream config whose `patch_len` doesn't match
the pretraining run's `patch_len` will silently lose only the two
`patch_len`-shaped tensors (`frontend.phase_tokenizer`, `.amplitude_tokenizer`)
at load time -- everything else (encoder, band_pe, the rest of the frontend)
still transfers. Verified directly (`scripts/verify_pretrained_load.py`,
job 371529): loading a `patch_len=200` checkpoint into a `patch_len=50` TUEV
model transferred 151/153 backbone tensors and cleanly skipped the 2
mismatched ones, rather than crashing or silently transferring garbage.

## 4. Token budget across our 9 corpora is not unified (and that's the design question)

Computed from actual training-log `(C, T)` shapes and each corpus's current
`patch_len`:

| corpus | current patch_len (matrix config) |
|---|---|
| tuab, tuev, tusz, chbmit, sleepedf, isruc, bci_iv_2a | 50 |
| faced, physionet_mi | 200 (raised from 50 after the collapse fix, docs/FINDINGS.md) |

Token counts (`C * n_bands * P`) range **960-5,760** across the matrix, a 6x
spread -- there was never an implicit constant-budget design here, unlike
LaBraM's deliberate 256-token cap. For **pretraining specifically** we chose
to fix `patch_len=200` globally (not 50), because 50 is the value that
already collapsed FACED and PhysioNet-MI under supervised CE, and Experiment
1 below confirms the same value is still the worse choice under the SSL
objective.

## 5. Pretraining corpus pool

All 9 corpora we already have preprocessed (`processed/`, ~350GB) are used
directly as an unlabeled pool -- zero new preprocessing work, and it already
covers every channel-count/montage regime in the benchmark. `data/tuh` holds
438GB more (tuar/tuep/tusl sub-corpora we don't currently use downstream, and
an empty/never-downloaded `tueg`), left out of this pilot: pulling in unused
TUH sub-corpora is a real lever for a bigger future pretraining run, but
isn't needed to answer the base/large/patch_len questions this plan targets,
and would cost real preprocessing time against a shared, space-constrained
disk (~101GB free, shared with other users).

`SpatialPE` (index mode) allocates an `nn.Embedding` sized to the pool's
*largest* channel count (PhysioNet-MI, 64ch) -- every pooled corpus indexes
into it with its own real channel count at forward time. `xyz` mode isn't an
option here since montage coordinates are corpus-specific and this pool mixes
montages.

## 6. Experiment 1 -- does the supervised collapse recur under the pretraining objective? (job 371510)

Pretrained a **base**-sized model (1.62M params) at `patch_len=50` -- the
exact value that caused FACED/PhysioNet-MI to collapse under supervised CE --
across the full 9-corpus pool for 3,000 steps, using
`crossfreq_aux_loss` (masked band-amplitude reconstruction) as a standalone
self-supervised objective, with every corpus's running loss logged
separately specifically to catch a repeat of that failure early.

**Result: partial repeat, but not the same corpus pair.**

- Every corpus except FACED converges cleanly into the 0.18-0.30 loss range
  by step 3000, **including PhysioNet-MI** (64ch) -- the corpus that also
  collapsed under supervised CE at this same `patch_len=50`. Under the SSL
  objective, PhysioNet-MI trains completely normally (0.94 -> 0.27 over the
  run).
- **FACED (32ch) does not converge**: loss descends from ~4.9 to a plateau
  around 3.2-3.4 and never approaches the other corpora's range.

This breaks the simple story that "high channel count breaks `patch_len=50`"
-- PhysioNet-MI has *more* channels than FACED (64 vs 32) and trains fine
under SSL, while FACED does not. Whatever makes FACED specifically hard is
not simply channel count, and is not simply "small training-sample corpus"
either (PhysioNet-MI's train split, 6,300 samples, is close to FACED's 6,720
and also converges fine). The supervised-CE collapse and the SSL plateau are
evidently two related but distinct failure modes sharing patch_len=50 as an
aggravating factor, not an identical mechanism -- worth a follow-up
diagnostic, but out of scope for this plan's base/large question.

**Conclusion for pretraining-pool design**: `patch_len=50` is not a safe
global choice for the pool (FACED still fails under it, under either
objective). `patch_len=200` -- already validated safe for FACED/PhysioNet-MI
under supervised finetuning -- is the correct default for Experiment 2.

## 7. Experiment 2 -- base vs. large pretraining (jobs 371511 / 371512)

Same 9-corpus pool, `patch_len=200`, 6,000 steps each:

| tier | params | d_model | depth | n_heads | wall time (1 GPU) |
|---|---:|---:|---:|---:|---:|
| base  | 1.643M | 128 | 6 | 4 | 17.5 min |
| large | 8.573M | 256 | 8 | 8 | 32.5 min |

Final per-corpus running loss (mean of last few logged windows):

| corpus | base | large | large/base ratio |
|---|---:|---:|---:|
| tuab | 0.110 | 0.070 | 0.64 |
| tuev | 0.130 | 0.083 | 0.64 |
| tusz | ~0.15-0.30 (noisy) | ~0.08 | -- |
| chbmit | 0.135 | 0.087 | 0.64 |
| sleepedf | 0.128 | 0.096 | 0.75 |
| isruc | 0.190 | 0.143 | 0.75 |
| physionet_mi | 0.175 | 0.119 | 0.68 |
| **faced** | **~2.6** | **~1.15** | **0.44** |
| **bci_iv_2a** | **0.150** | **0.073** | **0.49** |

**Result on the pretraining objective itself: capacity scaling helps, and
helps most exactly where the model is weakest.** Every corpus improves from
base to large, but the two hardest corpora in the pool (FACED, still far
above everything else in absolute loss; and BCI-IV-2a, the smallest corpus)
show roughly 2x the relative improvement of the easy, large corpora
(TUAB/TUEV/CHB-MIT improve ~35-36%; FACED improves ~56%, BCI-IV-2a ~51%). At
`patch_len=200`, FACED also no longer plateaus the way it did at
`patch_len=50` in Experiment 1 -- it keeps descending through the full
6,000-step budget in both tiers.

This is real, but it is a statement about the SSL reconstruction loss, not
about downstream classification accuracy. Section 8 shows those two things
do not move together at this pilot's scale.

## 8. Downstream finetuning pilot (jobs 371533/371536-371539)

Loading the pretrained backbone (`frontend` + `band_pe` + `encoder`;
`spatial_pe` and the classification head always reinitialize fresh --
electrode identity and label space are corpus-specific, see
`load_pretrained_backbone`'s docstring in `models/paclock/build.py`) into
finetuning configs for:

- FACED, base and large pretrained checkpoints (`patch_len=200` matches
  exactly -> full backbone transfer, 153/153 tensors)
- PhysioNet-MI, base and large (same full-match situation)
- TUEV, base checkpoint only, **on TUEV's native `patch_len=50`** (a
  deliberate partial-transfer case: 151/153 tensors transfer, only the two
  `patch_len`-shaped tokenizer convs reinitialize) -- to see whether a
  mismatched `patch_len` still yields useful transfer or whether it's not
  worth doing without matching the pretraining `patch_len`.

Compared against the existing 3-seed from-scratch `paclock_v2` numbers
already in the matrix for the same three corpora.

All five jobs completed (371539/TUEV took ~2h27m wall time -- a background
poll loop watching all five job IDs falsely reported completion at the ~12min
mark when an SSH connection to the cluster dropped mid-poll; caught by
cross-checking `scontrol show job` and `result.json` directly before trusting
it, and only TUEV was actually still running at that point).

| corpus | metric | from-scratch (3-seed mean) | pretrain+ft base (1 seed) | pretrain+ft large (1 seed) |
|---|---|---:|---:|---:|
| faced (9-class, chance .111) | balanced_acc | 0.1477 | 0.1279 | 0.1381 |
| physionet_mi (4-class, chance .25) | balanced_acc | 0.2739 | 0.2690 | 0.2716 |
| tuev (patch_len mismatch) | cohen_kappa | 0.5638 +/- 0.0193 | **0.7071** | -- |

**FACED and PhysioNet-MI: no win.** Both pretrain+finetune variants land at
or slightly below the from-scratch mean, kappa near 0.02-0.03 in both cases
(barely-above-chance for FACED, essentially chance for PhysioNet-MI's
balanced_acc). The base->large improvement visible in the *pretraining* loss
(section 7) does not show up here -- large edges out base on both corpora,
but neither tier clears the from-scratch baseline.

**TUEV: a real, large win** -- 0.7071 vs 0.5638+/-0.0193 is far outside the
3-seed noise band (>7 standard deviations by the naive estimate). But the
training curve is unusual: `verdict.status == "peaked-first-eval"` -- the
best validation score (0.6072) was reached after just 1 epoch, and every
subsequent epoch of finetuning (epochs 1-19, same lr/schedule as the
from-scratch recipe) *declined*, landing in the 0.40-0.49 range. Test
performance (0.7071) is reported from the epoch-0 checkpoint, since that's
what `best_val` selected. In effect the pretrained backbone starts in an
already-strong state for TUEV, and continued gradient updates under the
default from-scratch finetuning recipe actively degrade it -- the recipe
(learning rate, schedule, or how many epochs to run before evaluating) was
never tuned for finetuning *from* a pretrained checkpoint, only for training
from scratch, and this run used the from-scratch recipe unchanged.

**These three results are confounded three ways at once** and this pilot
cannot separate them with 3 datapoints: TUEV has ~10x more training data than
FACED/PhysioNet-MI (68,445 vs 6,720/6,300), TUEV was the corpus with a
`patch_len` *mismatch* (worse transfer, 151/153 tensors) while FACED/PMI had
a perfect match (153/153) yet did worse, and TUEV was already the easiest of
the three from-scratch (kappa 0.56 vs FACED/PMI hovering barely above
chance). Any of "more downstream data," "the mismatch forced a beneficial
partial reinit," or "TUEV's task is just more learnable" could explain why
TUEV transferred and the other two didn't. No causal claim beyond "it
happened" is warranted from this pilot.

## 9. Recommendation

- **Pretrain both base (1.6M) and large (8.6M), not huge.** Grounded in: (a)
  LaBraM's own authors flag their Huge as undertrained even at 2,500h, a
  scale we're in the same order of magnitude as but don't clearly exceed;
  (b) the pretraining-loss pilot shows base->large gives a real, and
  disproportionate, benefit on the SSL objective for the corpora that need it
  most (FACED, BCI-IV-2a) -- evidence for attempting large over base, without
  yet justifying a further jump to huge. Downstream finetuning has not yet
  shown large beating base on accuracy (section 8), so this is a bet
  justified by the pretraining signal, not yet confirmed by the finetuning
  signal.
- **Pretraining pool `patch_len` should be 200, not 50.** 50 leaves FACED
  unable to converge under either the supervised or the SSL objective; 200
  does not.
- **Do not yet claim "pretraining helps PACLock" as a general result.** It
  produced one large, real win (TUEV) and two flat-to-negative results
  (FACED, PhysioNet-MI) in this pilot. The honest current state is: pretrain
  + finetune is *not uniformly bad* (ruling out the worry that our
  architecture can't benefit from pretraining at all) and *not uniformly
  good* either. The next experiment this motivates is not "ship it" but
  "figure out why TUEV worked and the other two didn't" (section 10).
- **The finetuning recipe needs its own tuning pass separate from the
  from-scratch recipe.** TUEV's peaked-first-eval-then-declined curve is a
  concrete symptom: reusing the from-scratch lr/schedule for finetuning from
  a pretrained checkpoint is actively destroying a good starting point after
  epoch 0. A lower finetuning lr, a warmup-then-short-decay schedule, or
  evaluating (and stopping) more finely within the first epoch are all more
  likely explanations than "TUEV just doesn't benefit from more training" --
  worth fixing before drawing conclusions from any future finetuning run.

## 10. Open follow-ups (not blocking this plan's conclusion)

- **Highest priority**: tune a finetuning-specific lr/schedule (distinct from
  the from-scratch recipe) and rerun the TUEV/FACED/PhysioNet-MI finetuning
  pilot with it, before concluding anything about whether pretraining helps
  those corpora -- the current negative result on FACED/PMI may be a recipe
  artifact, not an architecture/data-scale limit, given what TUEV's curve
  shape suggests.
- Disentangle the three confounds in section 8's TUEV result (downstream
  data size vs `patch_len` match vs task difficulty) with a design that
  varies one at a time -- e.g. finetune TUEV at `patch_len=200` (full match)
  to isolate the mismatch variable, or finetune a `patch_len=200` corpus with
  a similarly large training split to isolate the data-size variable.
- Why FACED specifically plateaus under SSL reconstruction at `patch_len=50`
  while PhysioNet-MI (more channels, similar sample count) does not -- worth
  a dedicated diagnostic, not resolved here.
- 3-seed confirmation of both the pretraining pilot and the finetuning pilot
  before anything here is treated as a matrix-table result rather than a
  design decision input.
- Whether pulling in the unused TUH sub-corpora (tuar/tuep/tusl) or the
  never-downloaded `tueg` would move the needle further, once the finetuning
  recipe question above is resolved and pretraining is confirmed to help at
  all on more than one corpus.

## 11. Move to Bridges-2 (b2) and the finalized pretraining pool

Real pretraining runs at base/large scale, on the full data plan, are
happening on a second cluster (PSC Bridges-2, `ssh b2`,
`/ocean/projects/cis260249p/qren2/`) -- H100-80/V100/L40S GPUs (CUDA, not
ROCm), 1.1TB free project disk vs AMD's shared ~100GB, and, decisively, the
**full TUEG corpus already sitting on disk**: 27,074.4h / 1,643GB, v2.0.2
(README's own precise figures), no application needed -- superseding this
doc's earlier "would need to apply for TUEG access" note, which was wrong.

**What changed vs section 1-10's pilot, concretely:**

- Repo, `paclock_bench/models/build.py`'s checkpoint-loading path, and
  `training/pretrain.py` all synced over (b2's own checkout was stale at a
  pre-pilot commit with no network access to GitHub -- relayed directly
  file-by-file, not through git).
- New SLURM runners (`slurm/run_b2.slurm`, `run_cpu_b2.slurm`) -- this
  account is GPU-only (confirmed directly: every RM/RM-shared/RM-small
  partition submission fails "Invalid qos specification", and even a
  `--gpus=0` request on GPU-shared is refused "Access/permission denied"),
  so CPU-only utility work either runs on the login node directly (fine for
  anything finishing in seconds) or rides a minimal `v100-16:1` allocation.
  `--gpus=type:n` is Bridges-2's own required GPU request syntax, not
  `--gres`.
- Missing packages (`einops` for the vendored baseline code,
  `mne`/`openpyxl` for preprocessing/table scripts) live in a plain
  `--target` directory on `PYTHONPATH`, not a venv -- a venv layered on the
  provided `pytorch/26.05-2.11-py3` module with `--system-site-packages`
  does NOT inherit its torch, because the module is itself a venv on a base
  interpreter and site-package inheritance isn't transitive through more
  than one layer (confirmed directly by inspecting `sys.path`).
- **`preprocessing/tueg.py`** (new): the exclusion + subject-diverse
  sampling script this section's data decision required. Excludes every
  session TUEG shares with TUSZ (using TUEG's own
  `DOCS/sessions_tueg_common_with_tusz.list` -- exactly the
  eval-contamination check section 6's "excluding" follow-up asks for, now
  actually run rather than deferred) before sampling, then draws one file
  per subject across as many distinct subjects as the hour budget allows,
  round-robining to a second file per subject only once the budget can't be
  met from distinct subjects alone -- diversity first, not just hours.
  Found and fixed in passing: none of the existing `preprocessing/*.py`
  scripts (tuab.py etc.) ever called `paths.expand()` on their own
  `$PACLOCK_*`-templated config paths, unlike every training-time reader --
  latent since the portability migration templated those configs, harmless
  only because nothing had re-run preprocessing since. Fixed in `tueg.py`;
  the older scripts weren't touched since they're not on this plan's
  critical path (all 9 downstream corpora were already preprocessed on AMD
  and transferred as data, not re-run).
- **TUEG slice sized at 2,000h, not the full 27,074h or the originally
  drafted 5,000h.** At 5,000h the slice alone would out-window the entire
  rest of the 9-corpus pool combined (~1.8M windows vs ~1.2M), and
  `pretrain.py`'s per-corpus sampling is weighted by training-split size
  (deliberately, so a big corpus isn't swamped by many small ones) --
  meaning a 5,000h+ slice would flip that protection into the opposite
  failure, swamping FACED/PhysioNet-MI/BCI-IV-2a under a single dominant
  clinical source and undercutting the multi-paradigm-diversity argument
  section 9 makes for not chasing TUEG's full scale in the first place. At
  2,000h (~720k windows) TUEG becomes the largest single corpus in the pool
  but not an overwhelming majority of it. `steps` raised 6000 -> 9000 in
  both `configs/pretrain/size_{base,large}.yaml` to compensate: the small
  corpora should see roughly the same *absolute* number of draws as the
  9-corpus pilot, not a shrinking fraction of a now-larger pool.
- Final pool, both tiers: the original 9 downstream corpora + `tueg_slice`.
  The earlier idea of separately adding TUAR/TUEP/TUSL is dropped --
  TUEG v2.0.2's own changelog says it already "merged the TUSZ and TUEP
  changes into TUEG", so a TUEG slice already subsumes what those would
  have added; keeping them as separate pool entries too would double-count.

**Status of the b2 move**: complete. Repo synced, environment verified,
all 9 corpora's processed data on b2 (TUAB was re-preprocessed locally from
raw EDFs already present on b2 -- 4 minutes, vs an estimated 8.6h to
transfer; CHB-MIT was measured downloading from PhysioNet at ~49KB/s, 28x
slower than the running transfer, so transferring won there; TUSZ's official
annotations need a Temple/NEDC data-use agreement, so no local shortcut
existed). TUEG slice preprocessed: 706,817 windows / 1,963h / 5,224
subjects, 21 files excluded for missing channels.

## 12. The real pretraining run, and what it overturns

Sections 6-9 were all measured on a **6,000-step pilot**. That budget is now
known to have been far too small, and every negative conclusion drawn from
it about downstream transfer has been overturned by the real run.

**Step count, derived from our own data rather than copied.** LaBraM's
~50-epoch regime does not transfer -- discrete VQ-token classification and
our continuous masked-amplitude regression converge differently. From the
pilot's own per-corpus curves: tuab plateaus by ~step 3000-4000 (0.10-0.13,
no further descent), while FACED -- the hardest corpus -- is still
descending steadily at step 6000 (recent rate ~3.16e-4 loss/step) with no
plateau at all. Linear extrapolation of that rate to tuab's plateau level
needs ~7,700 more steps, and that is an underestimate because real decay
curves flatten rather than stay linear. A 2.5x margin for curve shape, then
1.6x for tueg_slice diluting each corpus's per-step draw probability, gives
~30k additional steps over the 6,000 baseline -> **60,000 steps**.

**Result (jobs 43534889 base / 43534890 large, b2, l40s-48):** base 1h39m,
large 4h25m. FACED converged: its loss now sits at 0.05-0.07, flat, in the
same band as every easy corpus -- versus still falling through 2.5 when the
pilot ran out of budget. The step-count derivation was correct, and FACED's
apparent "collapse" in the pilot was simply undertraining.

**Downstream finetuning with the properly-trained checkpoints** (seed 0;
each config is its own deliverable config with *only* `checkpoint` added, so
"pretrained vs from-scratch" is single-variable and lands next to its own
baseline in the matrix):

| corpus | from-scratch (3-seed mean) | pt base | pt large |
|---|---:|---:|---:|
| bci_iv_2a | 0.3588 | 0.3588 | 0.3623 |
| faced | 0.1477 | 0.1312 | **0.1742** |
| physionet_mi | 0.2833 | **0.3317** | 0.2965 |

FACED-large is +0.0265 over from-scratch whose own three seeds span only
0.1451-0.1523; PhysioNet-MI-base is +0.0484 over a from-scratch pair
spanning 0.2828-0.2839. Both are well outside their baselines' seed spread.

**This directly overturns section 8.** That section reported FACED and
PhysioNet-MI as flat-to-negative under pretraining and speculated the cause
was an untuned finetuning recipe. Two things are now known:

1. The recipe hypothesis was tested directly and **failed** (AMD jobs
   372507/372508/372509: lr 1e-4 -> 2e-5 plus warmup_epochs). TUEV's peak
   moved from epoch 0 to epoch 3 -- warmup does something real -- but the
   post-peak decline persisted and the final score dropped (kappa 0.644 vs
   0.707); FACED improved marginally (0.1408 vs 0.1279) but still trailed
   from-scratch; PhysioNet-MI got *worse* and was verdict-flagged
   mis-configured. So the recipe was not the explanation.
2. **Pretraining budget was.** The same corpora, same recipe, same
   architecture, with a 60k-step checkpoint instead of a 6k-step one, move
   from "no transfer" to a clear gain. Section 8's negative result was an
   artifact of an undertrained checkpoint, not a property of these corpora.

**Still open**: everything above is seed 0 only; 3-seed confirmation is
running. The remaining corpora (tuab, tusz, chbmit, isruc, sleepedf, tuev)
are in flight. And note 7 of 9 corpora finetune at their own `patch_len=50`
against a `patch_len=200` checkpoint, i.e. partial transfer (151/153
tensors, the two tokenizer convs reinitialize) -- deliberately, to keep the
comparison single-variable against each corpus's existing matrix row.

## 13. One cluster per matrix, and why

Every group A/B/C number already in `results/PACLock_baseline_matrix_filled.xlsx`
was produced on **amd** (ROCm, MI210, torch 2.7.1). The pretrained rows must
therefore also come from amd, or they are not comparable to the rows directly
above them -- which is the entire point of placing them under their own
from-scratch baseline.

This was not a theoretical concern. `tusz-paclock_pt_base` seed 0 was run on
both clusters from byte-identical configs (verified field-by-field: the only
differences were the checkpoint's path and the seed number) against the same
checkpoint and the same transferred data:

| cluster | torch / GPU | tusz pt_base seed 0 (AUC-PR) |
|---|---|---|
| amd | 2.7.1 ROCm / MI210 | 0.6989 |
| b2  | 2.11.0 CUDA / V100-32 | 0.6448 |

A 0.054 same-seed gap from hardware and library alone -- larger than several
of the pretraining effects this document reports. (tusz's own within-amd seed
spread is larger still, 0.576-0.699, so this corpus is high-variance
generally; but the gap is real, and there is no reason to import it into a
table where every other row is single-cluster.)

**Rule adopted:** every cell of the matrix is produced on amd. The 26
b2-sourced seed directories that had been copied in were identified by their
`checkpoint` path (`pretrain_runs/` = b2, `pretrain_runs_60k/` = amd) and
moved to `runs_b2_quarantine/`; all nine corpora x two tiers x three seeds
were then re-run on amd. b2's role ends at producing the two pretraining
checkpoints, which are just files and carry no cluster dependency once
written.

## 14. Baseline recipe deviation: CNN-Transformer on FACED / PhysioNet-MI

`cnn_transformer` was the one group-A model that collapsed to chance on FACED
and PhysioNet-MI -- 0 of 3 seeds admissible on either corpus. The cause is
not a mis-specified config on our side: all five group-A models use the
identical `lr=1e-3, batch_size=64` on FACED, and the other four train fine.
This architecture is simply unstable at that learning rate on these two
corpora.

A collapsed cell reports nothing about the baseline's real capability, so the
learning rate was swept (3e-4 and 1e-4, three seeds each, both corpora). Both
values fix the collapse outright -- 3/3 admissible everywhere, and scores far
above the chance floors (FACED chance 0.111 -> 0.13-0.27; PhysioNet-MI chance
0.25 -> 0.39-0.47). The better-scoring value per corpus is adopted, which is
the direction that favours the baseline rather than us.

This is a deliberate, minimal deviation from hard rule 2 (every baseline runs
its own repo's recipe): the repo recipe does not converge on these two
corpora, which its authors never ran, so there is no recipe to honour. It is
recorded here and in the workbook's `_填写记录` sheet rather than applied
silently.

---

# 附录 D(2026-08-18):扩模计划 —— 依据、成本、阶梯

原则:不盲目做大。每一级都要有(a)容量确实还在付钱的证据,(b)数据量跟上,
(c)算得起的成本。以下全部是实测数字,唯二的未知项单独标出。

## D.1 我们现在在哪

| 量 | 实测值 | 来源 |
|---|---|---|
| base | 1.63 M | result.json |
| large | 8.5 M(d256, depth8) | RAWPT 日志 |
| 60k 步成本(l40s-48 @24 SU/h) | base 1h39m ≈ **40 SU**;large 4h25m ≈ **106 SU** | b2 sacct 43534889/90 |
| 预训练池 | 9 语料 + 2,000h TUEG 切片(处理后 85 GB) | b2 du |
| 数据覆盖 | 60k 步 × batch32 × ~10s ≈ 5,300h 信号 ≈ **池的 ~1.1 个 epoch** | 配置推算 |
| b2 余额 | **147 / 700 SU** | projects |
| b2 存储 | 222 / 3,000 GB(余 2.78 TB) | projects |

关键读数:**当前 large 在 60k 步下把池子恰好过了一遍** —— 参数、步数、数据
此刻是平衡的。只加参数不加步数和数据,没有依据。

## D.2 对标(参数为我们矩阵里的实测值,不是论文自报)

| 模型 | 实测参数(as-run) | 预训练数据 |
|---|---|---|
| EEGPT | 25.7 M | (未核) |
| **CBraMod** | **19.1 M** | TUEG 级(具体小时数待从其论文核实) |
| **LaBraM-Base** | **5.8 M** | ~2,500h / ~20 数据集(vendor README 原文) |
| BIOT | 3.2 M | 5 数据集 |
| TFM-Tokenizer | 1.9 M | — |
| PACLock base / large | 1.63 / 8.5 M | 4,800h 池 |

**我们的 large(8.5M @ ~4,800h)已经与 LaBraM(5.8M @ 2,500h)同级或更大。**
下一个有意义的档位是 CBraMod 级(≈19 M)。

## D.3 容量还在付钱的证据(已有,不用再跑)

* 容量阶梯(raw, BCI):d32→64→128→256 = 0.2617→0.2780→0.4192→0.4437,单调不饱和;
* 60 epoch 下 large:ISRUC +0.0181(8.6σ)、TUEV +0.041 —— 20 epoch 时的负结果是没训够;
* 成本结构:encoder 只占步时 ~2%,size_large 的 5.33× FLOPs 只贵 7% 墙钟 ——
  **encoder 扩容在本栈上近乎免费,贵的是前端 GEMM(不随 d_model 显著增长)**。

## D.4 阶梯

**第 0 级(在跑,免费,AMD)**:hybrid 监督验证,TUEV/CHB-MIT/TUSZ/BCI 单 seed。
机制预测:处处 ≥ raw,TUEV 保住 PAC 优势。任一条不成立则 hybrid 出局。

**第 1 级(现在就付得起:~110 SU,余 147)**:hybrid-large(8.5 M + 51 K)
在**当前池**上 60k 步。前置条件有两个:
1. 第 0 级通过;
2. **hybrid 的配对行掩码**(见下 D.6)—— 未实现前 hybrid 不能预训练。
另:先在 AMD 用 200 步测 hybrid 的每步耗时(前端算两套 tokenizer,估计
+30–50%,**未实测,不许拍脑袋进成本表**)。

**第 2 级(付不起,需要 SU 补充)**:19 M 档 + 池扩到 ~10,000h
(TUEG 切片 2,000→7,000h,处理后约 85→300 GB,b2 存储足够)。
成本估算:每步 ~2×large、步数 ~120k(维持 ~1 epoch 覆盖)→ **~400–500 SU**,
另加扩片预处理的 GPU-shared 时长(**未知项二**:首次 2,000h 切片的预处理
job 耗时未留记录,扩片前先查 sacct 或用 500h 试片实测)。
**结论:第 2 级必须先申请 SU supplement(PSC 可申请)—— 这是 Zhizhe 层面的
动作,不是脚本能解决的。**

## D.5 为什么预训练留在 b2

AMD 免费但 `/work1` 配额 1.9 TB 已是紧约束(2026-08-18 清理后余 293 GB),
而 TUEG 原始数据 1.6 TB **只在 b2**,拷不动也放不下。数据在哪,预训练在哪。

## D.6 代码前置:hybrid 的配对行掩码(未实现,已设防)

交互行 j 直接携带 a_j。crossfreq 掩码若只遮 raw 行 j 而留交互行 j 可见,
幅度重建目标等于把答案递给模型。掩码必须**成对遮蔽**(raw 行 j 与交互行 j
同遮),损失只在 raw 行上计。实现前,frontend 的 `return_amp_target` 与
builder 的 `aux_recon` 都会对 hybrid 直接 raise(verify_hybrid.py 断言了
这两个 guard)—— 宁可当场失败,不许静默泄漏。
