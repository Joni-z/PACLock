# RMSNorm and GEGLU backbone screen

Status: implementation and combined single-card verification passed; the four
bounded training admissions are being prepared. The joint augmentation screen did not deliver a
TUEV kappa gain, so its automatic second-seed expansion remains held.

The next hypothesis concerns the encoder, while retaining the coupling/local
token construction. REVE reports an EEG ablation favoring RMSNorm plus GEGLU
over LayerNorm/GELU combinations on its tested tasks; this motivates a transfer
check, not an expectation that its gains will reproduce here. See
[REVE v1, Appendix D, Table 20](https://arxiv.org/html/2510.21585v1#A4.T20).

`model_kwargs.block_variant: rms_geglu` replaces the four pre-normalizations
in each tri-axial block with RMSNorm (epsilon 1e-5), and the 2D GELU FFN with
GEGLU at hidden width ceil(4D/3). At D=192, both FFNs have the same number of
matrix weights; normalization and bias counts differ slightly. The tokenizer,
positions, axis attentions, width, depth and final readout are unchanged.
This is one combined backbone change, not separate causal ablations of norm
and activation. It is not claimed as the paper's novel contribution.

The default `legacy` branch preserves the original constructor order, state
keys and arithmetic. Unknown variants fail explicitly. Scratch configurations
are used; compatibility with old pretraining checkpoints is not established.

## Verification before training

`smoke/smoke_rms_geglu.py` runs only through `slurm/smoke_gpu.slurm`. It checks
the actual full model against committed legacy source, exact initialization
and evaluation logits, and bounded float32 optimizer-step agreement including
an old-versus-old repeat. It also checks that the new module is reached through
the real builder, handles flat inputs, reloads a strict checkpoint, and has
finite nonzero normalization/FFN gradients on real training batches.

The first smoke, 417375, exited after 46 seconds at an overly strict bitwise
optimizer check: one of 3200 elements differed by 5.82e-11. Its log is retained.
Initialization and forward equality remain bitwise requirements; the retry
records both legacy self-repeat and new-legacy optimizer differences with
rtol 1e-6 and atol 1e-8. No production training setting was changed for this
verification. Retry 417377 passed the functional checks but failed the budget
check because a stochastic frequency-augmentation path first compiled after
the initial two batches. Its complete receipt is retained rather than dropped.

Only training arrays are opened by this combined smoke. Timing excludes two
warmup steps after explicitly warming every augmentation path, and each
projected training duration must fit under 80% of its
unchanged cap. The projection excludes data loading, evaluation and saving;
the receipt records source/config hashes and the actual hardware/runtime.

Smoke 417383 completed successfully in 1:26 on mi2101x. Initialization and
evaluation remained bit-exact; both the old-versus-old and new-legacy optimizer
comparisons had maximum absolute differences of 5.82e-11. All four real-batch
cases passed. TUEV legacy/new projected training times are 3.693/3.982 hours;
Sleep-EDF times are 2.662/2.851 hours. The new TUEV model has 3,612,886
parameters versus 3,616,726, with peak allocated memory 12.06 GiB versus
10.25 GiB. Timing includes all twelve post-warmup observations, with the same
observed augmentation sequence for each pair. The receipt is
`results/audits/rms-geglu-smoke-20260913.json`. The single-configuration smoke
also warms every augmentation path before its existing budget check.

## Bounded comparison

Four prepared seed-0 configurations are under `configs/backbone_gate/`:
TUEV and Sleep-EDF, each with legacy and RMSNorm/GEGLU. Both arms use joint
128/32/32 coupling/band/broadband content and the same augmentation list.
Within each corpus the only substantive change is the backbone variant.
These distinct run identities do not reuse the held augmentation-only plan.

TUEV keeps 20 epochs and a five-hour training cap within a six-hour allocation;
Sleep-EDF keeps 20 epochs and a four-hour cap within a five-hour allocation.
Use one mi2101x node per configuration, with a further real-batch budget smoke
before each trainer starts. Maximum requested wall time totals 22 single-card
node hours, not a verified conversion to the site's balance units. Do not
shorten epochs or extend caps to force a result into the complete-run table.

Before reading any new training result, the budget-advancement rule is fixed:
all arms must complete normally; TUEV selected-validation kappa must improve
by at least .02 with no balanced-accuracy decline, and Sleep-EDF kappa and
balanced accuracy must not decline. Class recalls at the same selected
checkpoint require explicit review. A failure prevents automatic expansion,
not permanent elimination of a model family based on one seed.

If this screen is promising, a second matched seed and further corpora are
required. It cannot establish a final candidate on ten-plus datasets, causal
physiology, a pretraining benefit or SOTA. No new pretraining is admitted.
