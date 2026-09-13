# Candidate decision and resource gates — 2026-09-13

**No final candidate is frozen.** The goal remains one defensible model with
SOTA or near-SOTA performance across more than ten datasets and a coherent
PAC/tokenizer explanation. A stronger isolated seed or a finished smoke test
is not sufficient to declare that goal achieved.

## Current validation evidence

The completed seed-0 factorized pilots on Sleep-EDF, TUAR and ISRUC favor f2:

| Dataset | f0: coupling control | f1: band-local lane | f2: broadband-local lane |
|---|---:|---:|---:|
| Sleep-EDF, completed 20 epochs | .6403 | .6416 | .6729 |
| TUAR, completed 20 epochs | .6049 | .6014 | .6231 |
| ISRUC, completed 20 epochs | .7232 | .7258 | .7584 |
| TUEV, completed 20 epochs | .5920 | .6294 | .6006 |

These are best logged **validation kappa**, not test scores. Completed pilots
have one seed; live rows are provisional. f2 has useful evidence on both sleep
and artifact corpora, but does not resolve the main TUEV weakness. f1 is closer
on TUEV: its seed-0 .6294 compares with historical rotation-only seed-0 .6339.
Historical rotation-only has three-seed validation mean .6504; duplex .6188.
Do not compare a new .64 validation number with an old .73 test number.

All three ISRUC arms completed without a budget stop despite differing
backfill time caps. Their learning recipes match except for content source;
all use index spatial embeddings. The completed f2–f1 validation difference
is +.032663. Original result hashes and execution differences are recorded
in `results/audits/isruc-factorized-completion-20260913.json`.

The TUAR seed-1 broadband-content confirmation, `tuar-factorized_f2_confirm`,
has completed all 20 epochs. Against the completed band-content control,
validation kappa improves .608668 to .630976 (+.022308), balanced accuracy
.717209 to .756006 (+.038797), and weighted F1 .773720 to .783590. Cohort
identities, class counts and recipes match after removing identity and
execution/descriptive metadata; both have two-hour caps and finish normally.
Both use mi2104x, but the control completed earlier on another allocation.
The original result hashes and checkpoint-selected validation metrics are in
`results/audits/tuar-f2-seed1-completion-20260913.json`. This confirms the TUAR
direction in a second seed without resolving the cross-corpus content tradeoff.

The separate mi2101x TUEV seed-0 scale pair has completed all 20 epochs.
Scale 1.0 achieves validation kappa .616866 and scale .3 achieves .644781
(+.027914). At their kappa-selected checkpoints, balanced accuracy changes
from .631361 to .569114 (-.062246); per-class recall shows a substantial
tradeoff. Configurations match except for the intended scale and run name,
with identical manifests and class counts. No test score informed admission.
See `results/audits/tuev-scale-seed0-completion-20260913.json`.

The TUEV seed-1 confirmation pair has **completed all 20 epochs** on mi2104x,
with trainer exit code 0 in both completion receipts. At each kappa-selected
checkpoint:

| Content scale | Validation kappa | Balanced accuracy | Weighted F1 |
|---|---:|---:|---:|
| 1.0 | .572936 | .512407 | .908817 |
| .3 | .620768 | .477718 | .918967 |

Scale .3 raises kappa by .047832 but lowers balanced accuracy by .034689,
repeating the direction of the seed-0 metric tradeoff. Configurations match
except content scale, identity and execution metadata; manifests and all class
counts match. Stop expanding the scale sweep while the joint-content gate
finishes. Do not pool the mi2101x seed-0 and mi2104x seed-1 cohorts silently.
Source at admission was `ccc0495`; full histories, original hashes and trainer
receipts are in `results/audits/tuev-scale-seed1-completion-20260913.json`.

Allocation 416484 has released its node after 12:49:52. Slurm records its
original packed batch as FAILED (1:0), consistent with previously stopped
original children, including TUSZ n3. The later scale-.3 trainer itself completed
normally with a full result and exit 0. This parent status must not trigger a
restart or invalidate that completed result. Allocation 416422 remains occupied
by other experiments; its scale-1 trainer has exited.

The fixed-width joint-content alternative, f4, has completed all 20 epochs on
both TUEV and TUAR at seed 1. TUEV finishes with validation kappa **.637154**,
balanced accuracy **.577796**, and weighted F1 **.926717**. Against the matched
band-content scale-1 control, these improve by .064218, .065389 and .017900;
against scale .3, by .016386, .100078 and .007750. Recipes, cohorts and parameter
counts match for the intended content contrast. The trainer exited with code
0 after 3.205 hours. Full histories, hashes and completion receipt are in
`results/audits/tuev-joint-completion-20260913.json`.

The final selected TUEV checkpoint has exactly the same SHA-256 as the immutable
snapshot used for the earlier content/preferred-phase probes. Those probe
receipts retain their original provisional observation times; their measured
interventions now describe the checkpoint selected at full completion as well.
They remain post-training interventions, not retrained ablations. This f4 result
still trails historical rotation-only seed-1 validation kappa .649116, whose
narrower model is not a matched-width causal control. SPSW recall remains zero.
No final candidate is frozen and no pretraining is resumed.

The TUEV f4 trainer used allocation 416422 GPU 3; the completed band-content
control used GPU 1. TUAR f4 used GPU 2. Their original admission checks are in
`results/audits/factorized-joint-admission-20260913.json`; no new allocation was
requested for these completed gates.

All three TUAR content arms completed 20 epochs with matching recipes, cohorts
and parameter counts. At their kappa-selected checkpoints:

| TUAR seed 1 content | Validation kappa | Balanced accuracy | Weighted F1 |
|---|---:|---:|---:|
| Band, 64 coordinates | .608668 | .717209 | .773720 |
| Broadband, 64 coordinates | .630976 | .756006 | .783590 |
| Joint, 32 + 32 coordinates | .615751 | .731577 | .773869 |

Joint content improves kappa by .007083 over band content but trails broadband
by .015225, with lower balanced accuracy and weighted F1 as well. Controls
completed earlier on different mi2104x allocations. This one seed has not
shown that splitting content capacity preserves the stronger TUAR result.
Do not expand joint content to other corpora yet. The now-completed TUEV gate
supports only the bounded TUEV/TUAR augmentation transfer below. Detailed config checks,
selected metrics and original hashes are in
`results/audits/tuar-joint-completion-20260913.json`.

f4 keeps 128 coupling coordinates and splits the existing 64 content coordinates
into 32 band-local and 32 broadband coordinates, with unchanged parameterization
and initialization. Each source has less capacity than its 64-coordinate control;
this is not a signal-invertibility claim. Contract checks cover exact coordinates,
finite random/flat/near-flat gradients, three-lane trainability and checkpoint
reload. Smokes measured about .257/.256 s per TUEV/TUAR step after two warmups,
10.31 GiB peak memory, and 3.05/.92 hours of projected training without loading
or evaluation. The completed TUEV result is recorded above.

A validation-only checkpoint diagnostic now probes the completed TUAR models.
All three unmodified checkpoints reproduce their selected validation metrics,
and parameters/checkpoint hashes remain unchanged. In f4, coordinate-zeroing
before positional embeddings gives:

| TUAR f4 intervention | Validation kappa |
|---|---:|
| Unmodified checkpoint | .615751 |
| Zero coupling coordinates | .176542 |
| Zero all content coordinates | .469919 |
| Zero band-content coordinates | .614158 |
| Zero broadband-content coordinates | .525218 |

The trained f4 prediction is sensitive to coupling and broadband coordinates;
its kappa barely changes when band content is removed. On the first validation
batch, band-content RMS is .6542 and broadband RMS .2941, so larger token scale
alone does not establish useful contribution. The result is consistent with
checking the fixed 32/32 capacity allocation next, but does not prove it caused
the gap to the 64-coordinate broadband control. Zeroing shifts the encoder's
input distribution and shared normalization, and coupling coordinates carry
more than a selective physical-PAC intervention. These are neither retrained
ablation scores nor proof that PAC itself causes the gain. No test split was
loaded or evaluated, and no optimizer or training was used.

The diagnostic ran on already-idle GPU 2 in allocation 416422; step 416422.1
completed normally in 2:05 while TUEV training remained live. Script revision
`c80edab` is on the factorized branch. Checkpoint/script hashes, all three models'
metrics, and measured 110.5-second evaluation runtime are in
`results/audits/tuar-content-knockout-20260913.json`. At that observation the
TUEV gate was still pending; its subsequent completion is recorded above.

The same coordinate diagnostic now covers TUEV using immutable snapshots of
still-running seed-1 checkpoints. The scale-1 and scale-.3 controls were observed
at 11 validations (selected epochs 6 and 0); f4 at nine validations (selected
epoch 1). All snapshot baselines exactly reproduce their saved validation kappa.
These are provisional checkpoint diagnostics, not complete-training results.
For TUEV f4, unmodified kappa is .637154, zero coupling .049324, zero all content
.541583, zero band content .614323, and zero broadband content .549926.

Both corpora's trained f4 checkpoints are sensitive to coupling and broadband
coordinates. Removing band content costs .022831 kappa on this TUEV snapshot
versus .001593 on the completed TUAR model, so the band lane cannot be declared
universally redundant from TUAR alone. The TUEV scale-.3 checkpoint also rises
from .620768 to .648009 after content zeroing, with balanced accuracy slightly
lower (.477718 to .474354). This post-hoc validation intervention is not a new
candidate score, a test result, or evidence for immediately deleting content.
The effects still confound information removal with changes to input statistics
and shared normalization; no specific causal PAC benefit has been established.

Step 416422.2 completed normally in 5:38 on previously idle GPU 2, with 324.4
seconds measured by the probe and no optimizer/test evaluation. Training
processes remained live afterward. The factorized-branch script revision is
`1573d3b`; immutable checkpoint/metadata snapshots remain alongside
`results/audits/tuev-content-knockout-snapshot-20260913.json` on AMD. The receipt
records their hashes and observed training state. Finish the original TUEV
training comparison before changing architecture or admitting more runs.

A more targeted preferred-phase diagnostic uses the existing frontend
`magnitude` and `scramble` controls on the same f4 checkpoints. The first
validation batch preserves content coordinates and coupling magnitudes exactly;
rotation-token modulus changes by at most 1.91e-6. Both measured-mode baselines
reproduce their saved validation scores, and weights/metadata hashes are unchanged.

| f4 checkpoint | Measured kappa | Preferred-phase alignment removed | Phase-edge scramble, mean ± SD |
|---|---:|---:|---:|
| TUAR, completed seed 1 | .615751 | .602219 | .610663 ± .005465 |
| TUEV, provisional seed 1 snapshot | .637154 | .171473 | .395576 ± .006057 |

The scramble seeds 0/1/2 are intervention draws on one trained checkpoint,
not three independent training seeds. TUEV scramble kappas are .390653,
.402340 and .393736. This supports task-dependent sensitivity to the measured
preferred-phase alignment: strong on this TUEV checkpoint, much smaller on
TUAR. It also shows why deleting all coupling coordinates overstates the
specific phase mechanism on TUAR. These interventions break the trained
alignment/gauge structure and change input directions; they do not prove a
causal physiological relationship or the gain from training with the prior.
A trained magnitude/scramble control would answer a different question.

The script is `smoke/diagnose_preferred_phase.py` at factorized revision
`20b132d`. Step 416422.3 completed normally in 3:32 on existing idle GPU 2;
the measured probe runtime was 198.3 seconds. TUEV training stayed live.
No test evaluation or new training allocation occurred. Detailed hashes,
metrics and control checks are in
`results/audits/preferred-phase-knockout-20260913.json`. This sharpens the
mechanism claim while the final candidate and broad performance remain open.

The combined diagnostic figure is generated from those three immutable audit
receipts by `python3 scripts/plot_content_diagnostics.py` (requires matplotlib).
PDF, SVG, PNG and the numerical/source-hash manifest are under
`results/figures/content-phase-diagnostics-20260913.*`. The figure labels TUEV
as an interim checkpoint and distinguishes intervention-draw SD from training
seed uncertainty. No table or checkpoint score was edited to produce it.

Historical from-scratch own-phase controls provide a different, necessary
comparison. Ten pairs match config (apart from identity/phase mode), seed,
manifest and class counts for the older duplex model. Measured-minus-own mean
validation effects are +.016224 kappa on IIIC (three seeds), -.028458 PR-AUC
on CHB (three), -.012056 PR-AUC on TUSZ (two), and -.000421 kappa on TUEV (two).
IIIC favors measured alignment in all three seeds; CHB/TUSZ are mixed and TUEV
is essentially tied. Measured TUEV seed 1 is absent; TUSZ seed 0 is excluded
because manifest timestamps differ, which alone does not prove different
examples. Historical source identity remains unverified.

These older trained controls cannot settle the current f4 architecture, but
warn against equating a trained checkpoint's sensitivity with the prior's
training benefit: an independently trained alternative may compensate. Own
still keeps each band's analytic phase/amplitude; it is not an amplitude-only
or trained magnitude/scramble control. Do not repeat the old sweep. Any final
new model's attribution needs its own matched training evidence. Exact pair
identities and hashes are in `results/audits/trained-own-controls-20260913.json`.

The 200-Hz f3 implementation remains blocked after its realized lowest filter
peaked at DC with gain 70.156. That invalidates the intended low-frequency
contrast; the completed TUAR f3 record is retained but excluded from candidate
selection. The 100-Hz Sleep f3 was diagnostic and completed. The detailed
measurement remains in `TUEV-DIAGNOSIS-2026-09-13.md` on the factorized branch.

CHB-MIT factorized f0 and f1 have both closed with ordinary patience after
four epochs: validation PR-AUC .587277 for zero content and .673026 for
band-local content (+.085749). The cohort and d192 recipe match except for
content source, identity/backfill and nonbinding time caps. This supports the
value of the band-local lane on CHB in one seed; broadband f2 is still running.
Historical d128 duplex scores .678728 and rotation .634410 on the same cohort,
so f1 is competitive but does not establish superiority over duplex or isolate
tokenization from model width. Allocation 416457 completed normally in 15:45:54
and released; f0's allocation 416484 subsequently released after its TUEV
confirmation finished, with the original packed-batch exit status noted above.
See `results/audits/chb-factorized-f1-completion-20260913.json`.

The CHB self-inclusive rotation run `chbmit-crofremo_s1`, seed 0, has now
completed with ordinary patience after seven epochs, validation PR-AUC
**.663388** and 12.456 training/evaluation hours. Its trainer exited with code
0. Allocation **416426** completed normally (0:0) after **19:04:06** and released
the node; no replacement allocation was requested.

Across four seed-0 configuration/cohort-matched self-inclusion contrasts,
CHB improves PR-AUC by .028978 and TUSZ by .005576, while TUEV kappa falls
.021605 and IIIC kappa falls .004542. Parameter counts match within each
contrast and all results stopped normally; historical source/runtime identity
is not established. These mixed directions do not justify expanding the
self-inclusion search. The CHB value also remains below historical duplex
seed-0 .678728, a different architectural contrast. Completion and cross-corpus
receipts are in `results/audits/chb-self-completion-20260913.json`.

Remaining CHB-MIT pilots continue or stop under their configured patience. Inspect
step validations as well as epoch ends. Their validation cadence differs from
TUEV; equal epoch numbers alone do not establish equal training progress.
Do not discard an entire model family from these seed-0 pilots.

The TUSZ factorized f2 pilot has now closed with ordinary patience after four
epochs. Validation PR-AUC is .376115 for broadband content, .377237 for zero
content and .349521 for band content; the other arms stopped after two and one
epochs respectively under the same patience policy. Recipes, d192 construction,
cohorts and 20,000-window validation caps match except content source, identity,
execution metadata and nonbinding time caps. Broadband has not surpassed zero
content in this seed. These capped-validation scores must not be ranked against
the full-validation 16-band TUSZ score .385660. No further content sweep is
admitted; details are in `results/audits/tusz-factorized-completion-20260913.json`.

A retrospective replay also shows why patience semantics cannot be silently
changed to save compute: stopping CHB f1 at its first 20-validation plateau
would retain .607448 rather than its later .673026 PR-AUC. Analogous missed
gains are .025978 for CHB duplex and .053972 for the TUSZ 16-band pilot.
These are replays of recorded curves, not new runs or proven compute savings.
Keep the existing epoch-boundary policy for current comparisons. A transient
plateau alone is insufficient evidence to discard a branch or erase its
in-memory best state. The same audit records all replay inputs and hashes.

The completed IIIC duplex+self-coupling control has validation kappa .547425
versus duplex .539373 (+.008053); the analogous TUEV seed-0 change is negative.
The added diagonal terms are not a self-only control. These single-seed results
remain mixed and do not justify a new sweep. Allocation 416453 completed with
exit code 0 and its controller released the node. Details are in
`results/audits/iiic-self-coupling-completion-20260913.json`.

Both 140k-step TUEG transfer gates have now completed all 50 finetuning epochs:
native validation kappa .403105 and coupling .424513. The coupling job 416830
completed normally in 5:21:22 and released its single-card allocation. Its
historical scratch counterpart uses a different duplex frontend, so this is
not a matched estimate of pretraining's effect. No new pretraining allocation
is justified by these gates. See `results/audits/cpl-140k-transfer-completion-20260913.json`.

Torch scheduling now defaults to `h200_public / torch_pr_63_general`. Slurm
rejected attempts to increase CPUs on pending jobs 17365674–17365677, which
retain their queue identities and eight CPUs. The three TUSZ transplant configs
now use eight workers and a 23-hour training cap for their 24-hour allocations;
all model, optimizer, loss and 50-epoch settings are unchanged. The two running
raw controls keep their already-loaded configuration. These experiments select
by AUROC, so maximum logged PR-AUC is not their selected checkpoint score.
Source hashes, atomic replacements/backups and scheduler responses are recorded
in `results/audits/torch-pending-cpus-20260913.json`.

The TUSZ 16-band rotation pilot has completed with ordinary patience after
three epochs. Its selected validation PR-AUC is .385660 versus .362138 for
the 8-band control (+.023523). Manifest timestamps, class counts and recipes
match except for band count and identity fields; both use the same patience
policy and finish below their wall-time caps. The 8-band control stopped after
two epochs, and historical source identity remains unverified. Combined with
the negative IIIC band-count result, this single-seed gain does not justify
another band sweep. Job 416484 still hosts other trainers; the completed child
does not release its node. See `results/audits/tusz-n1-completion-20260913.json`.

## Existing axisfree pretraining candidate

The inventory now includes both `cf2_v1d192` and `cf2_v1d192_ptR`, which were
previously absent from the monitor's family allowlist. The pretrained variant
now has eleven completed seed-0 datasets. Ten have matching manifest identities,
class counts and recipes against scratch after removing name/group/checkpoint:
four improve and six decline. TUEV kappa changes .521695 to .557578, while
TUAR changes .595250 to .526905. These single-seed observations do not establish
a stable pretraining benefit and do not justify more pretraining allocation.
TUSZ has equal counts but different manifest timestamps, so its apparent gain
is excluded from the matched count; the timestamp difference alone does not
prove different examples. Historical source/checkpoint provenance remains a
limitation. The added CHB-MIT result is PR-AUC .563290 versus scratch .691752
(-.128463), with an ordinary patience stop verified in the log and runtime
below its cap. Allocation 416043 completed normally and released. The initial
nine-pair audit and this additional completion are recorded in
`results/audits/axisfree-pretrain-validation-20260913.json` and
`results/audits/chb-axisfree-completion-20260913.json`.

TUAB axisfree pretraining transfer has now closed as a budget-censored result.
The log stops training at the 22.5-hour budget during epoch 3, step 200;
`stopped_by=time_budget` is already correct and no scores or metadata were
changed. `epochs_run=4` includes this interrupted fourth epoch, not four full
epochs. Allocation 416044 completed with exit code 0 in 22:46:46 and released.
Exclude this record from complete-training candidate ranking; no retry or
additional pretraining was admitted. See
`results/audits/tuab-axisfree-completion-20260913.json`.

## Historical readout check

The readout alternatives are already implemented and have historical results.
Configuration-matched three-seed attention pooling improves raw-token BCI-IV-2a
but reduces raw-token FACED validation performance in all three seeds. The
TUEV flagship seed-0 validation kappa .652144 combines 16 bands, a deep raw
stem, learned montage and gated mean/spatial pooling; it does not isolate the
head's effect against the duplex .609325 control. These records do not justify
relaunching pooling as an untested general remedy or freezing flagship.
Historical training-source identity remains unverified, including which
spatial-head runs used the eager-initialization fix. Pair identities, original
result hashes, metric names and configuration differences are preserved in
`results/audits/readout-history-20260913.json`. Continue the bounded factorized
content/scale confirmations before adding another readout sweep.

## Cross-dataset shortlist audit

The local `scripts/monitor/candidate_matrix.py` audit uses canonical result
identities and validation scores only. It excludes partial/budget runs,
training subsets, missing manifest identities and runs whose selected primary
validation score was not persisted. It refuses to average seeds with different
validation subsets, manifests, target counts or model construction. The JSON
artifact is `/Users/mr.z/PACLock-monitor/candidate-matrix.json`.

At this snapshot, rot2 has eligible completed evidence on 15 datasets, with
seeds 0/1/2 on 12; duplex has 16 datasets, with seeds 0/1/2 on 12. Ten datasets
have the same validation cohort and all three seeds for both families. Duplex
has the higher mean on seven (ADFD, CAUEEG, CHB-MIT, ISRUC, TUAR, TUEP, TUSZ);
rot2 has the higher mean on IIIC, Sleep-EDF and TUEV. These are descriptive
validation means, not significance tests or a SOTA comparison. TUSZ uses the
same 20k validation cap for both; the other paired rows use full validation.

This supports retaining a waveform-information lane in the shortlist while
addressing TUEV, rather than using TUEV alone to discard it. The new f1/f2
variants still have only seed-0 screening evidence and cannot replace that
broader comparison yet. Do not choose different content sources per dataset
and present them as one frozen candidate.

The historical rot2 label covers three configuration constructions: the main
11-corpus setup, index spatial embeddings on BCI-IV-2a/ISRUC, and 200-sample
patch/PAC windows on FACED/PhysioNet-MI. Duplex has two constructions differing
in spatial embedding choice. Configuration hashes do not establish historical
source identity. Final freeze must explicitly define the montage/patch policy.

Some official-loader baseline records lack manifest timestamps; others do not
persist a completion reason or selected primary validation metric. Exclusion
from this conservative audit is missing evidence, not evidence of a weak
baseline. These provenance gaps must be resolved before claiming near-SOTA
coverage; do not silently compare against only the baselines that remain.

The [TUEV protocol audit](TUEV-PROTOCOL-AUDIT-2026-09-13.md) confirms a substantive
cohort difference: frozen/LaBraM validation sets share only 9 of 58 subjects,
and test window counts are 29,421/26,067. This is more than a missing timestamp;
their historical validation scores must not be ranked as a matched comparison.
The native loaders now propagate manifests for future runs. Historical scores
and dataset files are preserved.

## Bounded scale comparison

The next controlled screen changes only the f1 local-lane multiplier: 1.0
versus 0.3, on TUEV and TUAR. The existing real-batch diagnostic found initial
coupling/local RMS .22795/.72937 on TUEV. A .3 multiplier brings the two lanes
near the same initial RMS while retaining all local coordinates. This tests
whether their relative scale contributes to the observed tradeoff; it does
not establish that scale caused the earlier performance decline.

All four configurations in `configs/factorized_scale/` use seed 0 and the
original 20-epoch recipe. Fresh controls run on the same mi2101x/rocBLAS setup
as the scaled arms, so a backend change is not silently attributed to scaling.
Prior f1 runtimes were 3.21 hours on TUEV and .97 on TUAR; approximately 8–10
single-card hours is an estimate for the paired screen, not an allocation-meter
conversion. Training caps are five hours per TUEV run and two per TUAR run.
Slurm limits are six and three hours respectively. A capped run cannot stand
in for a completed 20-epoch comparison.

`slurm/smoke_then_train.slurm` validates the output identity, checks a real
batch with finite gradients and parameters, and excludes two warmup steps.
Training starts only if the measured training-only estimate consumes less
than 80% of the configured training cap. This estimate excludes loader and
evaluation overhead. Per-job receipts preserve config/smoke hashes and commit.
These factorized experiments require the factorized branch; main does not
yet carry the experimental frontend. No B2 pretraining continuation is added.
One seed can motivate replication, not freeze or eliminate a model family.

Jobs **416840/416841** (TUEV, scale 1/.3) and **416842/416843** (TUAR, scale
1/.3) passed the single-card gate and launched training from `a832159`.
Measured steps were .312–.329 seconds, with 10.25 GiB peak allocated memory.
The revised training-only projection totals **9.88 single-card hours**;
loading/evaluation overhead is additional, so the earlier 8–10-hour total
estimate should not be treated as the measured total. The four training caps
sum to 14 hours and Slurm limits to 18 hours. Receipts and config-hash checks
are retained at `/Users/mr.z/PACLock-monitor/scale-screen-jobs.json`.

## Metric and identity safeguards

`best_val` is the **checkpoint-selection metric**, which is AUROC for some
CBraMod seizure experiments even when `primary_metric` is PR-AUC. The local
monitor records selection metric/score separately from the maximum logged
primary validation curve. It does not rank different metrics together or use
test scores to choose a candidate. Validation subsampling and data-manifest
identity also need to match before comparing recipes.

Four historical paths have misleading embedded run names: the three
`tuev-csbrain_pretrained_ls0` seeds and the archived
`_siena-paclock_duplex_focal_wrongloss/seed0`. The monitor flags and excludes
these aliases from canonical run identities, preserving their files. They
must not overwrite or be averaged with the canonical configuration.

## Three-cluster resource decision

- AMD: use the single-card/runtime rules in `AMD-SCHEDULING-2026-09-13.md`.
  Existing four-card allocations have available slots as pilots finish; use
  verified free slots for short diagnostics before requesting more nodes.
  Login-node activity is limited to scheduler/file access. Analysis and JSON
  aggregation run locally; model work runs inside allocations.
- Torch: two TUSZ additive-waveform seed replications are running; four paired
  coupling/transplant seeds are pending with `QOSGrpGRES`. They complete a
  useful attribution comparison and remain authorized. Result sync has been
  restored through the local monitor; each invocation fails visibly on rsync
  errors. New long jobs must not be assumed to start merely because GPUs look
  idle. Always use `h200_public` and `torch_pr_63_general`.
- B2: project continuation **45720335** was cancelled after verifying and
  exporting both native and coupling checkpoints at **140,000 steps**, inside
  its existing GPU allocation. Complete source checkpoints retain model,
  optimizer and scheduler state. The latest partial continuation beyond 140k
  is discarded; this is a budget pause, not a rejection from one seed. At the
  observed .93 seconds/step, 140k-to-219k would need about 20 additional GPU
  hours. Other shared-account jobs were not touched. Pretraining resume does
  not save RNG/sampler position, so it is not a bitwise continuation.

Exports are only about 20 MB per model, staged under
`pretrain_runs_cbramod/transfer_140k` on the factorized AMD worktree and Torch.
The 90-GB TUEG data stays on B2. Source validation, export sizes, hashes and the
cancelled-job accounting are retained by the local monitor. The initial paired
TUEV configs are `configs/pretrain_gate/tuev_{native,crofremo}_140k.yaml`.
Both exports passed strict backbone loading and five real TUEV training steps
in existing AMD allocation **416043**. Native matched 209 tensors and used
1.24 GiB at .073 seconds/step; coupling matched 202 tensors and used 8.01 GiB
at .325 seconds/step. Timings exclude two warmup steps. The sign-reversal probe
gave token relative L2 **.9839** on its masked batch: this implementation is
not sign-invariant in that probe. The proposed sign-invariance/raw-target
incompatibility is therefore not supported by this measurement.

`slurm/pretrain_gate.slurm native|crofremo` performs an individual GPU smoke
on mi2101x before starting each full 50-epoch seed-0 gate. Both configurations
have an 11-hour training cap within the site's 12-hour limit; a capped result
must be distinguished from a completed recipe. The roughly 68k training
windows and measured step times suggest the pair can fit independently on
single-card nodes; actual loader/evaluation overhead still needs observation.
Per-job smoke receipts preserve source/config/checkpoint hashes. A successful
load does not establish pretraining benefit; compare downstream validation.
The paired single-card jobs are **416829** (native) and **416830** (coupling),
submitted from factorized source commit `77aa69b`. Their run directories link
to canonical AMD `runs/` so the same passive monitor collects their progress.

## What changes the next decision

### Interim validation and control audit

An interim manually fetched snapshot contained 41 native and 9 coupling
validations in the 140k transfer pair. Through the same first 9 epochs, best
validation kappa is .403105 for native and .422662 for coupling. Neither has
completed its 50-epoch recipe at this observation. Native's historical scratch
run finished at .419841, but its runtime differs from the current run.

The historical `tuev-cbramod_crofremo_bands` scratch configuration is **duplex**,
whereas the current pretrained coupling configuration is **pac_interaction**
with rotation and bands represented as channels. The adapter forwards this
choice to `TriAxialFrontend`; these are distinct constructions. The historical
duplex result (.439489 at seed 0) is contextual evidence, not a matched scratch
control for the current coupling transfer. No completed exact counterpart was
found in the collected TUEV results. A promising transfer result would require
that counterpart before attributing a gain to pretraining or resuming B2 spend.

For the local-lane scale screen, equal-epoch best validation kappa is:

| Dataset | Completed epochs in both arms | Scale 1.0 | Scale 0.3 |
|---|---:|---:|---:|
| TUEV | 3 | .616866 | .644781 |
| TUAR | 11 | .598155 | .618194 |

Both pairs have matching manifests and class counts. They are unfinished,
seed-0 screens; retain the current bounded runs without expanding replication.
These observations do not select a final model or establish SOTA.

At the next observation, native transfer **416829** completed all **50 epochs**
with best validation kappa **.403105**, below the historical native scratch
seed-0 result **.419841**. Both have matching manifests and class counts, but
their runtime/backend and worker counts differ. This result does not justify
more native pretraining spend; it is not a causal rejection of pretraining
from one seed. Coupling transfer remains in progress (11 epochs, best .424513).
B2 remains paused and no replacement job is submitted for the freed allocation.

Secondary metrics also constrain the scale screen. Within the first four TUEV
epochs, the checkpoints selected by each arm's best kappa have balanced
accuracy **.631361** (scale 1) and **.569114** (scale .3). At fixed epoch index 2,
both kappa and balanced accuracy improve with scale .3 instead. This is an
unstable early checkpoint comparison, not evidence of a uniform improvement
or a stable minority-class regression. Continue the original bounded recipes
and report balanced accuracy and class recall at the kappa-selected checkpoint;
do not select separate checkpoints for each headline metric.

The TUAR scale pair subsequently completed all 20 epochs with matching data
manifests, class counts and recipes except run name and `content_scale`.
Scale .3 achieved validation kappa **.618194** versus **.598155** for scale 1;
balanced accuracy at those checkpoints was **.730853** versus **.731856**,
and weighted F1 **.782195** versus **.767187**. Both Slurm jobs completed with
exit code 0 and released their single-card nodes. Validation curves, source
hashes and scheduler receipts are in
`results/audits/tuar-scale-gate-20260913.json`. This is one completed positive
seed-0 screen. Retain scale .3 for confirmation; wait for full TUEV and additional
seeds before final selection or wider replication.

A bounded TUAR seed-1 pair is prepared for spare cards in existing mi2104x
allocations while the TUEV screen finishes. Each arm retains a two-hour training
cap and needs at least four hours left in the allocation. Real-batch smokes at
seed 1 passed finite-loss/gradient/parameter checks, measured about .22 seconds
per step after two warmups and used 10.31 GiB. Projected training alone is about
.8 hours per arm. Runtime differs from the mi2101x seed-0 pair, so compare the
new pair within its own runtime cohort rather than silently pooling results.
The admission plan and smoke/config hashes are in
`results/audits/tuar-seed1-confirm-plan-20260913.json`. No new Slurm allocation
is requested; added work can still extend the duration of an existing allocation.

Both seed-1 arms were admitted to existing allocation **416426**, on GPU 1
(scale .3) and GPU 2 (scale 1). Their processes and allocation membership were
verified. Admission receipts are in
`results/audits/tuar-seed1-confirm-admission-20260913.json`; source/config hashes
pin the comparison to branch commit `9665c7e`. No new Slurm job was submitted.

Both seed-1 arms subsequently completed all 20 epochs. Their validation kappa
was **.606875** (scale .3) versus **.608668** (scale 1), a difference of
**-.001793**. Balanced accuracy was .721983 versus .717209; weighted F1 was
.771903 versus .773720. Thus the positive TUAR kappa change observed at seed 0
did not reproduce in this seed/runtime cohort. Treat this pair as essentially
neutral and do not expand TUAR scale replication at this point. Keep the
existing TUEV pair running before deciding the shortlist. Full validation
curves and original-result checksums are recorded in
`results/audits/tuar-seed1-confirm-result-20260913.json`. Do not pool the two
runtime cohorts silently or claim a universal improvement from the first seed.

A source inspection found both TUEG and frozen TUEV preprocessing calling
`norm_div100`, with `WindowDataset` and the CBraMod classifier wrapper applying
no second scaling. This does not support a simple double-normalization or
unit-conversion explanation for weak transfer. Actual corpus amplitude
distributions were not measured in this inspection; no input rescaling or
additional pretraining was introduced on that hypothesis.

Direct process inspection found 19 training roots across the 12 active
four-card AMD allocations, with at least one trainer in every allocation.
The seven single-card allocations are separate. Idle cards alone are not a
reason to restart old trainers without optimizer checkpoints. No new work was
submitted in this audit. B2 has zero project jobs; Torch has two running and
four pending project jobs, now reporting `QOSMaxGRESPerUser`.

1. All three valid TUEV pilots have now completed 20 epochs. Compare their
   matching validation histories;
   retain the global f1/f2 tradeoff rather than choosing a different architecture
   for each dataset. Diagnose a proposed fix before expanding a sweep.
2. Validate the existing 140k pretraining exports against a matched native
   control. Check strict backbone loading, real-batch gradients/memory, timing
   after warmup, and whether the pretext target is compatible with tokenizer
   invariances. A load pass is not evidence of downstream improvement.
3. Run a bounded paired downstream gate using available compute, then replicate
   a promising shortlist with additional seeds and multiple corpora. Do not
   spend more B2 pretraining SUs until downstream evidence justifies continuation.
4. Before final freeze, audit one common architecture/source/config, multiple
   seeds, dataset/split/metric fidelity, budget stops, class behavior and the
   broader >10-dataset target. Preserve test evaluation as reporting evidence;
   do not use repeated best-test selection to manufacture a winning main row.

## TUEV effective input diversity and minority-class check

An exact-byte audit on the allocated compute node found **8,650 distinct
training inputs among 68,445 rows** and **1,028 validation inputs among 15,487
rows**. No identical input bytes crossed train and validation. Grouping uses
SHA-256 of the actual arrays; all class counts match the training manifest.
The read-only step `416422.4` completed in 10 seconds (9.01 seconds inside the
script), without a new allocation or any test-array access. Source revision
`2e1eccd` and full input/script hashes are recorded in
`results/audits/tuev-input-duplicates-20260913.json`.

SPSW has only **118 distinct training inputs** (526 rows) and **23 validation
inputs** (119 rows). The raw annotation audit attributes those rows to 20
training subjects and seven validation subjects. The three largest subjects
account for 44.49% and 68.91% of SPSW rows, respectively. Validation GPED is
even more concentrated: two subjects, one accounting for 95.88% of its rows.
The annotation audit exactly reconciles all train/validation class counts and
recording counts with the manifest; its window-key counts are slightly larger
than the exact-array counts and must not be substituted for them. See
`results/audits/tuev-annotation-multiplicity-20260913.json`.

Identical training inputs sometimes have different labels: 344 input groups,
covering 3,484 rows. For a fixed input-only classifier with smoothing .1, the
empirical row-weighted cross-entropy lower bound is **.436594**, rather than
the single-label smoothing entropy .420956. This is derived by averaging the
entropy of `.9 * empirical_label_distribution + .1 / 6` over input groups,
weighted by their row counts. Online training loss is measured while weights
and dropout change, so its proximity to this fixed-model bound is descriptive,
not an equality test or a generalization guarantee.

The current provisional f4 and both scale-control kappa-selected checkpoints
have zero SPSW recall. Across the observed f4 trajectory, even the highest
SPSW recall is .17647, with precision .08787; changing checkpoint selection
alone has not demonstrated a solution. The immutable interim histories and
derived checkpoint metrics are in
`results/audits/tuev-class-trajectory-snapshot-20260913.json`.
None of the 119 validation SPSW rows
shares an identical input with another label, so input-label contradiction
cannot explain away its zero recall. These observations favor investigating
generalization and effective sample diversity before merely increasing model
capacity. They do not establish that reweighting, deduplication, augmentation,
or pretraining will help. No training/evaluation data or recipe was changed;
the existing unweighted-CE benchmark and in-flight gates remain intact.

## Bounded augmentation gate, running

Historical r2/r4 results isolate the existing augmentation list at fixed
dropout .35 and weight decay .05 across seven matching seed-0 corpus pairs.
TUEV validation kappa rises .518495 to .609498 (+.091003). TUAR and ISRUC
decline by .005084 and .007002; Sleep-EDF and IIIC improve by .014195 and
.025303. ADFD and CAUEEG improve in **balanced accuracy**, their primary metric,
by .064558 and .012828. Do not average these mixed metrics. Recipes, selection
metrics, cohort identifiers and class counts match within each pair; historical
runtime/source identity is not established. Full checks and original hashes:
`results/audits/historical-augmentation-pairs-20260913.json`.

One bounded transfer of that existing augmentation list to f4 on TUEV
and TUAR at seed 1 is now running. Both keep f4's original model, dropout .2, weight decay
1e-5 and training recipe; only augmentation changes, apart from identity and
descriptive fields. This tests an augmentation-by-current-recipe interaction,
not a new architecture. Admission followed the full TUEV f4 completion and
validation-only review. Configured caps are five hours for TUEV and two for TUAR. The running
backfill controller actually applies `min(configured_cap, remaining_hours-1)`,
so require at least **5.5 hours remaining for TUEV** and **three for TUAR**.
This preserves at least 4.5/2 hours of training plus the controller's full
one-hour completion buffer. Both measured training projections fit within
80% of these minimum effective caps. Record any reduced execution cap and
exclude budget-stopped runs from completed-model comparisons. The prepared
plan is `results/audits/joint-augmentation-admission-plan-20260913.json`;
its original prepared-state record is retained.

Both tasks were admitted to existing allocation **416422**, with no new Slurm
job: TUAR on GPU 1 (PID 1362700, two-hour cap) and TUEV on GPU 2 (PID 1362701,
effective cap **4.57835 hours**). Owner, live process, configuration, allocation
cgroup and GPU selector were verified. Actual runtime configurations match the
smoked configs except recorded execution budget metadata. Source at admission
was `e519015`, with the model source unchanged from smoke revision `6d7f5de`.
Receipt: `results/audits/joint-augmentation-admission-20260913.json`.
Added work can extend this existing allocation. Do not admit further variants
or pretraining while this bounded pair is unresolved.

The training-only smoke passed on an already idle GPU in 416422, source
`6d7f5de`. It invokes each of the five configured augmentation modules, checks
finite gradients/parameters and learning in all three token lanes, and verifies
exact evaluation-time augmentation bypass. After two warmups, steps average
about .269 seconds with 10.31 GiB peak memory. Projected training alone is
3.20 hours on TUEV and .97 hours on TUAR; loading and evaluation are additional.
Receipts and source/config hashes are in
`results/audits/joint-augmentation-smoke-20260913.json`. No validation or test
arrays were used by this smoke, and no pretraining was started.

## Monitoring without model calls

`scripts/monitor/watch_clusters.py` runs locally every 900 seconds. It reads
scheduler state, project logs and result JSON, authenticates B2 using the
required expect helper, and invokes Torch's results-only sync. It separates
foreign B2 jobs, retains live handles after observation timeouts, reconciles
missing jobs with accounting, and reports new results, errors and identity
conflicts. It performs no training, automatic cancellation or LLM calls.

Current state is `/Users/mr.z/PACLock-monitor/{latest,summary}.json`; verify the
PID in `watch.pid` against the actual process before relying on the daemon.
The local machine/network and SSH sessions must remain available. Connection
failure is an alert, never evidence that training stopped. The active thread
goal remains open until the final-candidate evidence is sufficient.

## TUAR augmentation gate completed; TUEV pending

The TUAR f4 augmentation confirmation completed all 20 epochs normally in
0.973 hours, with trainer exit 0. At the validation-kappa-selected checkpoints:

| TUAR seed 1 | Validation kappa | Balanced accuracy | Weighted F1 |
|---|---:|---:|---:|
| Joint content, no augmentation | .615751 | .731577 | .773869 |
| Joint content, existing augmentation | .631884 | .755038 | .782292 |
| Broadband content, no augmentation | .630976 | .756006 | .783590 |

The matched joint pair differs only in augmentation plus identity/descriptive
and placement metadata. Manifests, class counts, parameter counts, time caps
and the remaining recipe match. Augmentation improves kappa by .016134,
balanced accuracy by .023461 and weighted F1 by .008423. Against broadband,
differences are +.000909, -.000968 and -.001298: close for this seed, not proof
of equivalence. That comparison changes both content and augmentation.

The result supports retaining augmentation as a recipe option, but does not
yet justify more datasets, a final candidate, or pretraining. Complete the
already-running TUEV counterpart before deciding on a second matched seed.
Trainer 1362700 has exited; allocation 416422 remains active for TUEV and CHB.
No new allocation was submitted. Full validation histories, configuration
checks, result hashes and the exit receipt are in
`results/audits/tuar-augmentation-completion-20260913.json`.

## Same-cohort, paired-seed baseline gaps

A local comparison now pairs each candidate with CBraMod, REVE and EEGPT
only where recorded validation cohorts agree and the same training seeds
exist. Means use that seed intersection; missing seeds do not become zeros.
Full per-seed scores, result hashes, construction metadata and missing-cell
reasons are in `results/audits/matched-baseline-gaps-20260913.json`.

Against the locally reproduced pretrained CBraMod, rotation-only has ten
complete three-seed pairs: six positive differences (ADFD, CAUEEG, IIIC, TUAB,
TUEP, TUEV) and four negative (BCI-IV-2a, ISRUC, Sleep-EDF, TUAR). Duplex has
nine such pairs: the same six positive datasets, with ISRUC, Sleep-EDF and
TUAR negative. These are descriptive validation comparisons across different
training recipes/pretraining budgets, not causal ablations or SOTA evidence.
No mixed-metric mean or retrospective near-SOTA threshold is defined.

FACED and PhysioNet-MI also show substantial gaps in the available incomplete
seed coverage, which calls for diagnosis and replication rather than family
elimination. New joint models have zero complete three-seed pairs. Historical
rotation/duplex coverage cannot be assigned to them. After the current TUEV
gate, a promising common joint recipe needs a second matched seed and sleep
corpora before wider expansion; retain the broader movement/affect gaps in
the final-model requirements. No training is admitted by this audit.

## Preserve metrics at the selected validation checkpoint

The trainer now writes `selected_validation` with the selection metric, tag,
zero-based validation index and all metrics from the exact evaluation that
selected `best_state`. The legacy `best_val` remains the selection score;
`val_curve` remains the primary-metric history. Optimizer, selection rule,
strict-tie behavior, patience and training budgets are unchanged. In particular,
a later higher PR-AUC cannot replace the PR-AUC paired with an AUROC-selected
checkpoint. No extra validation pass or training is needed.

The actual validation closures from both AMD and Torch source versions passed
a local controlled regression that separates AUROC and PR-AUC peaks, checks
selected weights and first-tie retention, and evaluates the serialized record.
Existing candidate-matrix tests still exclude legacy records without the
selected primary metric. This change preserves evidence for future consumers;
it does not relax their admission rules or rewrite historical results.

The minimal metadata patch was applied to both AMD worktrees and the distinct
Torch trainer without a broad pull. The four Torch pending jobs remained
pending before and after deployment and will load the added recording code.
The two already-running raw controls retain their loaded code. Tests, before
and after source hashes, queue evidence and scope are recorded in
`results/audits/selected-validation-recording-20260913.json`.

## Single-card CHB scheduling pilot closed at its budget

Job 416751 has completed with exit 0 after 11:06:13 and released its single-card
allocation. The configured 11-hour training budget triggered at epoch 3 step
200; the total duration includes validation, evaluation and saving around the
budget stop.
`epochs_run=4` means three complete epochs plus a truncated fourth. The selected
validation PR-AUC is .658491 at epoch 2 step 9800, with balanced accuracy .583333
and AUROC .972911. The near-budget improvement is retained in the checkpoint
history, but the run remains `scheduling_pilot` and `stopped_by=time_budget`.

Do not admit it into completed candidate ranking, relabel it as an uninterrupted
self-inclusion trial, or continue/restart it. The migration originally restarted
from seed without optimizer state and used the separate mi2101x software stack.
The original result, full validation history, migration provenance, log hashes,
pack exit receipt and Slurm closure are audited in
`results/audits/chb-single-pilot-completion-20260913.json`. AMD now has the two
remaining four-card allocations, 416422 and 416485; TUEV augmentation and the
other CHB trials continue under their existing caps.

## CBraMod task-head fidelity gap and prepared TUEV correction

A direct inspection of upstream revision `0ff6be918985689e7df679bc731ffb70e6c6224f`
found that the shared adapter had a fixed 800-wide hidden layer, while upstream
TUEV all_patch_reps uses 1000. The upstream CLI selects all_patch_reps by default.
At the same 16-channel, 1000-sample input and six-class output this is a difference
of 3,240,200 parameters. Upstream TUAB, CHB and FACED use 2000; Mumtaz uses 1000.
The old claim that 800 was the official TUAB head was incorrect and is removed.
Historical local CBraMod scores remain valid measurements of their recorded
adapter, but are not certified faithful upstream task reproductions. This
qualifies the earlier matched-baseline gaps; it must not be hidden by excluding
the affected baselines and reporting only favorable comparisons.

The adapter now accepts an explicit classifier_hidden_dim while retaining
800 for historical callers. A separate prepared configuration,
`configs/experiments/tuev_cbramod_pretrained_nativehead.yaml`, selects 1000 and
a new run identity. The original 50-epoch recipe remains otherwise intact.
This is a head correction only; it does not certify full native preprocessing
or optimization parity, and no baseline training has been submitted.

An allocated CPU check loaded the released pretrained backbone through the
actual builder, strictly copied the entire model state into upstream TUEV,
and obtained bit-identical logits on synthetic input. It also confirmed the
legacy default remains 800. Step 416422.6 completed with exit 0 in five seconds;
the check itself took .431 seconds, with no dataset arrays or training. TUEV
and CHB training stayed live in the allocation. The executed-source hashes and
post-check documentation-only edits are recorded separately, with identical
executable ASTs after removing docstrings. Evidence:
`results/audits/cbramod-tuev-head-parity-20260913.json` and
`results/audits/cbramod-native-head-inventory-20260913.json`.
