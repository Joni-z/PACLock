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

The prepared TUEV seed-1 confirmation pair is now **running** on mi2104x:
scale 1.0 on allocation 416422 GPU 1, and scale .3 on 416484 GPU 3. Both use
20 epochs and a five-hour training cap; the runtime configurations match the
GPU-smoked configurations after removing execution metadata. Owner, process,
configuration, GPU selector and Slurm cgroup were verified. The model-source
diff since their earlier smoke only adds the unused joint-content branch;
the existing band-content arithmetic is unchanged. Compare the paired arms
within this mi2104x cohort. No new Slurm allocation was requested, although
added work can extend existing allocation duration. The process receipt is
`results/audits/tuev-seed1-pair-admission-20260913.json`.

The fixed-width joint-content alternative, f4, is now **running** as a bounded
TUEV/TUAR seed-1 trial after both admission gates were reviewed. Both run on
existing allocation 416422: TUEV on GPU 3 and TUAR on GPU 2. The TUEV band
control is running on GPU 1 of the same allocation; the TUAR band control is
already complete. Runtime configs match their smokes, and owner/process/GPU/
cgroup checks passed. Model source matches smoke commit `1ac4ddd` exactly.
No new allocation or pretraining job was requested. Training caps are five
hours for TUEV and two for TUAR; added work may extend allocation duration.
See `results/audits/factorized-joint-admission-20260913.json`.

f4 keeps 128 coupling coordinates and splits the existing 64 content coordinates
into 32 band-local and 32 broadband coordinates, with unchanged parameterization
and initialization. Each source has less capacity than its 64-coordinate control;
this is not a signal-invertibility claim. Contract checks cover exact coordinates,
finite random/flat/near-flat gradients, three-lane trainability and checkpoint
reload. Smokes measured about .257/.256 s per TUEV/TUAR step after two warmups,
10.31 GiB peak memory, and 3.05/.92 hours of projected training without loading
or evaluation. Performance results for f4 remain pending.

The 200-Hz f3 implementation remains blocked after its realized lowest filter
peaked at DC with gain 70.156. That invalidates the intended low-frequency
contrast; the completed TUAR f3 record is retained but excluded from candidate
selection. The 100-Hz Sleep f3 was diagnostic and completed. The detailed
measurement remains in `TUEV-DIAGNOSIS-2026-09-13.md` on the factorized branch.

CHB-MIT/TUSZ pilots continue or stop under their configured patience. Inspect
step validations as well as epoch ends. Their validation cadence differs from
TUEV; equal epoch numbers alone do not establish equal training progress.
Do not discard an entire model family from these seed-0 pilots.

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
