# Candidate decision and resource gates — 2026-09-13

**No final candidate is frozen.** The goal remains one defensible model with
SOTA or near-SOTA performance across more than ten datasets and a coherent
PAC/tokenizer explanation. A stronger isolated seed or a finished smoke test
is not sufficient to declare that goal achieved.

## Current validation evidence

The completed seed-0 factorized pilots on Sleep-EDF and TUAR favor f2:

| Dataset | f0: coupling control | f1: band-local lane | f2: broadband-local lane |
|---|---:|---:|---:|
| Sleep-EDF, completed 20 epochs | .6403 | .6416 | .6729 |
| TUAR, completed 20 epochs | .6049 | .6014 | .6231 |
| ISRUC, live through epoch 6 | .7133 | .7211 | .7580 |
| TUEV, f0/f1 through epoch 16; f2 completed 20 epochs | .5920 | .6294 | .6006 |

These are best logged **validation kappa**, not test scores. Completed pilots
have one seed; live rows are provisional. f2 has useful evidence on both sleep
and artifact corpora, but does not resolve the main TUEV weakness. f1 is closer
on TUEV: its seed-0 .6294 compares with historical rotation-only seed-0 .6339.
Historical rotation-only has three-seed validation mean .6504; duplex .6188.
Do not compare a new .64 validation number with an old .73 test number.

The 200-Hz f3 implementation remains blocked after its realized lowest filter
peaked at DC with gain 70.156. That invalidates the intended low-frequency
contrast; the completed TUAR f3 record is retained but excluded from candidate
selection. The 100-Hz Sleep f3 was diagnostic and completed. The detailed
measurement remains in `TUEV-DIAGNOSIS-2026-09-13.md` on the factorized branch.

CHB-MIT/TUSZ pilots continue or stop under their configured patience. Inspect
step validations as well as epoch ends. Their validation cadence differs from
TUEV; equal epoch numbers alone do not establish equal training progress.
Do not discard an entire model family from these seed-0 pilots.

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

## What changes the next decision

1. Finish the imminent TUEV pilots and compare matching validation histories;
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
