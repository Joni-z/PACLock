# Factorized token content, 2026-09-13

## Objective and hypothesis

The target is one model competitive on at least ten corpora, with TUEV a
critical event-classification result. No new results are claimed yet.

Historical rot2 achieves TUEV kappa .7328 ± .0161 (three seeds), while duplex
achieves .6895 ± .0357. CHB-MIT favors duplex in the currently available runs.
Adding waveform content by an early sum has previously hurt TUEV; adding rows
also changes attention and readout. These observations motivate testing a
different location for the two contents, not a proven causal explanation.

The new token is `[interaction_128, local_64]`: one row per electrode/band/patch,
192 coordinates, original 128-dimensional rotation interaction untouched at
the tokenizer output. A linear projection of 50 waveform samples occupies
the other 64 coordinates. There is no learned sum, gate or extra token row.
Both lanes receive supervised gradients from the first step. The encoder
uses three-axis attention and learns their interactions after tokenization.

Only the projected features are retained as separate coordinates. The model
is not guaranteed to preserve the full input, biological PAC, or legacy
classification performance. The raw lane is phase-reference sensitive; an
invariance claim applies only to the nondegenerate coupling subspace with
the same lowest-band/fallback caveats as rot2. The encoder's LayerNorm and
attention mix coordinates, so eliminating an early sum does not eliminate
all possible interference.

## Controlled arms

| Arm | Coupling | Local coordinates | Filter initialization |
|---|---|---|---|
| f0 | 128-dimensional rot2 | 64 zeros | legacy |
| f1 | same | 64 band-waveform features | legacy |
| f2 | same | 64 broadband-waveform features | legacy |
| f3 | same | 64 band-waveform features | effective .5–min(75, Nyquist−1) Hz |

All four instantiate the local projector (including f0) and use the same
192-dimensional encoder, random-number initialization order and per-corpus
recipe. f0 controls the extra encoder width, but its zeroed projector has no
effective capacity; describe this explicitly. f1–f0 measures adding local
content at this width. f2–f1 tests input coverage. f3–f1 tests filter coverage.
Legacy sinc parameters start at f_min=2 plus min_low=1, so the first nominal
lower edge is 3 Hz; this is not a brick-wall absence of sub-3-Hz response.
f3 uses offset-corrected parameters and validates the effective edges; Sleep-EDF at 100 Hz uses .5–49 Hz. Its
201-tap filter still has finite frequency resolution.

The biological interpretation is deliberately limited: nonsinusoidal shape
can affect PAC measurements, so classification gains alone cannot distinguish
physiological coupling from useful morphology. See the primary study
[Discriminating Valid from Spurious Indices of Phase-Amplitude Coupling](https://pubmed.ncbi.nlm.nih.gov/28101528/).
The [CBraMod implementation](https://github.com/wjq-learning/CBraMod) is a
reference for complementary temporal/spectral content, not evidence this
particular new representation improves performance.

## Evaluation and execution

Pilot: TUEV, CHB-MIT, TUSZ, TUAR, Sleep-EDF, ISRUC; four arms each, seed 0,
one four-GPU packed node per corpus. This covers the critical event task,
two seizure tasks, artifact morphology and two sleep tasks. Configurations
for all 12 corpora are prepared, but only the six-corpus pilot is submitted.
No seed is selected by test performance and no family is definitively
eliminated from a single seed. Checkpoint selection uses existing validation
metrics. Inspect validation trajectories and resource cost; multi-seed
confirmation of promising comparisons precedes a final model decision.
The ten-corpus target cannot be fulfilled by cherry-picking different arms
per dataset. Already inspected test sets are development evidence, not
fresh architecture holdouts.

Batch, loss, learning rate, weight decay, patch length and epoch/early-stop
policy match recovered rot2 per corpus. The unsubmitted Siena configurations
use the documented batch 128/60-epoch recipe instead of the recovered batch-32
configuration; a matched baseline is required if that corpus is activated.

New implementation is isolated in the `codex/factorized-tokenizer-20260913`
worktree at `../PACLock-factorized`, leaving pending legacy jobs on their
original source. Existing raw/duplex/rot2 configurations keep their original
frontend class and arithmetic. The n2 dead-stem issue is handled by the
previous cancellation; this new frontend rejects unsupported raw_stem rather
than accepting another no-op setting.

`smoke/smoke_factorized.py` asserts exact legacy coupling coordinates, exact
local projections, row counts, finite outputs/gradients on flat and random
signals, effective filter edges, learning in both lanes and strict checkpoint
round trip. It then runs four complete optimization steps for all 24 pilot
configurations on real training batches. Timing excludes the first two steps.
No training arrays or torch are loaded on a login node.

The smoke runs through `slurm/smoke_gpu.slurm`, with `SMOKE_GPU_ID` permitting
an explicitly verified idle GPU inside an existing SLURM allocation. The
default dedicated-smoke behavior still selects GPU 0. Launch requires a
successful JSON receipt containing hashes of tested source and configuration
files and all 24 timing records; code changes invalidate that receipt.
`scripts/launch_factorized.py` records each submission and refuses a duplicate
launch. It submits exactly four configurations per allocated node.
