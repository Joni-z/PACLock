# AMD allocation and scheduling, 2026-09-13

AMD is allocation-limited. The site submission filter currently reports
`Used 1786.35 of 2250`, expiring 2027-06-30: 463.65 remaining before subsequent
usage is posted. Running cancellations consume elapsed allocation time;
cancelling a pending job consumes no runtime. Releasing one trainer while
retaining its exclusive node does not release that node's allocation.

## Node choice

| Partition | GPUs per exclusive node | CPUs to request | Configured NODE billing weight | Status |
|---|---:|---:|---:|---|
| mi2101x | 1 MI210 | 16 | 0.01 | Account accepted; runtime needs rocBLAS override |
| mi2104x | 4 MI210 | 128 | 0.04 | Existing runtime; require at least 4 independent configs |
| mi2508x | 8 logical devices | 128 | 0.08 | Submission test accepted; no idle allocation for hardware smoke |
| mi3501x | 1 MI350 | 24 | 0.0125 | Site submission filter rejects this account |

These are `scontrol show partition` weights, not a verified conversion of the
site's external balance meter. In particular, the eight-card weight is twice
the four-card weight: filling eight slots does not establish a twofold cost
saving. There are no GPU GRES on these partitions. Do not request `--gres` or
`--gpus`; select the partition, CPU count, one node and `--exclusive`.

## Commands

From the repository root, smoke the intended configuration before training:

```bash
sbatch -J smoke_one slurm/smoke_gpu.slurm smoke/smoke_amd_partition.py \
  --config configs/selfcoup/chbmit_crofremo_s2.yaml
python3 scripts/slurm/submit_packed.py --dry-run configs/selfcoup/chbmit_crofremo_s2.yaml:0
python3 scripts/slurm/submit_packed.py -J NAME --time 20:00:00 cfg.yaml:0
```

The submitter defaults to mi2101x and submits each configuration separately,
allowing its node to release independently. It checks configuration validity,
disabled experiments, duplicate outputs, and existing result/checkpoint/stop
artifacts before reserving a node. It never submits in `--dry-run` mode.

```bash
# Equivalent direct one-card submission:
sbatch slurm/train.slurm cfg.yaml 0
# Seeds 0, 1, 2, each on its own single-card node:
sbatch slurm/seeds_single.slurm cfg.yaml
# Legacy seeds_packed entry point now has the same one-card array default.
# Four-card pack:
SEED=0 sbatch -J NAME slurm/configs_packed.slurm cfg1 cfg2 cfg3 cfg4
# Eight-card pack (hardware smoke required before a real training launch):
python3 scripts/slurm/submit_packed.py -p mi2508x -J NAME --time 08:00:00 \
  cfg1 cfg2 cfg3 cfg4 cfg5 cfg6 cfg7 cfg8
```

The runner starts one trainer per visible GPU and refills completed slots
from additional configurations. It refuses underfilled multi-card packs and
checks the actual visible device count against the partition. Each trainer
has a separate log and MIOpen cache; pack receipts record GPU assignments,
exit codes, pending work, Python/PyTorch/ROCm, host, and commit. Single-card
jobs remain preferable for configurations with very different durations.

`B:TERM@120` forwards a wall-time warning to the trainer. The monitored trainer
saves best model weights and progress and handles cooperative stop. These are
selection checkpoints, **not optimizer-resume checkpoints**. Existing processes
started before the monitoring change do not gain the feature from a source
edit. Migration of those processes requires a restart, not `scontrol requeue`
presented as a resume. Preserve their logs and record the restart explicitly.

## Runtime findings

Torch 2.7.1 is compiled for CPython **3.9**, while 2.10.0 targets CPython 3.12.
The single-card MI210 VM lacks the old module's hardcoded rocm/6.3.1 dependency.
`amd_env.sh` uses available rocm/6.4.1 with the same shared Torch 2.7.1 wheel
and its bundled ROCm 6.3 libraries. Four-card nodes use their existing module.

The MI210 VM's default hipBLASLt path fails in matmul and linear layers.
The single-card environment selects rocBLAS through the package's conditional
runtime initialization, sets `ROCBLAS_USE_HIPBLASLT=0`, and disables the fused
Lt addmm path with `DISABLE_ADDMM_CUDA_LT=1`. The latter is implemented in
[PyTorch 2.7.1 Blas.cpp](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/Blas.cpp);
the backend selector is described in [PyTorch's backend documentation](https://docs.pytorch.org/docs/main/backends.html).
This changes the numerical backend and is recorded in new run receipts.

Torch 2.10.0 also failed on the MI210 VM and is not the default there. Isolated
CPython 3.12 supplemental dependencies were installed under
`/work1/chenyuyou/yifanwang/Zhizhe/PACLock-runtime/py312-v1`; their pinned list is
`slurm/amd-py312-requirements.txt`. MI350 support remains unverified because
the account cannot submit to that partition. No dependencies were installed
over the cluster's shared Torch trees. All GPU calculations ran in SLURM jobs.

## Verification record

- CPU scheduler tests cover 1/4/8-card plans, duplicate/disabled/existing output
  rejection, eight concurrent slots, ten-config refill, and failure propagation.
- Eight-card submission `--test-only` was accepted. This did not allocate GPUs.
- Short runtime probe 416739 passed GPU matmul, convolution, FFT and backward
  with explicit rocBLAS. Full model and trainer smoke results are recorded below.
- Full CHB-MIT duplex/self-coupling model smoke **416745 passed**, batch
  `(32,16,2000)`, five optimizer steps. After excluding two warmup steps, step
  times were 0.668/0.668/0.665 seconds; peak allocated GPU memory was 26.95 GiB.
- Actual trainer smoke **416746 passed** on mi2101x: normal and monitored runs
  give identical metrics, SIGTERM and STOP preserve selection weights without
  scoring the test split, checkpoints reload, inherited workers terminate,
  binary heads work, and existing/disabled runs are rejected.
