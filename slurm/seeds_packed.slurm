#!/bin/bash
#SBATCH --job-name=paclock-seeds-packed
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Usage: sbatch slurm/seeds_packed.slurm configs/experiments/<exp>.yaml
#
# Runs all three seeds of one cell CONCURRENTLY on one node, one GPU each.
#
# Why this rather than the 3-task array in seeds.slurm: this cluster does not
# expose GPUs as a Slurm GRES (`scontrol show node` reports Gres=(null)), so
# `--gpus=1` is rejected and the only way to get a GPU is `--exclusive`, which
# hands over the whole node. An mi2104x node carries **four** MI210s, so the
# array form was taking three whole nodes and leaving 9 of the 12 GPUs idle.
#
# Nothing about the numerics changes: each seed is still an independent
# single-GPU process with its own HIP_VISIBLE_DEVICES, exactly as before. This
# is not data parallelism -- the batch size, and therefore the optimisation
# trajectory, is untouched. That matters here because the whole point of the
# rerun is to stop deviating from the validated recipe.
#
# Measured: PACLock/TUEV is 99.6% GPU-bound (0.2% dataloading, 0.2% H2D), so
# three processes on three distinct GPUs do not contend for the host, and 32
# CPUs covers 3 x 8 dataloader workers.
set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/seeds_packed.slurm <config.yaml>}
SEEDS=${2:-"0 1 2"}

module load pytorch/2.7.1

cd "$SLURM_SUBMIT_DIR"

pids=()
for s in $SEEDS; do
    # MIOpen caches tuned kernels in a SQLite DB; concurrent processes sharing
    # one file over the shared filesystem corrupt it ("database disk image is
    # malformed"). Per-process, node-local cache dirs keep them apart.
    HIP_VISIBLE_DEVICES=$s \
    MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}_${s}" \
    MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen_${SLURM_JOB_ID}_${s}" \
    python3 -u -m paclock_bench.training.train \
        --config "$CONFIG" --seed "$s" \
        > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-seed${s}.out" 2>&1 &
    pids+=($!)
    echo "launched seed $s on GPU $s (pid ${pids[-1]})"
done

# Wait for all three and fail the job if any seed failed, so a partially
# completed cell cannot look like a success to the table writer.
rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
echo "all seeds finished, rc=$rc"
exit $rc
