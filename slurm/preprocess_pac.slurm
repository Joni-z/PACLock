#!/bin/bash
#SBATCH --job-name=prep-pac
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Usage: sbatch slurm/preprocess.slurm <dataset>
# CPU-only, but the CPU partitions are the same nodes; no GPU is requested.
set -euo pipefail

DATASET=${1:?usage: sbatch slurm/preprocess.slurm <dataset>}

# MIOpen caches tuned conv kernels in a SQLite DB under $HOME by default.
# Concurrent array tasks then write the same file over a shared filesystem and
# corrupt it -- observed as "database disk image is malformed" /
# miopenStatusInternalError, and as a bare Bus error. Give every task its own
# cache on node-local disk. Costs a short re-tune per job; buys determinism.
export MIOPEN_USER_DB_PATH="/tmp/miopen-$USER-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT

module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

python3 -m preprocessing."$DATASET" \
    --config configs/datasets_pac/"$DATASET".yaml \
    --jobs "$SLURM_CPUS_PER_TASK"
