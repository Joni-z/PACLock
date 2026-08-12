#!/bin/bash
#SBATCH --job-name=paclock-train
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Usage: sbatch slurm/train.slurm configs/experiments/<exp>.yaml [seed]
set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/train.slurm <config.yaml> [seed]}
SEED=${2:-0}

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
mkdir -p logs runs

echo "host=$(hostname) config=$CONFIG seed=$SEED"
rocm-smi --showproductname 2>/dev/null | grep -E "Card Series|GFX Version" || true

python3 -m paclock_bench.training.train --config "$CONFIG" --seed "$SEED"
