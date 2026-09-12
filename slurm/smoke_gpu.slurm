#!/bin/bash
#SBATCH --job-name=smoke_gpu
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Usage: sbatch slurm/smoke_gpu.slurm <script.py> (one-card mi2101x node).
# Override -p mi3501x to use the MI350 module; inherited allocations use their own partition.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/slurm/amd_env.sh"
cd "$SLURM_SUBMIT_DIR"
HIP_VISIBLE_DEVICES=${SMOKE_GPU_ID:-0} MIOPEN_USER_DB_PATH=/tmp/miopen_smoke_${SLURM_JOB_ID}_${SLURM_STEP_ID:-batch} MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_smoke_${SLURM_JOB_ID}_${SLURM_STEP_ID:-batch} "$PACLOCK_PYTHON" -u "$@"
