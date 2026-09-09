#!/bin/bash
#SBATCH --job-name=smoke_gpu
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Usage: sbatch slurm/smoke_gpu.slurm <script.py>  -- one-GPU smoke test on a compute node (never on the login node)
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
HIP_VISIBLE_DEVICES=0 MIOPEN_USER_DB_PATH=/tmp/miopen_smoke_$SLURM_JOB_ID MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_smoke_$SLURM_JOB_ID python3 -u "$@"
