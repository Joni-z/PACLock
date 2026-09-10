#!/bin/bash
#SBATCH --job-name=amd_pretrain
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Usage: sbatch -J name slurm/amd_pretrain.slurm <module> [args...]   (one exclusive node; use --dp to span its 4 GPUs)
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
export MIOPEN_USER_DB_PATH=/tmp/miopen_pre_$SLURM_JOB_ID MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_pre_$SLURM_JOB_ID
python3 -u -m "$@"
