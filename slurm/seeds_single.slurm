#!/bin/bash
#SBATCH --job-name=paclock-seeds
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --array=0-2
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Three independent seeds; each one-card node releases as soon as its seed ends.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec bash slurm/configs_packed.slurm "${1:?config.yaml}:${SLURM_ARRAY_TASK_ID}"
