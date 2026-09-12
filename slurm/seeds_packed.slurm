#!/bin/bash
#SBATCH --job-name=paclock-seeds
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.out
# Backward-compatible entry point: three seeds now use three single-card nodes.
# Usage: sbatch slurm/seeds_packed.slurm config.yaml ["0 1 2"]
# For a different number of seeds, override --array=0-(N-1) as well.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
read -ra seeds <<< "${2:-0 1 2}"
seed=${seeds[${SLURM_ARRAY_TASK_ID:?array allocation required}]:?seed index out of range}
exec bash slurm/configs_packed.slurm "${1:?config.yaml}:$seed"
