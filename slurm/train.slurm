#!/bin/bash
#SBATCH --job-name=paclock-train
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Usage: sbatch slurm/train.slurm configs/experiments/<exp>.yaml [seed]
set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/train.slurm <config.yaml> [seed]}
SEED=${2:-0}
cd "$SLURM_SUBMIT_DIR"
# Reuse the same validation, receipt and signal handling as packed jobs.
exec bash slurm/configs_packed.slurm "$CONFIG:$SEED"
