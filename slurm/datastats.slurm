#!/bin/bash
#SBATCH --job-name=datastats
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
# devel nodes cannot load the ROCm-based pytorch module and this needs only numpy
cd "$SLURM_SUBMIT_DIR"
python3 -u scripts/datastats.py
