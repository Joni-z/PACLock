#!/bin/bash
#SBATCH --job-name=params
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=logs/params-%j.out
#SBATCH --error=logs/params-%j.out
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -m tests.test_baseline_params
