#!/bin/bash
#SBATCH --job-name=bench_fe
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -u scripts/bench_frontend.py
