#!/bin/bash
#SBATCH --job-name=verify-pac
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -u scripts/verify_pac_decouple.py
