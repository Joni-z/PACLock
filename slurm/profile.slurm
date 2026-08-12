#!/bin/bash
#SBATCH --job-name=profile-step
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
export MIOPEN_USER_DB_PATH=/tmp/miopen_prof MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_prof
echo "### batch 32"; python3 -u scripts/profile_step.py configs/_bench/E_ram_b32.yaml 30
echo; echo "### batch 128"; python3 -u scripts/profile_step.py configs/_bench/F_ram_b128.yaml 15
