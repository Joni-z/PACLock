#!/bin/bash
#SBATCH --job-name=FEAT_siena
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:29:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
for seed in 0 1 2; do python -m scripts.feat_baseline --dataset siena --seed $seed || echo "FAILED siena seed$seed"; done
echo ALLDONE
