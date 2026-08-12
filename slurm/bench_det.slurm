#!/bin/bash
#SBATCH --job-name=bench_det
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
for m in on off bench; do
  HIP_VISIBLE_DEVICES=0 DET_MODE=$m python3 -u -m scripts.bench_determinism
done
