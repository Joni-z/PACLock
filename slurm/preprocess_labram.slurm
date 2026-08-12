#!/bin/bash
#SBATCH --job-name=prep-labram
#SBATCH --partition=mi2104x
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
DATASET=${1:?usage: sbatch slurm/preprocess_labram.slurm <tuab|tuev>}
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -m preprocessing.labram_native \
    --dataset "$DATASET" \
    --out /work1/chenyuyou/yifanwang/Zhizhe/processed_labram/"$DATASET" \
    --jobs "$SLURM_CPUS_PER_TASK"
