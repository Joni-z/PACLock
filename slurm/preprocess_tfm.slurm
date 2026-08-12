#!/bin/bash
#SBATCH --job-name=prep-tfm
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# BIOT-native preprocessing for group B.
#   sbatch slurm/preprocess_biot.slurm <tuab|tuev>
set -euo pipefail
DATASET=${1:?usage: sbatch slurm/preprocess_biot.slurm <tuab|tuev>}
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -m preprocessing.tfm_native \
    --dataset "$DATASET" \
    --out /work1/chenyuyou/yifanwang/Zhizhe/processed_tfm/"$DATASET" \
    --jobs "$SLURM_CPUS_PER_TASK"
