#!/bin/bash
#SBATCH --job-name=prep_labram_tuev
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -u -m preprocessing.labram_native --dataset tuev --out /work1/chenyuyou/yifanwang/Zhizhe/processed_labram/tuev --jobs 32
