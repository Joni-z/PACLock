#!/bin/bash
#SBATCH --job-name=smoke_pt
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --time=00:12:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -u -m scripts.smoke_pretrain
python3 -u -m scripts.smoke_cand
