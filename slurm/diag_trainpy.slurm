#!/bin/bash
#SBATCH --job-name=diag_trainpy
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
date +%s
HIP_VISIBLE_DEVICES=0 python3 -u -m paclock_bench.training.train --config configs/_prof/trainpy.yaml --seed 0
