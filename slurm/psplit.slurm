#!/bin/bash
#SBATCH --job-name=psplit
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=00:08:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -u -m scripts.param_split
