#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Run any module in this repo on a compute node.
#
#     sbatch --job-name=smoke slurm/run.slurm scripts.smoke_paths
#     sbatch --job-name=bench --time=1:00:00 slurm/run.slurm scripts.bench_ab
#     sbatch slurm/run.slurm scripts.verify_frontend
#
# Replaces eighteen files that were this file with one module name changed.
# Nothing heavier than an editor belongs on the login node -- that includes
# anything that imports torch or touches the preprocessed arrays.
set -euo pipefail
MODULE=${1:?usage: sbatch slurm/run.slurm <module.path> [args...]}
shift
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 -u -m "$MODULE" "$@"
