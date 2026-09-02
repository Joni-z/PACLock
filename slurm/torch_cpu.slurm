#!/bin/bash
#SBATCH --job-name=crofremo-cpu
#SBATCH --account=torch_pr_63_general
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# NYU Torch CPU job (preprocessing, downloads, audits).
# Usage: sbatch slurm/torch_cpu.slurm <command...>
#   e.g. sbatch slurm/torch_cpu.slurm python3 -m preprocessing.tuev --config configs/datasets/tuev.yaml
set -euo pipefail
source /scratch/zz5070/miniconda3/etc/profile.d/conda.sh
conda activate py312
export PACLOCK_DATA=${PACLOCK_DATA:-/scratch/zz5070/data/raw}
export PACLOCK_PROC=${PACLOCK_PROC:-/scratch/zz5070/data}
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "host=$(hostname) cpus=$SLURM_CPUS_PER_TASK cmd=$*"
"$@"
