#!/bin/bash
#SBATCH --job-name=crofremo
#SBATCH --account=torch_pr_63_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# NYU Torch launcher: ONE config on ONE GPU per job (Torch schedules single-GPU
# jobs far sooner than whole nodes, the opposite of amd where --exclusive is
# forced and packing four processes per node is the only way not to waste it).
#
# Usage:
#   sbatch slurm/torch_run.slurm <cfg.yaml> [seed]            # one seed
#   sbatch --array=0-2 slurm/torch_run.slurm <cfg.yaml>       # seeds 0 1 2
#   sbatch -p h200_public -A torch_pr_63_general slurm/torch_run.slurm <cfg>
#
# Accounts/partitions this user can reach (checked 2026-09-02):
#   torch_pr_63_tandon_advanced -> h100_tandon, h200_tandon, a100_tandon
#   torch_pr_63_general         -> h200_public, l40s_public
# Job QOS is set by walltime: <=48h -> gpu48 (16 GPUs/user), <=7d -> gpu168 (4).
set -euo pipefail

CONFIG=${1:?usage: sbatch slurm/torch_run.slurm <config.yaml> [seed]}
SEED=${2:-${SLURM_ARRAY_TASK_ID:-0}}

source /scratch/zz5070/miniconda3/etc/profile.d/conda.sh
conda activate py312
export PACLOCK_DATA=${PACLOCK_DATA:-/scratch/zz5070/data/raw}
export PACLOCK_PROC=${PACLOCK_PROC:-/scratch/zz5070/data}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "host=$(hostname) gpu=$(nvidia-smi -L | head -1) config=$CONFIG seed=$SEED"
python3 -u -m paclock_bench.training.train --config "$CONFIG" --seed "$SEED"
