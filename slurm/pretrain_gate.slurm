#!/bin/bash
#SBATCH --job-name=pt140k-gate
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Usage: sbatch slurm/pretrain_gate.slurm native|crofremo
set -euo pipefail
kind=${1:?native or crofremo required}
case "$kind" in native|crofremo) ;; *) exit 2 ;; esac
[[ "$SLURM_JOB_PARTITION" == mi2101x ]] || exit 2
cd "$SLURM_SUBMIT_DIR"
source slurm/amd_env.sh
export HIP_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MIOPEN_USER_DB_PATH="/tmp/miopen_gate_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
config="configs/pretrain_gate/tuev_${kind}_140k.yaml"
# Check output identity before smoke; all model work stays in the allocation.
"$PACLOCK_PYTHON" scripts/slurm/run_packed.py --plan-only "$config:0"
"$PACLOCK_PYTHON" -u smoke/smoke_pretrain_140k.py --kind "$kind" \
  --out "results/pretrain_140k_smoke_${SLURM_JOB_ID}.json"
# An unsuccessful load/gradient smoke exits before any full fine-tuning.
exec bash slurm/configs_packed.slurm "$config:0"
