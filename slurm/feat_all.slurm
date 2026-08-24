#!/bin/bash
#SBATCH --job-name=FEAT_all
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
for ds in tuev tusz chbmit tuab tuep adfd; do
  for seed in 0 1 2; do
    echo "=== $ds seed $seed ==="
    python -m scripts.feat_baseline --dataset $ds --seed $seed || echo "FAILED $ds seed$seed"
  done
done
echo ALLDONE
