#!/bin/bash
#SBATCH --job-name=prof-loader
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
for w in 0 8 16; do
  echo "### num_workers=$w"
  python3 -u scripts/profile_loader.py configs/_bench/E_ram_b32.yaml $w
done
