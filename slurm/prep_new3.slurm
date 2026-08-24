#!/bin/bash
#SBATCH --job-name=prep_iiic_caueeg
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
echo "== iiic =="
python -m preprocessing.iiic --config configs/datasets/iiic.yaml
echo "== caueeg =="
python -m preprocessing.caueeg --config configs/datasets/caueeg.yaml
echo "== sizes =="
du -sh /work1/chenyuyou/yifanwang/Zhizhe/processed/iiic /work1/chenyuyou/yifanwang/Zhizhe/processed/caueeg
df -h /work1/chenyuyou/yifanwang | tail -1
echo DONE
