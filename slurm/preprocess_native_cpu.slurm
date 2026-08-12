#!/bin/bash
#SBATCH --job-name=prep-native
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
# mi2101x is 384 CPUs across 24 nodes -- 16 per node, so asking for the 32
# the GPU-partition scripts use is rejected outright.
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Usage: sbatch slurm/preprocess_native_cpu.slurm <biot|labram> <tuab|tuev|tusz>
#
# Same work as preprocess_biot.slurm / preprocess_labram.slurm, moved off the
# GPU partitions. Neither biot_native.py nor labram_native.py imports torch --
# they are mne/scipy/numpy over EDF files, entirely CPU-bound -- so running them
# on mi2104x meant holding four MI210s idle for hours while the training queue
# waited behind them. mi2101x cannot load the ROCm pytorch module, which is
# exactly why it is free, and the system python already carries numpy 1.26.4,
# scipy 1.13.1 and mne 1.8.0.
set -euo pipefail

TOOL=${1:?usage: sbatch slurm/preprocess_native_cpu.slurm <biot|labram> <dataset>}
DATASET=${2:?usage: sbatch slurm/preprocess_native_cpu.slurm <biot|labram> <dataset>}

case "$TOOL" in
    biot)   MOD=preprocessing.biot_native;   OUT=processed_biot ;;
    labram) MOD=preprocessing.labram_native; OUT=processed_labram ;;
    *) echo "tool must be biot or labram, got $TOOL" >&2; exit 2 ;;
esac

cd "$SLURM_SUBMIT_DIR"

# These corpora are large and the quota is per-user, not per-filesystem: /work1
# shows 142T free but our own allowance is 1.9T with ~276G left. Report it going
# in so a job that cannot possibly fit is visible at the top of the log rather
# than as a truncated array eight hours later.
echo "space before:"; df -h /work1/chenyuyou/yifanwang/Zhizhe | tail -1

python3 -u -m "$MOD" \
    --dataset "$DATASET" \
    --out /work1/chenyuyou/yifanwang/Zhizhe/"$OUT"/"$DATASET" \
    --jobs "$SLURM_CPUS_PER_TASK"

echo "space after:"; df -h /work1/chenyuyou/yifanwang/Zhizhe | tail -1
du -sh /work1/chenyuyou/yifanwang/Zhizhe/"$OUT"/"$DATASET"
