#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
#   sbatch slurm/preprocess.slurm <protocol> <dataset>
#
#   frozen  0.3-75 Hz band-pass + 60 Hz notch, 16 bipolar, 200 Hz, /100
#           -> $PACLOCK_PROC/processed/       groups A, C, D and CBraMod
#   pac     high-pass 0.5 Hz, NO notch, everything else copied from frozen
#           -> $PACLOCK_PROC/processed_pac/   the PAC-protocol sensitivity arm
#   biot    unfiltered 200 Hz, q95 normalisation
#           -> $PACLOCK_PROC/processed_biot/  BIOT and TFM-Tokenizer
#   labram  0.1-75 Hz + 50 Hz notch, 23 unipolar channels
#           -> $PACLOCK_PROC/processed_labram/
#
# Four protocols exist because hard rule 2 says every model runs its own repo's
# preprocessing; feeding a foundation model our pipeline instead of its own is
# what turned 0.6772 into 0.4436 once. See docs/PROTOCOLS.md.
#
# Replaces five near-identical scripts.
set -euo pipefail

PROTOCOL=${1:?usage: sbatch slurm/preprocess.slurm <frozen|pac|biot|labram> <dataset>}
DATASET=${2:?usage: sbatch slurm/preprocess.slurm <protocol> <dataset>}
PROC=${PACLOCK_PROC:-/work1/chenyuyou/yifanwang/Zhizhe}

case "$PROTOCOL" in
    frozen) MOD="preprocessing.$DATASET";        OUT=processed        ;;
    pac)    MOD="preprocessing.$DATASET";        OUT=processed_pac    ;;
    biot)   MOD=preprocessing.biot_native;       OUT=processed_biot   ;;
    labram) MOD=preprocessing.labram_native;     OUT=processed_labram ;;
    *) echo "protocol must be frozen|pac|biot|labram, got '$PROTOCOL'" >&2; exit 2 ;;
esac

# MIOpen caches tuned kernels in a SQLite DB; concurrent processes sharing one
# file over a shared filesystem corrupt it ("database disk image is malformed").
export MIOPEN_USER_DB_PATH="/tmp/miopen-$USER-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
trap 'rm -rf "$MIOPEN_USER_DB_PATH"' EXIT

module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"

echo "protocol=$PROTOCOL dataset=$DATASET -> $PROC/$OUT/$DATASET"
echo "space before:"; df -h "$PROC" | tail -1

if [ "$PROTOCOL" = "pac" ]; then
    # the PAC protocol is the frozen one with a single axis changed; its config
    # is derived rather than written by hand, so the contrast stays single-variable
    python3 -m scripts.make_pac_protocol
    CFG=configs/datasets_pac/"$DATASET".yaml
else
    CFG=configs/datasets/"$DATASET".yaml
fi

if [ "$PROTOCOL" = "biot" ] || [ "$PROTOCOL" = "labram" ]; then
    # the native preprocessors take an explicit --out
    python3 -u -m "$MOD" --dataset "$DATASET" --out "$PROC/$OUT/$DATASET" \
        --jobs "$SLURM_CPUS_PER_TASK"
else
    # frozen/pac read their destination from the dataset config, not from a flag
    python3 -u -m "$MOD" --config "$CFG" --jobs "$SLURM_CPUS_PER_TASK"
fi

echo "space after:"; df -h "$PROC" | tail -1
du -sh "$PROC/$OUT/$DATASET"
echo
echo "verify against the frozen manifest:"
echo "  sbatch slurm/run.slurm scripts.verify_processed --dataset $DATASET"
