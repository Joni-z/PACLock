#!/bin/bash
#SBATCH --job-name=paclock-configs-packed
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Usage: sbatch slurm/configs_packed.slurm <cfg1.yaml> [cfg2.yaml] [cfg3.yaml] [cfg4.yaml]
#
# Same packing as seeds_packed.slurm, but across DIFFERENT configs at one seed
# instead of one config at three seeds. That is the shape a diagnostic wants:
# three corpora, one hypothesis, one node, one answer -- rather than three whole
# nodes each running a single GPU because --exclusive hands over the whole node
# whether or not you asked for it.
#
# Not for matrix cells. A cell needs three seeds of one config and must go
# through seeds_packed.slurm so a partially completed cell fails the job.
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 4 ]; then
    echo "usage: sbatch slurm/configs_packed.slurm <cfg1> [cfg2] [cfg3] [cfg4]" >&2
    exit 2
fi
SEED=${SEED:-0}

module load pytorch/2.7.1

cd "$SLURM_SUBMIT_DIR"

pids=()
gpu=0
for spec in "$@"; do
    # <cfg.yaml> or <cfg.yaml>:<seed> -- a per-config seed lets one node pack
    # e.g. a rerun of one corpus with extra seeds of another.
    cfg=${spec%%:*}; s=${spec#*:}; [ "$s" = "$spec" ] && s=$SEED
    tag=$(basename "$cfg" .yaml)-s$s
    # Per-process MIOpen cache: concurrent processes sharing one SQLite DB on
    # the shared filesystem corrupt it ("database disk image is malformed").
    HIP_VISIBLE_DEVICES=$gpu \
    MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}_${gpu}" \
    MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen_${SLURM_JOB_ID}_${gpu}" \
    python3 -u -m paclock_bench.training.train \
        --config "$cfg" --seed "$s" \
        > "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${tag}.out" 2>&1 &
    pids+=($!)
    echo "launched $tag on GPU $gpu (pid ${pids[-1]})"
    gpu=$((gpu + 1))
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
echo "all configs finished, rc=$rc"
exit $rc
