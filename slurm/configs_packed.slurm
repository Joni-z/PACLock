#!/bin/bash
#SBATCH --job-name=paclock-configs-packed
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# Four-card default: sbatch slurm/configs_packed.slurm cfg1 cfg2 cfg3 cfg4
# Eight-card pack: sbatch -p mi2508x slurm/configs_packed.slurm cfg1 ... cfg8
# One card: sbatch -p mi2101x -c 16 --time=08:00:00 slurm/configs_packed.slurm cfg.yaml:seed
# This account limits mi2101x to 12 hours; override the four-card 24-hour default.
# More configs than cards form a queue; each finished slot gets the next config.
# Underfilled multi-GPU packs and duplicate (run name, seed) outputs are rejected.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$SLURM_SUBMIT_DIR/slurm/amd_env.sh"
"$PACLOCK_PYTHON" -u scripts/slurm/run_packed.py "$@" &
runner=$!
trap 'kill -TERM "$runner" 2>/dev/null || true' TERM INT
rc=0
wait "$runner" || rc=$?
# An interrupted wait must not let the batch shell exit before its runner.
if kill -0 "$runner" 2>/dev/null; then wait "$runner" || rc=$?; fi
exit "$rc"
