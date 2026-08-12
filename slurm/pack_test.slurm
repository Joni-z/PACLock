#!/bin/bash
#SBATCH --job-name=pack-test
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Does packing N processes onto one node collapse throughput? The step profile
# measured a 535 samples/s ceiling running ALONE; every packed production run has
# come in at ~31. Same config, same node, only the number of concurrent
# processes changes.
set -euo pipefail
module load pytorch/2.7.1
cd $SLURM_SUBMIT_DIR
N=${1:-4}
pids=()
for g in $(seq 0 $((N-1))); do
  HIP_VISIBLE_DEVICES=$g \
  MIOPEN_USER_DB_PATH=/tmp/mio_pk$g MIOPEN_CUSTOM_CACHE_DIR=/tmp/mio_pk$g \
  python3 -u scripts/profile_step.py configs/_bench/E_ram_b32.yaml 30 \
    > logs/packtest-${SLURM_JOB_ID}-n${N}-g${g}.out 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
echo done
