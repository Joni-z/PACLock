#!/bin/bash
#SBATCH --job-name=isruc-dl
#SBATCH --partition=mi2104x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# ISRUC lives on Mega, which rate-limits anonymous downloads to roughly 5 GB per
# 5.5 h; the corpus is ~14 GB, so it needs several windows. This job resumes
# across them (mega_dl.js skips files already complete on disk) and sleeps for
# exactly as long as the API says the limit lasts.
#
# It runs under Slurm rather than as a login-node daemon: the login node is for
# I/O, compilation and talking to the scheduler, nothing long-running.
#
#   sbatch slurm/download_isruc.slurm
set -uo pipefail

DEST=${PACLOCK_DATA:-/work1/chenyuyou/yifanwang/data}/isruc
LOG=${PACLOCK_DATA:-/work1/chenyuyou/yifanwang/data}/.download_logs/isruc.log
URL="https://mega.nz/folder/QJgDQDDZ#ZMDj3w82msavACurqP48IA"

export PATH=/home1/yifanwang/local/node-v20.17.0-linux-x64/bin:$PATH
mkdir -p "$(dirname "$LOG")"
cd /home1/yifanwang/local/megajs_proj

for attempt in $(seq 1 20); do
    have=$(find "$DEST" -name '*.rec' | wc -l)
    echo "[$(date -Is)] attempt $attempt: $have/100 .rec present"
    if [ "$have" -ge 100 ]; then
        echo "[$(date -Is)] COMPLETE"
        exit 0
    fi

    node mega_dl.js "$URL" "$DEST" 6 > "$LOG" 2>&1 || true

    if ! grep -q "FAILED" "$LOG"; then
        echo "[$(date -Is)] pass finished with no failures"
        continue
    fi
    wait_s=$(grep -o 'Bandwidth limit reached: [0-9]*' "$LOG" | head -1 | grep -o '[0-9]*')
    wait_s=$(( ${wait_s:-1800} + 120 ))
    echo "[$(date -Is)] rate limited, sleeping ${wait_s}s"
    sleep "$wait_s"
done

echo "[$(date -Is)] exiting, $(find "$DEST" -name '*.rec' | wc -l)/100 .rec"
