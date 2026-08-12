#!/bin/bash
#SBATCH --job-name=B-cbramod-dl
#SBATCH --partition=mi2104x
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# CBraMod ships its pretrained weights on HuggingFace rather than in the repo.
set -uo pipefail
module load pytorch/2.7.1
DST="$SLURM_SUBMIT_DIR/vendor/cbramod/pretrained_weights"
mkdir -p "$DST"
python3 - "$DST" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
dst = sys.argv[1]
p = snapshot_download(repo_id="weighting666/CBraMod", local_dir=dst)
print("downloaded to", p)
for f in sorted(os.listdir(p)):
    full = os.path.join(p, f)
    if os.path.isfile(full):
        print("  %-40s %8.1f MB" % (f, os.path.getsize(full)/1e6))
PY
