#!/bin/bash
#SBATCH --job-name=dl_reve2
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --time=0:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
cd /work1/chenyuyou/yifanwang/Zhizhe/PACLock/vendor
rm -rf reve && git clone -q https://github.com/elouayas/reve_eeg reve && ls reve
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("brain-bzh/reve-positions", repo_type="model",
                      local_dir="/work1/chenyuyou/yifanwang/Zhizhe/PACLock/vendor/reve_positions")
import os
for root, _, files in os.walk(p):
    for f in files:
        if not f.startswith("."):
            print(os.path.join(root.replace(p,""), f))
PYEOF
echo DONE
