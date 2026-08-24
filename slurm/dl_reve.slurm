#!/bin/bash
#SBATCH --job-name=dl_reve
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --time=0:29:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd /work1/chenyuyou/yifanwang/Zhizhe/PACLock/vendor
if [ ! -d reve ]; then git clone -q https://github.com/brain-bzh/reve 2>/dev/null || git clone -q https://github.com/brain-bzh/REVE reve; fi
ls reve 2>/dev/null | head -12
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("brain-bzh/reve-base",
                      local_dir="/work1/chenyuyou/yifanwang/Zhizhe/PACLock/vendor/reve_weights")
print("downloaded to", p)
import os
for f in sorted(os.listdir(p)):
    print(" ", f, os.path.getsize(os.path.join(p, f)))
PYEOF
echo DONE
