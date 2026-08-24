#!/bin/bash
#SBATCH --job-name=dl_iiic
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:29:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v aws >/dev/null 2>&1 || python3 -m pip install --user -q awscli
AP="arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-restricted-access-point"
D=/work1/chenyuyou/yifanwang/data/iiic
mkdir -p "$D/sparcnet_data"
aws s3 sync "s3://$AP/sparcnet_data/" "$D/sparcnet_data/" --no-progress
du -sh "$D"
python3 - <<'PYEOF'
import numpy as np, os
d = "/work1/chenyuyou/yifanwang/data/iiic/sparcnet_data"
for f in sorted(os.listdir(d)):
    p = os.path.join(d, f)
    try:
        a = np.load(p, mmap_mode="r", allow_pickle=False)
        print("%-24s %s %s" % (f, a.shape, a.dtype))
    except Exception as e:
        print("%-24s (not plain npy: %s)" % (f, str(e)[:60]))
PYEOF
echo DONE
