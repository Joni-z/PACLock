#!/bin/bash
#SBATCH --job-name=peek_iiic
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
python3 - <<'PYEOF'
import numpy as np
d = "/work1/chenyuyou/yifanwang/data/iiic/sparcnet_data"
k = np.load(d + "/all_train_key.npy")
print("keys sample:", k[:5].tolist())
print("unique keys:", len(np.unique(k)))
import re
pats = sorted({s.split("_")[0] for s in k[:2000].tolist()})
print("prefix guess (first field):", pats[:5], "…")
y = np.load(d + "/all_train_Y.npy")
print("Y row sums (votes?):", np.unique(y.sum(1))[:10])
print("Y example rows:", y[:3].tolist())
x = np.load(d + "/all_train_X.npy", mmap_mode="r")
import numpy.linalg as _
v = np.asarray(x[:8], dtype=np.float32)
print("X abs p50/p99:", round(float(np.percentile(np.abs(v),50)),3), round(float(np.percentile(np.abs(v),99)),3))
PYEOF
