#!/bin/bash
#SBATCH --job-name=iiic_split_dry
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
module load pytorch/2.7.1
cd "$SLURM_SUBMIT_DIR"
python3 - <<'PYEOF'
import numpy as np, os, sys
sys.path.insert(0, ".")
from preprocessing.iiic import balanced_patient_split
d = "/work1/chenyuyou/yifanwang/data/iiic/sparcnet_data"
keys = np.load(d + "/all_train_key.npy")
votes = np.load(d + "/all_train_Y.npy")
_, first = np.unique(keys, return_index=True); first = np.sort(first)
v = votes[first]; top = v.max(axis=1, keepdims=True)
tied = (v == top).sum(axis=1) > 1
keep = first[~tied]; labels = v[~tied].argmax(axis=1)
pat = np.array([k.split("_")[0] for k in keys[keep]])
g = balanced_patient_split(pat, labels, 0.7, 0.5, 6)
glob_mix = np.bincount(labels, minlength=6) / len(labels)
print("global   ", " ".join("%.3f" % x for x in glob_mix))
for name in ("train","val","test"):
    sel = np.isin(pat, sorted(g[name]))
    mix = np.bincount(labels[sel], minlength=6) / sel.sum()
    print("%-9s" % name, " ".join("%.3f" % x for x in mix),
          "| patients %4d windows %6d (%.1f%%)" % (len(g[name]), sel.sum(), 100*sel.sum()/len(pat)))
    print("          max |Δ| vs global: %.3f" % np.abs(mix - glob_mix).max())
PYEOF
