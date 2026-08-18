"""Screen every corpus for FACED-style artefact windows.

FACED's training split turned out to contain two end-to-end corrupted subjects
at ~5000x normal amplitude, contributing 2.6% of windows but dominating the
loss. That was found by accident while chasing something else; nothing in the
pipeline would have flagged it, so the same defect could be sitting in any
other corpus. This screens all nine on the same statistic (per-window
worst-channel RMS vs the split median) so the answer is on record either way.
"""
import numpy as np

from paclock_bench.paths import expand

CORPORA = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc",
           "physionet_mi", "faced", "bci_iv_2a"]

print("%-14s %-6s %-9s %-9s %-9s %s" % (
    "corpus", "split", "median", "p99", "max", "frac>20x"))
for ds in CORPORA:
    for split in ("train", "test"):
        path = expand("$PACLOCK_PROC/processed/%s/%s_signals.npy" % (ds, split))
        try:
            X = np.load(path, mmap_mode="r")
        except FileNotFoundError:
            continue
        # subsample: enough to see a >1% population, cheap on the big corpora
        n = min(4000, X.shape[0])
        idx = np.linspace(0, X.shape[0] - 1, n).astype(int)
        r = np.empty(n)
        for i in range(0, n, 256):
            blk = np.asarray(X[idx[i:i + 256]], dtype=np.float64)
            if blk.ndim == 4:                 # ISRUC (N, seq, C, T)
                blk = blk.reshape(blk.shape[0], -1, blk.shape[-1])
            r[i:i + 256] = np.sqrt((blk ** 2).mean(axis=-1)).max(axis=1)
        med = np.median(r)
        frac = (r > 20 * med).mean()
        flag = "  <-- CHECK" if frac > 0.005 else ""
        print("%-14s %-6s %-9.3f %-9.3f %-9.1f %.2f%%%s"
              % (ds, split, med, np.percentile(r, 99), r.max(), 100 * frac, flag))
