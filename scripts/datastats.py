"""Amplitude statistics of each processed corpus, for comparison.

PACLock does not fit the training set at all on FACED / PhysioNet-MI /
BCI-IV-2a, and the loss trajectory there is smooth and monotone rather than
noisy -- the shape of a signal that is too small to produce useful gradients
through the sinc filterbank and Hilbert transform, not of a bad learning rate.
This prints the scale each corpus actually arrives at so that guess can be
checked rather than argued about.
"""
from paclock_bench.paths import processed
import glob
import os

import numpy as np

ROOT = processed("processed")

print(f"{'corpus':14s} {'shape':>22s} {'std':>10s} {'p99.9|x|':>10s} "
      f"{'mean':>10s} {'zeros%':>7s}")
for ds in sorted(os.listdir(ROOT)):
    path = os.path.join(ROOT, ds, "train_signals.npy")
    if not os.path.exists(path):
        continue
    a = np.load(path, mmap_mode="r")
    # a bounded sample: these arrays run to tens of GB and only the scale is
    # wanted, so take a few hundred windows rather than the whole corpus
    n = min(200, a.shape[0])
    idx = np.linspace(0, a.shape[0] - 1, n).astype(int)
    x = np.asarray(a[idx], dtype=np.float64)
    print(f"{ds:14s} {str(a.shape):>22s} {x.std():>10.4f} "
          f"{np.percentile(np.abs(x), 99.9):>10.4f} {x.mean():>10.4f} "
          f"{100.0 * (x == 0).mean():>7.2f}")
