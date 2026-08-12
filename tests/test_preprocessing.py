"""Correctness checks for the preprocessing primitives, run before any GPU job.

    python -m tests.test_preprocessing

Covers the parts where a silent bug would corrupt every downstream number:
filter/resample ordering, the half-open overlap labelling rule, windowing tail
accounting, the three normalisation schemes, and manifest leak detection.
"""

from __future__ import annotations

import numpy as np

from preprocessing.common import (
    Manifest,
    compute_train_stats,
    intervals_overlap_labels,
    norm_div100,
    norm_q95,
    norm_with_stats,
    preprocess_signal,
    resample_to,
    window_signal,
)

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, bool(cond), detail))


# --------------------------------------------------------------------------- #
# resampling / filtering
# --------------------------------------------------------------------------- #
fs_in, fs_out, dur = 250.0, 200.0, 4.0
t = np.arange(int(fs_in * dur)) / fs_in
x = np.sin(2 * np.pi * 10 * t)[None, :]                     # 10 Hz, 1 channel

y = resample_to(x, fs_in, fs_out)
check("resample length", y.shape[1] == int(fs_out * dur),
      f"{y.shape[1]} vs {int(fs_out * dur)}")

# a 10 Hz tone must survive band-pass + resample with its amplitude intact
z = preprocess_signal(x, fs_in, fs_out=fs_out, band=(0.3, 75.0), notch_freq=60.0)
mid = z[0, len(z[0]) // 4: -len(z[0]) // 4]                 # ignore edge transients
check("bandpass keeps in-band tone", 0.8 < mid.max() < 1.2, f"peak={mid.max():.3f}")

# content above the target Nyquist must not alias back in
hi = np.sin(2 * np.pi * 95 * t)[None, :]                    # 95 Hz > 100 Hz Nyquist/2 margin
zh = preprocess_signal(hi, fs_in, fs_out=fs_out, band=(0.3, 75.0), notch_freq=None)
mh = zh[0, len(zh[0]) // 4: -len(zh[0]) // 4]
check("out-of-band tone suppressed", np.abs(mh).max() < 0.2, f"peak={np.abs(mh).max():.4f}")

# notch must actually remove line noise
line = np.sin(2 * np.pi * 60 * t)[None, :]
zl = preprocess_signal(line, fs_in, fs_out=fs_out, band=(0.3, 75.0), notch_freq=60.0)
ml = zl[0, len(zl[0]) // 4: -len(zl[0]) // 4]
check("60 Hz notch suppresses line", np.abs(ml).max() < 0.3, f"peak={np.abs(ml).max():.4f}")

check("output dtype float32", z.dtype == np.float32, str(z.dtype))

# --------------------------------------------------------------------------- #
# windowing
# --------------------------------------------------------------------------- #
sig = np.arange(16 * 2500, dtype=np.float64).reshape(16, 2500)
W, tail = window_signal(sig, win=2000, stride=2000)
check("window count", W.shape == (1, 16, 2000), str(W.shape))
check("tail reported", tail == 500, str(tail))

W2, tail2 = window_signal(sig, win=1000, stride=500)
check("strided window count", W2.shape[0] == (2500 - 1000) // 500 + 1, str(W2.shape))

short = np.zeros((16, 100))
W3, tail3 = window_signal(short, win=2000, stride=2000)
check("too-short returns empty", W3.shape[0] == 0 and tail3 == 100, str(W3.shape))

# --------------------------------------------------------------------------- #
# overlap labelling: the CHB-MIT / TUSZ rule
# --------------------------------------------------------------------------- #
fs = 200.0
# windows: [0,10) [10,20) [20,30) seconds
lab = intervals_overlap_labels(3, win=2000, stride=2000,
                               intervals=[(12.0, 15.0)], fs=fs)
check("overlap hits middle window", lab.tolist() == [0, 1, 0], str(lab.tolist()))

# a seizure that fully covers a window must be positive -- the bug the protocol
# calls out in TFM's original implementation
lab = intervals_overlap_labels(3, win=2000, stride=2000,
                               intervals=[(0.0, 30.0)], fs=fs)
check("seizure covering window labels positive", lab.tolist() == [1, 1, 1],
      str(lab.tolist()))

# half-open boundary: an interval ending exactly at the window start does not count
lab = intervals_overlap_labels(2, win=2000, stride=2000,
                               intervals=[(5.0, 10.0)], fs=fs)
check("half-open boundary excludes touching interval", lab.tolist() == [1, 0],
      str(lab.tolist()))

lab = intervals_overlap_labels(2, win=2000, stride=2000,
                               intervals=[(10.0, 15.0)], fs=fs)
check("interval starting at boundary hits later window", lab.tolist() == [0, 1],
      str(lab.tolist()))

# sub-sample overlap still counts (> 0 rule, not >= 1 sample)
lab = intervals_overlap_labels(2, win=2000, stride=2000,
                               intervals=[(9.999, 10.5)], fs=fs)
check("tiny overlap counts", lab.tolist() == [1, 1], str(lab.tolist()))

# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
a = np.full((2, 3, 10), 250.0)
check("div100", np.allclose(norm_div100(a), 2.5))

b = np.random.RandomState(0).randn(4, 3, 1000) * 50
nb = norm_q95(b)
q = np.quantile(np.abs(nb), 0.95, axis=-1)
check("q95 normalises per window+channel", np.allclose(q, 1.0, atol=1e-3),
      f"q95 range [{q.min():.4f}, {q.max():.4f}]")

tr = np.random.RandomState(1).randn(50, 2, 100) * 3 + 7
mean, std = compute_train_stats(tr)
ntr = norm_with_stats(tr, mean, std)
check("train stats zero-mean unit-std",
      abs(ntr.mean()) < 1e-6 and abs(ntr.std() - 1) < 1e-6,
      f"mean={ntr.mean():.2e} std={ntr.std():.6f}")
check("train stats are per-channel", mean.shape == (2, 1), str(mean.shape))

# val/test must be transformed with train statistics, not their own
va = np.random.RandomState(2).randn(20, 2, 100) * 3 + 9
nva = norm_with_stats(va, mean, std)
check("val keeps its offset under train stats", abs(nva.mean()) > 0.1,
      f"mean={nva.mean():.3f}")

# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
m = Manifest(dataset="t", protocol={})
m.add_split("train", subjects=["a", "b"], n_windows=10, class_counts={0: 6, 1: 4})
m.add_split("test", subjects=["c"], n_windows=5, class_counts={0: 3, 1: 2})
try:
    m.check_disjoint()
    check("disjoint splits pass", True)
except ValueError as e:
    check("disjoint splits pass", False, str(e))

m.add_split("val", subjects=["b"], n_windows=3, class_counts={0: 3})
try:
    m.check_disjoint()
    check("subject leakage detected", False, "no error raised")
except ValueError:
    check("subject leakage detected", True)

# --------------------------------------------------------------------------- #
print("=" * 68)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
print("=" * 68)
nfail = sum(1 for _, ok, _ in results if not ok)
print(f"{len(results) - nfail}/{len(results)} passed")
raise SystemExit(1 if nfail else 0)
