"""Is the FACED array actually EEG-shaped, and does it carry channel structure?

The alpha-topography check came back flat (4% spread across all 32 channels),
which is not what scalp EEG looks like. Before concluding anything about
channel ORDER, confirm the array has a plausible 1/f EEG spectrum at all and
that channels differ from each other somewhere. Compared against TUAB, whose
spatial structure we have never had reason to doubt.
"""
import numpy as np

from paclock_bench.paths import expand

fs = 200.0


def probe(name, path, nch):
    X = np.load(path, mmap_mode="r")
    idx = np.linspace(0, X.shape[0] - 1, min(500, X.shape[0])).astype(int)
    seg = np.asarray(X[idx], dtype=np.float64)
    if seg.ndim == 4:                      # ISRUC-style (N, seq, C, T)
        seg = seg.reshape(-1, seg.shape[-2], seg.shape[-1])
    f = np.fft.rfftfreq(seg.shape[-1], 1.0 / fs)
    psd = (np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(axis=0)

    print("\n=== %s === shape=%s  amp: mean|x|=%.4f std=%.4f" % (
        name, X.shape, np.abs(seg).mean(), seg.std()))
    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    print("  band power by channel (first 6 ch):")
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        v = psd[:, m].sum(axis=1)
        print("    %2d-%2dHz  %s" % (lo, hi, "  ".join("%.3e" % x for x in v[:6])))
    tot = psd[:, (f >= 1) & (f < 45)].sum(axis=1)
    print("  broadband 1-45Hz across all channels: min=%.3e max=%.3e ratio=%.2f"
          % (tot.min(), tot.max(), tot.max() / tot.min()))
    # is the spectrum 1/f like real EEG?
    lo_p = psd[:, (f >= 1) & (f < 4)].sum(axis=1).mean()
    hi_p = psd[:, (f >= 30) & (f < 45)].sum(axis=1).mean()
    print("  delta/gamma power ratio = %.1f  (real EEG: >>1)" % (lo_p / hi_p))


probe("FACED", expand("$PACLOCK_PROC/processed/faced/train_signals.npy"), 32)
probe("TUAB (reference)", expand("$PACLOCK_PROC/processed/tuab/train_signals.npy"), 16)
probe("PhysioNet-MI", expand("$PACLOCK_PROC/processed/physionet_mi/train_signals.npy"), 64)
