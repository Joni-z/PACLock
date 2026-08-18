"""Which FACED channels/windows carry the 857x power spread?

The FACED array's std after norm_div100 is 29.9 against TUAB's 0.75, and its
per-channel broadband power spans a factor of 857. Either a few electrodes are
pathological, or a few windows are, and the two have very different fixes.
Any model that mixes across channels (ours does, via attention on the space
axis) is dominated by whichever channel is loudest, so this has to be pinned
down before blaming the architecture for FACED.
"""
import numpy as np

from paclock_bench.paths import expand

ASSUMED = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
           "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8",
           "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8",
           "PO3", "PO4", "Oz", "O1", "O2", "A2", "A1"]

X = np.load(expand("$PACLOCK_PROC/processed/faced/train_signals.npy"), mmap_mode="r")
print("shape", X.shape)

# per (window, channel) RMS over the whole train split, in chunks
N, C, T = X.shape
rms = np.empty((N, C), dtype=np.float64)
step = 512
for i in range(0, N, step):
    blk = np.asarray(X[i:i + step], dtype=np.float64)
    rms[i:i + step] = np.sqrt((blk ** 2).mean(axis=-1))

ch_med = np.median(rms, axis=0)
print("\nper-channel median RMS (sorted, loud -> quiet):")
for i in np.argsort(-ch_med):
    print("  ch%-3d %-5s %.4f" % (i, ASSUMED[i], ch_med[i]))

print("\nchannel median RMS spread: max/min = %.1f" % (ch_med.max() / ch_med.min()))

w_max = rms.max(axis=1)
print("\nwindow-level worst-channel RMS percentiles:")
for q in (50, 90, 99, 99.9, 100):
    print("   p%-5s %.3f" % (q, np.percentile(w_max, q)))

frac = (w_max > 10 * np.median(w_max)).mean()
print("\nwindows whose worst channel exceeds 10x the median window: %.2f%%" % (100 * frac))
print("VERDICT:", "a few WINDOWS are extreme (artefact epochs)"
      if ch_med.max() / ch_med.min() < 5 else "a few CHANNELS are systematically extreme")
