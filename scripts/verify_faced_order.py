"""Verify FACED's 32-channel order from the data itself before trusting it.

FACED's raw release is gone from this cluster and the official channel order
lives only in the paper's supplementary table, so the order has to be assumed
from torcheeg's FACED_CHANNEL_LIST (30 scalp electrodes) plus the two mastoids
whose position in the file is NOT documented there. Guessing wrong is worse
than doing nothing: SpatialPE would then hand the model a scrambled geometry,
which is actively misleading rather than merely uninformative.

The check uses the most reliable spatial gradient in scalp EEG: posterior
alpha. Occipital/parietal electrodes carry far more 8-13 Hz power than frontal
ones. If the assumed order is right, the alpha-power ranking should put the
O/PO/P channels at the top and the frontal ones near the bottom; if the order
is shifted or permuted, that structure will not line up.

Mastoids are also identifiable: referenced against a common average they carry
markedly less scalp-generated signal, so they should sit at an extreme of the
broadband-power ranking rather than in the middle.
"""
import numpy as np

from paclock_bench.paths import expand

ASSUMED = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8",
           "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8",
           "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8",
           "PO3", "PO4", "Oz", "O1", "O2", "A2", "A1"]

POSTERIOR = {"Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2"}
FRONTAL = {"Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6"}

X = np.load(expand("$PACLOCK_PROC/processed/faced/train_signals.npy"), mmap_mode="r")
print("data shape:", X.shape)
n = min(2000, X.shape[0])
idx = np.linspace(0, X.shape[0] - 1, n).astype(int)
seg = np.asarray(X[idx], dtype=np.float64)          # (n, 32, T)

fs = 200.0
freqs = np.fft.rfftfreq(seg.shape[-1], 1.0 / fs)
psd = (np.abs(np.fft.rfft(seg, axis=-1)) ** 2).mean(axis=0)   # (32, F)

alpha = psd[:, (freqs >= 8) & (freqs <= 13)].sum(axis=1)
broad = psd[:, (freqs >= 1) & (freqs <= 45)].sum(axis=1)
rel_alpha = alpha / broad

order = np.argsort(-rel_alpha)
print("\nrelative alpha power, high -> low (assuming the order above):")
for r, i in enumerate(order, 1):
    nm = ASSUMED[i] if i < len(ASSUMED) else "?"
    tag = "POST" if nm in POSTERIOR else ("front" if nm in FRONTAL else "")
    print("  %2d. ch%-3d %-5s %.4f  %s" % (r, i, nm, rel_alpha[i], tag))

post_i = [i for i, nm in enumerate(ASSUMED) if nm in POSTERIOR]
front_i = [i for i, nm in enumerate(ASSUMED) if nm in FRONTAL]
pm, fm = rel_alpha[post_i].mean(), rel_alpha[front_i].mean()
print("\nmean relative alpha: posterior=%.4f frontal=%.4f  ratio=%.2f" % (pm, fm, pm / fm))
print("VERDICT:", "consistent with the assumed order"
      if pm > fm * 1.2 else "NOT consistent -- do not trust this order")

print("\nbroadband power (mastoid check), low -> high:")
bo = np.argsort(broad)
for i in bo[:4]:
    print("   ch%-3d %-5s %.3e" % (i, ASSUMED[i], broad[i]))
