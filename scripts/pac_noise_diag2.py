"""Class-conditioned follow-up to pac_noise_diag: on TUEV (where the PAC
tokenizer wins by +0.172 kappa) and BCI-IV-2a (where it loses), does the
patch-PAC statistic differ BY CLASS -- i.e. is |Z| / its phase carrying label
information, even though pooled preferred phase looks uniform?

Reports per class: pooled R_pref (mean over the 15 band pairs), mean |Z|
(coupling magnitude), and the carrier gain mean/CV.

Run:  python3 -m scripts.pac_noise_diag2 tuev bci_iv_2a
"""
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from paclock_bench.paths import expand

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 75)]
FS = 200
PATCH = 200
PER_CLASS = 40
SEED = 0


def band_decompose(xs):
    """(N, T) -> unit phase (nb, N, P, PATCH) complex, debiased amp same shape."""
    N, T = xs.shape
    P = T // PATCH
    nb = len(BANDS)
    ph = np.empty((nb, N, P, PATCH), dtype=np.complex128)
    am = np.empty((nb, N, P, PATCH))
    for b, (lo, hi) in enumerate(BANDS):
        sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
        f = sosfiltfilt(sos, xs, axis=-1)[..., : P * PATCH]
        z = hilbert(f, axis=-1).reshape(N, P, PATCH)
        a = np.abs(z)
        ph[b] = z / np.maximum(a, 1e-12)
        am[b] = a - a.mean(axis=-1, keepdims=True)
    return ph, am


def run(ds):
    root = expand(f"$PACLOCK_PROC/processed/{ds}")
    X = np.load(f"{root}/train_signals.npy", mmap_mode="r")
    y = np.load(f"{root}/train_labels.npy")
    rng = np.random.default_rng(SEED)
    nb = len(BANDS)
    print(f"\n=== {ds}")
    print(f"    {'class':>5} {'n_win':>5} {'R_pref':>7} {'mean|Z|':>8} "
          f"{'carrier':>8} {'car CV':>7}")
    for cls in np.unique(y):
        pool = np.where(y == cls)[0]
        idx = np.sort(rng.choice(pool, min(PER_CLASS, len(pool)), replace=False))
        xs = np.asarray(X[idx], dtype=np.float64)        # (n, C, T)
        n, C, T = xs.shape
        ph, am = band_decompose(xs.reshape(n * C, T))
        Z = np.einsum("ispt,jspt->ijsp", ph, am) / PATCH
        iu, ju = np.tril_indices(nb, k=-1)               # (i<j) as (j_row, i_col)
        pairs = Z[ju, iu]                                # (15, s, P)
        unit = pairs / np.maximum(np.abs(pairs), 1e-15)
        r_pref = np.abs(unit.mean(axis=(1, 2))).mean()
        mean_mag = np.abs(pairs).mean()
        u_mean = ph.mean(axis=-1)                        # (nb, s, P)
        gains = []
        for j in range(1, nb):
            mag = np.abs(Z[:j, j])
            alpha = mag / np.maximum(mag.sum(axis=0, keepdims=True), 1e-15)
            rot = np.conj(Z[:j, j] / np.maximum(mag, 1e-15))
            gains.append(np.abs((alpha * rot * u_mean[:j]).sum(axis=0)))
        g = np.concatenate([v.ravel() for v in gains])
        print(f"    {cls:>5} {n:>5} {r_pref:>7.3f} {mean_mag:>8.4f} "
              f"{g.mean():>8.3f} {g.std() / g.mean():>7.2f}")


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)
