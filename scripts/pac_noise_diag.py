"""Is the patch-level PAC statistic signal or noise, per dataset?

The pac_interaction token for band j>0 is  a_j * sum_{i<j} alpha_ij
exp(-i angle Z_ij) p_i  -- the amplitude feature rides on a carrier built
entirely from the per-patch coupling estimate Z_ij. If angle(Z_ij) is stable
across patches, the carrier is a consistent code; if it is uniform random,
the carrier multiplies the amplitude by per-patch noise.

This measures, on real windows of each corpus, with the same estimator the
frontend uses (1 s patch, dPAC debiasing):

  R_pref(i,j)  resultant length of unit(Z_ij) pooled over (window, channel,
               patch). 1 = same preferred phase everywhere, 0 = uniform.
               Null level for N pooled samples is ~sqrt(pi)/2/sqrt(N).
  carrier CV   coefficient of variation of |sum_i alpha_ij e^{-i angle Z_ij}
               u_i| across patches, u_i = the patch's mean unit phase of band
               i (stand-in for the phase feature direction). High CV = the
               multiplicative gain on the amplitude token swings per patch.

Run:  python3 -m scripts.pac_noise_diag bci_iv_2a physionet_mi faced tuev
"""
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from paclock_bench.paths import expand

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 75)]
FS = 200
PATCH = 200          # 1 s -- the model's patch_len / default pac_patch_len
N_WIN = 48
SEED = 0


def analytic(x):
    """x: (N, T) -> unit phase (complex), amplitude, both (N, T)."""
    z = hilbert(x, axis=-1)
    amp = np.abs(z)
    ph = z / np.maximum(amp, 1e-12)
    return ph, amp


def run(ds):
    path = expand(f"$PACLOCK_PROC/processed/{ds}/train_signals.npy")
    X = np.load(path, mmap_mode="r")
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(X.shape[0], min(N_WIN, X.shape[0]), replace=False))
    xs = np.asarray(X[idx], dtype=np.float64)            # (n, C, T)
    n, C, T = xs.shape
    P = T // PATCH
    xs = xs[..., : P * PATCH].reshape(n * C, P * PATCH)

    nb = len(BANDS)
    ph = np.empty((nb, n * C, P * PATCH), dtype=np.complex128)
    am = np.empty((nb, n * C, P * PATCH))
    for b, (lo, hi) in enumerate(BANDS):
        sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
        f = sosfiltfilt(sos, xs, axis=-1)
        ph[b], am[b] = analytic(f)

    ph = ph.reshape(nb, n * C, P, PATCH)
    am = am.reshape(nb, n * C, P, PATCH)
    am = am - am.mean(axis=-1, keepdims=True)            # dPAC debiasing
    # Z[i,j,s,p] = mean_t phase_i * amp_j, per signal s and patch p
    Z = np.einsum("ispt,jspt->ijsp", ph, am) / PATCH

    print(f"\n=== {ds}  ({n} windows x {C} ch x {P} patches, "
          f"{Z.shape[2] * Z.shape[3]} pooled samples/pair)")
    null = np.sqrt(np.pi) / 2 / np.sqrt(Z.shape[2] * Z.shape[3])
    print(f"    null R_pref ~ {null:.3f}")

    print("    R_pref (phase-consistency of Z_ij; rows j=target, i<j):")
    for j in range(1, nb):
        vals = []
        for i in range(j):
            u = Z[i, j] / np.maximum(np.abs(Z[i, j]), 1e-15)
            vals.append(np.abs(u.mean()))
        print(f"      band{j} ({BANDS[j][0]:>4.1f}-{BANDS[j][1]:>4.1f} Hz): "
              + "  ".join(f"{v:.3f}" for v in vals))

    # carrier gain per (target band, signal, patch)
    print("    carrier |sum alpha e^{-i angle Z} u_i|: mean, CV across patches:")
    u_mean = ph.mean(axis=-1)                            # (nb, s, P) complex
    for j in range(1, nb):
        mag = np.abs(Z[:j, j])                           # (j, s, P)
        alpha = mag / np.maximum(mag.sum(axis=0, keepdims=True), 1e-15)
        rot = np.conj(Z[:j, j] / np.maximum(mag, 1e-15))
        carrier = (alpha * rot * u_mean[:j]).sum(axis=0) # (s, P)
        g = np.abs(carrier)
        print(f"      band{j}: mean {g.mean():.3f}  CV {g.std() / g.mean():.2f}")


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)
