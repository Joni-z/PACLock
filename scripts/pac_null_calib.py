"""Calibrate the null level of the patch-PAC estimator, and predict what a
significance gate would do to each corpus.

Why this has to be measured rather than derived. The gate we want is

    w_ij = relu(1 - 1/lambda_ij),   lambda_ij = |Z_ij|^2 / E_null|Z_ij|^2

so it needs E_null|Z|^2. The textbook value is Var(A_j)/L for L independent
samples, but a band-limited envelope is heavily autocorrelated: over a 1 s
patch a 4 Hz-wide band carries nothing like 200 independent samples. Using the
nominal L would inflate lambda by the ratio of nominal to effective d.o.f. and
declare almost every edge significant -- the gate would then be a no-op and we
would have "implemented" it without changing anything.

So the null is measured directly with a circular-shift surrogate: roll the
amplitude against the phase within each patch. That destroys any phase-amplitude
relationship while preserving both marginals AND both autocorrelations, which is
exactly the null we need. Reported against two analytic predictions:

    L_nominal = patch samples
    L_eff     = (bw_i + bw_j) * patch_seconds

L_eff is the spectral-spread argument: A~_j(t) exp(i phi_i(t)) has a spectrum
centred at f_i with width ~ bw_i + bw_j, so its decorrelation time is
1/(bw_i+bw_j) and a T-second patch holds ~(bw_i+bw_j)T independent draws. If the
surrogate agrees with L_eff, the frontend can compute the null from its own sinc
bandwidths with no calibration table.

Then the payoff question: with the surrogate-calibrated null, what fraction of
edges survive the gate, per class? The gate is only worth implementing if it
keeps TUEV's edges (where PAC wins +0.172) and drops BCI-IV-2a's (where PAC
collapses to chance).

    sbatch slurm/run_cpu.slurm scripts.pac_null_calib tuev bci_iv_2a
"""
import sys

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from paclock_bench.paths import expand

BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 75)]
FS = 200
PATCH = 200          # 1 s -- the model's patch_len / default pac_patch_len
PER_CLASS = 40
N_SURR = 12
SEED = 0


def decompose(xs):
    """(N, T) -> unit phase and debiased amplitude, both (nb, N, P, PATCH)."""
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
        am[b] = a - a.mean(axis=-1, keepdims=True)      # dPAC debiasing
    return ph, am


def pac(ph, am):
    """Z[i, j, s, p] -- low-band-i phase driving band-j amplitude."""
    return np.einsum("ispt,jspt->ijsp", ph, am) / PATCH


def run(ds):
    root = expand(f"$PACLOCK_PROC/processed/{ds}")
    X = np.load(f"{root}/train_signals.npy", mmap_mode="r")
    y = np.load(f"{root}/train_labels.npy")
    rng = np.random.default_rng(SEED)
    nb = len(BANDS)
    patch_sec = PATCH / FS
    bw = np.array([hi - lo for lo, hi in BANDS])

    print(f"\n{'=' * 72}\n=== {ds}\n{'=' * 72}", flush=True)
    calib_rows, gate_rows = [], []

    for cls in np.unique(y):
        pool = np.where(y == cls)[0]
        idx = np.sort(rng.choice(pool, min(PER_CLASS, len(pool)), replace=False))
        xs = np.asarray(X[idx], dtype=np.float64)
        n, C, T = xs.shape
        ph, am = decompose(xs.reshape(n * C, T))
        Z = pac(ph, am)                                  # (nb, nb, s, P)
        var_a = (am ** 2).mean(axis=-1)                  # (nb, s, P) per band j

        # --- null by circular shift: preserves both marginals and both ACFs ---
        null_sq = np.zeros_like(np.abs(Z) ** 2)
        for _ in range(N_SURR):
            shift = rng.integers(PATCH // 4, 3 * PATCH // 4)
            null_sq += np.abs(pac(ph, np.roll(am, shift, axis=-1))) ** 2
        null_sq /= N_SURR                                # E_null|Z_ij|^2

        # effective d.o.f. the surrogate implies: E_null|Z|^2 = Var(A_j)/L_eff
        l_eff_meas = var_a[None, :, :, :] / np.maximum(null_sq, 1e-30)
        l_eff_pred = np.maximum((bw[:, None] + bw[None, :]) * patch_sec, 2.0)

        iu, ju = np.tril_indices(nb, k=-1)               # valid edges: i < j
        m_meas = l_eff_meas[iu, ju].mean(axis=(1, 2))    # per edge, pooled
        m_pred = l_eff_pred[iu, ju]
        calib_rows.append((cls, m_meas, m_pred))

        # --- what the gate would do, with the measured null ---
        lam = np.abs(Z) ** 2 / np.maximum(null_sq, 1e-30)
        w = np.maximum(0.0, 1.0 - 1.0 / np.maximum(lam, 1e-12))
        we = w[iu, ju]                                   # (n_edge, s, P)
        g = w[iu, ju].max(axis=0)                        # strongest driver per patch
        gate_rows.append((cls, n, (we > 0).mean(), we.mean(), g.mean(),
                          np.abs(Z[iu, ju]).mean()))

    print("\n  [1] effective d.o.f. per edge: measured by surrogate vs "
          "(bw_i+bw_j)*T prediction")
    print(f"      L_nominal = {PATCH}")
    iu, ju = np.tril_indices(nb, k=-1)
    print("      " + "  ".join(f"{BANDS[i][0]:.0f}>{BANDS[j][0]:.0f}"
                               for i, j in zip(iu, ju)))
    meas_all = np.mean([r[1] for r in calib_rows], axis=0)
    print("  meas " + "  ".join(f"{v:>7.1f}" for v in meas_all))
    print("  pred " + "  ".join(f"{v:>7.1f}" for v in calib_rows[0][2]))
    print("  rati " + "  ".join(f"{m / p:>7.2f}" for m, p
                                in zip(meas_all, calib_rows[0][2])))

    print("\n  [2] gate behaviour with the measured null")
    print(f"      {'class':>5} {'n_win':>5} {'frac w>0':>9} {'mean w':>7} "
          f"{'mean g':>7} {'mean|Z|':>8}")
    for cls, n, frac, mw, mg, mz in gate_rows:
        print(f"      {cls:>5} {n:>5} {frac:>9.3f} {mw:>7.3f} {mg:>7.3f} "
              f"{mz:>8.4f}")


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)
