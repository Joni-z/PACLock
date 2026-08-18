"""Is band power linearly readable from a single token? One probe, four tokenizers.

Three fusion fixes have now failed on BCI-IV-2a (product 0.2581/0.2589/0.2604,
uniform 0.2639, rotation 0.2724, all against a 0.25 chance level, while
`tokenizer_mode: raw` reaches ~0.45). `rotation` failing is the informative one:
it makes |h_jk| = |a_jk| hold exactly, so the amplitude magnitude is provably
present in the token and the model still cannot use it. Something other than
"the amplitude is missing or corrupted in magnitude" is wrong.

The candidate mechanism, stated so it can be falsified rather than argued: after
`view_as_real`, an interaction token's components are a_k*cos(theta_k) and
a_k*sin(theta_k), where theta_k is the carrier's phase and varies per patch.
Recovering a_k needs sqrt(x^2 + y^2) -- QUADRATIC in the token. Every path out of
the frontend starts with a linear map, and on 2160 training windows a network has
little chance of learning a per-dimension modulus. Under `raw` no such rotation is
applied, so band power should be far more accessible.

This measures accessibility directly and with no training loop involved: take an
UNTRAINED frontend (so the question is about the representation, not about
optimisation), and fit a linear ridge from one token to that token's own log band
power -- the quantity `return_amp_target=True` already computes. Every (window,
electrode, band, patch) is one sample, D features, one target.

Reported per tokenizer:
  R2_lin   ridge from the token's D real components -> its log band power
  R2_quad  ridge from the SQUARED components -> same target. If the linear probe
           collapses and the quadratic one does not, the information is present
           but behind a nonlinearity, which is the claim.
  R2_pool  linear probe from patch-mean-pooled tokens -> mean log band power,
           since ClassificationHead pools over patches: averaging a_k*e^{i theta_k}
           over patches with varying theta drives it toward zero, so pooling should
           cost the interaction modes more than it costs raw.

    sbatch slurm/run.slurm scripts.probe_readability bci_iv_2a tuev
"""
import sys

import numpy as np
import torch

from paclock_bench.models.paclock.frontend.triaxial import TriAxialFrontend
from paclock_bench.paths import expand

BASE = dict(n_bands=8, hidden_dim=128, sample_rate=200, patch_len=200)
MODES = [
    ("raw", dict(tokenizer_mode="raw")),
    ("product", dict(tokenizer_mode="pac_interaction", interaction_mode="product")),
    ("rotation", dict(tokenizer_mode="pac_interaction", interaction_mode="rotation")),
    ("concat", dict(tokenizer_mode="pac_interaction", interaction_mode="concat")),
]
N_WIN = 96
RIDGE = 1.0


def r2_ridge(X, y):
    """Out-of-sample R^2 of a ridge fit, 50/50 split, features standardised."""
    n = len(X)
    tr, te = slice(0, n // 2), slice(n // 2, n)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    ym = y[tr].mean()
    A = Xtr.T @ Xtr + RIDGE * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (y[tr] - ym))
    pred = Xte @ w + ym
    ss_res = float(((y[te] - pred) ** 2).sum())
    ss_tot = float(((y[te] - y[te].mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def run(ds):
    root = expand(f"$PACLOCK_PROC/processed/{ds}")
    X = np.load(f"{root}/train_signals.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(X.shape[0], min(N_WIN, X.shape[0]), replace=False))
    xb = torch.from_numpy(np.asarray(X[idx], dtype=np.float32))
    print(f"\n=== {ds}  ({xb.shape[0]} windows, {xb.shape[1]} electrodes)", flush=True)
    print(f"    {'tokenizer':>10} {'R2_lin':>8} {'R2_quad':>8} {'R2_pool':>8}")

    for name, over in MODES:
        torch.manual_seed(0)
        fe = TriAxialFrontend(**BASE, **over).eval()
        with torch.no_grad():
            tok, _, _, amp_t = fe(xb, return_amp_target=True)
        # tok (B,C,nb,P,D); amp_t (B,C,nb,P) = log mean amplitude of that token
        D = tok.shape[-1]
        T = tok.reshape(-1, D).double().numpy()
        y = amp_t.reshape(-1).double().numpy()
        keep = np.isfinite(y) & np.isfinite(T).all(axis=1)
        T, y = T[keep], y[keep]
        perm = rng.permutation(len(T))
        T, y = T[perm], y[perm]

        r_lin = r2_ridge(T, y)
        r_quad = r2_ridge(T ** 2, y)

        tp = tok.mean(dim=3)                      # pool over patches, as the head does
        yp = amp_t.mean(dim=3)
        Tp = tp.reshape(-1, D).double().numpy()
        yp = yp.reshape(-1).double().numpy()
        keep = np.isfinite(yp) & np.isfinite(Tp).all(axis=1)
        Tp, yp = Tp[keep], yp[keep]
        perm = rng.permutation(len(Tp))
        r_pool = r2_ridge(Tp[perm], yp[perm])

        print(f"    {name:>10} {r_lin:>8.3f} {r_quad:>8.3f} {r_pool:>8.3f}", flush=True)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        run(ds)
