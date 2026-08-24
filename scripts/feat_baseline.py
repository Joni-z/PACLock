"""Classical-feature baseline: bandpower + Hjorth + spectral entropy -> LR/LDA.

    python -m scripts.feat_baseline --dataset tuev --seed 0

The 2026 critique literature's first demand (and gate 1 of the four-gate
attribution protocol, arXiv:2607.24519): a foundation model must beat the
handcrafted-feature family under the same split before any representation
claim. Features are the "simple family" those papers use -- per channel:
five canonical band powers (log), three Hjorth parameters, spectral entropy
-- 9 x C per window. Two classical heads (multinomial logistic regression
and LDA) are trained; the one with the better VALIDATION primary metric is
reported on test, and result.json records which won. One run directory,
``runs/<ds>-feat_best/seed<k>/``, in the trainer's result format so
scripts/fill_xlsx.py ingests it unchanged.

Determinism: features and LDA are deterministic; LR takes the seed. Seeds
exist to satisfy the 3-seed reporting convention -- their spread is ~0 and
the notes sheet will show that honestly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from paclock_bench.paths import expand
from paclock_bench.training.metrics import compute_metrics, primary_metric

BANDS = [(0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 50.0)]


def features(X: np.ndarray, fs: float) -> np.ndarray:
    """(N, C, T) -> (N, C*9) float64. Chunk-friendly, pure numpy."""
    N, C, T = X.shape
    Xf = X.astype(np.float64)
    # band powers from the one-sided periodogram
    spec = np.abs(np.fft.rfft(Xf, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    bp = np.empty((N, C, len(BANDS)))
    for i, (lo, hi) in enumerate(BANDS):
        sel = (freqs >= lo) & (freqs < hi)
        bp[:, :, i] = np.log(spec[:, :, sel].mean(axis=-1) + 1e-12)
    # Hjorth activity / mobility / complexity
    d1 = np.diff(Xf, axis=-1)
    d2 = np.diff(d1, axis=-1)
    var0 = Xf.var(axis=-1) + 1e-12
    var1 = d1.var(axis=-1) + 1e-12
    var2 = d2.var(axis=-1) + 1e-12
    activity = np.log(var0)
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    # spectral entropy over the 0.5-50 Hz range
    sel = (freqs >= 0.5) & (freqs < 50.0)
    p = spec[:, :, sel]
    p = p / (p.sum(axis=-1, keepdims=True) + 1e-24)
    sent = -(p * np.log(p + 1e-24)).sum(axis=-1)
    out = np.concatenate(
        [bp, activity[..., None], mobility[..., None], complexity[..., None],
         sent[..., None]], axis=-1)
    return out.reshape(N, -1)


def load_split(root: str, split: str, fs: float, chunk: int = 4096):
    X = np.load(os.path.join(root, f"{split}_signals.npy"), mmap_mode="r")
    y = np.load(os.path.join(root, f"{split}_labels.npy"))
    feats = np.concatenate(
        [features(np.asarray(X[i:i + chunk], dtype=np.float32), fs)
         for i in range(0, len(X), chunk)])
    return feats, np.asarray(y).ravel()


def scores_of(clf, F: np.ndarray, ncls: int) -> np.ndarray:
    """Probability-ish scores shaped the way compute_metrics expects."""
    p = clf.predict_proba(F)
    logp = np.log(p + 1e-12)
    return logp if ncls > 2 else logp        # (N, ncls) softmax-compatible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    root = expand(args.data_root or f"$PACLOCK_PROC/processed/{args.dataset}")
    man = json.load(open(os.path.join(root, "manifest.json")))
    fs = float(man["protocol"].get("sample_rate", 200))
    key = primary_metric(args.dataset)

    t0 = time.time()
    Ftr, ytr = load_split(root, "train", fs)
    Fva, yva = load_split(root, "val", fs)
    Fte, yte = load_split(root, "test", fs)
    ncls = int(max(ytr.max(), yva.max(), yte.max())) + 1
    print(f"[{args.dataset}] features {Ftr.shape} train / {Fva.shape[0]} val / "
          f"{Fte.shape[0]} test, {ncls} classes, extracted in "
          f"{time.time()-t0:.0f}s", flush=True)

    scaler = StandardScaler().fit(Ftr)
    Ftr, Fva, Fte = scaler.transform(Ftr), scaler.transform(Fva), scaler.transform(Fte)

    cands = {
        "logreg": LogisticRegression(max_iter=500, C=1.0,
                                     class_weight="balanced", n_jobs=-1,
                                     random_state=args.seed),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    }
    results = {}
    for name, clf in cands.items():
        t1 = time.time()
        clf.fit(Ftr, ytr)
        val = compute_metrics(yva, scores_of(clf, Fva, ncls), ncls)
        results[name] = (clf, val)
        print(f"  {name}: val {key} {val[key]:.4f}  ({time.time()-t1:.0f}s)",
              flush=True)

    best = max(results, key=lambda n: results[n][1][key])
    clf, val = results[best]
    test = compute_metrics(yte, scores_of(clf, Fte, ncls), ncls)
    print(f"  -> {best} wins on val; test {key} {test[key]:.4f}", flush=True)

    n_params = sum(np.prod(getattr(clf, a).shape)
                   for a in ("coef_", "intercept_") if hasattr(clf, a))
    run_dir = os.path.join(args.out, f"{args.dataset}-feat_best",
                           f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump({
        "name": f"{args.dataset}-feat_best",
        "dataset": args.dataset,
        "model": "feat_best",
        "group": "A0",
        "seed": args.seed,
        "n_params_M": float(n_params) / 1e6,
        "primary_metric": key,
        "best_val": float(val[key]),
        "verdict": {"ok": True, "status": "ok", "reason": "ok",
                    "n_evals": 1, "peak_index": 0, "best_val": float(val[key]),
                    "chance": 0.0},
        "epochs_run": 1, "stopped_by": "closed-form",
        "wall_time_sec": time.time() - t0,
        "winner": best,
        "val": {k: float(v) for k, v in val.items()},
        "test": {k: float(v) for k, v in test.items()},
        "class_counts": {"train": sorted(Counter(ytr.tolist()).items())},
        "config": {"name": f"{args.dataset}-feat_best", "group": "A0",
                   "dataset": args.dataset, "model": "feat_best",
                   "loss": "closed-form", "lr": None, "batch_size": None,
                   "epochs": 1, "patience": None, "eval_every_steps": None,
                   "features": "bandpower5+hjorth3+spec_entropy per channel",
                   "candidates": list(cands), "winner": best,
                   "seed": args.seed},
    }, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    print("wrote", run_dir, flush=True)


if __name__ == "__main__":
    main()
