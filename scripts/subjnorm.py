"""Per-subject z-scored copies of the tier-3 corpora.

All three tier-3 corpora are evaluated across subjects (PhysioNet-MI and FACED
hold out whole subjects; BCI-IV-2a holds out a whole session of the same
subjects), and our pipeline applies only a global divide-by-100. Every
subject's own amplitude offset and scale therefore survive into the model,
which on 2k-7k training windows is an easier thing to fit than the task.
Per-subject standardisation is routine in both the emotion (FACED/SEED) and
BCI (motor imagery) literature for exactly this reason.

Each subject occupies a contiguous block of windows, and the block size is
derived from the manifest (n_windows / n_subjects) and asserted to divide
evenly rather than assumed -- a wrong block size would mix subjects and
silently produce a worse-than-useless normalisation.

No information crosses the split boundary: a subject is normalised using only
its own windows, within its own split.

    python -m scripts.subjnorm <dataset> [<dataset> ...]
"""
import json
import os
import sys

import numpy as np

from paclock_bench.paths import expand


def build(ds):
    src = expand("$PACLOCK_PROC/processed/%s" % ds)
    dst = expand("$PACLOCK_PROC/processed/%s_subjnorm" % ds)
    os.makedirs(dst, exist_ok=True)
    man = json.load(open("%s/manifest.json" % src))
    print("=== %s ===" % ds)

    for split in ("train", "val", "test"):
        xp = "%s/%s_signals.npy" % (src, split)
        if not os.path.exists(xp):
            continue
        X = np.load(xp, mmap_mode="r")
        y = np.load("%s/%s_labels.npy" % (src, split))
        n = X.shape[0]
        n_subj = man["splits"][split]["n_subjects"]
        if n % n_subj:
            raise SystemExit(
                "%s/%s: %d windows over %d subjects is not an even block; "
                "per-subject normalisation would mix subjects" % (ds, split, n, n_subj))
        per = n // n_subj

        out = np.empty(X.shape, dtype=np.float32)
        for s in range(n_subj):
            lo, hi = s * per, (s + 1) * per
            blk = np.asarray(X[lo:hi], dtype=np.float64)
            axes = tuple(i for i in range(blk.ndim) if i != blk.ndim - 2)
            mu = blk.mean(axis=axes, keepdims=True)
            sd = blk.std(axis=axes, keepdims=True)
            out[lo:hi] = ((blk - mu) / np.maximum(sd, 1e-6)).astype(np.float32)

        np.save("%s/%s_signals.npy" % (dst, split), out)
        np.save("%s/%s_labels.npy" % (dst, split), y)
        print("  %-6s %5d win / %3d subj (%d each)  std %.3f -> %.3f"
              % (split, n, n_subj, per, np.asarray(X[:256]).std(), out[:256].std()))

    man["derived_from"] = src
    man["normalisation"] = {
        "rule": "per (subject, channel) z-score within the subject's own block",
        "reason": "cross-subject / cross-session evaluation; subject-level "
                  "amplitude and scale otherwise dominate a small-data task",
        "leakage": "none -- each subject normalised from its own windows only",
    }
    json.dump(man, open("%s/manifest.json" % dst, "w"), indent=2, ensure_ascii=False)
    print("  -> %s" % dst)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        build(ds)
