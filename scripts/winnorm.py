"""Per-window, per-channel z-scored copies of the tier-3 corpora.

Same motivation as the per-subject version: all three tier-3 corpora are
evaluated across subjects (or across sessions of the same subjects), and our
pipeline applies only a global divide-by-100, so recording-level amplitude and
scale reach the model intact and are easier to fit than the task on 2k-7k
windows.

Per-WINDOW rather than per-subject because PhysioNet-MI's subjects do not have
equal window counts (90 / 91.26 / 90.15 per subject across splits) and the
arrays carry no per-window subject id, so fixed-size blocking would mix
subjects -- an assertion in scripts/subjnorm.py caught exactly that. Per-window
normalisation is well defined for every corpus, needs no subject metadata, and
cannot leak: each window is standardised using only its own samples.

    python -m scripts.winnorm <dataset> [<dataset> ...]
"""
import json
import os
import sys

import numpy as np

from paclock_bench.paths import expand


def build(ds):
    src = expand("$PACLOCK_PROC/processed/%s" % ds)
    dst = expand("$PACLOCK_PROC/processed/%s_winnorm" % ds)
    os.makedirs(dst, exist_ok=True)
    man = json.load(open("%s/manifest.json" % src))
    print("=== %s ===" % ds)

    for split in ("train", "val", "test"):
        xp = "%s/%s_signals.npy" % (src, split)
        if not os.path.exists(xp):
            continue
        X = np.load(xp, mmap_mode="r")
        y = np.load("%s/%s_labels.npy" % (src, split))
        out = np.empty(X.shape, dtype=np.float32)
        for i in range(0, X.shape[0], 512):
            blk = np.asarray(X[i:i + 512], dtype=np.float64)
            mu = blk.mean(axis=-1, keepdims=True)          # per (window, channel)
            sd = blk.std(axis=-1, keepdims=True)
            out[i:i + 512] = ((blk - mu) / np.maximum(sd, 1e-6)).astype(np.float32)
        np.save("%s/%s_signals.npy" % (dst, split), out)
        np.save("%s/%s_labels.npy" % (dst, split), y)
        print("  %-6s %6d win  std %.3f -> %.3f"
              % (split, X.shape[0], np.asarray(X[:256]).std(), out[:256].std()))

    man["derived_from"] = src
    man["normalisation"] = {
        "rule": "per (window, channel) z-score",
        "reason": "cross-subject / cross-session evaluation; recording-level "
                  "amplitude and scale otherwise dominate a small-data task",
        "leakage": "none -- each window standardised from its own samples only",
    }
    json.dump(man, open("%s/manifest.json" % dst, "w"), indent=2, ensure_ascii=False)
    print("  -> %s" % dst)


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        build(ds)
