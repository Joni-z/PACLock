"""Drop FACED's non-physiological training windows into a cleaned copy.

Two subjects in the FACED training split are corrupted end to end (slot 23:
84/84 windows, slot 30: 57/84) with worst-channel RMS around 805 against a
0.166 median -- roughly 5000x, i.e. not EEG. They are 2.6% of the training
windows but dominate the loss of anything trained on them. val/test are clean
(0.12% / 0.05%, and those outliers are ~25x, not ~5000x), so this degrades
learning without invalidating the test metric.

The raw FACED release is no longer on this cluster, so the clean copy is
derived by dropping windows from the existing processed arrays rather than by
re-running preprocessing/faced.py. That is a strictly-subtractive operation on
data every model already reads, and the dropped indices are recorded in the
manifest so the operation is auditable and reversible.

Threshold: worst-channel RMS > 20x the split median. Chosen from the observed
gap -- p90 is 0.42 and the artefact population starts at 4.2, so anything from
~5x to ~20x separates the same set; 20x is the conservative end, keeping
borderline-but-plausible windows.

train only. val/test are left byte-identical so reported scores stay
comparable to every number already in the matrix.
"""
import json
import os
import shutil

import numpy as np

from paclock_bench.paths import expand

SRC = expand("$PACLOCK_PROC/processed/faced")
DST = expand("$PACLOCK_PROC/processed/faced_clean")
FACTOR = 20.0

os.makedirs(DST, exist_ok=True)


def worst_rms(X):
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(0, X.shape[0], 512):
        blk = np.asarray(X[i:i + 512], dtype=np.float64)
        out[i:i + 512] = np.sqrt((blk ** 2).mean(axis=-1)).max(axis=1)
    return out


report = {}
for split in ("train", "val", "test"):
    Xp = "%s/%s_signals.npy" % (SRC, split)
    yp = "%s/%s_labels.npy" % (SRC, split)
    X = np.load(Xp, mmap_mode="r")
    y = np.load(yp)

    if split != "train":
        shutil.copyfile(Xp, "%s/%s_signals.npy" % (DST, split))
        shutil.copyfile(yp, "%s/%s_labels.npy" % (DST, split))
        report[split] = {"kept": int(len(y)), "dropped": 0,
                         "note": "untouched so scores stay comparable"}
        print("%-6s copied unchanged (%d windows)" % (split, len(y)))
        continue

    r = worst_rms(X)
    thr = FACTOR * np.median(r)
    keep = r <= thr
    Xc = np.asarray(X[keep])
    yc = y[keep]
    np.save("%s/%s_signals.npy" % (DST, split), Xc)
    np.save("%s/%s_labels.npy" % (DST, split), yc)
    dropped = int((~keep).sum())
    report[split] = {
        "kept": int(keep.sum()), "dropped": dropped,
        "threshold_worst_channel_rms": float(thr),
        "dropped_indices": np.flatnonzero(~keep).tolist(),
        "class_counts_before": np.bincount(y, minlength=9).tolist(),
        "class_counts_after": np.bincount(yc, minlength=9).tolist(),
    }
    print("%-6s kept %d, dropped %d (%.2f%%), threshold=%.3f"
          % (split, keep.sum(), dropped, 100 * dropped / len(keep), thr))

man = json.load(open("%s/manifest.json" % SRC))
man["derived_from"] = SRC
man["cleaning"] = {
    "rule": "drop train windows whose worst-channel RMS exceeds %.0fx the "
            "split median" % FACTOR,
    "reason": "two subjects' recordings are non-physiological (~5000x normal "
              "amplitude); they dominated the training loss",
    "applied_to": "train only; val/test byte-identical to the source",
    "report": report,
}
json.dump(man, open("%s/manifest.json" % DST, "w"), indent=2, ensure_ascii=False)
print("\nwrote %s" % DST)
