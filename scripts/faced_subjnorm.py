"""Per-subject z-scored copy of FACED.

FACED is split across subjects (train 0-79, val 80-99, test 100-122), and
cross-subject emotion recognition is dominated by inter-subject amplitude and
spectral differences -- per-subject standardisation is routine in the
FACED/SEED literature for exactly this reason. Our pipeline applies only a
global divide-by-100, so every subject's own offset and scale survive into the
model, and with 80 training subjects the model can fit subject identity more
easily than emotion.

Statistics are computed per (subject, channel) and applied within that subject
only, so no information crosses the split boundary: each split's subjects are
disjoint by construction, and a subject's own statistics are available at test
time in any realistic deployment (you always have the recording you are
classifying).

FACED windows are emitted 3-per-video, 28 videos per subject, subjects in
order, so a subject occupies a contiguous block of 84 windows -- verified
against the manifest's per-split subject counts before use.
"""
import json
import os

import numpy as np

from paclock_bench.paths import expand

SRC = expand("$PACLOCK_PROC/processed/faced")
DST = expand("$PACLOCK_PROC/processed/faced_subjnorm")
PER_SUBJECT = 3 * 28          # 3 windows x 28 videos

os.makedirs(DST, exist_ok=True)
man = json.load(open("%s/manifest.json" % SRC))

for split in ("train", "val", "test"):
    X = np.load("%s/%s_signals.npy" % (SRC, split), mmap_mode="r")
    y = np.load("%s/%s_labels.npy" % (SRC, split))
    n, C, T = X.shape
    n_subj = n // PER_SUBJECT
    assert n % PER_SUBJECT == 0, "%s: %d windows is not a whole number of subjects" % (split, n)
    assert n_subj == man["splits"][split]["n_subjects"], (
        "%s: derived %d subjects, manifest says %d"
        % (split, n_subj, man["splits"][split]["n_subjects"]))

    out = np.empty((n, C, T), dtype=np.float32)
    for s in range(n_subj):
        lo, hi = s * PER_SUBJECT, (s + 1) * PER_SUBJECT
        blk = np.asarray(X[lo:hi], dtype=np.float64)          # (84, C, T)
        mu = blk.mean(axis=(0, 2), keepdims=True)             # per channel
        sd = blk.std(axis=(0, 2), keepdims=True)
        out[lo:hi] = ((blk - mu) / np.maximum(sd, 1e-6)).astype(np.float32)

    np.save("%s/%s_signals.npy" % (DST, split), out)
    np.save("%s/%s_labels.npy" % (DST, split), y)
    print("%-6s %d windows / %d subjects  std before=%.3f after=%.3f"
          % (split, n, n_subj, np.asarray(X[:256]).std(), out[:256].std()))

man["derived_from"] = SRC
man["normalisation"] = {
    "rule": "per (subject, channel) z-score, computed and applied within one "
            "subject's own 84 windows",
    "reason": "cross-subject split; inter-subject amplitude/scale differences "
              "otherwise dominate a 9-class emotion signal",
    "leakage": "none -- splits are subject-disjoint and each subject is "
               "normalised using only its own data",
}
json.dump(man, open("%s/manifest.json" % DST, "w"), indent=2, ensure_ascii=False)
print("wrote %s" % DST)
