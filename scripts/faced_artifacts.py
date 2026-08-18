"""Characterise FACED's artefact windows across all three splits.

train has ~2.9% of windows whose worst channel is ~3000x the typical window
(p90 RMS 0.42 vs p99 1301). Before deciding on a fix, establish:
  * does it hit val/test too (if so, test scores are partly noise-limited for
    every model, not just ours)
  * is it a whole-subject / whole-video problem or scattered epochs
  * how the labels distribute over the bad windows -- if the artefact
    correlates with class, dropping them changes the task rather than cleaning it
"""
import numpy as np

from paclock_bench.paths import expand

ROOT = expand("$PACLOCK_PROC/processed/faced")


def rms_per_window(X):
    N = X.shape[0]
    out = np.empty((N, X.shape[1]), dtype=np.float64)
    for i in range(0, N, 512):
        blk = np.asarray(X[i:i + 512], dtype=np.float64)
        out[i:i + 512] = np.sqrt((blk ** 2).mean(axis=-1))
    return out


for split in ("train", "val", "test"):
    X = np.load("%s/%s_signals.npy" % (ROOT, split), mmap_mode="r")
    y = np.load("%s/%s_labels.npy" % (ROOT, split))
    r = rms_per_window(X).max(axis=1)
    med = np.median(r)
    bad = r > 20 * med
    print("=== %s === n=%d  median worst-ch RMS=%.3f" % (split, len(r), med))
    print("    windows >20x median: %d (%.2f%%)" % (bad.sum(), 100 * bad.mean()))
    if bad.any():
        print("    their RMS: min=%.1f median=%.1f max=%.1f"
              % (r[bad].min(), np.median(r[bad]), r[bad].max()))
        # FACED windows are emitted 3-per-video, videos in order, subjects in
        # order -- so index/(3*28) recovers the subject slot within the split.
        subj = np.arange(len(r)) // (3 * 28)
        us, cnt = np.unique(subj[bad], return_counts=True)
        tot = np.bincount(subj, minlength=us.max() + 1)
        print("    affected subject-slots: %d of %d" % (len(us), subj.max() + 1))
        worst = np.argsort(-cnt)[:5]
        print("    worst slots (bad/total): %s"
              % ", ".join("%d:%d/%d" % (us[i], cnt[i], tot[us[i]]) for i in worst))
        cls = np.bincount(y[bad], minlength=9) / np.maximum(np.bincount(y, minlength=9), 1)
        print("    per-class bad rate: %s" % " ".join("%.3f" % v for v in cls))
    print()
