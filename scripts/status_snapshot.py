"""One table per corpus: best baseline, our frozen config, our best variant.

Feeds docs/STATUS.md so the progress document quotes measured numbers rather
than remembered ones.

    sbatch slurm/run_cpu.slurm scripts.status_snapshot
"""
import glob
import json
import os
import statistics as st

KEY = {"tuab": "auroc", "tuev": "cohen_kappa", "tusz": "pr_auc",
       "chbmit": "pr_auc", "sleepedf": "cohen_kappa", "isruc": "cohen_kappa",
       "physionet_mi": "balanced_acc", "faced": "balanced_acc",
       "bci_iv_2a": "balanced_acc", "tuar": "cohen_kappa"}
CHANCE = {"tuev": 0.0, "faced": 0.111, "bci_iv_2a": 0.25, "physionet_mi": 0.25}
OURS = "paclock"


def cells(ds):
    out = {}
    for d in sorted(glob.glob("runs/%s-*" % ds)):
        name = os.path.basename(d).replace(ds + "-", "")
        vals = []
        for f in sorted(glob.glob(d + "/*/result.json")):
            v = json.load(open(f))["test"].get(KEY[ds])
            if v is not None:
                vals.append(v)
        if vals:
            out[name] = (st.mean(vals), len(vals),
                         st.stdev(vals) if len(vals) > 1 else 0.0)
    return out


print("%-14s %-12s | %-26s | %-22s | %-26s" %
      ("corpus", "metric", "best baseline", "ours (frozen v2)", "ours (best variant)"))
print("-" * 118)
for ds, k in KEY.items():
    c = cells(ds)
    if not c:
        continue
    base = {n: v for n, v in c.items() if not n.startswith(OURS)}
    ours = {n: v for n, v in c.items() if n.startswith(OURS)}
    fmt = lambda n, v: "%s %.4f±%.4f(%d)" % (n, v[0], v[2], v[1])
    b = max(base.items(), key=lambda kv: kv[1][0]) if base else None
    o = max(ours.items(), key=lambda kv: kv[1][0]) if ours else None
    v2 = ours.get("paclock_v2")
    print("%-14s %-12s | %-26s | %-22s | %-26s" % (
        ds, k,
        fmt(*b) if b else "-",
        ("%.4f±%.4f(%d)" % (v2[0], v2[2], v2[1])) if v2 else "-",
        fmt(*o) if o else "-"))

print()
print("=== every PACLock variant that beats the best baseline ===")
for ds, k in KEY.items():
    c = cells(ds)
    base = {n: v for n, v in c.items() if not n.startswith(OURS)}
    if not base:
        continue
    top = max(v[0] for v in base.values())
    win = {n: v for n, v in c.items() if n.startswith(OURS) and v[0] > top}
    if win:
        for n, v in sorted(win.items(), key=lambda kv: -kv[1][0]):
            print("  %-14s %-28s %.4f  (best baseline %.4f, +%.4f, n=%d)"
                  % (ds, n, v[0], top, v[0] - top, v[1]))
