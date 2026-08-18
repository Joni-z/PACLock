import glob, json

DS = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc", "physionet_mi",
     "faced", "bci_iv_2a"]
MODELS = ["biot_scratch", "labram_scratch", "cbramod_scratch", "paclock_v2"]

print("%-14s %-18s %-18s %-18s %-18s" % ("dataset", *MODELS))
missing = []
for ds in DS:
    row = []
    for m in MODELS:
        ps = sorted(glob.glob("runs/%s-%s/seed*/result.json" % (ds, m)))
        if not ps:
            row.append("MISSING")
            missing.append((ds, m, "no runs dir"))
            continue
        rs = [json.load(open(p)) for p in ps]
        ok = [r for r in rs if r["verdict"]["ok"]]
        status = "%d/3 seed" % len(rs)
        if len(ok) < len(rs):
            status += " (%d bad)" % (len(rs) - len(ok))
        if len(ok) < 3:
            missing.append((ds, m, status))
        row.append(status)
    print("%-14s %-18s %-18s %-18s %-18s" % (ds, *row))

print()
print("=== cells not yet complete (3/3 valid seeds) ===")
for ds, m, why in missing:
    print("  %-14s %-18s %s" % (ds, m, why))
