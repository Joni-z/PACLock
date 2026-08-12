import json, os
R = "/work1/chenyuyou/yifanwang/Zhizhe/processed"
for ds in ["tuab","tuev","tusz","chbmit","sleepedf","isruc","physionet_mi","bci_iv_2a","faced"]:
    p = os.path.join(R, ds, "manifest.json")
    if not os.path.exists(p):
        print("== %s: no manifest ==" % ds); continue
    m = json.load(open(p))
    print("="*72); print("== %s ==" % ds)
    for k, v in (m.get("protocol") or {}).items():
        print("   %-16s %s" % (k, str(v)[:150]))
    for sp, d in (m.get("splits") or {}).items():
        print("   [%-5s] n=%-8s shape=%-12s subj=%-5s classes=%s" % (
            sp, d.get("n_windows"), d.get("shape"),
            len(d.get("subjects") or []), d.get("class_counts")))
    qc = m.get("qc") or {}
    if qc: print("   qc: %s" % str(qc)[:160])
