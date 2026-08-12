import glob, json, statistics as st

DS = ["tuev", "isruc", "sleepedf", "faced", "bci_iv_2a", "physionet_mi", "tusz", "chbmit", "tuab"]

def get(cell):
    ps = sorted(glob.glob("runs/%s/seed*/result.json" % cell))
    if not ps:
        return None
    rs = [json.load(open(p)) for p in ps]
    k = rs[0]["primary_metric"]
    ok = [r for r in rs if r["verdict"]["ok"]]
    vals = [r["test"][k] for r in rs]
    return dict(metric=k, n=len(rs), n_ok=len(ok),
                mean=st.mean(vals), sd=st.stdev(vals) if len(vals) > 1 else 0.0,
                per_seed=[(r["seed"], round(r["test"][k], 4), r["verdict"]["ok"]) for r in rs])

print("%-13s %-9s %-28s %-28s" % ("dataset", "metric", "TFM-Tokenizer", "PACLock v2"))
print("-" * 82)
for ds in DS:
    tfm = get("%s-tfm_pretrained" % ds)
    ours = get("%s-paclock_v2" % ds)
    def fmt(r):
        if not r:
            return "--"
        s = "%.4f+-%.4f (%d/%d ok)" % (r["mean"], r["sd"], r["n_ok"], r["n"])
        return s
    print("%-13s %-9s %-28s %-28s" % (ds, (tfm or ours or {}).get("metric", "?"),
          fmt(tfm), fmt(ours)))
    if ours and ours["n_ok"] < ours["n"]:
        print("    per-seed:", ours["per_seed"])
