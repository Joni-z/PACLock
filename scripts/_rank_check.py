import glob, json, statistics as st

DS = ["tuab", "tuev", "tusz", "chbmit", "sleepedf", "isruc", "physionet_mi",
     "faced", "bci_iv_2a"]

# group-B pretrained foundation models -- the set CBraMod's own paper compares
# against (BIOT, LaBraM) plus what our matrix adds (EEGPT, TFM), and our own
# from-scratch group C for reference.
MODELS = ["biot_pretrained", "labram_pretrained", "cbramod_pretrained",
         "eegpt_pretrained", "tfm_pretrained", "paclock_v2",
         "biot_scratch", "labram_scratch", "cbramod_scratch"]


def get(cell):
    for root in ("runs", "runs_wrong_recipe", "archive/runs_paclock_p200"):
        ps = sorted(glob.glob("%s/%s/seed*/result.json" % (root, cell)))
        if ps:
            rs = [json.load(open(p)) for p in ps]
            ok = [r for r in rs if r["verdict"]["ok"]]
            if len(ok) < len(rs):
                continue
            k = rs[0]["primary_metric"]
            vals = [r["test"][k] for r in rs]
            flag = " *off-recipe*" if root == "runs_wrong_recipe" else ""
            return st.mean(vals), len(vals), k, flag
    return None


for ds in DS:
    rows = []
    for m in MODELS:
        r = get("%s-%s" % (ds, m))
        if r:
            rows.append((r[0], m, r[1], r[2], r[3]))
    if not rows:
        continue
    rows.sort(reverse=True)
    print("=== %s ===" % ds)
    for rank, (val, m, n, k, flag) in enumerate(rows, 1):
        mark = "  <-- CBraMod" if "cbramod" in m else ""
        print("  %d. %-20s %.4f (%s, %d seed)%s%s" % (rank, m, val, k, n, flag, mark))
    print()
