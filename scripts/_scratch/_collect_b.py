import json, os, glob
import numpy as np
R = "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/runs"
PRIMARY = {"tuab":"auroc","tuev":"cohen_kappa","tusz":"pr_auc","chbmit":"pr_auc",
           "sleepedf":"cohen_kappa","isruc":"cohen_kappa","physionet_mi":"balanced_acc",
           "bci_iv_2a":"balanced_acc","faced":"balanced_acc"}
DS = ["tuab","tuev","tusz","chbmit","sleepedf","isruc","physionet_mi","bci_iv_2a","faced"]
MODELS = ["biot_prest16","labram_pretrained","cbramod_pretrained","eegpt_pretrained"]
for ds in DS:
    key = PRIMARY[ds]
    line = []
    for m in MODELS:
        fs = sorted(glob.glob(os.path.join(R, "%s-%s" % (ds, m), "seed*", "result.json")))
        if not fs:
            line.append("%-22s" % "-"); continue
        vals = []
        for f in fs:
            r = json.load(open(f))
            t = r.get("test") or r.get("test_metrics") or {}
            if key in t: vals.append(t[key])
        if not vals:
            line.append("%-22s" % "(no %s)" % key); continue
        line.append("%-22s" % ("%.4f±%.4f n=%d" % (np.mean(vals), np.std(vals), len(vals))))
    print("%-13s %-13s %s" % (ds, key, " ".join(line)))
print()
print("列顺序: BIOT / LaBraM / CBraMod / EEGPT")
