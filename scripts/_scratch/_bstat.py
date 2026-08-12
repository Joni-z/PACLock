import json, os, glob
R = "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/runs"
DS = ["tuab","tuev","tusz","chbmit","sleepedf","isruc","physionet_mi","bci_iv_2a","faced"]
MODELS = ["biot_prest16","labram_pretrained","cbramod_pretrained","eegpt_pretrained"]
print("%-14s %s" % ("dataset", "  ".join("%-18s" % m for m in MODELS)))
for ds in DS:
    row = []
    for m in MODELS:
        d = os.path.join(R, "%s-%s" % (ds, m))
        n = len(glob.glob(os.path.join(d, "seed*", "result.json")))
        row.append("%-18s" % ("%d/3" % n if n else "-"))
    print("%-14s %s" % (ds, "  ".join(row)))
