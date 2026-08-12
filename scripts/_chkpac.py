import json, os
P="/work1/chenyuyou/yifanwang/Zhizhe/processed_pac"
F="/work1/chenyuyou/yifanwang/Zhizhe/processed"
for ds in ["tuab","tuev","tusz","chbmit","sleepedf","isruc","physionet_mi","bci_iv_2a"]:
    try:
        m=json.load(open(os.path.join(P,ds,"manifest.json")))
        f=json.load(open(os.path.join(F,ds,"manifest.json")))
        ms={k:v.get("n_windows") for k,v in m.get("splits",{}).items()}
        fs={k:v.get("n_windows") for k,v in f.get("splits",{}).items()}
        pr=m.get("protocol",{})
        same = ms==fs
        print("%-13s hp=%-5s notch=%-5s  窗口数 %s  %s" % (
            ds, pr.get("hp"), pr.get("notch"), ms,
            "= 冻结协议 ✓" if same else "≠ 冻结协议 %s" % fs))
    except Exception as e:
        print("%-13s ERR %s" % (ds, str(e)[:60]))
