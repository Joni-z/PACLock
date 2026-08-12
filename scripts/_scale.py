import numpy as np, json, os
R = "/work1/chenyuyou/yifanwang/Zhizhe/processed"
for ds in ["faced","bci_iv_2a","physionet_mi","sleepedf","isruc","chbmit","tuab"]:
    x = np.load(os.path.join(R, ds, "train_signals.npy"), mmap_mode="r")
    s = np.asarray(x[:200], dtype=np.float32)
    m = json.load(open(os.path.join(R, ds, "manifest.json")))
    norm = (m.get("protocol") or {}).get("normalization")
    print("%-13s norm=%-12s std=%9.5f  absmax=%9.4f  -> LaBraM loader /100 => std %.2e"
          % (ds, str(norm), s.std(), np.abs(s).max(), s.std()/100))
