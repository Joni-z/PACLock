import sys, torch
sys.path.insert(0, "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench")
from paclock_bench.models.foundation.tfm_adapter import build_tfm
CASES = [("tuab",16,10*200,2),("tuev",16,5*200,6),("tusz",16,10*200,2),("chbmit",16,10*200,2),
         ("sleepedf",2,30*200,5),("isruc",6,30*200,5),
         ("physionet_mi",64,4*200,4),("bci_iv_2a",22,4*200,4),("faced",32,10*200,9)]
for name, C, T, K in CASES:
    try:
        m = build_tfm(K, name, pretrained=True, setting="multiple", n_channels=C).eval()
        with torch.no_grad():
            out = m(torch.randn(2, C, T))
        p = sum(q.numel() for q in m.parameters())/1e6
        print("%-13s C=%-3d out=%-8s %.2fM  %s" % (name, C, tuple(out.shape), p,
              "OK" if tuple(out.shape)==(2,K) else "SHAPE?"))
    except Exception as e:
        print("%-13s C=%-3d FAIL %s: %s" % (name, C, type(e).__name__, str(e)[:110]))
