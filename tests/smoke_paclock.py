import sys, torch, yaml, glob, os
sys.path.insert(0, "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench")
from paclock_bench.models.build import build_model
SHAPES = {"tuab":(16,2000),"tuev":(16,1000),"tusz":(16,2000),"chbmit":(16,2000),
          "sleepedf":(2,3000),"isruc":(6,6000),"physionet_mi":(64,800),
          "bci_iv_2a":(22,800),"faced":(32,2000)}
for ds,(C,T) in SHAPES.items():
    cfg = yaml.safe_load(open("configs/experiments/%s_paclock_full.yaml" % ds))
    try:
        m = build_model(cfg, (C,T)).eval()
        with torch.no_grad(): out = m(torch.randn(2,C,T))
        p = sum(q.numel() for q in m.parameters())/1e6
        print("%-13s C=%-3d T=%-5d out=%-8s %.3fM  %s" % (ds,C,T,tuple(out.shape),p,
              "OK" if tuple(out.shape)==(2,cfg["num_classes"]) else "SHAPE?"))
    except Exception as e:
        print("%-13s FAIL %s: %s" % (ds, type(e).__name__, str(e)[:100]))
