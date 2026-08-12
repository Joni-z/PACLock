import sys, torch
sys.path.insert(0, "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench")
from paclock_bench.models.foundation.biot_adapter import build_biot
from paclock_bench.models.foundation.labram_adapter import build_labram

CASES = [
    ("sleepedf",      2, 30*200, 15*200, 5),
    ("isruc",         6, 30*200, 15*200, 5),
    ("physionet_mi", 64,  4*200,  4*200, 4),
    ("bci_iv_2a",    22,  4*200,  4*200, 4),
    ("faced",        32, 10*200, 10*200, 9),
    ("chbmit",       16, 10*200, 10*200, 2),
]
shown = False
for name, C, T_native, T_model, K in CASES:
    x = torch.randn(2, C, T_native)
    for tag, fn in [
        ("BIOT",   lambda: build_biot(K, C, checkpoint="prest16", seq_len=T_model)),
        ("LaBraM", lambda: build_labram(K, C, montage_mode="positional", seq_len=T_model)),
    ]:
        try:
            mm = fn().eval()
            if tag == "LaBraM" and not shown:
                pe = mm.model.pos_embed
                print("LaBraM pos_embed %s -> max %d time patches (= %d samples @200Hz)"
                      % (tuple(pe.shape), pe.shape[1]-1, (pe.shape[1]-1)*200))
                shown = True
            with torch.no_grad():
                out = mm(x)
            p = sum(q.numel() for q in mm.parameters())/1e6
            print("%-13s %-7s in=%-14s out=%-8s %.2fM  %s" % (
                name, tag, tuple(x.shape), tuple(out.shape), p,
                "OK" if tuple(out.shape) == (2, K) else "SHAPE?"))
        except Exception as e:
            print("%-13s %-7s FAIL %s: %s" % (name, tag, type(e).__name__, str(e)[:100]))
