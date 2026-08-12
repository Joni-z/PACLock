"""Smoke-test EEGPT across every corpus shape in the matrix.

    python -m tests.smoke_eegpt
"""

from paclock_bench.models.foundation.eegpt_adapter import build_eegpt, DATASETS

CASES = [("sleepedf",2,30*200,5), ("isruc",6,30*200,5), ("physionet_mi",64,4*200,4),
         ("bci_iv_2a",22,4*200,4), ("faced",32,10*200,9), ("chbmit",16,10*200,2),
         ("tuab",16,10*200,2), ("tuev",16,5*200,6), ("tusz",16,10*200,2)]
for name, C, T, K in CASES:
    try:
        m = build_eegpt(K, C, name).eval()
        x = torch.randn(2, C, T)
        with torch.no_grad():
            out = m(x)
        p = sum(q.numel() for q in m.parameters())/1e6
        bb = sum(q.numel() for q in m.target_encoder.parameters())/1e6
        print("%-13s in=%-14s out=%-8s total=%6.2fM backbone=%6.2fM %s" % (
            name, tuple(x.shape), tuple(out.shape), p, bb,
            "OK" if tuple(out.shape) == (2, K) else "SHAPE?"))
    except Exception as e:
        print("%-13s FAIL %s: %s" % (name, type(e).__name__, str(e)[:120]))
