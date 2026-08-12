import sys, torch
sys.path.insert(0, "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench")
from paclock_bench.models.paclock.build import build_model

BASE = dict(arch="triaxial", tokenizer_mode="pac_interaction", pac_token_mode="measured",
            interaction_mode="product", freq_mixer="attention",
            sample_rate=200, augmentations=[], dropout=0.1, n_heads=4)

print("目标: xlsx 的 PACLock (from scratch, full) = 1.64M")
for d_model in (64, 96, 128, 160):
    for depth in (4, 6, 8):
        for n_bands in (8,):
            cfg = dict(BASE, d_model=d_model, depth=depth, n_bands=n_bands,
                       n_channels=16, seq_len=2000, num_classes=2, dataset="tuab")
            try:
                m = build_model(cfg)
                p = sum(q.numel() for q in m.parameters()) / 1e6
                with torch.no_grad():
                    out = m(torch.randn(2, 16, 2000))
                print("  d_model=%-4d depth=%-2d n_bands=%d -> %.3fM  out=%s" % (
                    d_model, depth, n_bands, p, tuple(out.shape)))
            except Exception as e:
                print("  d_model=%-4d depth=%-2d n_bands=%d FAIL %s: %s" % (
                    d_model, depth, n_bands, type(e).__name__, str(e)[:80]))
