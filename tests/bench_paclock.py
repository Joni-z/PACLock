import sys, time, torch
sys.path.insert(0, "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench")
from paclock_bench.models.paclock.build import build_model

cfg = dict(arch="triaxial", tokenizer_mode="pac_interaction", pac_token_mode="measured",
           interaction_mode="product", freq_mixer="attention", sample_rate=200,
           augmentations=[], dropout=0.2, n_heads=4, d_model=128, depth=6, n_bands=8,
           kernel_size=201, patch_len=200, n_channels=16, seq_len=1000,
           num_classes=6, dataset="tuev", spatial_pe="xyz", band_pe="index")
dev = "cuda"
m = build_model(cfg).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
crit = torch.nn.CrossEntropyLoss()

for bs in (32, 128):
    x = torch.randn(bs, 16, 1000, device=dev)
    y = torch.randint(0, 6, (bs,), device=dev)
    for _ in range(3):                      # warmup
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    torch.cuda.synchronize()

    # full step
    t = time.time()
    for _ in range(10):
        opt.zero_grad(); crit(m(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    full = (time.time() - t) / 10

    # frontend only (sinc + Hilbert + PAC token construction)
    t = time.time()
    for _ in range(10):
        with torch.no_grad():
            m.frontend(m.augment(x) if hasattr(m, "augment") else x)
    torch.cuda.synchronize()
    fe = (time.time() - t) / 10

    print("batch=%-4d  完整step %.1f ms   frontend(no_grad) %.1f ms  (%.0f%%)  "
          "样本/秒 %.0f" % (bs, full*1000, fe*1000, 100*fe/full, bs/full))
