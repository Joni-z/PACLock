import sys, time, yaml, torch
sys.path.insert(0, "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench")
from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.models.build import build_model
from paclock_bench.training.losses import build_loss

cfg = yaml.safe_load(open("/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/configs/experiments/tuev_paclock_full.yaml"))
tr, va, te, info = build_dataloaders(cfg)
dev = "cuda"
m = build_model(cfg, info["input_shape"]).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
crit = build_loss(cfg)
print("GPU:", torch.cuda.get_device_name(0), "| batch", cfg["batch_size"],
      "| steps/epoch", len(tr))

t_data = t_h2d = t_fwd = t_bwd = 0.0
n = 0
it = iter(tr)
X, y = next(it)                                     # warm
for _ in range(3):
    Xg, yg = X.to(dev), y.to(dev).long()
    opt.zero_grad(); crit(m(Xg), yg).backward(); opt.step()
torch.cuda.synchronize()

t0 = time.time()
for i in range(60):
    ta = time.time()
    try: X, y = next(it)
    except StopIteration: break
    t_data += time.time() - ta

    ta = time.time()
    Xg = X.to(dev, non_blocking=True); yg = y.to(dev, non_blocking=True).long()
    torch.cuda.synchronize(); t_h2d += time.time() - ta

    ta = time.time()
    loss = crit(m(Xg), yg)
    torch.cuda.synchronize(); t_fwd += time.time() - ta

    ta = time.time()
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize(); t_bwd += time.time() - ta
    n += 1
wall = time.time() - t0

print("%d steps in %.1f s  ->  %.1f ms/step" % (n, wall, wall/n*1000))
for name, t in [("数据加载", t_data), ("H2D 传输", t_h2d), ("前向", t_fwd), ("反向+优化", t_bwd)]:
    print("  %-10s %7.1f ms/step  (%4.1f%%)" % (name, t/n*1000, 100*t/wall))
print("一个 epoch 预计 %.1f 分钟" % (wall/n*len(tr)/60))
