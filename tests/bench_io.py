import sys, time, numpy as np, torch
sys.path.insert(0, "/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench")
import yaml
from paclock_bench.data.datasets import build_dataloaders

cfg = yaml.safe_load(open("/work1/chenyuyou/yifanwang/Zhizhe/paclock-bench/configs/experiments/tuev_paclock_full.yaml"))
for nw in (16,):
    cfg["num_workers"] = nw
    tr, va, te, info = build_dataloaders(cfg)
    print("batch=%d num_workers=%d  train=%d" % (cfg["batch_size"], nw, info["n_samples"]["train"]))
    it = iter(tr)
    next(it)                                   # warm
    t = time.time(); n = 0
    for i, (X, y) in enumerate(it):
        n += X.shape[0]
        if i >= 30: break
    dt = time.time() - t
    print("  纯数据加载: %.1f ms/batch  %.0f 样本/秒  -> 一个 epoch 需 %.1f 分钟"
          % (dt/31*1000, n/dt, info["n_samples"]["train"]/(n/dt)/60))
