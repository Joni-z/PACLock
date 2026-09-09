import sys, os, yaml, torch, time
sys.path.insert(0, os.getcwd())
from paclock_bench.models.build import build_model
dev = "cuda"
def run(cfg_path, C, T, bs=2, big_bs=None):
    cfg = yaml.safe_load(open(cfg_path)); cfg["device"] = dev
    m = build_model(cfg, (C, T)).to(dev)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.randn(bs, C, T, device=dev) * 0.5
    y = m(x); y.float().sum().backward()
    print(f"{os.path.basename(cfg_path)}: out {tuple(y.shape)} params {n:.2f}M finite={bool(torch.isfinite(y).all())}", flush=True)
    if big_bs:
        m.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
        x = torch.randn(big_bs, C, T, device=dev) * 0.5; t0 = time.time()
        y = m(x); y.float().sum().backward(); torch.cuda.synchronize()
        print(f"   batch {big_bs}: peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB, {time.time()-t0:.1f}s fwd+bwd", flush=True)
    del m; torch.cuda.empty_cache()
run("configs/cf1/tuev_reve_crofremo.yaml", 16, 1000, big_bs=16)
run("configs/cf1/iiic_reve_crofremo.yaml", 16, 2000, big_bs=16)
run("configs/cf1/tuev_labram_crofremo.yaml", 23, 1000, big_bs=16)
run("configs/experiments/tuev_reve_scratch.yaml", 16, 1000)
print("SMOKE_OK")
