import sys, os, torch, time
sys.path.insert(0, os.getcwd())
from paclock_bench.models.foundation.cbramod_pretrain import build_mpm, generate_mask
dev = "cuda"
for kind, kw in [("native", {}), ("crofremo", {"tokenizer_mode": "pac_interaction", "interaction_mode": "rotation"})]:
    m = build_mpm(kind, **kw).to(dev)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.randn(8, 16, 10, 200, device=dev) * 0.5
    mask = generate_mask(8, 16, 10, 0.5, dev)
    y = m(x, mask); loss = torch.nn.functional.mse_loss(y[mask == 1], x[mask == 1]); loss.backward()
    print(f"{kind}: out {tuple(y.shape)} params {n:.2f}M loss {loss.item():.4f} finite={bool(torch.isfinite(y).all())}", flush=True)
    sd = m.export_state_dict(); print(f"   export keys: {len(sd)} e.g. {list(sd)[:2]}", flush=True)
    m.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    x = torch.randn(128, 16, 10, 200, device=dev) * 0.5; mask = generate_mask(128, 16, 10, 0.5, dev); t0 = time.time()
    y = m(x, mask); torch.nn.functional.mse_loss(y[mask == 1], x[mask == 1]).backward(); torch.cuda.synchronize()
    print(f"   batch 128: peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB, {time.time()-t0:.2f}s fwd+bwd", flush=True)
    del m; torch.cuda.empty_cache()
# round-trip: the exported crofremo state dict must load into the finetune backbone
from paclock_bench.models.foundation.cbramod_paclockfe_adapter import PACLockCBraModBackbone
from paclock_bench.models.foundation.cbramod_adapter import _import_cbramod, BACKBONE_ARGS
m = build_mpm("crofremo", tokenizer_mode="pac_interaction", interaction_mode="rotation")
fb = PACLockCBraModBackbone(tokenizer_mode="pac_interaction", interaction_mode="rotation", band_mode="channels")
missing, unexpected = fb.load_state_dict(m.export_state_dict(), strict=False)
print("crofremo round-trip: missing", len(missing), missing[:3], "unexpected", len(unexpected), unexpected[:3], flush=True)
mn = build_mpm("native"); CB = _import_cbramod()(**BACKBONE_ARGS); CB.load_state_dict(mn.export_state_dict(), strict=True); print("native round-trip: strict OK", flush=True)
print("SMOKE_OK")
