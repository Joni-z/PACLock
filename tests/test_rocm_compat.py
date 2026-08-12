"""ROCm compatibility check for the three risks flagged in BIG_CLUSTER_HANDOFF.md sec.7:
FFT (Hilbert transform), complex arithmetic, and BF16/DDP.
"""
import torch
import torch.nn as nn

dev = "cuda"
print("torch", torch.__version__, "| hip", torch.version.hip)
print("device:", torch.cuda.get_device_name(0), "| count", torch.cuda.device_count())
print()

ok = {}

# 1. basic matmul
try:
    a = torch.randn(512, 512, device=dev)
    b = a @ a.T
    torch.cuda.synchronize()
    ok["matmul fp32"] = f"OK {tuple(b.shape)}"
except Exception as e:
    ok["matmul fp32"] = f"FAIL {e}"

# 2. rfft/irfft -- the Hilbert transform path in models/frontend/analytic.py
try:
    x = torch.randn(8, 16, 2000, device=dev)
    X = torch.fft.rfft(x, dim=-1)
    y = torch.fft.irfft(X, n=2000, dim=-1)
    torch.cuda.synchronize()
    err = (y - x).abs().max().item()
    ok["rfft/irfft roundtrip"] = f"OK max_err={err:.3e}"
except Exception as e:
    ok["rfft/irfft roundtrip"] = f"FAIL {e}"

# 3. full complex fft + complex arithmetic (analytic signal / PAC coupling Z_ij)
try:
    x = torch.randn(8, 16, 2000, device=dev)
    n = x.shape[-1]
    Xf = torch.fft.fft(x, dim=-1)
    h = torch.zeros(n, device=dev)
    h[0] = 1
    h[1:(n + 1) // 2] = 2
    if n % 2 == 0:
        h[n // 2] = 1
    analytic = torch.fft.ifft(Xf * h, dim=-1)
    amp = analytic.abs()
    phase = torch.angle(analytic)
    # complex exp used by the PAC token: exp(-i * angle)
    z = amp * torch.exp(-1j * phase)
    torch.cuda.synchronize()
    ok["hilbert analytic + complex"] = f"OK amp{tuple(amp.shape)} dtype={z.dtype}"
except Exception as e:
    ok["hilbert analytic + complex"] = f"FAIL {e}"

# 4. complex autograd (gradient must flow through the coupling computation)
try:
    x = torch.randn(4, 8, 512, device=dev, requires_grad=True)
    Xf = torch.fft.fft(x, dim=-1)
    loss = (Xf.abs() ** 2).mean()
    loss.backward()
    torch.cuda.synchronize()
    gn = x.grad.norm().item()
    ok["complex autograd"] = f"OK grad_norm={gn:.4f}"
except Exception as e:
    ok["complex autograd"] = f"FAIL {e}"

# 5. BF16 autocast
try:
    m = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256)).to(dev)
    x = torch.randn(32, 256, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
        loss = y.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    ok["bf16 autocast"] = f"OK dtype={y.dtype}"
except Exception as e:
    ok["bf16 autocast"] = f"FAIL {e}"

# 6. FFT under autocast (known to be finicky: fft does not support half/bf16)
try:
    x = torch.randn(4, 8, 512, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Xf = torch.fft.fft(x.float(), dim=-1)
    torch.cuda.synchronize()
    ok["fft inside autocast (float cast)"] = f"OK dtype={Xf.dtype}"
except Exception as e:
    ok["fft inside autocast (float cast)"] = f"FAIL {e}"

# 7. scaled_dot_product_attention (encoder attention path)
try:
    q = torch.randn(4, 8, 128, 64, device=dev)
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()
    ok["sdpa attention"] = f"OK {tuple(o.shape)}"
except Exception as e:
    ok["sdpa attention"] = f"FAIL {e}"

# 8. conv1d with large kernel (sinc frontend uses kernel_size=201)
try:
    conv = nn.Conv1d(16, 128, kernel_size=201, padding=100, groups=1).to(dev)
    x = torch.randn(8, 16, 2000, device=dev)
    y = conv(x)
    torch.cuda.synchronize()
    ok["conv1d k=201"] = f"OK {tuple(y.shape)}"
except Exception as e:
    ok["conv1d k=201"] = f"FAIL {e}"

print("=" * 60)
for k, v in ok.items():
    status = "PASS" if v.startswith("OK") else "FAIL"
    print(f"[{status}] {k}: {v}")
print("=" * 60)
nfail = sum(1 for v in ok.values() if not v.startswith("OK"))
print(f"{len(ok) - nfail}/{len(ok)} passed")
