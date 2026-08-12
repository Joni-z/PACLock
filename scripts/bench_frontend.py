"""Time each stage of a PACLock step separately, on realistic shapes.

Three measured facts motivate this and none of them point at the encoder:

  * size_large (d_model 128->256, depth 6->8 = 5.3x the encoder FLOPs) costs
    only 7% wall time -- solving 35.3/32.8 = 1 + 4.3f puts the encoder at ~1.8%
    of the step.
  * patch400 (27.1 samp/s) is SLOWER than patch100 (39.5) even though it has 4x
    FEWER tokens. Token count is not what costs.
  * the whole model sustains ~1% of the MI210 fp32 peak.

What patch_len does change is the kernel size of the tokenizer convolutions,
and every conv in this frontend has in_channels=1, which is the shape both
MIOpen and cuDNN handle worst. This script measures that directly instead of
inferring it.
"""

from __future__ import annotations

import time
import torch

torch.backends.cudnn.benchmark = True
dev = "cuda"
B, C, nb, T, d = 32, 16, 8, 1000, 128
K = d // 2
PATCH = 200


def timeit(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


x = torch.randn(B, C, T, device=dev)
filt = torch.randn(B, C, nb, T, device=dev)

sinc = torch.nn.Conv1d(1, nb, kernel_size=201, padding=100, bias=False).to(dev)
tok = torch.nn.Conv1d(1, K, kernel_size=PATCH, stride=PATCH, bias=False).to(dev)

xs = x.reshape(B * C, 1, T)
fs = filt.reshape(B * C * nb, 1, T)

print("shapes: sinc in %s | tokenizer in %s" % (tuple(xs.shape), tuple(fs.shape)))
print()

res = {}
res["1. sinc filterbank  Conv1d(1,8,k=201)"] = timeit(lambda: sinc(xs))
res["2. hilbert  fft+ifft complex"] = timeit(
    lambda: torch.fft.ifft(torch.fft.fft(filt, dim=-1), dim=-1))
res["3. tokenizer x3  Conv1d(1,64,k=200,s=200)"] = timeit(
    lambda: (tok(fs), tok(fs), tok(fs)))

# the same tokenizer written as reshape + matmul: mathematically identical for
# stride == kernel_size, but one big GEMM instead of 4096 single-channel convs
W = tok.weight.detach().reshape(K, PATCH).t().contiguous()
res["3b. tokenizer x3 as reshape+matmul"] = timeit(
    lambda: [fs.reshape(-1, T // PATCH, PATCH) @ W for _ in range(3)])

# encoder-equivalent: one big GEMM over all tokens
N = C * nb * (T // PATCH)
h = torch.randn(B * N, d, device=dev)
Wq = torch.randn(d, 3 * d, device=dev)
res["4. one encoder-sized GEMM (B*N,d)@(d,3d)"] = timeit(lambda: h @ Wq)

tot = sum(v for k, v in res.items() if not k.startswith("3b"))
for k in sorted(res):
    share = "" if k.startswith("3b") else "  %5.1f%% of the four" % (res[k] / tot * 100)
    print("  %-46s %8.2f ms%s" % (k, res[k], share))
print()
print("  tokenizer conv / matmul speedup: %.1fx" % (
    res["3. tokenizer x3  Conv1d(1,64,k=200,s=200)"]
    / res["3b. tokenizer x3 as reshape+matmul"]))
