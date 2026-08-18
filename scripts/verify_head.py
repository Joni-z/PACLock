"""Gates for head mode 'spatial'.

  1. mean/band/attn are bit-identical to the pre-change head -- every frozen
     config uses `mean`, so drift here would move finished cells.
  2. the spatial projection is in model.parameters() BEFORE any forward runs.
     This is the bug the change fixes: the layer used to be created inside
     forward(), after the optimizer had already captured the parameter list, so
     it stayed at its random initialisation for the entire run and the mode
     would have benchmarked as "tried and failed" without ever being trained.
  3. it actually receives gradient, and the grid-size guard fires on a mismatch.

    sbatch slurm/run.slurm scripts.verify_head
"""
import torch

from paclock_bench.models.paclock.head import ClassificationHead
from paclock_bench.models.paclock._head_prev import (
    ClassificationHead as PrevHead,
)

D, NB, C, P, NCLS = 32, 8, 22, 4, 4
ok = True


def check(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{'  ' + detail if detail else ''}",
          flush=True)


x = torch.randn(3, C * NB * P, D)
grid = (C, NB, P)

print("=== 1. mean/band/attn unchanged", flush=True)
for mode in ("mean", "band", "attn"):
    torch.manual_seed(0)
    new = ClassificationHead(D, NCLS, mode=mode, n_bands=NB, n_channels=C).eval()
    torch.manual_seed(0)
    old = PrevHead(D, NCLS, mode=mode, n_bands=NB).eval()
    old.load_state_dict(new.state_dict(), strict=True)
    with torch.no_grad():
        d = (new(x, grid) - old(x, grid)).abs().max().item()
    check(f"{mode} bit-identical", d == 0.0, f"max|diff| = {d:.3e}")

print("\n=== 2. spatial projection exists before the first forward", flush=True)
h = ClassificationHead(D, NCLS, mode="spatial", n_bands=NB, n_channels=C)
names = [n for n, _ in h.named_parameters()]
n_par = sum(p.numel() for p in h.parameters())
check("proj.weight in parameters()", "proj.weight" in names, str(names))
check("param count is C*D*ncls + ...", n_par == C * D * NCLS + NCLS + 2 * D,
      f"{n_par} params")

print("\n=== 3. it trains", flush=True)
opt = torch.optim.SGD(h.parameters(), lr=0.1)      # captured BEFORE any forward
before = h.proj.weight.detach().clone()
loss = torch.nn.functional.cross_entropy(h(x, grid), torch.tensor([0, 1, 2]))
loss.backward()
check("proj.weight has grad", h.proj.weight.grad is not None
      and h.proj.weight.grad.abs().sum().item() > 0)
opt.step()
moved = (h.proj.weight - before).abs().max().item()
check("optimizer step moves it", moved > 0, f"max|delta| = {moved:.3e}")

print("\n=== 4. grid guard", flush=True)
try:
    h(torch.randn(2, (C + 1) * NB * P, D), (C + 1, NB, P))
    check("wrong C raises", False)
except ValueError as e:
    check("wrong C raises", True, str(e)[:60])

print(f"\n{'ALL CHECKS PASSED' if ok else 'FAILURES ABOVE'}")
raise SystemExit(0 if ok else 1)
