"""Build + one forward for every new-corpus baseline config, one per process.

A single process cannot check them all: EEGPT's adapter puts vendor/eegpt on
sys.path, whose top-level `models` package then shadows vendor/tfm's, so TFM
reports a bogus ModuleNotFoundError. Each real job is its own process, so the
gate must be too.
"""
import glob, json, subprocess, sys

SHAPES = {}
for ds in ("tuep", "tuar", "adfd"):
    m = json.load(open("/work1/chenyuyou/yifanwang/Zhizhe/processed/%s/manifest.json" % ds))
    SHAPES[ds] = tuple(m["splits"]["train"]["shape"])

ONE = '''
import sys, yaml, torch
from paclock_bench.models.build import build_model
cfg = yaml.safe_load(open(sys.argv[1]))
shape = tuple(int(x) for x in sys.argv[2].split(","))
m = build_model(cfg, input_shape=shape); m.eval()
with torch.no_grad(): y = m(torch.randn(2, *shape))
assert torch.isfinite(y).all() and y.shape[0] == 2, "bad output"
print("%.2fM %s" % (sum(p.numel() for p in m.parameters())/1e6, tuple(y.shape)))
'''

bad = 0
for cfg_path in sorted(glob.glob("configs/experiments/*.yaml")):
    ds = cfg_path.split("/")[-1].split("_")[0]
    if ds not in SHAPES:
        continue
    name = cfg_path.split("/")[-1][:-5]
    shape = SHAPES[ds]
    r = subprocess.run([sys.executable, "-c", ONE, cfg_path,
                        ",".join(str(x) for x in shape)],
                       capture_output=True, text=True, timeout=900)
    if r.returncode == 0:
        print("  %-28s OK   %s" % (name, r.stdout.strip().splitlines()[-1]), flush=True)
    else:
        bad += 1
        err = [l for l in r.stderr.strip().splitlines() if l.strip()][-1][:110]
        print("  %-28s FAIL %s" % (name, err), flush=True)
print("\n%d problems" % bad)
sys.exit(1 if bad else 0)
