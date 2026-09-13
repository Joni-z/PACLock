"""Bound the head-fidelity pair with real training batches; no validation/test arrays."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
from paclock_bench.models.build import build_model
from paclock_bench.paths import expand
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import amp_context, set_seed

assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
torch.set_num_threads(8)
output = ROOT/'results/audits/cbramod-head-training-smoke-20260913.json'
assert not output.exists()
records = []
for width in (800, 1000):
    path = ROOT/f'configs/baseline_head_confirm/tuev_cbramod_h{width}_s1.yaml'
    cfg = yaml.safe_load(path.read_text())
    set_seed(cfg['seed'])
    data = Path(expand(cfg['data_root']))
    signals = np.load(data/'train_signals.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(data/'train_labels.npy', mmap_mode='r', allow_pickle=False)
    indices = np.random.default_rng(cfg['seed']).choice(len(labels), cfg['batch_size'], replace=False)
    x = torch.from_numpy(np.array(signals[indices], copy=True)).cuda()
    y = torch.from_numpy(np.array(labels[indices], copy=True)).long().cuda()
    model = build_model(cfg, tuple(signals.shape[1:])).cuda().train()
    criterion = build_loss(cfg)
    head_lr = .001*(cfg['batch_size']/256)**.5
    optimizer = torch.optim.AdamW([
        {'params': list(model.backbone_parameters()), 'lr': cfg['lr']},
        {'params': list(model.head_parameters()), 'lr': head_lr}],
        weight_decay=cfg['weight_decay'])
    steps = math.ceil(len(labels)/cfg['batch_size'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs']*steps)
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for i in range(10):
        torch.cuda.synchronize(); started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with amp_context(cfg, torch.device('cuda')):
            loss = criterion(model(x), y)
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        assert torch.isfinite(norm)
        optimizer.step(); scheduler.step()
        torch.cuda.synchronize(); durations.append(time.perf_counter()-started)
        assert all(torch.isfinite(p).all() for p in model.parameters())
    projected = statistics.mean(durations[2:])*steps*cfg['epochs']
    records.append(dict(config=str(path.relative_to(ROOT)), config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        width=width, seed=cfg['seed'], input_shape=list(x.shape), train_rows=len(labels),
        steps_per_epoch=steps, epochs=cfg['epochs'], optimizer_lrs=[cfg['lr'],head_lr],
        warmup_steps_excluded=2, steady_step_seconds=durations[2:],
        projected_training_seconds=projected, peak_GiB=torch.cuda.max_memory_allocated()/2**30,
        finite_loss_gradients_parameters=True, training_cap_hours=cfg['max_hours'],
        fits_80_percent_of_cap=projected < .8*cfg['max_hours']*3600))
    del model, optimizer, scheduler, x, y, signals, labels
    gc.collect(); torch.cuda.empty_cache()
files = [Path(__file__), ROOT/'paclock_bench/models/build.py',
         ROOT/'paclock_bench/models/foundation/cbramod_adapter.py',
         ROOT/'paclock_bench/training/train.py', ROOT/'paclock_bench/training/losses.py']
record = dict(allocation=os.environ['SLURM_JOB_ID'], step=os.environ.get('SLURM_STEP_ID'),
    host=os.uname().nodename, device=torch.cuda.get_device_name(), torch=torch.__version__,
    gpu_selector=os.environ.get('HIP_VISIBLE_DEVICES'), records=records,
    all_fit=all(r['fits_80_percent_of_cap'] for r in records),
    projection_limitation='Training compute only; excludes loading, validation, final evaluation and saving.',
    validation_or_test_arrays_loaded=False,
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2), flush=True)
