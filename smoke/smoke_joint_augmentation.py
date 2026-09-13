"""Exercise every existing augmentation on real training batches only."""
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
import yaml
from paclock_bench.models.build import build_model
from paclock_bench.paths import expand
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import amp_context, set_seed

assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
output = Path('results/audits/joint-augmentation-smoke-20260913.json')
assert not output.exists(), output
records = []
paths = [__file__, 'paclock_bench/models/paclock/augment.py',
         'paclock_bench/models/paclock/build.py', 'paclock_bench/models/build.py',
         'paclock_bench/models/paclock/frontend/factorized.py',
         'paclock_bench/models/paclock/frontend/triaxial.py',
         'paclock_bench/models/paclock/triaxial.py',
         'paclock_bench/training/train.py', 'paclock_bench/training/losses.py']
for dataset in ('tuev', 'tuar'):
    path = Path('configs/factorized_joint_augment') / (dataset + '_f4_aug_s1.yaml')
    paths.append(str(path))
    cfg = yaml.safe_load(path.read_text())
    set_seed(cfg['seed'])
    root = Path(expand(cfg['data_root']))
    signals = np.load(root / 'train_signals.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(root / 'train_labels.npy', mmap_mode='r', allow_pickle=False)
    indices = np.random.default_rng(cfg['seed']).choice(len(labels), cfg['batch_size'], replace=False)
    x = torch.from_numpy(np.array(signals[indices], copy=True)).cuda()
    y = torch.from_numpy(np.array(labels[indices], copy=True)).cuda().long()
    model = build_model(cfg, tuple(signals.shape[1:])).cuda().train()
    criterion = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    modules = model.augment.augs
    specs = cfg['model_kwargs']['augmentations']
    torch.cuda.reset_peak_memory_stats()
    measured = []
    for step in range(2 + len(modules)):
        model.augment.augs = modules if step < 2 else torch.nn.ModuleList([modules[step - 2]])
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with amp_context(cfg, 'cuda'):
            logits = model(x)
        loss = criterion(logits.float(), y)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        grad = model.frontend.local_projection.weight.grad
        assert grad[:32].abs().sum() > 0 and grad[32:].abs().sum() > 0
        for param in (model.frontend.phase_tokenizer.weight, model.frontend.amplitude_tokenizer.weight):
            assert param.grad is not None and param.grad.abs().sum() > 0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        assert torch.isfinite(norm)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if step >= 2:
            measured.append(dict(module=specs[step - 2], seconds=elapsed, loss=float(loss.detach())))
    model.augment.augs = modules
    model.eval()
    with torch.no_grad():
        predicted = model(x)
        model.augment = torch.nn.Identity()
        torch.testing.assert_close(model(x), predicted, rtol=0, atol=0)
    projected = statistics.mean(r['seconds'] for r in measured) * math.ceil(len(labels) / cfg['batch_size']) * cfg['epochs']
    assert projected < cfg['max_hours'] * 3600 * .8, ('Training projection exceeds 80% of cap', projected)
    record = dict(dataset=dataset, seed=cfg['seed'], config=str(path), batch=list(x.shape),
                  measured=measured, warmup_steps_excluded=2, all_three_lanes_receive_gradients=True,
                  eval_augmentation_bypass_exact=True, peak_GiB=torch.cuda.max_memory_allocated() / 2**30,
                  projected_training_seconds=projected,
                  projection_note='Training only; excludes loading, validation and final evaluation.')
    records.append(record)
    print(json.dumps(record), flush=True)
    del model, optimizer, criterion, x, y, logits, loss, predicted, modules, signals, labels, grad, param
    gc.collect()
    torch.cuda.empty_cache()
receipt = dict(ok=True, job=os.environ['SLURM_JOB_ID'], host=os.uname().nodename,
               git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               training_batches_only=True, runs=records,
               sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(receipt, indent=2) + '\n')
print('AUGMENTATION_SMOKE_OK', output, flush=True)
