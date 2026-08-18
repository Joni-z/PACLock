"""Confirm train_subsample actually reduces the split and keeps every class."""
import numpy as np
from paclock_bench.data.datasets import build_dataloaders

base = dict(data_root='$PACLOCK_PROC/processed/tuev', batch_size=32, num_workers=0)
full = build_dataloaders(dict(base))[3]
print('full   ', full['n_samples']['train'], full['class_counts']['train'])
for cap in (2160, 6720, 20000):
    info = build_dataloaders(dict(base, train_subsample=cap))[3]
    cc = info['class_counts']['train']
    print('cap=%-6d %6d %s  min_class=%d' % (cap, info['n_samples']['train'], cc, min(cc)))
    assert info['n_samples']['train'] == cap, 'cap not honoured'
    assert min(cc) >= 1, 'a class was dropped'
    assert info['input_shape'] == full['input_shape'], 'shape changed'
print('OK')
