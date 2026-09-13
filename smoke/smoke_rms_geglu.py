"""Legacy equivalence and train-only budget smoke, via slurm/smoke_gpu.slurm."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml
import paclock_bench.models.paclock.build as pacbuild
from paclock_bench.models.build import build_model
from paclock_bench.models.paclock.triaxial import GEGLUFFN, TriAxialBlock
from paclock_bench.paths import expand
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import amp_context, set_seed

assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
torch.set_num_threads(8)
output = ROOT/'results'/('rms-geglu-smoke-'+os.environ['SLURM_JOB_ID']+'.json')
assert not output.exists()
baseline_commit = '345fd9cb60a08da0875717e8d5ac5cb66bfa2697'
reference_source = subprocess.check_output([
    'git', 'show', baseline_commit+':paclock_bench/models/paclock/triaxial.py'])
assert hashlib.sha256(reference_source).hexdigest() == '08ea20014236aaaf72393b960339774d99c25d0761839c5fe097e3f9a68011a4'
reference = types.ModuleType('paclock_reference_triaxial')
reference.__package__ = 'paclock_bench.models.paclock'
exec(compile(reference_source, '<committed legacy backbone>', 'exec'), reference.__dict__)

cfg = yaml.safe_load((ROOT/'configs/backbone_gate/tuev_legacy_s0.yaml').read_text())
cfg['model_kwargs']['augmentations'] = []
set_seed(17)
with patch.object(pacbuild, 'TriAxialEncoder', reference.TriAxialEncoder):
    before = build_model(cfg, (16, 1000)).cuda()
set_seed(17)
after = build_model(cfg, (16, 1000)).cuda()
assert before.state_dict().keys() == after.state_dict().keys()
for k, v in before.state_dict().items():
    torch.testing.assert_close(v, after.state_dict()[k], rtol=0, atol=0)
x = torch.randn(2, 16, 1000, device='cuda')
y = torch.tensor([0, 1], device='cuda')
before.eval(); after.eval()
with torch.no_grad():
    torch.testing.assert_close(before(x), after(x), rtol=0, atol=0)
for m in (before, after):
    m.train(); set_seed(23)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-5)
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
for k, v in before.state_dict().items():
    torch.testing.assert_close(v, after.state_dict()[k], rtol=0, atol=0)
legacy_equivalence = dict(initial_state=True, eval_logits=True, dropout_optimizer_step=True)
del before, after, opt, x, y
gc.collect(); torch.cuda.empty_cache()

try:
    TriAxialBlock(192, freq_mixer='attention', block_variant='misspelled')
except ValueError:
    pass
else:
    raise AssertionError('unknown block variant was silently accepted')

records = []
config_paths = []
for dataset in ('tuev', 'sleepedf'):
    for variant in ('legacy', 'rms_geglu'):
        path = ROOT/f'configs/backbone_gate/{dataset}_{variant}_s0.yaml'
        config_paths.append(path)
        cfg = yaml.safe_load(path.read_text())
        set_seed(cfg['seed'])
        data = Path(expand(cfg['data_root']))
        signals = np.load(data/'train_signals.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(data/'train_labels.npy', mmap_mode='r', allow_pickle=False)
        indices = np.random.default_rng(cfg['seed']).choice(len(labels), cfg['batch_size'], replace=False)
        x = torch.from_numpy(np.array(signals[indices], copy=True)).cuda()
        y = torch.from_numpy(np.array(labels[indices], copy=True)).long().cuda()
        model = build_model(cfg, tuple(signals.shape[1:])).cuda()
        if variant == 'rms_geglu':
            assert len(model.encoder.blocks) == cfg['model_kwargs']['depth']
            for block in model.encoder.blocks:
                assert all(isinstance(getattr(block, n), torch.nn.RMSNorm)
                           for n in ('n_time', 'n_space', 'n_freq', 'n_ffn'))
                assert isinstance(block.ffn, GEGLUFFN)
            model.eval()
            with torch.no_grad():
                assert torch.isfinite(model(torch.zeros_like(x[:2]))).all()
                assert torch.isfinite(model(torch.full_like(x[:2], 1e-7))).all()
                expected = model(x[:2])
            clone = build_model(cfg, tuple(signals.shape[1:])).cuda().eval()
            clone.load_state_dict(model.state_dict(), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(clone(x[:2]), expected, rtol=0, atol=0)
            del clone, expected
        model.train()
        criterion = build_loss(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        steps = math.ceil(len(labels)/cfg['batch_size'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs']*steps)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for i in range(6):
            torch.cuda.synchronize(); start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with amp_context(cfg, torch.device('cuda')):
                loss = criterion(model(x), y)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            assert torch.isfinite(norm)
            optimizer.step(); scheduler.step()
            torch.cuda.synchronize(); durations.append(time.perf_counter()-start)
            if variant == 'rms_geglu':
                for block in model.encoder.blocks:
                    for param in (block.n_ffn.weight, block.ffn.in_proj.weight, block.ffn.out_proj.weight):
                        assert param.grad is not None and torch.isfinite(param.grad).all()
                        assert param.grad.abs().sum() > 0
            assert all(torch.isfinite(p).all() for p in model.parameters())
        projected = statistics.mean(durations[2:])*steps*cfg['epochs']
        records.append(dict(config=str(path.relative_to(ROOT)), dataset=dataset, variant=variant,
            parameters=sum(p.numel() for p in model.parameters()), input_shape=list(x.shape),
            sample_rate=cfg['sample_rate'], train_rows=len(labels), steps_per_epoch=steps,
            warmup_steps_excluded=2, steady_step_seconds=durations[2:],
            projected_training_seconds=projected, peak_GiB=torch.cuda.max_memory_allocated()/2**30,
            finite_loss_gradients_parameters=True, training_cap_hours=cfg['max_hours'],
            fits_80_percent_of_cap=projected < .8*cfg['max_hours']*3600))
        del model, optimizer, scheduler, loss, norm, x, y, signals, labels
        gc.collect(); torch.cuda.empty_cache()
for dataset in ('tuev', 'sleepedf'):
    a, b = [r for r in records if r['dataset'] == dataset]
    assert abs(b['parameters']-a['parameters'])/a['parameters'] < .01
files = [Path(__file__), ROOT/'paclock_bench/models/paclock/triaxial.py',
         ROOT/'paclock_bench/models/paclock/build.py', ROOT/'paclock_bench/models/paclock/frontend/factorized.py',
         ROOT/'paclock_bench/training/train.py', ROOT/'paclock_bench/training/losses.py'] + config_paths
result = dict(job=os.environ['SLURM_JOB_ID'], step=os.environ.get('SLURM_STEP_ID'),
    partition=os.environ.get('SLURM_JOB_PARTITION'), host=os.uname().nodename,
    torch=torch.__version__, rocm=torch.version.hip, device=torch.cuda.get_device_name(),
    baseline_commit=baseline_commit, legacy_bit_exact=legacy_equivalence,
    unknown_variant_rejected=True, new_variant_flat_finite_and_strict_checkpoint_roundtrip=True,
    validation_or_test_arrays_loaded=False, records=records,
    all_fit=all(r['fits_80_percent_of_cap'] for r in records),
    projection_limitation='Training compute only; excludes data loading, validation, final evaluation and saving.',
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
assert result['all_fit'], 'One or more configurations fail the planned budget check'
