"""Exercise the actual trainer on a tiny CUDA task, including cooperative stop."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from paclock_bench.training import train

assert torch.cuda.is_available(), 'Run through slurm/smoke_gpu.slurm'
root = Path(tempfile.mkdtemp(prefix='paclock-control-smoke-'))
g = torch.Generator().manual_seed(41)
X = torch.randn(72, 1, 8, generator=g)
y = torch.arange(72) % 6
info = dict(input_shape=(1,8), n_samples=dict(train=72,val=72,test=72),
            manifest=dict(created_utc='synthetic-smoke'),
            class_counts={s:[12]*6 for s in ['train','val','test']})

def loaders(cfg):
    return tuple(DataLoader(TensorDataset(X,y),batch_size=8,shuffle=(s=='train'))
                 for s in ['train','val','test']) + (info,)

class Tiny(torch.nn.Module):
    def __init__(self, mode, directory):
        super().__init__()
        self.linear = torch.nn.Linear(8,6)
        self.calls, self.mode, self.directory = 0, mode, directory
    def forward(self, x):
        if self.training:
            self.calls += 1
            if self.calls == 11:
                if self.mode == 'signal': os.kill(os.getpid(), signal.SIGTERM)
                elif self.mode == 'file': (self.directory/'STOP').touch()
        return self.linear(x.flatten(1))

results = {}
for mode in ['unmonitored','normal','signal','file']:
    cfg = dict(name='smoke_'+mode, dataset='tuev', model='paclock', seed=0,
               device='cuda', num_classes=6, loss='cross_entropy',
               optimizer='adamw', lr=1e-3, weight_decay=1e-5,
               label_smoothing=.1, batch_size=8, epochs=4, patience=0,
               scheduler='cosine', num_workers=0, run_monitor=mode!='unmonitored')
    directory=root/cfg['name']/'seed0'
    cfgfile=root/(mode+'.yaml');cfgfile.write_text(yaml.safe_dump(cfg))
    with patch.object(train,'build_dataloaders',loaders), \
         patch.object(train,'build_model',lambda c,shape:Tiny(mode,directory)), \
         patch.object(sys,'argv',['train','--config',str(cfgfile),'--out',str(root)]):
        train.main()
    if mode in ['signal','file']:
        assert not (directory/'result.json').exists()
        assert not (directory/'test_scores.npz').exists()
        stopped=json.loads((directory/'stopped.json').read_text())
        assert stopped['stopped_by']=='operator_stop' and not stopped['test_evaluated']
        assert len(json.loads((directory/'progress.json').read_text())['history'])<4
    else:
        results[mode]=json.loads((directory/'result.json').read_text())
    if mode!='unmonitored':
        cp=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        Tiny('',directory).load_state_dict(cp['model'],strict=True)
        progress=json.loads((directory/'progress.json').read_text())
        assert all(sum(e['pred_counts'])==72 and sum(e['target_counts'])==72 for e in progress['history'])
        assert not list(directory.glob('*.tmp.*'))
assert results['normal']['test']==results['unmonitored']['test']
assert results['normal']['val_curve']==results['unmonitored']['val_curve']
print('PASS: actual CUDA trainer; identical monitored/unmonitored metrics; signal and file stops; no test scoring on operator stop; checkpoint strict reload; complete class accounting.')
print('SMOKE_ARTIFACTS',root)

# Workers inherit Python signal handlers when forked: they must still terminate.
from paclock_bench.training.run_control import RunControl
control=RunControl(root/'worker-signal',dict(name='worker-test',num_classes=2),{})
pid=os.fork()
if pid==0:
    os.kill(os.getpid(),signal.SIGTERM)
    os._exit(99)
_,status=os.waitpid(pid,0)
assert os.WIFSIGNALED(status) and os.WTERMSIG(status)==signal.SIGTERM
assert control.reason is None
import numpy as np
for logits in [np.array([-.5,.5]),np.array([[-.5],[.5]]),np.array([[.5,-.5],[-.5,.5]])]:
    control.record(tag='binary',val_loss=0.,metrics={},logits=logits,labels=np.array([0,1]),
                   lr=[.001],train_loss=0.,best=1.)
    assert control.history[-1]['confusion']==[[1,0],[0,1]]
control.restore_handlers()
print('PASS: inherited worker SIGTERM exits normally; binary head shapes score correctly.')
try:
    RunControl(root/'smoke_normal'/'seed0',dict(name='existing',num_classes=6),{})
except FileExistsError:
    pass
else:
    raise AssertionError('Existing run must not be overwritten or mixed with a stop receipt')
print('PASS: existing completed run is protected from overwrite.')

blocked=root/'blocked.yaml'
blocked.write_text(yaml.safe_dump(dict(disabled_reason='verified diagnostic block')))
with patch.object(sys,'argv',['train','--config',str(blocked)]):
    try:
        train.main()
    except ValueError as exc:
        assert 'Disabled experiment' in str(exc)
    else:
        raise AssertionError('Blocked configurations must fail before constructing loaders/models')
print('PASS: blocked configurations fail before loading data or constructing a model.')
