"""Short hardware/real-batch check; run only through smoke_gpu.slurm."""
import argparse,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.models.build import build_model
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import set_seed
assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
ap=argparse.ArgumentParser()
ap.add_argument('--config',default='configs/selfcoup/tuev_crofremo_s2.yaml')
ap.add_argument('--require-budget-fit',action='store_true')
args=ap.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text());cfg['num_workers']=2
set_seed(int(cfg.get('seed',0)))
assert not cfg.get('disabled_reason'),cfg.get('disabled_reason')
tr,va,te,info=build_dataloaders(cfg)
x,y=next(iter(tr));x,y=x.cuda(),y.cuda().long()
m=build_model(cfg,info['input_shape']).cuda().train()
opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
loss_fn=build_loss(cfg);times=[]
for i in range(5):
 torch.cuda.synchronize();t=time.monotonic()
 opt.zero_grad(set_to_none=True);loss=loss_fn(m(x),y)
 assert torch.isfinite(loss)
 loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.get('grad_clip',1.0))
 assert torch.isfinite(norm)
 opt.step()
 torch.cuda.synchronize();times.append(time.monotonic()-t)
 assert all(torch.isfinite(p).all() for p in m.parameters())
projected_train_seconds=statistics.mean(times[2:])*len(tr)*cfg['epochs']
if args.require_budget_fit:
 assert cfg.get('max_hours') and projected_train_seconds<cfg['max_hours']*3600*.8, \
  ('Training alone would consume over 80% of the budget',projected_train_seconds,cfg.get('max_hours'))
p=Path('results')/('amd-partition-smoke-'+os.environ['SLURM_JOB_ID']+'.json')
p.write_text(json.dumps(dict(ok=True,partition=os.environ.get('SLURM_JOB_PARTITION'),
 job=os.environ['SLURM_JOB_ID'],config=args.config,seed=int(cfg.get('seed',0)),host=os.uname().nodename,torch=torch.__version__,rocm=torch.version.hip,
 device=torch.cuda.get_device_name(),visible_gpus=torch.cuda.device_count(),
 input_shape=list(x.shape),steps=5,warmup_excluded=2,steady_step_seconds=times[2:],
 peak_GiB=torch.cuda.max_memory_allocated()/2**30,
 projected_train_seconds=projected_train_seconds,
 projection_note='Excludes data loading and evaluation; not a completion guarantee.',
 config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
 smoke_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2)+'\n')
print(p.read_text())
