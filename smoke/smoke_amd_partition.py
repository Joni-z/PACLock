"""Short hardware/real-batch check; run only through smoke_gpu.slurm."""
import argparse,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.models.build import build_model
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import set_seed
assert torch.cuda.is_available()
set_seed(0)
ap=argparse.ArgumentParser()
ap.add_argument('--config',default='configs/selfcoup/tuev_crofremo_s2.yaml')
args=ap.parse_args()
cfg=yaml.safe_load(Path(args.config).read_text());cfg['num_workers']=2
assert not cfg.get('disabled_reason'),cfg.get('disabled_reason')
tr,va,te,info=build_dataloaders(cfg)
x,y=next(iter(tr));x,y=x.cuda(),y.cuda().long()
m=build_model(cfg,info['input_shape']).cuda().train()
opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
loss_fn=build_loss(cfg);times=[]
for i in range(5):
 torch.cuda.synchronize();t=time.monotonic()
 opt.zero_grad(set_to_none=True);loss=loss_fn(m(x),y);loss.backward();opt.step()
 torch.cuda.synchronize();times.append(time.monotonic()-t)
 assert torch.isfinite(loss)
p=Path('results')/('amd-partition-smoke-'+os.environ['SLURM_JOB_ID']+'.json')
p.write_text(json.dumps(dict(ok=True,partition=os.environ.get('SLURM_JOB_PARTITION'),
 job=os.environ['SLURM_JOB_ID'],config=args.config,host=os.uname().nodename,torch=torch.__version__,rocm=torch.version.hip,
 device=torch.cuda.get_device_name(),visible_gpus=torch.cuda.device_count(),
 input_shape=list(x.shape),steps=5,warmup_excluded=2,steady_step_seconds=times[2:],
 peak_GiB=torch.cuda.max_memory_allocated()/2**30),indent=2)+'\n')
print(p.read_text())
