"""Verify matched 140k exports and real training batches inside a GPU allocation."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID')
ap=argparse.ArgumentParser()
ap.add_argument('--kind',choices=('native','crofremo'))
ap.add_argument('--out',type=Path,default=Path('results/pretrain_140k_smoke.json'))
args=ap.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,yaml
from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.models.build import build_model
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import set_seed
from paclock_bench.models.foundation.cbramod_adapter import VENDOR
assert torch.cuda.is_available() and torch.cuda.device_count()==1
rows=[];hashes={}
for kind in ([args.kind] if args.kind else ('native','crofremo')):
 config=Path('configs/pretrain_gate/tuev_'+kind+'_140k.yaml');cfg=yaml.safe_load(config.read_text())
 cfg['num_workers']=2
 path=Path(cfg['pretrained_path']);ck=torch.load(path,map_location='cpu',weights_only=False)
 assert ck['step']==140000 and ck['cfg']['kind']==kind
 for p in (config,path):hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 set_seed(0);tr,va,te,info=build_dataloaders(cfg)
 x,y=next(iter(tr));model=build_model(cfg,info['input_shape'])
 # Classifier construction removes proj_out; every remaining backbone tensor
 # must match the exported checkpoint, including the tokenizer.
 state=model.backbone.state_dict()
 missing=[k for k in state if k not in ck['model']]
 mismatch=[k for k in state if k in ck['model'] and not torch.equal(state[k],ck['model'][k])]
 assert not missing and not mismatch,(kind,missing,mismatch)
 del ck,state
 torch.cuda.reset_peak_memory_stats();model=model.cuda();x=x.cuda();y=y.cuda()
 row=dict(kind=kind,step=140000,matched_backbone_tensors=len(model.backbone.state_dict()),input_shape=list(x.shape))
 if kind=='crofremo':
  model.eval()
  with torch.no_grad():
   xx=x[:2].clone();patches=xx.reshape(2,xx.shape[1],-1,200)
   mask=torch.rand(patches.shape[:-1],device=x.device)<.5
   masked=patches.masked_fill(mask[...,None],0).reshape_as(xx)
   a=model.backbone.frontend(masked)[0];b=model.backbone.frontend(-masked)[0]
   row['masked_signflip_token_relative_l2']=float((a-b).norm()/a.norm().clamp_min(1e-12))
   row['masked_target_sign_pair_zero_prediction_mse']=float(patches[mask].square().mean())
   del xx,patches,mask,masked,a,b
 model.train();criterion=build_loss(cfg)
 groups=[dict(params=model.backbone_parameters(),lr=cfg['lr']),dict(params=model.head_parameters(),lr=cfg['lr']*5)]
 opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay']);times=[]
 for i in range(5):
  torch.cuda.synchronize();start=time.monotonic();opt.zero_grad(set_to_none=True)
  loss=criterion(model(x),y);assert torch.isfinite(loss)
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']);opt.step()
  torch.cuda.synchronize();times.append(time.monotonic()-start)
 row.update(steady_seconds=times[2:],warmup_excluded=2,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
 assert row['peak_GiB']<55,row
 rows.append(row);print(json.dumps(row),flush=True)
 del opt,model,x,y,tr,va,te,loss;gc.collect();torch.cuda.empty_cache()
for p in ('smoke/smoke_pretrain_140k.py','paclock_bench/models/foundation/cbramod_adapter.py','paclock_bench/models/foundation/cbramod_paclockfe_adapter.py','paclock_bench/models/paclock/frontend/triaxial.py'):
 hashes[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p in Path(VENDOR).rglob('*.py'):
 hashes['vendor/cbramod/'+str(p.relative_to(VENDOR))]=hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(ok=True,job=os.environ['SLURM_JOB_ID'],step=os.environ.get('SLURM_STEP_ID'),host=os.uname().nodename,torch=torch.__version__,rows=rows,sha256=hashes)
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(report,indent=2)+'\n')
print('PRETRAIN_140K_SMOKE_PASS',flush=True)
