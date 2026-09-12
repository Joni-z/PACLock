"""Read initial lane scales and gradients on one fixed real training batch."""
import gc,json,os,sys
from pathlib import Path
sys.path.insert(0,os.getcwd())
import torch,yaml
from smoke.filter_response import check_low_band_response
from paclock_bench.training.train import set_seed
from paclock_bench.models.build import build_model
from paclock_bench.data.datasets import build_dataloaders
from paclock_bench.training.losses import build_loss
assert torch.cuda.is_available()
cfg=yaml.safe_load(Path('configs/factorized/tuev_f0.yaml').read_text());cfg['num_workers']=0
set_seed(0)
tr,va,te,info=build_dataloaders(cfg)
x,y=next(iter(tr));x,y=x.cuda(),y.cuda().long()
rows=[]
for arm in ['f0','f1','f2','f3']:
 cfg=yaml.safe_load(Path(f'configs/factorized/tuev_{arm}.yaml').read_text())
 set_seed(0);m=build_model(cfg,info['input_shape']).cuda().train()
 saved={}
 def hook(fe,inputs,output):
  saved['tokens']=output[0];output[0].retain_grad()
 h=m.frontend.register_forward_hook(hook)
 logits=m(x);loss=build_loss(cfg)(logits,y);loss.backward()
 z=saved['tokens'];dz=z.grad
 def rms(a):return float(a.detach().float().square().mean().sqrt())
 try:
  check_low_band_response(m.frontend);response_gate='pass'
 except AssertionError as exc:
  response_gate=str(exc)
 assert (response_gate=='pass') == (arm!='f3'), (arm,response_gate)
 response=torch.fft.rfft(m.frontend.sinc.filters().detach().squeeze(1),n=16384).abs()
 peak,idx=response.max(-1)
 rows.append(dict(response_gate=response_gate,filter_peak_gain=peak.cpu().tolist(),filter_dc_gain=response[:,0].cpu().tolist(),
                  filter_peak_hz=(idx*200/16384).cpu().tolist(),arm=arm,loss=float(loss),coupling_rms=rms(z[...,:128]),local_rms=rms(z[...,128:]),
                  coupling_grad_rms=rms(dz[...,:128]),local_grad_rms=rms(dz[...,128:]),
                  local_projector_grad_rms=rms(m.frontend.local_projection.weight.grad),
                  amplitude_projector_grad_rms=rms(m.frontend.amplitude_tokenizer.weight.grad),
                  finite=bool(torch.isfinite(z).all() and torch.isfinite(dz).all())))
 h.remove();del saved,m,z,dz,logits,loss;gc.collect();torch.cuda.empty_cache()
Path('results/factorized-scale-diagnostic-20260913.json').write_text(json.dumps(dict(
 scope='initialization only, one fixed TUEV training batch; not trained-checkpoint diagnosis',
 class_counts=info['class_counts'],rows=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2))
