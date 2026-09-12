"""Run a full 1/4/8-GPU pack, refilling slots until its config queue drains."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

PARTITIONS={'mi2101x':1,'mi3501x':1,'mi2104x':4,'mi2508x':8}

def plan(specs, partition, default_seed, slots=None):
    import yaml
    if partition not in PARTITIONS:
        raise ValueError('Supported partitions: '+', '.join(PARTITIONS))
    capacity=PARTITIONS[partition]
    if slots is not None and slots!=capacity:
        raise ValueError('GPU count must match the partition')
    if len(specs)<capacity:
        raise ValueError(f'{partition} needs at least {capacity} configs; use mi2101x for a single config')
    out=[];seen=set()
    for index,spec in enumerate(specs):
        path,sep,seed=spec.rpartition(':')
        if not sep:path,seed=spec,str(default_seed)
        seed=int(seed)
        if seed<0:raise ValueError('seed must be nonnegative')
        p=Path(path).resolve();cfg=yaml.safe_load(p.read_text())
        if cfg.get('disabled_reason'):raise ValueError(f'{p}: {cfg["disabled_reason"]}')
        key=(cfg['name'],seed)
        if key in seen:raise ValueError(f'Duplicate output run: {key}')
        directory=Path('runs')/cfg['name']/f'seed{seed}'
        existing=[n for n in ('STOP','progress.json','best.pt','result.json','stopped.json') if (directory/n).exists()]
        if existing:raise ValueError(f'Existing output artifacts in {directory}: {existing}; choose a new run name')
        seen.add(key)
        out.append(dict(index=index,config=str(p),name=cfg['name'],seed=seed))
    return capacity,out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('specs',nargs='+');ap.add_argument('--partition',default=os.environ.get('SLURM_JOB_PARTITION','mi2104x'))
    ap.add_argument('--seed',type=int,default=int(os.environ.get('SEED','0')))
    ap.add_argument('--plan-only',action='store_true')
    args=ap.parse_args()
    capacity,items=plan(args.specs,args.partition,args.seed)
    if args.plan_only:
        print(json.dumps(dict(partition=args.partition,slots=capacity,runs=items),indent=2));return
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run inside a SLURM allocation')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=capacity:
        raise RuntimeError(f'{args.partition} requires {capacity} visible GPUs; found {torch.cuda.device_count()}')
    job=os.environ['SLURM_JOB_ID'];name=os.environ.get('SLURM_JOB_NAME','packed')
    runtime=dict(python=sys.version,torch=torch.__version__,rocm=torch.version.hip,
                 host=os.uname().nodename,torch_module=os.environ.get('PACLOCK_TORCH_MODULE'),
                 rocblas_override=os.environ.get('PACLOCK_USE_ROCBLAS','0'),
                 device=torch.cuda.get_device_name(0),git_commit=subprocess.check_output(
                     ['git','rev-parse','HEAD'],text=True).strip())
    Path('logs').mkdir(exist_ok=True);Path('runs').mkdir(exist_ok=True)
    receipt=Path('logs')/f'{name}-{job}-pack.json'
    pending=list(items);active={};done=[];stopping=[False]
    def snapshot():
        tmp=receipt.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(dict(job=job,partition=args.partition,slots=capacity,runtime=runtime,
                                      pending=pending,running=[v[1] for v in active.values()],done=done),indent=2)+'\n')
        tmp.replace(receipt)
    def stop(sig,frame):
        stopping[0]=True
        # Signal trainer parents: their handlers checkpoint and exit; do not
        # terminate their DataLoader workers before the trainer can finalize.
        for proc,_,_ in active.values():
            if proc.poll() is None:proc.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while pending or active:
        for gpu,(proc,item,log) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();item.update(exit_code=rc,finished_at=time.time())
            done.append(item);del active[gpu];snapshot()
        if stopping[0]:
            if not active:break
        else:
            for gpu in sorted(set(range(capacity))-set(active)):
                if not pending:break
                item=pending.pop(0)
                env=os.environ.copy();env['HIP_VISIBLE_DEVICES']=str(gpu)
                for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:env[k]='1'
                cache=f'/tmp/miopen_{job}_{gpu}_{item["index"]}'
                env['MIOPEN_USER_DB_PATH']=cache;env['MIOPEN_CUSTOM_CACHE_DIR']=cache
                logpath=Path('logs')/f'{name}-{job}-{item["index"]}-{Path(item["config"]).stem}-s{item["seed"]}.out'
                log=logpath.open('w')
                proc=subprocess.Popen([sys.executable,'-u','-m','paclock_bench.training.train',
                                       '--config',item['config'],'--seed',str(item['seed'])],
                                      env=env,stdout=log,stderr=subprocess.STDOUT)
                item.update(gpu=gpu,pid=proc.pid,started_at=time.time(),log=str(logpath))
                active[gpu]=(proc,item,log);snapshot()
                print(f'launched {item["name"]} seed {item["seed"]} on GPU {gpu}, pid {proc.pid}',flush=True)
        if active:time.sleep(1)
    snapshot()
    raise SystemExit(1 if stopping[0] or any(x['exit_code']!=0 for x in done) else 0)

if __name__=='__main__':main()
