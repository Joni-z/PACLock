"""Validate a pack on the login node before asking Slurm for GPU resources."""
import argparse,datetime,json,re,subprocess
from pathlib import Path
from run_packed import PARTITIONS,plan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('specs',nargs='+')
    ap.add_argument('-p','--partition',choices=sorted(PARTITIONS),default='mi2101x')
    ap.add_argument('-J','--name',default='PAC')
    ap.add_argument('--time',default='08:00:00')
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    match=re.fullmatch(r'(\d+):([0-5]\d):([0-5]\d)',args.time)
    if not match:ap.error('--time must use HH:MM:SS')
    hours,minutes,seconds=map(int,match.groups())
    duration=3600*hours+60*minutes+seconds
    if duration<=0:ap.error('--time must be positive')
    if args.partition=='mi2101x' and duration>12*3600:
        ap.error('AMD limits mi2101x to 12 hours for this account; long runs need optimizer resume')
    root=Path(__file__).resolve().parents[2]
    import os
    os.chdir(root)
    capacity=PARTITIONS[args.partition]
    _,items=plan(args.specs,args.partition,args.seed)
    # Single-card nodes release individually, so one slow run cannot hold an
    # otherwise idle 4/8-card node. The default is intentionally one-card jobs.
    groups=[[r] for r in items] if capacity==1 else [items]
    cpus={'mi2101x':16,'mi3501x':24,'mi2104x':128,'mi2508x':128}[args.partition]
    commands=[]
    for group in groups:
        commands.append(['sbatch','--parsable','-p',args.partition,'-c',str(cpus),
                         '--time',args.time,'-J',args.name,'slurm/configs_packed.slurm']+
                        [r['config']+':'+str(r['seed']) for r in group])
    if args.dry_run:
        print(json.dumps(dict(commands=commands,slots_per_node=capacity),indent=2));return
    Path('logs').mkdir(exist_ok=True)
    dest=root/'results'/'submissions';dest.mkdir(parents=True,exist_ok=True)
    receipt=dest/(datetime.datetime.now().strftime('%Y%m%dT%H%M%S%f')+'.json')
    results=[]
    for cmd in commands:
        output=subprocess.check_output(cmd,text=True,cwd=root).strip()
        # The AMD submission filter prints allocation usage before --parsable.
        ids=[s for s in output.splitlines() if re.fullmatch(r'\d+(?:;[\w.-]+)?',s)]
        if len(ids)!=1:
            raise RuntimeError(f'Cannot identify submitted job; inspect squeue before retrying: {output}')
        job=ids[0].split(';')[0]
        results.append(dict(job_id=job,command=cmd,submission_output=output))
        receipt.write_text(json.dumps(dict(jobs=results),indent=2)+'\n')
        print(job,flush=True)
    print('receipt:',receipt)

if __name__=='__main__':main()
