"""Local metadata collector. No training, remote analysis, or automatic cancellation.

python3 scripts/monitor/watch_clusters.py --state /Users/mr.z/PACLock-monitor
Add --watch for a persistent 15-minute loop; a file lock prevents duplicates.
"""
import argparse,concurrent.futures,fcntl,hashlib,io,json,os,re,shlex,subprocess,tarfile,time
from pathlib import Path
HOSTS={
 'amd':'/work1/chenyuyou/yifanwang/Zhizhe/PACLock',
 'torch':'/scratch/zz5070/PACLock',
 'b2':'/ocean/projects/cis260249p/qren2/Zhizhe/PACLock',
}
B2_AUTH=Path.home()/'.claude/jobs/18b2571f/tmp/b2_master.exp'
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'}

def command(args,timeout=90,binary=False):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
 if p.returncode:raise RuntimeError(f'command exit {p.returncode}: '+p.stderr.decode(errors='replace')[-1000:])
 return p.stdout if binary else p.stdout.decode(errors='replace')

def ssh(host,cmd,**kw):
 return command(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,cmd],**kw)

def atomic(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)

def extract(data,dest):
 # Only regular files beneath the cache; no symlink or traversal extraction.
 with tarfile.open(fileobj=io.BytesIO(data)) as archive:
  for member in archive:
   p=Path(member.name)
   if not member.isfile():continue
   if p.is_absolute() or '..' in p.parts or member.size>20_000_000:raise ValueError('unexpected archive member')
   target=dest/p;target.parent.mkdir(parents=True,exist_ok=True)
   tmp=target.with_suffix(target.suffix+'.download')
   tmp.write_bytes(archive.extractfile(member).read());tmp.replace(target)

def result_row(x):
 cfg=x.get('config',{});primary=x['primary_metric']
 curve=x.get('val_curve',[])
 return dict(name=x['name'],seed=x['seed'],dataset=x['dataset'],primary_metric=primary,
  selection_metric=cfg.get('select_metric') or primary,selection_score=x.get('best_val'),
  best_logged_primary=max(curve) if curve else None,epochs=x.get('epochs_run'),
  stopped_by=x.get('stopped_by'),val_subsample=cfg.get('val_subsample'),
  manifest=x.get('data_manifest_created'),group=x.get('group'),
  blocked=bool(cfg.get('disabled_reason') or (cfg.get('group')=='factorized' and
       cfg.get('sample_rate')==200 and x['name'].endswith('_f3'))),
  diagnostic_only=x.get('group')=='scheduling_pilot')

def collect(host,root,state,previous):
 dest=state/'hosts'/host;dest.mkdir(parents=True,exist_ok=True)
 out=dict(host=host,at=time.time(),ok=False,errors=[],jobs=[],foreign_jobs=[])
 try:
  if host=='b2':command(['expect',str(B2_AUTH)],timeout=45)
  raw=ssh(host,'squeue -r -u "$USER" -h -o "%i|%P|%j|%T|%M|%N|%Z|%r"')
  for line in raw.splitlines():
   fields=line.split('|',7)
   if len(fields)!=8:raise ValueError('unrecognized squeue row')
   job=dict(zip(('id','partition','name','state','elapsed','node','workdir','reason'),fields))
   if job['workdir']==root or job['workdir'].startswith(root+'-'):out['jobs'].append(job)
   else:out['foreign_jobs'].append({k:job[k] for k in ('id','state')})
  out['queue_verified']=True
  oldids={j['id'] for j in previous.get('jobs',[])};ids={j['id'] for j in out['jobs']}
  missing=sorted(j for j in oldids-ids if re.fullmatch(r'\d+(?:_\d+)?',j))
  if missing:
   out['accounting']=ssh(host,'sacct -X -n -P -j '+','.join(missing)+' -o JobID,State,ExitCode,Elapsed')
  # Read result files; all JSON parsing/comparison runs on this local machine.
  if host!='b2':
   data=ssh(host,'cd '+shlex.quote(root)+' && find runs -type f \\( -name result.json -o -name progress.json -o -name stopped.json -o -name design_stop.json \\) -print0 | tar --null -T - -cf -',binary=True)
   extract(data,dest)
  if host=='amd':
   q=root+'-backfill-20260913'
   data=ssh(host,'cd '+shlex.quote(q)+' && tar -cf - logs running done deferred',binary=True)
   with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    out['backfill_reported_running']=[json.load(archive.extractfile(m)) for m in archive if m.isfile() and m.name.startswith('running/') and m.name.endswith('.json')]
   extract(data,dest/'backfill')
   for run in out['backfill_reported_running']:
    log=dest/'backfill/logs'/Path(run.get('log','')).name
    if log.is_file():
     txt=log.read_text(errors='replace')
     val=re.findall(r'epoch\s+\d+(?: step \d+)?\s*\|\s*val [^\n]+',txt)
     run['latest_validation']=val[-1] if val else None
     if re.search(r'Traceback|out of memory|loss\s*[=: ]\s*(?:nan|inf)\b',txt,re.I):
      out['errors'].append('trainer log requires review: '+run['name'])
  for job in out['jobs']:
   if not re.fullmatch(r'\d+(?:_\d+)?',job['id']):continue
   meta=ssh(host,'scontrol show job '+job['id']+' -o')
   job['metadata']=meta.strip()
   log=re.search(r'(?:^| )StdOut=(\S+)',meta)
   if log and log.group(1).startswith(root) and job['state'] in ('RUNNING','COMPLETING'):
    try:job['tail']=ssh(host,'tail -n 40 '+shlex.quote(log.group(1)))
    except Exception as exc:out['errors'].append(f"log {job['id']}: {exc}")
  if host=='torch':
   try:
    # Existing site helper: no overwrites; transfer only results/json/npz.
    out['sync_output']=ssh(host,'bash -eo pipefail /scratch/zz5070/sync/push_runs.sh',timeout=120).strip()
   except Exception as exc:out['errors'].append('torch result sync: '+str(exc))
  out['ok']=True
 except Exception as exc:
  out['errors'].append(str(exc))
  if not out.get('queue_verified'):
   # A failed observation is not evidence that jobs finished. Retain handles
   # so a later successful poll can still reconcile them against sacct.
   out['jobs']=previous.get('jobs',[]);out['jobs_are_cached']=True
 return out

def snapshot(state):
 latest=state/'latest.json';old=json.loads(latest.read_text()) if latest.exists() else {}
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  tasks={h:pool.submit(collect,h,r,state,old.get('hosts',{}).get(h,{})) for h,r in HOSTS.items()}
  hosts={h:f.result() for h,f in tasks.items()}
 results={};conflicts=[];alerts=[]
 for host,report in hosts.items():
  for err in report['errors']:alerts.append(dict(kind='connection_or_read_error',host=host,message=err))
  for line in report.get('accounting','').splitlines():
   f=line.split('|')
   if len(f)>2 and f[1].split()[0] in TERMINAL and f[1]!='COMPLETED':
    alerts.append(dict(kind='terminal_job',host=host,job=f[0],state=f[1],exit=f[2]))
  for p in (state/'hosts'/host/'runs').glob('*/seed*/result.json'):
   try:
    x=json.loads(p.read_text());key=x['name']+':'+str(x['seed']);row=result_row(x)
    if p.parent.parent.name!=x['name'] or p.parent.name!='seed'+str(x['seed']):
     alerts.append(dict(kind='result_identity_mismatch',path=str(p),declared=key));continue
    scientific={k:x.get(k) for k in ('test','val_curve','best_val','data_manifest_created','class_counts')}
    digest=hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
    if key in results and results[key]['digest']!=digest:conflicts.append(key)
    elif key not in results:results[key]=dict(row,host=host,digest=digest,path=str(p))
   except Exception as exc:alerts.append(dict(kind='result_read_error',path=str(p),message=str(exc)))
 newkeys=sorted(set(results)-set(old.get('results',{}))) if old else []
 changed=sorted(k for k in set(results)&set(old.get('results',{})) if results[k]['digest']!=old['results'][k]['digest'])
 known_alerts={json.dumps(a,sort_keys=True) for a in old.get('alerts',[])}
 new_alerts=[a for a in alerts if json.dumps(a,sort_keys=True) not in known_alerts]
 report=dict(at=time.time(),initial_baseline=not bool(old),hosts=hosts,results=results,new_results=newkeys,changed_results=changed,conflicts=sorted(set(conflicts)),alerts=alerts,new_alerts=new_alerts)
 atomic(latest,report)
 for key in newkeys:
  with (state/'events.jsonl').open('a') as f:f.write(json.dumps(dict(at=report['at'],event='result',result=results[key]))+'\n')
 summary=dict(at=report['at'],hosts={h:dict(ok=x['ok'],ours=len(x['jobs']),foreign=len(x['foreign_jobs'])) for h,x in hosts.items()},new_results=newkeys,changed_results=changed,conflicts=report['conflicts'],new_alerts=new_alerts,alerts=alerts)
 atomic(state/'summary.json',summary)
 print(json.dumps(dict(at=report['at'],hosts=summary['hosts'],new_results=len(newkeys),conflicts=len(conflicts),alerts=len(alerts))),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path,required=True);ap.add_argument('--watch',action='store_true');ap.add_argument('--interval',type=int,default=900);a=ap.parse_args()
 a.state.mkdir(parents=True,exist_ok=True)
 lock=(a.state/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (a.state/'watch.pid').write_text(str(os.getpid()))
 while True:
  try:snapshot(a.state)
  except Exception as exc:print(json.dumps(dict(at=time.time(),error=str(exc))),flush=True)
  if not a.watch:break
  time.sleep(max(60,a.interval))
