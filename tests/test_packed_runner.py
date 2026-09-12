"""Scheduler regression checks without reserving GPUs."""
import importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pack',ROOT/'scripts/slurm/run_packed.py')
pack=importlib.util.module_from_spec(spec);spec.loader.exec_module(pack)

class PackedTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.old=Path.cwd();os.chdir(self.tmp.name)
  self.cfg=Path('c.yaml');self.cfg.write_text('name: example\n')
 def tearDown(self):os.chdir(self.old);self.tmp.cleanup()
 def test_partition_capacity_and_rejections(self):
  for part,n in pack.PARTITIONS.items():
   slots,rows=pack.plan([f'c.yaml:{s}' for s in range(n)],part,0)
   self.assertEqual(slots,n);self.assertEqual(len(rows),n)
  for specs in [['c.yaml']*4,['c.yaml:0','c.yaml:1']]:
   with self.assertRaises(ValueError):pack.plan(specs,'mi2104x',0)
  self.cfg.write_text('name: example\ndisabled_reason: known design defect\n')
  with self.assertRaises(ValueError):pack.plan(['c.yaml'],'mi2101x',0)
 def test_completed_run_is_rejected_before_allocation(self):
  p=Path('runs/example/seed0');p.mkdir(parents=True);(p/'result.json').write_text('{}')
  with self.assertRaises(ValueError):pack.plan(['c.yaml'],'mi2101x',0)
 def test_eight_slots_refill_and_failures(self):
  launched=[];live=set();peak=[0]
  class Proc:
   def __init__(self,cmd,env,**kw):
    self.pid=1000+len(launched);self.gpu=int(env['HIP_VISIBLE_DEVICES']);self.calls=0
    self.seed=int(cmd[-1]);self.dead=False
    if self.gpu in live:raise AssertionError('two trainers bound to one GPU')
    live.add(self.gpu);peak[0]=max(peak[0],len(live));launched.append(self)
   def poll(self):
    self.calls+=1
    if self.calls<2:return None
    if not self.dead:live.remove(self.gpu);self.dead=True
    return 1 if self.seed==9 else 0
   def send_signal(self,sig):self.calls=2
  fake=SimpleNamespace(__version__='test',version=SimpleNamespace(hip='test'),cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:8,get_device_name=lambda _: 'test'))
  argv=['run_packed','--partition','mi2508x']+[f'c.yaml:{s}' for s in range(10)]
  with patch.dict(sys.modules,{'torch':fake}),patch.dict(os.environ,{'SLURM_JOB_ID':'test','SLURM_JOB_NAME':'test'}),patch.object(sys,'argv',argv),patch.object(pack.subprocess,'Popen',Proc),patch.object(pack.subprocess,'check_output',return_value='test\n'),patch.object(pack.time,'sleep',lambda t:None),patch.object(pack.signal,'signal'):
   with self.assertRaises(SystemExit) as out:pack.main()
  self.assertEqual(out.exception.code,1)
  self.assertEqual(peak[0],8);self.assertEqual(len(launched),10)
  self.assertEqual([p.gpu for p in launched[:8]],list(range(8)))
  receipt=json.loads(Path('logs/test-test-pack.json').read_text())
  self.assertEqual(len(receipt['done']),10);self.assertEqual(receipt['pending'],[])
  self.assertEqual(sum(r['exit_code']!=0 for r in receipt['done']),1)

if __name__=='__main__':unittest.main()
