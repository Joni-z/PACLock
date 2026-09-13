import importlib.util,io,json,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('monitor',Path(__file__).resolve().parents[1]/'scripts/monitor/watch_clusters.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class MonitorTests(unittest.TestCase):
 def pack_archive(self,pack):
  buf=io.BytesIO();raw=json.dumps(pack).encode()
  with tarfile.open(fileobj=buf,mode='w') as t:
   member=tarfile.TarInfo('repo/logs/trial-123-pack.json');member.size=len(raw);t.addfile(member,io.BytesIO(raw))
  return buf.getvalue()
 def test_smoke_wrapper_receipt_reaches_actual_training_log(self):
  job=dict(id='123',workdir='/repo',metadata='Command=/repo/slurm/smoke_then_train.slurm cfg.yaml StdOut=/repo/logs/trial-123.out',tail='smoke passed')
  receipts=m.packed_receipt_paths([job])
  run=dict(log='logs/trial-123-0-cfg-s0.out',name='model',seed=0,gpu=0,pid=456)
  paths=m.receipt_log_paths(self.pack_archive(dict(job='123',running=[run],done=[])),receipts)
  rows=m.packed_log_rows('==> /repo/logs/trial-123-0-cfg-s0.out <==\nepoch 2 | val cohen_kappa=0.65\n',paths)
  self.assertEqual(rows[0]['latest_validation'],'epoch 2 | val cohen_kappa=0.65')
  self.assertEqual(rows[0]['process_liveness'],'not inferred from log')
  self.assertEqual(rows[0]['launch_pid'],456)
  for broken in (dict(job='999',running=[run]),dict(job='123',running=[dict(run,log='../elsewhere.out')])):
   with self.assertRaises(ValueError):m.receipt_log_paths(self.pack_archive(broken),receipts)
 def test_legacy_packed_logs_still_work(self):
  paths=m.packed_log_paths([dict(id='122',metadata='Command=/repo/slurm/configs_packed.slurm StdOut=/repo/logs/old.out',tail='launched old_cfg-s0 on GPU 2 (pid 789)')])
  self.assertEqual(paths['/repo/logs/old-old_cfg-s0.out']['launch_pid'],789)
 def test_selection_metric_is_not_mislabeled(self):
  row=m.result_row(dict(name='tusz-control',seed=0,dataset='tusz',primary_metric='pr_auc',best_val=.91,val_curve=[.2,.3],config={'select_metric':'auroc'}))
  self.assertEqual((row['selection_metric'],row['selection_score']),('auroc',.91))
  self.assertEqual(row['best_logged_primary'],.3)
 def test_failed_poll_retains_live_handles(self):
  jobs=[dict(id='123',state='RUNNING')]
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'ssh',side_effect=TimeoutError('observation timeout')):
   row=m.collect('amd','/repo',Path(tmp),dict(jobs=jobs))
  self.assertFalse(row['ok']);self.assertEqual(row['jobs'],jobs)
  self.assertTrue(row['jobs_are_cached']);self.assertNotIn('accounting',row)
 def test_archive_timeout_still_checks_live_logs_and_sync(self):
  commands=[]
  def remote(host,cmd,**kwargs):
   commands.append(cmd)
   if cmd.startswith('squeue '):return '123|h200_public|trial|RUNNING|1:00|gh116|/repo|None\n'
   if 'find runs ' in cmd:raise TimeoutError('archive read timeout')
   if cmd.startswith('scontrol '):return 'JobId=123 JobState=RUNNING StdOut=/repo/logs/trial.out'
   if cmd.startswith('tail '):return 'epoch 24 | val auroc=0.88\n'
   if 'push_runs.sh' in cmd:return 'sync completed'
   raise AssertionError(cmd)
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'ssh',side_effect=remote):
   row=m.collect('torch','/repo',Path(tmp),{})
  self.assertTrue(row['queue_verified']);self.assertFalse(row['results_verified'])
  self.assertEqual(row['jobs'][0]['tail'],'epoch 24 | val auroc=0.88\n')
  self.assertEqual(row['sync_output'],'sync completed')
  self.assertEqual(row['errors'],['result archive: archive read timeout'])
  self.assertTrue(any('push_runs.sh' in cmd for cmd in commands))
if __name__=='__main__':unittest.main()
