import importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('monitor',Path(__file__).resolve().parents[1]/'scripts/monitor/watch_clusters.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class MonitorTests(unittest.TestCase):
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
if __name__=='__main__':unittest.main()
