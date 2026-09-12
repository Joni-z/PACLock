import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('matrix', Path(__file__).resolve().parents[1]/'scripts/monitor/candidate_matrix.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CandidateMatrixTests(unittest.TestCase):
    def summarize(self, records):
        with tempfile.TemporaryDirectory() as tmp:
            rows = {}
            for seed, overrides in enumerate(records):
                raw = dict(name='tusz-paclock_duplex', dataset='tusz', seed=seed,
                           primary_metric='pr_auc', best_val=.3, epochs_run=20,
                           stopped_by='epochs', data_manifest_created='same-split',
                           class_counts={'val': [90,10]}, config={'model':'paclock'})
                raw.update(overrides)
                path = Path(tmp)/f'{seed}.json'
                path.write_text(json.dumps(raw))
                rows[str(seed)] = dict(name=raw['name'], dataset=raw['dataset'],
                                      seed=seed, path=str(path))
            return m.summarize(dict(at=0, results=rows))

    def test_different_validation_subsets_are_not_averaged(self):
        x = self.summarize([{}, {'config':{'model':'paclock','val_subsample':50}}])
        self.assertEqual(x['cells'], [])
        self.assertIn('incompatible', x['excluded'][0]['reasons'][0])

    def test_auroc_selection_is_not_ranked_as_pr_auc(self):
        x = self.summarize([{'best_val':.95, 'config':{'model':'paclock','select_metric':'auroc'}}])
        self.assertEqual(x['cells'], [])
        self.assertIn('selected primary', x['excluded'][0]['reasons'][0])

if __name__ == '__main__':
    unittest.main()
