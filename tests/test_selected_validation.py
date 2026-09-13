"""Exercise the real validation closure without GPU/data dependencies."""
import ast
import contextlib
import io
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


class SelectedValidationTests(unittest.TestCase):
    def test_metrics_follow_selected_weights_and_first_tie(self):
        source = Path(os.environ.get('PACLOCK_TRAIN_SOURCE',
            Path(__file__).resolve().parents[1] / 'paclock_bench/training/train.py'))
        main = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        validate = next(n for n in main.body
                        if isinstance(n, ast.FunctionDef) and n.name == 'validate')
        initializers = [n for n in main.body if isinstance(n, ast.Assign)
                        and any(isinstance(v, ast.Name) and v.id in
                                {'best', 'best_val_metrics', 'patience'}
                                for target in n.targets for v in ast.walk(target))]
        body = '\n'.join(ast.unparse(n) for n in initializers + [validate])
        factory = 'def factory():\n' + '\n'.join('    '+line for line in body.splitlines())
        factory += '\n    return validate, lambda: (best, best_state, val_curve, best_val_metrics, best_val_tag, best_val_index)\n'
        observations = [dict(pr_auc=.2, auroc=.9), dict(pr_auc=.6, auroc=.85),
                        dict(pr_auc=.4, auroc=.95), dict(pr_auc=.7, auroc=.95)]

        class Tensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def clone(self): return self.value

        class Model:
            step = -1
            def state_dict(self): return {'step': Tensor(self.step)}
            def train(self): pass

        model = Model()
        def evaluate(*args, **kwargs):
            model.step += 1
            result = (0., observations[model.step])
            return result + (None, None) if kwargs.get('return_raw') else result

        env = dict(np=SimpleNamespace(inf=math.inf), cfg={'num_classes': 2},
                   evaluate=evaluate, model=model, val_loader=None, device=None,
                   criterion=None, key='pr_auc', select_key='auroc', control=None)
        exec(compile(factory, str(source), 'exec'), env)
        run, state = env['factory']()
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(len(observations)): run(f'epoch {i} |')
        best, weights, curve, metrics, tag, index = state()
        self.assertEqual(best, .95)
        self.assertEqual(weights, {'step': 2})
        self.assertEqual(metrics, {'pr_auc': .4, 'auroc': .95})
        self.assertEqual((tag, index), ('epoch 2 |', 2))
        self.assertEqual(curve[index], metrics['pr_auc'])
        self.assertEqual(max(curve), .7)
        result = next(n.value for n in main.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'result' for t in n.targets))
        saved = next(value for key, value in zip(result.keys, result.values)
                     if isinstance(key, ast.Constant) and key.value == 'selected_validation')
        record = eval(compile(ast.Expression(saved), str(source), 'eval'), dict(
            select_key='auroc', best_val_metrics=metrics,
            best_val_tag=tag, best_val_index=index))
        self.assertEqual(record, dict(selection_metric='auroc', tag=tag,
                                      evaluation_index=2, metrics=metrics))
        observations[2]['pr_auc'] = -1
        self.assertEqual(metrics['pr_auc'], .4)


if __name__ == '__main__':
    unittest.main()
