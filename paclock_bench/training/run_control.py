"""Durable selection checkpoints and cooperative stopping; no optimizer resume."""
from pathlib import Path
import json
import os
import signal
import time

import numpy as np
import torch


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


class RunControl:
    def __init__(self, directory, cfg, info):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cfg, self.info = cfg, info
        self.pid = os.getpid()
        self.reason = None
        self.started = time.time()
        self.history = []
        self.previous = {}
        # Never let a stopped rerun inherit an older completed result.json.
        # Resuming optimizer state is deliberately outside this utility.
        existing = [n for n in ('STOP', 'progress.json', 'best.pt', 'result.json',
                                'stopped.json') if (self.directory / n).exists()]
        if existing:
            raise FileExistsError(f'Existing run artifacts {existing} in {self.directory}; '
                                  'choose a new experiment name or output directory')
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self.request)

    def request(self, signum, frame):
        # Forked DataLoader workers inherit handlers, but do not run the
        # trainer's polling loop. Preserve normal termination in those workers.
        if os.getpid() != self.pid:
            signal.signal(signum, self.previous[signum])
            os.kill(os.getpid(), signum)
            return
        self.reason = f'signal:{signal.Signals(signum).name}'

    def requested(self):
        if self.reason is None and (self.directory / 'STOP').exists():
            self.reason = 'stop_file'
        return self.reason is not None

    def record(self, *, tag, val_loss, metrics, logits, labels, lr,
               train_loss, best, best_state=None):
        labels = np.asarray(labels).reshape(-1).astype(int)
        pred = np.asarray(logits).argmax(-1) if logits.ndim > 1 and logits.shape[-1] > 1 else (logits.reshape(-1) > 0).astype(int)
        k = int(self.cfg['num_classes'])
        cm = np.bincount(labels * k + pred, minlength=k*k).reshape(k, k)
        denom = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(k), where=denom > 0)
        entry = dict(tag=tag, elapsed_sec=time.time()-self.started,
                     val_loss=float(val_loss), train_loss=train_loss,
                     metrics=metrics, lr=lr, best_val=float(best),
                     confusion=cm.tolist(), class_f1=f1.tolist(),
                     target_counts=cm.sum(1).tolist(), pred_counts=cm.sum(0).tolist())
        if best_state is not None:
            target = self.directory / 'best.pt'
            tmp = target.with_name(target.name + f'.tmp.{os.getpid()}')
            torch.save(dict(model=best_state, config=self.cfg, best_val=float(best),
                            tag=tag, purpose='selection checkpoint, not optimizer resume'), tmp)
            os.replace(tmp, target)
        self.history.append(entry)
        atomic_json(self.directory / 'progress.json', dict(
            pid=os.getpid(), name=self.cfg['name'], seed=self.cfg.get('seed', 0),
            config=self.cfg, class_counts=self.info.get('class_counts'),
            data_manifest_created=self.info.get('manifest', {}).get('created_utc'),
            history=self.history))

    def finish_stop(self, best):
        atomic_json(self.directory / 'stopped.json', dict(
            stopped_by='operator_stop', reason=self.reason, best_val=float(best),
            elapsed_sec=time.time()-self.started, test_evaluated=False,
            checkpoint='best.pt' if (self.directory/'best.pt').exists() else None))

    def restore_handlers(self):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)
