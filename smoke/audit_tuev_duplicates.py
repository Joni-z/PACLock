"""Read-only train/validation input audit; run inside a Slurm allocation."""
import argparse
import collections
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np


def signature(path):
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, inode=stat.st_ino)


def summarize(groups, smoothing):
    counts = [sum(v[c] for v in groups.values()) for c in range(6)]
    total = sum(counts)
    entropy = 0.0
    for counter in groups.values():
        n = sum(counter.values())
        probs = [(1 - smoothing) * counter[c] / n + smoothing / 6 for c in range(6)]
        entropy += n * sum(-q * math.log(q) for q in probs if q)
    mixed = [v for v in groups.values() if len(v) > 1]
    return dict(
        samples=total, class_counts=counts, unique_inputs=len(groups),
        multiplicities=dict(sorted(collections.Counter(sum(v.values()) for v in groups.values()).items())),
        mixed_label_inputs=len(mixed), mixed_label_rows=sum(sum(v.values()) for v in mixed),
        deterministic_accuracy_upper_bound=sum(max(v.values()) for v in groups.values()) / total,
        smoothed_cross_entropy_lower_bound=entropy / total,
        classes=[dict(index=c, samples=counts[c], unique_inputs=sum(v[c] > 0 for v in groups.values()),
                      mixed_label_rows=sum(v[c] for v in mixed)) for c in range(6)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('This data audit must run in a Slurm allocation, never on the login node')
    if args.output.exists():
        raise FileExistsError(args.output)
    start = time.monotonic()
    raw_manifest = (args.root / 'manifest.json').read_bytes()
    manifest = json.loads(raw_manifest)
    assert manifest['dataset'] == 'tuev'
    smoothing = manifest['protocol']['label_smoothing']
    all_groups, rows = {}, {}
    for split in ('train', 'val'):
        signal_path, label_path = (args.root / (split + '_' + key + '.npy') for key in ('signals', 'labels'))
        before = {str(p): signature(p) for p in (signal_path, label_path)}
        signals = np.load(signal_path, mmap_mode='r', allow_pickle=False)
        labels = np.load(label_path, allow_pickle=False)
        assert signals.shape == (len(labels), 16, 1000)
        assert labels.shape == (len(labels),) and np.issubdtype(labels.dtype, np.integer)
        groups = collections.defaultdict(collections.Counter)
        stream = hashlib.sha256()
        for i, label in enumerate(labels):
            if i % 512 == 0 and time.monotonic() - start > 240:
                raise TimeoutError('Input audit exceeded four-minute internal budget')
            assert 0 <= label < 6
            data = signals[i].tobytes(order='C')
            stream.update(data)
            groups[hashlib.sha256(data).digest()][int(label)] += 1
        after = {str(p): signature(p) for p in (signal_path, label_path)}
        assert before == after, 'Data files changed during the audit'
        summary = summarize(groups, smoothing)
        expected = manifest['splits'][split]
        assert summary['samples'] == expected['n_windows']
        assert summary['class_counts'] == [expected['class_counts'].get(str(i), 0) for i in range(6)]
        rows[split] = dict(summary, shape=list(signals.shape), dtype=str(signals.dtype),
                           input_data_sha256=stream.hexdigest(),
                           labels_file_sha256=hashlib.sha256(label_path.read_bytes()).hexdigest(),
                           file_signatures=before)
        all_groups[split] = groups
        print(split, summary, flush=True)
    overlap = set(all_groups['train']) & set(all_groups['val'])
    report = dict(dataset='tuev', job=os.environ['SLURM_JOB_ID'], host=os.uname().nodename,
                  manifest_created=manifest['created_utc'], manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  label_smoothing=smoothing, splits=rows, cross_split_identical_inputs=len(overlap),
                  cross_split_identical_rows={s:sum(sum(all_groups[s][k].values()) for k in overlap) for s in all_groups},
                  elapsed_sec=time.monotonic()-start,
                  interpretation=['Exact input bytes grouped by SHA-256; no test arrays read.',
                      'Loss bound is for a deterministic input-only classifier on empirical row-weighted labels; it does not estimate generalization.',
                      'Duplicate counts alone do not authorize changing the benchmark or deduplicating evaluation.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('saved', args.output, 'elapsed', report['elapsed_sec'], flush=True)


if __name__ == '__main__':
    main()
