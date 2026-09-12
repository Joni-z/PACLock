"""Summarize canonical validation evidence locally; never select using test scores."""
import argparse, collections, hashlib, json, statistics
from pathlib import Path

FAMILIES = (
    'paclock_rot2', 'paclock_duplex', 'factorized_f0', 'factorized_f1',
    'factorized_f2', 'crofremo_s1', 'crofremo_s2', 'crofremo_n1',
    'cf2_v0d192', 'cf2_v0d192cs', 'labram_pretrained', 'cbramod_pretrained',
    'biot_prest16', 'reve_pretrained', 'eegpt_pretrained',
)

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def summarize(snapshot):
    if snapshot.get('conflicts'):
        raise ValueError('Resolve canonical result conflicts before aggregating')
    cells = collections.defaultdict(list)
    excluded = []
    for key, row in snapshot['results'].items():
        family = row['name'].removeprefix(row['dataset'] + '-')
        if family not in FAMILIES:
            continue
        raw = json.loads(Path(row['path']).read_text())
        cfg = raw['config']
        reasons = []
        if raw['name'] != row['name'] or raw['seed'] != row['seed']:
            raise ValueError('Result identity changed: ' + key)
        if row.get('blocked') or row.get('diagnostic_only'):
            reasons.append('blocked or diagnostic')
        if raw.get('stopped_by') not in ('epochs', 'patience'):
            reasons.append('completion reason absent or budget/operator stop')
        if cfg.get('train_subsample'):
            reasons.append('training subset')
        selection = cfg.get('select_metric') or raw['primary_metric']
        if selection != raw['primary_metric']:
            reasons.append('selected primary validation metric not persisted')
        if raw.get('data_manifest_created') is None:
            reasons.append('missing manifest identity')
        if reasons:
            excluded.append(dict(key=key, reasons=reasons))
            continue
        # Equal validation caps are necessary: a 5k-window curve is not a
        # full-validation curve. Include target counts and manifest identity.
        cohort = dict(dataset=raw['dataset'], metric=selection,
                      manifest=raw['data_manifest_created'],
                      val_subsample=cfg.get('val_subsample') or 0,
                      val_counts=raw.get('class_counts', {}).get('val'))
        architecture = dict(model=cfg['model'], kwargs=cfg.get('model_kwargs', {}))
        cells[(family, raw['dataset'])].append(dict(
            seed=raw['seed'], score=raw['best_val'], cohort=cohort,
            architecture=fingerprint(architecture), construction=architecture,
            path=row['path'], epochs=raw['epochs_run'], stop=raw['stopped_by']))
    result = []
    for (family, dataset), rows in sorted(cells.items()):
        cohorts = {fingerprint(r['cohort']) for r in rows}
        architectures = {r['architecture'] for r in rows}
        if len(cohorts) != 1 or len(architectures) != 1:
            excluded.append(dict(key=family + '/' + dataset,
                                 reasons=['incompatible seed cohorts or constructions']))
            continue
        scores = [r['score'] for r in rows]
        result.append(dict(family=family, dataset=dataset,
                           seeds=sorted(r['seed'] for r in rows),
                           mean=statistics.mean(scores),
                           sd=statistics.stdev(scores) if len(scores)>1 else None,
                           cohort=rows[0]['cohort'], architecture=rows[0]['architecture'],
                           construction=rows[0]['construction'], rows=rows))
    coverage = {}
    for family in FAMILIES:
        rows = [r for r in result if r['family'] == family]
        coverage[family] = dict(
            datasets=len(rows), datasets_three_seeds=sum(set((0,1,2)) <= set(r['seeds']) for r in rows),
            architecture_variants=len({r['architecture'] for r in rows}))
    return dict(snapshot_at=snapshot['at'], coverage=coverage, cells=result,
                excluded=excluded, caveats=[
                    'Validation evidence only; no SOTA or final-model claim.',
                    'Match cohort identity before comparison; recipes may still differ.',
                    'Configuration fingerprints do not prove historical source identity.',
                    'Missing selected-primary metrics are excluded, not treated as poor scores.',
                    'Single-seed cells are screening evidence only.'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--state', type=Path, required=True)
    a = ap.parse_args()
    result = summarize(json.loads((a.state/'latest.json').read_text()))
    (a.state/'candidate-matrix.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['coverage'], indent=2))
    print('excluded:', len(result['excluded']))
