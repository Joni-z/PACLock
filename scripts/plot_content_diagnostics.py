"""Generate the checkpoint-sensitivity figure from immutable audit JSON only."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    audits = args.root / 'results/audits'
    paths = [audits / name for name in (
        'tuar-content-knockout-20260913.json',
        'tuev-content-knockout-snapshot-20260913.json',
        'preferred-phase-knockout-20260913.json')]
    sources = [json.loads(p.read_text()) for p in paths]
    phase = {r['dataset']: r for r in sources[2]['rows']}
    labels = ['Original', 'Zero coupling\ncoordinates', 'Zero band\ncontent',
              'Zero broadband\ncontent', 'Remove\npreferred phase',
              'Shuffle\nphase pairing']
    data = []
    for dataset, content in zip(('tuar', 'tuev'), sources[:2]):
        row = next(r for r in content['rows'] if r['name'] == dataset + '-factorized_f4_joint_confirm')
        pr = phase[dataset]
        assert row['checkpoint_sha256'] == pr['hashes']['checkpoint']
        values = [row['evaluations'][key]['metrics']['cohen_kappa']
                  for key in ('full', 'zero_coupling', 'zero_band', 'zero_broadband')]
        original = next(r for r in pr['interventions'] if r['mode'] == 'measured')
        assert abs(original['metrics']['cohen_kappa'] - values[0]) < 1e-12
        magnitude = next(r for r in pr['interventions'] if r['mode'] == 'magnitude')
        draws = [r for r in pr['interventions'] if r['mode'] == 'scramble']
        assert sorted(r['intervention_seed'] for r in draws) == [0, 1, 2]
        shuffle = [r['metrics']['cohen_kappa'] for r in draws]
        values += [magnitude['metrics']['cohen_kappa'], statistics.mean(shuffle)]
        data.append(dict(dataset=dataset, completed_training=pr['completed_training'],
                         selected_tag=pr['selected_tag'], values=values,
                         errors=[0.] * 5 + [statistics.stdev(shuffle)]))
    assert data[0]['completed_training'] and not data[1]['completed_training']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.titlesize': 12, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    colors = ['#214A72', '#9BA7B1', '#54A6A0', '#397B77', '#8864A5', '#B49ACA']
    for ax, row in zip(axes, data):
        bars = ax.bar(range(6), row['values'], color=colors, width=.68)
        ax.errorbar(5, row['values'][5], yerr=row['errors'][5], fmt='none',
                    ecolor='#303840', capsize=4, linewidth=1.1)
        for index, (bar, value) in enumerate(zip(bars, row['values'])):
            ax.text(bar.get_x() + bar.get_width()/2, value + row['errors'][index] + .014,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        ax.set_xticks(range(6), labels, fontsize=8.5)
        status = 'completed training' if row['completed_training'] else 'interim checkpoint'
        ax.set_title(row['dataset'].upper() + ' — ' + status, loc='left', pad=13, fontweight='bold')
        ax.set_ylim(0, .72)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#E1E5E8', linewidth=.6)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#CBD2D7')
        ax.tick_params(axis='both', length=0, pad=8)
    axes[0].set_ylabel('Validation Cohen’s κ', labelpad=12)
    fig.suptitle('Sensitivity of the trained f4 model to token interventions',
                 x=.055, ha='left', fontsize=15, fontweight='bold')
    fig.subplots_adjust(left=.055, right=.985, top=.83, bottom=.31, wspace=.12)
    fig.text(.055, .15, 'One trained seed per dataset. TUEV uses a saved checkpoint selected at epoch 1; its training was still running.', fontsize=9, color='#48535C')
    fig.text(.055, .105, 'Shuffle error bars: sample SD across three intervention seeds, not training seeds. All scores use validation data.', fontsize=9, color='#48535C')
    fig.text(.055, .06, 'Inference-only interventions change model inputs; these are not retrained ablations or causal physiological evidence.', fontsize=9, color='#48535C')
    dest = args.root / 'results/figures'
    dest.mkdir(parents=True, exist_ok=True)
    stem = dest / 'content-phase-diagnostics-20260913'
    for suffix in ('pdf', 'svg', 'png'):
        fig.savefig(stem.with_suffix('.' + suffix), dpi=180, facecolor='white')
    plt.close(fig)
    receipt = dict(labels=labels, rows=data, matplotlib=matplotlib.__version__,
                   generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   sources={str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    stem.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(stem.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
