"""Validation-only preferred-phase interventions on immutable checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from torch.utils.data import DataLoader
from paclock_bench.data.datasets import WindowDataset, load_manifest
from paclock_bench.models.build import build_model
from paclock_bench.paths import expand
from paclock_bench.training.losses import build_loss
from paclock_bench.training.train import evaluate, set_seed


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', action='append', required=True, help='checkpoint:metadata')
    ap.add_argument('--out', required=True)
    ap.add_argument('--job', required=True)
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID') == args.job
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    dest = Path(args.out)
    assert not dest.exists()
    started = time.monotonic()
    rows = []
    for pair in args.pair:
        checkpoint, metadata = map(Path, pair.split(':', 1))
        hashes = {'checkpoint': sha(checkpoint), 'metadata': sha(metadata)}
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        source = json.loads(metadata.read_text())
        cfg = saved['config']
        assert cfg == source['config']
        assert cfg['model_kwargs']['tokenizer_mode'] == 'factorized'
        assert cfg['model_kwargs']['pac_token_mode'] == 'measured'
        assert cfg['model_kwargs']['interaction_mode'] == 'rotation'
        assert cfg['model_kwargs']['freq_mixer'] == 'attention'
        completed = 'stopped_by' in source
        if completed:
            assert source['stopped_by'] == 'epochs' and source['epochs_run'] == 20
            reference = source['best_val']
        else:
            matched = [r for r in source['history'] if r['tag'] == saved['tag']]
            assert len(matched) == 1
            reference = matched[0]['metrics']['cohen_kappa']
        assert abs(reference - saved['best_val']) < 1e-12
        root = expand(cfg['data_root'])
        assert load_manifest(root)['created_utc'] == source['data_manifest_created']
        dataset = WindowDataset(root, 'val', preload=True)
        assert dataset.class_counts().tolist() == source['class_counts']['val']
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
        set_seed(1)
        model = build_model(cfg, dataset.shape).cuda().eval()
        model.load_state_dict(saved['model'], strict=True)
        criterion = build_loss(cfg)
        x, _ = next(iter(loader))
        x = x.cuda()
        with torch.no_grad():
            for _ in range(2):
                model(x)
            reference_tokens, reference_coupling, reference_hz = model.frontend(x)
        reference_modulus = reference_tokens[..., :128].reshape(*reference_tokens.shape[:-1], 64, 2).norm(dim=-1)
        controls = [('measured', 0), ('magnitude', 0)] + [('scramble', seed) for seed in (0, 1, 2)]
        interventions = []
        for mode, seed in controls:
            model.frontend.pac_token_mode = mode
            set_seed(seed)
            with torch.no_grad():
                tokens, coupling, hz = model.frontend(x)
                assert torch.isfinite(tokens).all()
                assert torch.equal(tokens[..., 128:], reference_tokens[..., 128:])
                assert torch.equal(coupling, reference_coupling) and torch.equal(hz, reference_hz)
                modulus = tokens[..., :128].reshape(*tokens.shape[:-1], 64, 2).norm(dim=-1)
                modulus_error = (modulus - reference_modulus).abs().max().item()
                assert modulus_error < 1e-5, ('Rotation magnitude changed', modulus_error)
            set_seed(seed)
            before = time.monotonic()
            loss, metrics = evaluate(model, loader, 'cuda', criterion, cfg['num_classes'], cfg)
            assert np.isfinite(loss) and all(np.isfinite(v) for v in metrics.values())
            if mode == 'measured':
                assert abs(metrics['cohen_kappa'] - reference) < 1e-6, (metrics, reference)
            interventions.append(dict(mode=mode, intervention_seed=seed, metrics=metrics,
                                      val_loss=loss, elapsed_seconds=time.monotonic() - before,
                                      first_batch_max_modulus_error=modulus_error,
                                      first_batch_content_and_coupling_magnitude_unchanged=True))
            print(cfg['name'], mode, seed, json.dumps(metrics), flush=True)
        model.frontend.pac_token_mode = 'measured'
        with torch.no_grad():
            torch.testing.assert_close(model.frontend(x)[0], reference_tokens, rtol=0, atol=0)
        assert all(torch.equal(value.cpu(), saved['model'][key]) for key, value in model.state_dict().items())
        assert sha(checkpoint) == hashes['checkpoint'] and sha(metadata) == hashes['metadata']
        rows.append(dict(name=cfg['name'], dataset=cfg['dataset'], seed=cfg['seed'],
                         completed_training=completed, selected_tag=saved['tag'],
                         checkpoint=str(checkpoint), metadata=str(metadata), hashes=hashes,
                         interventions=interventions))
        del model, loader, dataset, saved
        torch.cuda.empty_cache()
    result = dict(rows=rows, job=args.job, host=os.uname().nodename,
                  gpu_selector=os.environ.get('HIP_VISIBLE_DEVICES'), torch=torch.__version__,
                  script_sha256=sha(__file__), elapsed_seconds=time.monotonic() - started,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  training_performed=False, test_evaluated=False,
                  limitations='Post-training sensitivity, not retrained ablation or causal physiological evidence. Magnitude removes preferred-phase alignment and breaks gauge invariance; scramble additionally permutes edge phases per patch. First-batch invariants do not measure every batch. Intervention seeds are not training seeds.')
    temp = dest.with_name(dest.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(dest)
    print('PHASE_DIAGNOSTIC_COMPLETE', dest, flush=True)


if __name__ == '__main__':
    main()
