"""Validation-only coordinate knockouts of completed TUAR checkpoints.

Run through smoke_gpu.slurm in an existing allocation. These interventions
measure sensitivity after training; they are not retrained ablations or a
specific test of physical PAC versus other information in coupling tokens.
"""
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
    ap.add_argument('--out', required=True)
    ap.add_argument('--job', required=True)
    args = ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID') == args.job
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    dest = Path(args.out)
    assert not dest.exists(), 'Never overwrite a diagnostic receipt'
    started = time.monotonic()
    rows = []
    dataset = None
    for name in ('tuar-factorized_f1_scale1_confirm',
                 'tuar-factorized_f2_confirm',
                 'tuar-factorized_f4_joint_confirm'):
        directory = Path('runs') / name / 'seed1'
        result = json.loads((directory / 'result.json').read_text())
        checkpoint = directory / 'best.pt'
        checkpoint_hash = sha(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = saved['config']
        assert cfg == result['config'] and cfg['dataset'] == 'tuar'
        assert result['stopped_by'] == 'epochs' and result['epochs_run'] == 20
        assert cfg['model_kwargs']['freq_mixer'] == 'attention'
        assert abs(saved['best_val'] - result['best_val']) < 1e-12
        root = expand(cfg['data_root'])
        assert load_manifest(root)['created_utc'] == result['data_manifest_created']
        if dataset is None:
            dataset = WindowDataset(root, 'val', preload=True)
        assert dataset.class_counts().tolist() == result['class_counts']['val']
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
        set_seed(1)
        model = build_model(cfg, dataset.shape).cuda().eval()
        model.load_state_dict(saved['model'], strict=True)
        criterion = build_loss(cfg)
        first_x, _ = next(iter(loader))
        first_x = first_x.cuda()
        with torch.no_grad():
            for _ in range(2):
                model(first_x)
            tokens = model.frontend(first_x)[0]
            assert tokens.shape[-1] == 192 and torch.isfinite(tokens).all()
            lane_rms = {label: tokens[..., lo:hi].square().mean().sqrt().item()
                        for label, lo, hi in (('coupling', 0, 128),
                                             ('content_first32', 128, 160),
                                             ('content_last32', 160, 192))}
        modes = {'full': None, 'zero_coupling': (0, 128), 'zero_content': (128, 192)}
        if cfg['model_kwargs']['content_source'] == 'band_broadband':
            modes.update(zero_band=(128, 160), zero_broadband=(160, 192))
        evaluations = {}
        for mode, coordinate_range in modes.items():
            def intervene(module, inputs, output):
                values = output[0].clone()
                lo, hi = coordinate_range
                values[..., lo:hi] = 0
                return (values,) + tuple(output[1:])
            hook = model.frontend.register_forward_hook(intervene) if coordinate_range else None
            torch.cuda.synchronize()
            before = time.monotonic()
            try:
                loss, metrics = evaluate(model, loader, 'cuda', criterion, 3, cfg)
            finally:
                if hook is not None:
                    hook.remove()
            torch.cuda.synchronize()
            assert np.isfinite(loss) and all(np.isfinite(v) for v in metrics.values())
            if mode == 'full':
                assert abs(metrics['cohen_kappa'] - result['best_val']) < 1e-6, (name, metrics, result['best_val'])
            evaluations[mode] = dict(val_loss=loss, metrics=metrics,
                                     elapsed_seconds=time.monotonic() - before)
            print(name, mode, json.dumps(metrics), flush=True)
        # Hooks did not mutate parameters or checkpoint files.
        assert all(torch.equal(v.cpu(), saved['model'][k]) for k, v in model.state_dict().items())
        assert sha(checkpoint) == checkpoint_hash
        rows.append(dict(name=name, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                         result_sha256=sha(directory / 'result.json'), selected_tag=saved['tag'],
                         first_validation_batch_token_rms=lane_rms, evaluations=evaluations))
        del model, loader, saved
        torch.cuda.empty_cache()
    receipt = dict(job=args.job, host=os.uname().nodename, gpu_selector=os.environ.get('HIP_VISIBLE_DEVICES'),
                   git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   script_sha256=sha(__file__), torch=torch.__version__, rows=rows,
                   elapsed_seconds=time.monotonic() - started, test_evaluated=False, training_performed=False,
                   interpretation='Post-training coordinate-zeroing before positional embeddings; attention ignores the side coupling matrix. Distribution shifts and shared normalization preclude interpreting drops as causal PAC benefit or retrained-ablation scores.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(receipt, indent=2) + '\n')
    temp.replace(dest)
    print('DIAGNOSTIC_COMPLETE', dest, flush=True)


if __name__ == '__main__':
    main()
