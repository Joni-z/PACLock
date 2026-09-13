"""Check native TUEV head parity on an allocated node, using synthetic inputs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run only inside an allocation'
    assert not args.output.exists(), 'Preserve existing evidence'
    import torch
    import yaml
    from paclock_bench.models.build import build_model
    from paclock_bench.paths import vendored

    torch.set_num_threads(1)
    torch.manual_seed(0)
    started = time.time()
    config = ROOT/'configs/experiments/tuev_cbramod_pretrained_nativehead.yaml'
    cfg = yaml.safe_load(config.read_text())
    candidate = build_model(cfg, (16, 1000)).eval()
    from models.model_for_tuev import Model
    reference = Model(SimpleNamespace(use_pretrained_weights=False,
        classifier='all_patch_reps', dropout=.1, num_of_classes=6)).eval()
    reference.load_state_dict(candidate.state_dict(), strict=True)
    legacy_cfg = copy.deepcopy(cfg)
    legacy_cfg['pretrained'] = False
    legacy_cfg['model_kwargs'].pop('classifier_hidden_dim')
    legacy = build_model(legacy_cfg, (16, 1000)).eval()
    assert candidate.classifier[1].out_features == reference.classifier[1].out_features == 1000
    assert legacy.classifier[1].out_features == 800
    delta = sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in legacy.parameters())
    assert delta == 3_240_200
    x = torch.randn(2, 16, 1000)
    with torch.no_grad():
        actual = candidate(x)
        expected = reference(x.reshape(2, 16, 5, 200))
    assert actual.shape == (2, 6) and torch.isfinite(actual).all()
    assert torch.equal(actual, expected), (actual-expected).abs().max().item()
    files = [config, Path(__file__), ROOT/'paclock_bench/models/build.py',
             ROOT/'paclock_bench/models/foundation/cbramod_adapter.py']
    vendor = Path(vendored('cbramod'))
    files += [vendor/'models/model_for_tuev.py', vendor/'datasets/tuev_dataset.py',
              vendor/'preprocessing/preprocessing_tuev.py', vendor/'finetune_main.py']
    record = dict(allocation=os.environ['SLURM_JOB_ID'], step=os.environ.get('SLURM_STEP_ID'),
        host=os.uname().nodename, device='cpu', torch_version=torch.__version__,
        pretrained_backbone_loaded_by_builder=True, full_state_strict_load=True,
        reference_logits_bit_equal=True, corrected_hidden_width=1000,
        legacy_default_width=800, parameter_increase=delta,
        parameters=sum(p.numel() for p in candidate.parameters()),
        elapsed_sec=time.time()-started, input='synthetic standard normal, shape (2,16,1000)',
        training_performed=False, dataset_arrays_loaded=False,
        vendor_revision=subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
