"""What tokenizer and geometry does a pretraining checkpoint carry?"""
import sys
import torch

for n in sys.argv[1:]:
    c = torch.load('pretrain_runs/%s/checkpoint.pt' % n, map_location='cpu')
    m = c['model'] if 'model' in c else c
    meta = {k: v for k, v in c.items() if k != 'model' and not hasattr(v, 'shape')}
    tk = sorted(k for k in m if 'tokenizer' in k)
    print('=== %s' % n)
    print('    meta:', {k: v for k, v in meta.items() if not isinstance(v, dict)})
    if 'config' in meta and isinstance(meta['config'], dict):
        cf = meta['config']
        mk = cf.get('model_kwargs', cf)
        print('    tokenizer_mode=%s interaction_mode=%s d_model=%s depth=%s' % (
            mk.get('tokenizer_mode'), mk.get('interaction_mode'),
            mk.get('d_model'), mk.get('depth')))
        print('    corpora:', cf.get('corpora') or cf.get('data_roots'))
    print('    tokenizer params:', [(k, tuple(m[k].shape)) for k in tk])
    enc = [k for k in m if k.startswith('encoder.')]
    print('    n encoder tensors: %d   total tensors: %d' % (len(enc), len(m)))
    print()
