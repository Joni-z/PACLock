"""Which checkpoint, at how many steps, did each pretrained finetune use?"""
import glob, json, os
import torch

steps = {}
for p in sorted(glob.glob('pretrain_runs*/*/checkpoint.pt')):
    name = p
    c = torch.load(p, map_location='cpu')
    m = c['model'] if 'model' in c else c
    tok = 'raw' if any('frontend.tokenizer.' in k for k in m) else 'pac'
    d = m.get('frontend.tokenizer.weight', m.get('frontend.amplitude_tokenizer.weight'))
    steps[name] = (c.get('step'), tok, d.shape[0] if d is not None else None)
    print('%-52s step=%-7s tokenizer=%-4s tok_out=%s' % (name, *steps[name]))

print()
print('%-42s %-26s %s' % ('run', 'checkpoint', 'pretrain_steps'))
for d in sorted(glob.glob('runs/*')):
    fs = sorted(glob.glob(d + '/*/result.json'))
    if not fs:
        continue
    cfg = json.load(open(fs[0])).get('config', {})
    ck = cfg.get('checkpoint') or (cfg.get('model_kwargs') or {}).get('checkpoint')
    if not ck:
        continue
    name = str(ck)
    st = steps.get(name, (None, None, None))
    print('%-42s %-52s %s (%s)' % (os.path.basename(d), name, st[0], st[1]))
