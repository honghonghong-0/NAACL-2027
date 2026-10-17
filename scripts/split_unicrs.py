"""Re-split UniCRS's DBpedia-linked ReDial (all 11,348 conversations) by OUR ReDial-MS user split:
train = 'history' sessions, valid = our valid sessions, test = our test sessions (users.json, conversationId order).
Writes src/data/redial_ms/{train,valid,test}_data_dbpedia.jsonl + copies KG files, then runs UniCRS process.py there."""
import json, shutil, subprocess, collections
from pathlib import Path
R = Path('.'); S = R / 'baselines/unicrs/src/data'; src = S / 'redial'; dst = S / 'redial_ms'
dst.mkdir(exist_ok=True)
split = {s['conv_id']: s['split'] for ss in json.load(open(R / 'processed/users.json')).values() for s in ss}
out = {k: open(dst / f'{k}_data_dbpedia.jsonl', 'w') for k in ['train', 'valid', 'test']}; cnt = collections.Counter()
for f in ['train', 'valid', 'test']:
    for line in open(src / f'{f}_data_dbpedia.jsonl'):
        d = json.loads(line); sp = split.get(int(d['conversationId']))
        k = {'history': 'train', 'valid': 'valid', 'test': 'test'}.get(sp)
        if k is None: cnt['missing'] += 1; continue
        out[k].write(line); cnt[k] += 1
for o in out.values(): o.close()
for f in ['entity2id.json', 'dbpedia_subkg.json', 'relation2id.json', 'process.py']: shutil.copy(src / f, dst / f)
print(dict(cnt))
subprocess.run(['python', 'process.py'], cwd=dst, check=True)
