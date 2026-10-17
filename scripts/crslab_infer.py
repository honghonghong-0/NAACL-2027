"""Score OUR ReDial-MS eval turns with a trained CRSLab KBRD (restored from save/). Context entities = DBpedia
entities+movies (UniCRS linking) of all messages before the turn, de-duplicated in order (same rule CRSLab uses).
Output: <out>.npz with keys (user, conv_id, turn_index) and float16 scores over item_ids (6,281 movie entities).
Usage (from baselines/crslab): python ../../scripts/crslab_infer.py <config.yaml> <split> <out_prefix>"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '.')
from crslab.config import Config
from crslab.data import get_dataset, get_dataloader
from crslab.system import get_system
B = Path('.'); U = B / 'baselines/unicrs/src/data/redial'
cfg, split, outp = sys.argv[1], sys.argv[2], sys.argv[3]
opt = Config(cfg, gpu='0')
ds = get_dataset(opt, opt['tokenize'], False, False)
dl = lambda d: get_dataloader(opt, d, ds.vocab)
system = get_system(opt, dl(ds.train_data), dl(ds.valid_data), dl(ds.test_data), ds.vocab, ds.side_data, restore_system=True)
model = system.model; model.eval(); e2i = ds.vocab['entity2id']
items = json.load(open(B / 'processed/pool_a.json'))['item_ids']; idx = torch.tensor(items, device=system.device)
raw = {}
for f in ['train', 'valid', 'test']:
    for line in open(U / f'{f}_data_dbpedia.jsonl'):
        d = json.loads(line); raw[int(d['conversationId'])] = d['messages']
keys, scores = [], []
turns = [json.loads(l) for l in open(B / f'processed/eval_turns_{split}.jsonl')]
with torch.no_grad():
    for i in range(0, len(turns), 256):
        chunk = turns[i:i + 256]; ctx = []
        for t in chunk:
            seen, ents = set(), []
            for m in raw[t['conv_id']][:t['turn_index']]:
                for e in m['entity'] + m['movie']:
                    if e in e2i and e2i[e] not in seen: seen.add(e2i[e]); ents.append(e2i[e])
            ctx.append(ents)
        _, sc = model.recommend({'context_entities': ctx, 'item': torch.zeros(len(chunk), dtype=torch.long, device=system.device)}, 'test')
        scores.append(sc[:, idx].float().cpu().numpy().astype(np.float16)); keys += [(t['user'], t['conv_id'], t['turn_index']) for t in chunk]
np.savez_compressed(outp, keys=np.array(keys), scores=np.concatenate(scores), item_ids=np.array(items))
print('saved', outp, len(keys), 'turns')
