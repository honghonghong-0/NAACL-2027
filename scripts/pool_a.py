"""Pool A (decision 2026-10-12): canonical item = CRSLab/UniCRS movie entity id (UniCRS entity2id numbering, the id
space of KBRD/KGSF/UniCRS/UCCR as trained here). ReDial @id -> movies_merged name -> entity2link URI -> UniCRS entity2id;
pool A = canonical ids that are movie items (item_ids, 6,281) AND reachable from our catalog. Writes processed/pool_a.json:
{'redial2canon': {...}, 'pool': [...], 'item_ids': [...]}"""
import csv, json
from pathlib import Path
R = Path('./data/redial'); B = Path('.')
U = B / 'baselines/unicrs/src/data/redial_ms'
name = {r['databaseId']: r['movieName'] for r in csv.DictReader(open(R / 'movies_merged.csv'))}
link = json.load(open(R / 'entity2link.json')); e2i = json.load(open(U / 'entity2id.json')); items = json.load(open(U / 'item_ids.json')); iset = set(items)
ours = [r['movieId'] for r in csv.DictReader(open(B / 'raw_unzip/movies_with_mentions.csv'))]
r2c = {}
for m in ours:
    u = link.get(name.get(m, ''))
    eid = e2i.get(f'<{u}>') if u else None
    if eid is not None and eid in iset: r2c[m] = eid
pool = sorted(set(r2c.values()))
json.dump({'redial2canon': r2c, 'pool': pool, 'item_ids': items}, open(B / 'processed/pool_a.json', 'w'))
for sp in ['valid', 'test']:
    T = [json.loads(l) for l in open(B / f'processed/eval_turns_{sp}.jsonl')]; tg = [g for t in T for g in t['targets']]
    print(sp, 'targets in pool A', sum(g in r2c for g in tg), '/', len(tg), '| turns without in-pool target', sum(1 for t in T if not any(g in r2c for g in t['targets'])))
print('mapped redial ids', len(r2c), 'pool A size', len(pool), 'of item_ids', len(items))
