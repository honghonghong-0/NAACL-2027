"""ReDial movie id (@databaseId) -> CRSLab/KGSF movie entity id. Chain: movies_merged.csv (databaseId -> movieName)
-> entity2link (name -> DBpedia URI) -> KGSF entity2id ('<URI>' -> id) -> keep if id in movie_ids (6,281 recommendable).
Reports catalog coverage and how many valid/test targets fall outside the CRSLab item set (decision: pool A = intersection)."""
import csv, json
from pathlib import Path
R = Path('./data'); B = Path('.')
name = {r['databaseId']: r['movieName'] for r in csv.DictReader(open(R / 'raw/redial/movies_merged.csv'))}
link = json.load(open(R / 'raw/redial/entity2link.json')); e2i = json.load(open(R / 'kgsf_redial/entity2id.json')); mv = set(json.load(open(R / 'kgsf_redial/movie_ids.json')))
ours = [r['movieId'] for r in csv.DictReader(open(B / 'raw_unzip/movies_with_mentions.csv'))]
mp, why = {}, {'no_name': 0, 'no_link': 0, 'no_entity': 0, 'not_movie_item': 0}
for m in ours:
    n = name.get(m)
    if n is None: why['no_name'] += 1; continue
    u = link.get(n)
    if u is None: why['no_link'] += 1; continue
    eid = e2i.get(f'<{u}>')
    if eid is None: why['no_entity'] += 1; continue
    if eid not in mv: why['not_movie_item'] += 1; continue
    mp[m] = eid
print('ours', len(ours), 'mapped', len(mp), 'distinct crslab ids', len(set(mp.values())), 'unmapped reasons', why)
for sp in ['valid', 'test']:
    T = [json.loads(l) for l in open(B / f'processed/eval_turns_{sp}.jsonl')]
    tg = [g for t in T for g in t['targets']]; inn = sum(g in mp for g in tg)
    turns_all_out = sum(1 for t in T if not any(g in mp for g in t['targets']))
    print(sp, 'targets', len(tg), 'in pool A', inn, f'({inn/len(tg):.1%})', '| turns with no in-pool target', turns_all_out, '/', len(T))
json.dump(mp, open(B / 'processed/redial_to_crslab_movie.json', 'w'))
