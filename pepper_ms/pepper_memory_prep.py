"""PEPPER-MS preparation for our method (steps 1-2).
(1) Memory from the PAST session (S1) of each simulated user: v6 extraction on SEEKER utterances only
    (Reflected_Preferences is simulator-internal and is never read), consolidated into memory entries with concepts.
(2) Item side: IMDb id -> Tag Genome scored movie -> attribute x and population experience yhat.
Outputs (PEPPER/research-notes/pepper-ms-odkg-100/ours/): memory.json, items_imdb.npz, report.json
"""
import os
import json, re, glob, asyncio, collections, sys
import numpy as np
from pathlib import Path
R = Path('./pepper_ms/PEPPER'); RN = R / 'research-notes'
B = Path('.'); IC = B / 'processed/item_chain'
GEN = Path('./data/tag_genome_2021')
OUT = RN / 'pepper-ms-odkg-100/ours'; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(B / 'scripts'))
import extract_v6 as X   # reuse prompt + validator (SYSTEM, VERIFY, validate, norm)

# ---- S1 dialogues (seeker utterances only)
dlg = {}
for f in glob.glob(str(RN / 'e2e-10x20-ms100-s1-shard*/baseline-*/dialogues/*.json')):
    d = json.load(open(f)); uid = Path(f).stem; lines = []
    for t in d[1:]:
        (k, v), = t.items()
        if isinstance(v, str): lines.append(('RECOMMENDER', v)); continue
        lines.append(('USER', v.get('Seeker', ''))); lines.append(('RECOMMENDER', v.get('Recommender', '')))
    dlg[uid] = lines
print('S1 dialogues', len(dlg))

async def extract():
    from openai import AsyncOpenAI
    cli = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=120, max_retries=3)
    sem = asyncio.Semaphore(12); recs = {}; use = [0, 0]
    async def one(uid, lines):
        roles = {i + 1: l for i, l in enumerate(lines)}
        text = '\n'.join(f'[{i}] {r}: {t}' for i, (r, t) in roles.items())
        msgs = [{'role': 'system', 'content': X.SYSTEM}, {'role': 'user', 'content': text}]; raws = []
        for a in range(2):
            async with sem: r = await cli.chat.completions.create(model='gpt-4o-mini', messages=msgs, temperature=0)
            use[0] += r.usage.prompt_tokens; use[1] += r.usage.completion_tokens
            raw = r.choices[0].message.content or ''; raws.append(raw); items, errs = X.validate(raw, roles)
            if not errs: break
            msgs += [{'role': 'assistant', 'content': raw}, {'role': 'user', 'content': 'Some lines were invalid: ' + '; '.join(errs[:5]) + '. Output the full corrected list again in the exact format.'}]
        exp = [it for it in items if it['kind'] == 'experience']
        if exp:
            claims = '\n'.join(f'{k+1}. claimed: {it["target"]} | evidence: "{it["evidence"]}"' for k, it in enumerate(exp))
            async with sem: v = await cli.chat.completions.create(model='gpt-4o-mini', messages=[{'role': 'system', 'content': X.VERIFY}, {'role': 'user', 'content': claims}], temperature=0)
            use[0] += v.usage.prompt_tokens; use[1] += v.usage.completion_tokens
            ok = {int(m.group(1)) for m in re.finditer(r'(\d+)\W+VALID', v.choices[0].message.content or '')}
            for k, it in enumerate(exp): it['verified'] = (k + 1) in ok
        recs[uid] = {'items': items, 'raw_responses': raws, 'errors_after_retry': errs}
    await asyncio.gather(*[one(u, l) for u, l in dlg.items()]); await cli.close()
    return recs, round(use[0] * .15e-6 + use[1] * .6e-6, 4)
recs, usd = asyncio.run(extract())
json.dump(recs, open(OUT / 'extraction_s1.json', 'w'), ensure_ascii=False)

# ---- item side
z = np.load(IC / 'item_chain_v1.npz'); gpos = {g: k for k, g in enumerate(z['items'].tolist())}
yh = 1 / (1 + np.exp(-(((z['X'] - z['mu']) / z['sd']) @ z['W'] + z['b'])))
imdb2g = {}
for l in open(GEN / 'raw/metadata_updated.json'):
    d = json.loads(l)
    if d['item_id'] in gpos: imdb2g['tt' + d['imdbId'].zfill(7)] = gpos[d['item_id']]
idm = json.load(open(RN / 'rating7-imdb-20261008/candidate-identity-map-final.json'))
info = json.load(open(R / 'crs_data/opendialkg/id2info.json'))
cands = sorted({v['identity_key'] for v in idm.values() if v.get('identity_key', '').startswith('tt')})
ids = [c for c in cands if c in imdb2g]
np.savez_compressed(OUT / 'items_imdb.npz', imdb=np.array(ids), X=np.clip(z['X'][[imdb2g[c] for c in ids]], 0, 1).astype(np.float32),
                    Y=yh[[imdb2g[c] for c in ids]].astype(np.float32), ymean=yh.mean(0).astype(np.float32))

# ---- memory entries (entity phrases linked to IMDb ids through catalog titles)
def tn(s):
    s = re.sub(r'\(\d{4}\)', '', s.lower()); s = re.sub(r'^(the|a|an) ', '', s.strip()); s = re.sub(r'[^a-z0-9 ]', ' ', s); return ' '.join(s.split())
title2imdb = collections.defaultdict(set)
for k, v in idm.items():
    if v.get('identity_key', '').startswith('tt'): title2imdb[tn(v['title'])].add(v['identity_key'])
mem, st = {}, collections.Counter()
for uid, r in recs.items():
    ents = {}
    for it in r['items']:
        if it['kind'] == 'experience' and it.get('verified') is False: continue
        kind = 'entity' if it['kind'] in ('movie', 'genre', 'person', 'franchise') else it['kind']
        imdb = None
        if kind == 'entity':
            c = title2imdb.get(tn(it['target']), set()); imdb = next(iter(c)) if len(c) == 1 else None
            st['entity_linked' if imdb else 'entity_unlinked'] += 1
        key = f"{kind}|{imdb or tn(it['target'])}|{it['direction']}"
        e = ents.setdefault(key, {'kind': kind, 'target': it['target'], 'imdb': imdb, 'direction': it['direction'], 'evidence': []})
        e['evidence'].append({'turn': it['turn'], 'quote': it['evidence']})
    mem[uid] = list(ents.values())
json.dump(mem, open(OUT / 'memory.json', 'w'), ensure_ascii=False, indent=0)
rep = {'s1_users': len(dlg), 'extraction_usd': usd, 'catalog_imdb': len(cands), 'catalog_with_chain': len(ids),
       'chain_coverage': round(len(ids) / max(len(cands), 1), 3), **st,
       'entries_per_user_median': float(np.median([len(v) for v in mem.values()])),
       'kinds': dict(collections.Counter(e['kind'] for v in mem.values() for e in v))}
json.dump(rep, open(OUT / 'report.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
