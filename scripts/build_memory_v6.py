"""Memory store v6: store by affordance + update without deletion (our modification of LightMem's update step,
which deletes the older memory on conflict).
Chain: entity -> property -> affordance (experience). Value layer dropped (0.1% of items).
Kinds are collapsed to the three chain layers: movie/genre/person/franchise -> entity (a movie is an entity);
whether an entity is a movie is decided by linking to a ReDial movie id, not by the LLM label.
1) link: movie entries -> ReDial movie id (session mentions first, then global title index; ambiguous -> unlinked)
2) concept map: property / experience phrases -> Tag Genome concepts by MiniLM cosine, top-3 kept with their cosine
   (soft weights, no threshold); movies -> experiences through the population affordance (yhat, top-5 centered)
3) update: merge key = (kind, normalized target, direction, normalized condition) -> evidence list grows;
   same (kind, target, condition) with opposite directions -> both kept, flagged 'conflict';
   different conditions -> coexist.
4) affordance index: experience concept -> entry ids (via explicit experience entries, property entries through
   the population affordance, and movie entries through yhat)
Also writes a flat edge file compatible with alpha_train / residual_fuse (relational_memory_v6.json)
and a v5-compatible extraction file for current-session deltas (extract_v6/compat_v5.jsonl).
"""
import json, re, csv, collections
import numpy as np
from pathlib import Path
B = Path('.'); P = B / 'processed'; IC = P / 'item_chain'
EMB = 'sentence-transformers/all-MiniLM-L6-v2'

def tn(s):
    s = re.sub(r'\(\d{4}\)', '', s.lower()); s = re.sub(r'^(the|a|an) ', '', s.strip())
    s = re.sub(r'[^a-z0-9 ]', ' ', s); return ' '.join(s.split())

users = json.load(open(P / 'users.json'))
names = {r['movieId']: r['movieName'].strip() for r in csv.DictReader(open(B / 'raw_unzip/movies_with_mentions.csv'))}
gidx = collections.defaultdict(set)
for k, v in names.items(): gidx[tn(v)].add(k)
recs = [json.loads(l) for l in open(B / 'extract_v6' / 'full_redial_v6.jsonl')]

vocab = json.load(open(IC / 'item_chain_v1_vocab.json')); PROPS, EXPS = vocab['properties'], vocab['experiences']
z = np.load(IC / 'item_chain_v1.npz'); gpos = {g: k for k, g in enumerate(z['items'].tolist())}
yh = 1 / (1 + np.exp(-(((z['X'] - z['mu']) / z['sd']) @ z['W'] + z['b']))); yc = yh - yh.mean(0)
r2g = json.load(open(IC / 'redial2genome.json'))

from sentence_transformers import SentenceTransformer
enc = SentenceTransformer(str(EMB), device='cpu')
def E(xs): return enc.encode(xs, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
EP, EE = E(PROPS), E(EXPS)
phr = sorted({it['target'] for r in recs for it in r['items'] if it['kind'] in ('property', 'experience')})
PE = dict(zip(phr, E(phr))) if phr else {}
# population affordance in property space: which experiences a property tends to give (W column signs)
W = z['W']  # [P, A]

def concepts(it):
    if it['kind'] == 'experience':
        s = EE @ PE[it['target']]; top = np.argsort(-s)[:3]
        return [('experience', EXPS[k], round(float(s[k]), 3)) for k in top]
    if it['kind'] == 'property':
        s = EP @ PE[it['target']]; top = np.argsort(-s)[:3]
        out = [('property', PROPS[k], round(float(s[k]), 3)) for k in top]
        k0 = top[0]; ex = np.argsort(-W[k0])[:3]
        out += [('experience_via_property', EXPS[a], round(float(s[k0] * W[k0, a]), 3)) for a in ex if W[k0, a] > 0]
        return out
    return []

def link_movie(target, sess):
    t = tn(target)
    if not t: return None
    local = [m for m in sess.get('mentions', {}) if tn(names.get(m, '')) and (t in tn(names[m]) or tn(names[m]) in t)]
    if len(local) == 1: return local[0]
    g = gidx.get(t, set())
    return next(iter(g)) if len(g) == 1 else None

stores, flat, compat, stat = {}, {}, [], collections.Counter()
sidx = {(u, s['conv_id']): (k, s) for u, ss in users.items() for k, s in enumerate(ss)}
for r in recs:
    _, u, cid = r['key'].split(':'); k, s = sidx[(u, int(cid))]
    st = stores.setdefault(u, {'entries': {}, 'affordance_index': collections.defaultdict(list)})
    citems = []
    for it in r['items']:
        if it['kind'] == 'experience' and it.get('verified') is False:
            stat['experience_rejected_by_verifier'] += 1; continue
        if it['kind'] in ('movie', 'genre', 'person', 'franchise'): it = {**it, 'kind': 'entity'}
        mid = link_movie(it['target'], s) if it['kind'] == 'entity' else None
        if it['kind'] == 'entity': stat['entity_linked_to_movie' if mid else 'entity_unlinked'] += 1
        tgt = mid if mid else tn(it['target'])
        cond = tn(it['condition']) if it['condition'] else None
        key = f"{it['kind']}|{tgt}|{it['direction']}|{cond}"
        ev = {'session_idx': k, 'conv_id': int(cid), 'split': r['split'], 'turn': it['turn'], 'quote': it['evidence']}
        if key in st['entries']:
            st['entries'][key]['evidence'].append(ev); stat['merged'] += 1
        else:
            st['entries'][key] = {'kind': it['kind'], 'target': it['target'], 'movie_id': mid, 'direction': it['direction'],
                                  'condition': it['condition'], 'evidence': [ev], 'concepts': concepts(it), 'conflict': False}
            stat['new_entry'] += 1
        citems.append({'relation': 'preference', 'target_type': 'movie' if mid else it['kind'],
                       'layer': {'entity': 'L1', 'property': 'L2', 'experience': 'L3'}[it['kind']],
                       'target': names.get(mid, it['target']) if mid else it['target'], 'valence': it['direction'], 'turn': it['turn']})
    compat.append({'key': r['key'], 'items': citems})
for u, st in stores.items():
    by = collections.defaultdict(list)
    for key, e in st['entries'].items():
        kind, tgt, d, cond = key.split('|'); by[(kind, tgt, cond)].append(key)
    for ks in by.values():
        if len({st['entries'][x]['direction'] for x in ks}) > 1:
            for x in ks: st['entries'][x]['conflict'] = True
            stat['conflict_groups'] += 1
    for key, e in st['entries'].items():
        if e['movie_id'] and e['movie_id'] in r2g and r2g[e['movie_id']] in gpos:
            v = yc[gpos[r2g[e['movie_id']]]]
            for a in np.argsort(-v)[:5]: st['affordance_index'][EXPS[a]].append(key)
        for typ, name, w in e['concepts']:
            if typ in ('experience', 'experience_via_property'): st['affordance_index'][name].append(key)
    st['affordance_index'] = dict(st['affordance_index'])
    flat[u] = [{'rel': 'preference', 'source': 'utterance', 'valence': e['direction'], 'movie_id': e['movie_id'],
                'session_idx': ev['session_idx'], 'conv_id': ev['conv_id'], 'split': ev['split'], 'kind': e['kind'],
                'target': e['target'], 'concepts': e['concepts'], 'conflict': e['conflict'],
                'layer': {'property': 'L2', 'experience': 'L3'}.get(e['kind'], 'L1')}
               for e in st['entries'].values() for ev in e['evidence']]
json.dump(stores, open(P / 'memory_store_v6.json', 'w'))
json.dump(flat, open(P / 'relational_memory_v6.json', 'w'))
with open(B / 'extract_v6' / 'compat_v5.jsonl', 'w') as f:
    for c in compat: f.write(json.dumps(c) + '\n')
ne = [len(st['entries']) for st in stores.values()]; na = [len(st['affordance_index']) for st in stores.values()]
print(json.dumps({'users': len(stores), **stat, 'entries_per_user_median': float(np.median(ne)),
                  'affordances_per_user_median': float(np.median(na))}, default=int))
ex = next(iter(stores.values()))
print('example affordance index:', {k: len(v) for k, v in list(ex['affordance_index'].items())[:8]})
