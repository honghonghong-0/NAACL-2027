"""Inspired (single session) evaluation of the current-session part of the method. DEV split only (test not read).
Pool: every movie title in the Inspired 'movies' column (train+dev). Turns: recommender utterances that mention
movies not mentioned earlier in the dialog; targets = those new movies; prefix = earlier utterances.
Backbone: co-mention from TRAIN dialogs + 1e-3 popularity.  Chain: Tag Genome via movie_database imdb ids.
q = unit(q_implicit from movies the seeker liked/disliked earlier) + unit(q_explicit from experience/attribute
statements earlier in the dialog; v5 extraction, verifier-rejected experiences dropped; phrases -> Tag Genome
concepts by MiniLM top-3 cosine).  rho_pop = q^T Wbar x_i ; rho_cur = q^T Delta x_i with Delta from current-session
evidence (movies +-1, attributes, experiences; kappa = 5).  s = z(cooc) + g1 z(rho_pop) + g2 z(rho_cur),
gammas cross-fitted over a 2-fold dialog split; paired dialog bootstrap.
"""
import csv, json, re, math, random, collections
import numpy as np
from pathlib import Path
D = Path('./data/inspired')
B = Path('.'); IC = B / 'processed/item_chain'
GEN = Path('./data/tag_genome_2021')
EMB = 'sentence-transformers/all-MiniLM-L6-v2'
KAPPA = 5.0

def tn(s):
    s = re.sub(r'\(\d{4}\)', '', s.lower()); s = re.sub(r'^(the|a|an) ', '', s.strip()); s = re.sub(r'[^a-z0-9 ]', ' ', s); return ' '.join(s.split())
def split_titles(x): return [t.strip() for t in re.split(r';|\|', x) if t.strip()] if x else []

# ---- item chain
z = np.load(IC / 'item_chain_v1.npz'); gpos = {g: k for k, g in enumerate(z['items'].tolist())}
yh = 1 / (1 + np.exp(-(((z['X'] - z['mu']) / z['sd']) @ z['W'] + z['b']))); yc = yh - yh.mean(0)
Xg = np.clip(z['X'], 0, 1); SBAR = (yh.T @ Xg) / len(Xg)
voc = json.load(open(IC / 'item_chain_v1_vocab.json')); PROPS, EXPS = voc['properties'], voc['experiences']
pi, ai = {p: k for k, p in enumerate(PROPS)}, {a: k for k, a in enumerate(EXPS)}
Wp = np.clip(z['W'], 0, None); PROF_P = Wp / (Wp.max(1, keepdims=True) + 1e-9); PROF_A = (Wp / (Wp.max(0, keepdims=True) + 1e-9)).T
imdb2g = {}
for l in open(GEN / 'raw/metadata_updated.json'):
    d = json.loads(l)
    if d['item_id'] in gpos: imdb2g[d['imdbId'].lstrip('0')] = d['item_id']
tyear2imdb = {}
for r in csv.DictReader(open(D / 'movie_database.tsv'), delimiter='\t'):
    tyear2imdb[(tn(r['title']), r['year'][:4])] = r['imdb_id'].replace('tt', '').lstrip('0')

# ---- dialogs
def load(split):
    dl = collections.OrderedDict()
    for r in csv.DictReader(open(D / f'dialog_data/{split}.tsv'), delimiter='\t'):
        dl.setdefault(r['dialog_id'], []).append(r)
    return dl
TR, DV = load('train'), load('dev')
titles = sorted({t for dl in (TR, DV) for rows in dl.values() for r in rows for t in split_titles(r['movies'])})
IDX = {t: k for k, t in enumerate(titles)}; N = len(titles)
def gidx(t):
    m = re.search(r'\((\d{4})\)', t); y = m.group(1) if m else ''
    imdb = tyear2imdb.get((tn(t), y)); return gpos.get(imdb2g.get(imdb)) if imdb else None
G = [gidx(t) for t in titles]; COVER = np.array([g is not None for g in G])
EC = np.zeros((N, yc.shape[1]), np.float32); XN = np.zeros((N, Xg.shape[1]), np.float32); YN = np.zeros_like(EC)
for k, g in enumerate(G):
    if g is not None: EC[k], XN[k], YN[k] = yc[g], Xg[g], yh[g]
import os
if os.environ.get('REVIEW_FILL') == '1':   # D5: review-derived attributes for movies without Tag Genome scores
    rf = np.load(IC / 'review_fill_inspired_v1.npz'); rpos = {g: k for k, g in enumerate(rf['items'].tolist())}
    t2g = json.load(open(IC / 'inspired_title2genome_unscored.json')); nfill = 0
    for t, g in t2g.items():
        k = IDX.get(t)
        if k is None or COVER[k] or g not in rpos: continue
        x = rf['X'][rpos[g]]; y = 1 / (1 + np.exp(-(((x - z['mu']) / z['sd']) @ z['W'] + z['b'])))
        XN[k], YN[k], EC[k] = x, y, y - yh.mean(0); COVER[k] = True; nfill += 1
    print('review-filled movies', nfill)
print('pool', N, 'chain coverage', round(COVER.mean(), 3))

pop = np.zeros(N); co = collections.defaultdict(collections.Counter)
for rows in TR.values():
    ms = list(dict.fromkeys(t for r in rows for t in split_titles(r['movies'])))
    for r in rows:
        if r['speaker'] == 'RECOMMENDER':
            for t in split_titles(r['movies']): pop[IDX[t]] += 1
    for a in ms:
        for b in ms:
            if a != b: co[a][b] += 1

EXT = {json.loads(l)['key']: json.loads(l)['items'] for l in open(B / 'extract_v4/full_inspired_v5.jsonl')}
from sentence_transformers import SentenceTransformer
enc = SentenceTransformer(str(EMB), device='cpu')
phr = sorted({it['target'] for v in EXT.values() for it in v if it['layer'] in ('L2', 'L3')})
PE = dict(zip(phr, enc.encode(phr, batch_size=256, normalize_embeddings=True, show_progress_bar=False)))
EP = enc.encode(PROPS, normalize_embeddings=True, show_progress_bar=False); EE = enc.encode(EXPS, normalize_embeddings=True, show_progress_bar=False)
def unit(v): n = np.linalg.norm(v); return v / n if n > 0 else v

rows_out = []; info = collections.Counter()
for did, rows in DV.items():
    items = EXT.get(f'inspired:{did}', [])
    seen = []
    for i, r in enumerate(rows):
        ms = split_titles(r['movies'])
        new = [t for t in dict.fromkeys(ms) if t not in seen]
        if r['speaker'] == 'RECOMMENDER' and new and i > 0:
            cur = list(dict.fromkeys(seen))
            pre = [it for it in items if it['turn'] - 1 < i]
            seeker = {t for rr in rows[:i] if rr['speaker'] == 'SEEKER' for t in split_titles(rr['movies'])}
            dlt = {t: (0.5 if t in seeker else 0.0) for t in cur}
            for it in pre:
                if it['layer'] == 'L1' and it['target_type'] == 'movie' and it['relation'] in ('appraisal', 'preference') and it['valence'] in '+-':
                    for t in cur:
                        if tn(it['target']) and (tn(it['target']) in tn(t) or tn(t) in tn(it['target'])): dlt[t] = 1.0 if it['valence'] == '+' else -1.0
            q_imp = sum((w * EC[IDX[t]] for t, w in dlt.items() if w and COVER[IDX[t]]), np.zeros(EC.shape[1], np.float32))
            q_exp = np.zeros(len(EXPS), np.float32); S = np.zeros_like(SBAR); n = 0.; npos = 0.
            for it in pre:
                d = 1.0 if it['valence'] == '+' else -1.0 if it['valence'] == '-' else 0.0
                if not d: continue
                if it['layer'] == 'L3' and it.get('verified') is not False:
                    s_ = EE @ PE[it['target']]
                    for a in np.argsort(-s_)[:3]: q_exp[a] += d * s_[a]; S[a, :] += d * s_[a] * PROF_A[a]; n += s_[a]
                elif it['layer'] == 'L2':
                    s_ = EP @ PE[it['target']]; p = int(np.argmax(s_))
                    q_exp += d * s_[p] * PROF_P[p]; S[:, p] += d * s_[p] * PROF_P[p]; n += s_[p]
            for t, w in dlt.items():
                if abs(w) == 1.0 and COVER[IDX[t]]:
                    k = IDX[t]; S += w * np.outer(YN[k], XN[k]); n += 1; npos += w
            q = unit(q_imp) + unit(q_exp)
            rp = EC @ q if np.abs(q).sum() else np.zeros(N)
            rc = XN @ (((S - npos * SBAR) / (n + KAPPA)).T @ q) if (n > 0 and np.abs(q).sum()) else np.zeros(N)
            cv = np.zeros(N)
            for a in cur:
                for b, c in co[a].items(): cv[IDX[b]] += c
            cv = cv / (cv.max() + 1e-9) + 1e-3 * pop / pop.max()
            tg = [IDX[t] for t in new]
            rows_out.append(dict(did=did, cur=[IDX[t] for t in cur], tg=tg, cv=cv, rp=rp, rc=rc, exp=bool(np.abs(q_exp).sum()), cnt=n))
            info['turns'] += 1; info['q_exp'] += bool(np.abs(q_exp).sum()); info['q_any'] += bool(np.abs(q).sum()); info['cur_evidence'] += n > 0
        seen += ms
print(dict(info))

def zc(v, m):
    o = np.zeros(N); s = v[m].std(); o[m] = (v[m] - v[m].mean()) / s if s > 0 else 0; return o
def ndcg(g1, g2):
    pu = collections.defaultdict(list); tag = collections.defaultdict(list)
    for x in rows_out:
        m = np.ones(N, bool); m[x['cur']] = False
        sc = zc(x['cv'], m) + g1 * zc(x['rp'], m & COVER) + g2 * zc(x['rc'], m & COVER)
        sc = sc + np.random.default_rng(0).random(N) * 1e-7; sc[~m] = -np.inf
        rk = np.empty(N, int); rk[np.argsort(-sc)] = np.arange(1, N + 1)
        for g in x['tg']:
            if m[g]: pu[x['did']].append(1 / math.log2(rk[g] + 1) if rk[g] <= 10 else 0.); tag[x['did']].append(x['exp'])
    return pu, tag
GR = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
grids = {'nochain': [(0, 0)], 'pop': [(g, 0) for g in GR], 'pop+pers': [(a, b) for a in GR for b in GR]}
res = {c: ndcg(*c) for c in {c for v in grids.values() for c in v}}
TAG = res[(0, 0)][1]
ds = sorted(res[(0, 0)][0]); random.Random(0).shuffle(ds); halves = [set(ds[::2]), set(ds[1::2])]
def mean(pu, S_, f=None): xs = [x for d in S_ for x, t in zip(pu[d], TAG[d]) if f is None or f(t)]; return float(np.mean(xs)) if xs else float('nan')
final = {}
for cond, grid in grids.items():
    pu = {}; picks = []
    for h in (0, 1):
        pk = max(grid, key=lambda c: mean(res[c][0], halves[h])); picks.append(pk)
        for d in halves[1 - h]: pu[d] = res[pk][0][d]
    final[cond] = (pu, picks)
def boot(a, b, f=None):
    rng = random.Random(1); out = []
    for _ in range(2000):
        s_ = rng.choices(ds, k=len(ds)); va, vb = mean(final[a][0], s_, f), mean(final[b][0], s_, f)
        if not math.isnan(va): out.append(vb - va)
    return np.round(np.percentile(out, [2.5, 97.5]), 4)
print('== Inspired dev, backbone co-mention (targets', sum(len(v) for v in final['nochain'][0].values()), ')')
for c in grids:
    m = mean(final[c][0], ds); line = f'  {c:9s} NDCG@10 {m:.4f} gammas {final[c][1]}'
    if c != 'nochain': line += f'  Delta {m - mean(final["nochain"][0], ds):+.4f} CI95 {boot("nochain", c)}'
    print(line)
print(f'  pop+pers vs pop  Delta {mean(final["pop+pers"][0], ds) - mean(final["pop"][0], ds):+.4f} CI95 {boot("pop", "pop+pers")}')
for nm, f in [('explicit experience/attribute in prefix', lambda t: t), ('none', lambda t: not t)]:
    print(f'  slice [{nm}] pop+pers - nochain {mean(final["pop+pers"][0], ds, f) - mean(final["nochain"][0], ds, f):+.4f} CI95 {boot("nochain", "pop+pers", f)}')
