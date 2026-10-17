"""M-v5 applicability alpha(m, c_t) = Pr(memory m pushes the true target above the average candidate | context).
Counterfactual weak label (train = history-session turns of memory users, leave-one-conv-out co-mention):
  contribution_m(i) = delta_m * q^T (yhat_m (x) x_m - Sbar) x_i      (single-memory personal realization)
  y = 1  if contribution_m(target) > mean_i contribution_m(i) over candidates (all chain-covered non-current movies)
Context-only features (no target): cos(yhat_m - mean, q), cooc(m, current movies) (log), sessions ago, direction, |q|, genre cos(m, current).
Model: logistic regression; calibration checked on held-out users (reliability bins, ECE). Saves coefficients for realize_v4.
"""
import json, sys, re, math, random, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, './scripts')
import baselines_simple as bs
from baselines_simple import users, movies, IDX, N, P, CAT
random.seed(0); np.random.seed(0)
IC = P / 'item_chain'
z = np.load(IC / 'item_chain_v1.npz'); gpos = {g: k for k, g in enumerate(z['items'].tolist())}
yh = 1 / (1 + np.exp(-(((z['X'] - z['mu']) / z['sd']) @ z['W'] + z['b']))); yc = yh - yh.mean(0)
r2g = json.load(open(IC / 'redial2genome.json'))
XN = np.zeros((N, z['X'].shape[1]), np.float32); YN = np.zeros((N, yh.shape[1]), np.float32); EC = np.zeros_like(YN); COVER = np.zeros(N, bool)
for rid, g in r2g.items():
    if rid in IDX and g in gpos:
        k = IDX[rid]; XN[k] = np.clip(z['X'][gpos[g]], 0, 1); YN[k] = yh[gpos[g]]; EC[k] = yc[gpos[g]]; COVER[k] = True
Xg = np.clip(z['X'], 0, 1); SBAR = (yh.T @ Xg) / len(Xg)
import os
MEM = json.load(open(P / os.environ.get('MEM_FILE', 'relational_memory_v5.1.json')))
EXT = {json.loads(l)['key']: json.loads(l)['items'] for l in open(P.parent / os.environ.get('EXT_FILE', 'extract_v4/full_redial_v5.jsonl'))}
pop, co, _ = bs.stats('valid')

def tn(s):
    s = re.sub(r'\(\d{4}\)', '', s.lower()); s = re.sub(r'[^a-z0-9 ]', ' ', s); return ' '.join(s.split())
def deltas(u, s, i, cur):
    msgs = s['messages'][:i]
    seeker = {x for m in msgs if m['role'] == 'seeker' for x in re.findall(r'@(\d+)', m['text'])}
    titles = {x: tn(s.get('mentions', {}).get(x, '')) for x in cur}
    d = {x: (0.5 if x in seeker else 0.0) for x in cur}
    for it in EXT.get(f"redial:{u}:{s['conv_id']}", []):
        if it['turn'] - 1 >= i or it['layer'] != 'L1' or it['target_type'] != 'movie': continue
        if it['relation'] not in ('appraisal', 'preference') or it['valence'] not in '+-': continue
        tg = tn(it['target'])
        for x, ti in titles.items():
            if tg and ti and (tg in ti or ti in tg): d[x] = 1.0 if it['valence'] == '+' else -1.0
    return d
def turns_of(s):
    seen, out = set(), []
    for i, m in enumerate(s['messages']):
        ids = [x for x in re.findall(r'@(\d+)', m['text']) if x in IDX]
        if m['role'] == 'recommender':
            tg = [x for x in dict.fromkeys(ids) if x not in seen]
            if tg and i > 0: out.append((i, tg))
        seen.update(ids)
    return out
def mem_items(u, idx):
    out = {}
    for e in MEM.get(u, []):
        if e['source'] == 'utterance' and e['session_idx'] < idx and e['split'] == 'history' and e['rel'] in ('preference', 'appraisal') \
           and e['movie_id'] in IDX and COVER[IDX[e['movie_id']]] and e['valence'] in '+-':
            out[e['movie_id']] = (1.0 if e['valence'] == '+' else -1.0, idx - e['session_idx'])
    return out
def context(u, s, idx, i):
    cur = list(dict.fromkeys(movies(s['messages'][:i])))
    d = deltas(u, s, i, cur)
    q = sum((w * EC[IDX[x]] for x, w in d.items() if w != 0 and COVER[IDX[x]]), np.zeros(EC.shape[1], np.float32))
    return cur, q
def feats(m, dm, ago, cur, q, own=None):
    k = IDX[m]; e = EC[k]; qn = np.linalg.norm(q) + 1e-9
    c = sum(co[x].get(m, 0) for x in cur) - (len(cur) if own and m in own else 0)
    ec = sum((EC[IDX[x]] for x in cur if COVER[IDX[x]]), np.zeros_like(e))
    return [float(e @ q / (np.linalg.norm(e) + 1e-9) / qn) * dm, math.log1p(max(c, 0)), 1 / (1 + ago), dm,
            math.log1p(qn), float(e @ ec / (np.linalg.norm(e) * np.linalg.norm(ec) + 1e-9)) * dm]
FEATS = ['cos(mem_exp,q)*dir', 'log cooc(mem,cur)', '1/(1+sessions_ago)', 'direction', 'log|q|', 'cos(mem_exp,cur_exp)*dir']

if __name__ == '__main__':
    Xr, yr, gr = [], [], []
    cand_all = np.nonzero(COVER)[0]
    for u in MEM:
        ss = users[u]
        for idx, s in enumerate(ss):
            if s['split'] != 'history' or idx == 0: continue
            mi = mem_items(u, idx)
            if not mi: continue
            own = set(movies(s['messages']))
            for i, tg in turns_of(s):
                cur, q = context(u, s, idx, i)
                if not np.abs(q).sum(): continue
                tgs = [IDX[g] for g in tg if g in IDX and COVER[IDX[g]] and g not in cur]
                if not tgs: continue
                base = XN[cand_all]                     # [C,P]
                for m, (dm, ago) in mi.items():
                    if m in cur: continue
                    k = IDX[m]
                    D = dm * (np.outer(YN[k], XN[k]) - SBAR)   # [A,P] single-memory personal deviation
                    v = D.T @ q                                  # [P]
                    contrib_mean = float((base @ v).mean())
                    lab = float(np.mean([XN[t] @ v for t in tgs]) > contrib_mean)
                    Xr.append(feats(m, dm, ago, cur, q, own)); yr.append(lab); gr.append(u)
    Xr, yr = np.array(Xr), np.array(yr); print('pairs', len(yr), 'positive rate', yr.mean().round(3))
    us = sorted(set(gr)); rng = random.Random(0); hold = set(rng.sample(us, len(us) // 5)); gr = np.array(gr)
    te = np.isin(gr, list(hold)); tr = ~te
    mu, sd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-9
    clf = LogisticRegression(C=1.0, max_iter=2000).fit((Xr[tr] - mu) / sd, yr[tr])
    p = clf.predict_proba((Xr[te] - mu) / sd)[:, 1]
    from sklearn.metrics import roc_auc_score
    bins = np.clip((p * 10).astype(int), 0, 9); ece = sum(abs(p[bins == b].mean() - yr[te][bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    print('held-out users AUC', round(roc_auc_score(yr[te], p), 4), 'ECE', round(float(ece), 4))
    for f, w in zip(FEATS, clf.coef_[0]): print(f'  {f:28s} {w:+.3f}')
    json.dump({'feats': FEATS, 'mu': mu.tolist(), 'sd': sd.tolist(), 'coef': clf.coef_[0].tolist(), 'intercept': float(clf.intercept_[0]),
               'heldout_auc': float(roc_auc_score(yr[te], p)), 'ece': float(ece), 'n_pairs': int(len(yr))},
              open(IC / os.environ.get('ALPHA_FILE', 'alpha_v1.json'), 'w'), indent=1)
