"""Paired comparison on Pool A (valid only), same protocol as eval_poola.py (current-dialogue movies excluded, per-target NDCG@10).
- paired user-level bootstrap CI of Delta = NDCG(B) - NDCG(A)  (2,000 resamples)
- candidate diagnostics for reranker npz (non-candidates carry score <= -9): candidate recall, oracle NDCG@10, conditional NDCG@10
- PRE-REGISTERED slices (fixed 2026-10-13 before seeing personal-realization results) + interaction effect
Usage: paired_eval.py A=path.npz B=path.npz [C=path.npz ...]   (first = reference; cooc available as A=cooc)"""
import json, sys, math, random, collections, re
import numpy as np
sys.path.insert(0, './scripts')
import baselines_simple as bs
from baselines_simple import users, movies, IDX, P, co_vec
from baselines_simple import norm_max as _nm
class sp: norm = staticmethod(_nm)

args = dict(a.split('=', 1) for a in sys.argv[1:]); names = list(args)
PA = json.load(open(P / 'pool_a.json')); r2c = PA['redial2canon']; pool = PA['pool']; col = {c: i for i, c in enumerate(pool)}; M = len(pool)
r_idx = np.array([IDX[r] for r in r2c]); c_idx = np.array([col[c] for c in r2c.values()])
def to_pool(sc):
    out = np.full(M, -np.inf); np.maximum.at(out, c_idx, sc[r_idx]); return out
pop, co, _ = bs.stats('valid')
sessions = {(int(u), s['conv_id']): s for u, ss in users.items() for s in ss}
turns = [json.loads(l) for l in open(P / 'eval_turns_valid.jsonl')]
EX = {json.loads(l)['key']: json.loads(l)['items'] for l in open(P.parent / 'extract_v4' / 'full_redial_v5.jsonl')}
MEM = json.load(open(P / 'relational_memory_v5.1.json'))
S = {}
for n, p in args.items():
    if p == 'cooc': S[n] = None; continue
    z = np.load(p); pos = {c: i for i, c in enumerate(z['item_ids'].tolist())}; sel = np.array([pos[c] for c in pool])
    S[n] = {tuple(k): z['scores'][i][sel].astype(np.float32) for i, k in enumerate(z['keys'].tolist())}

def slices(t, s, cur):
    key = f"redial:{t['user']}:{t['conv_id']}"; ti = t['turn_index']
    pre = [it for it in EX.get(key, []) if it['turn'] - 1 < ti]
    exp_now = any(it['layer'] in ('L3', 'L4') or it['relation'] == 'experience_request' for it in pre)
    mov_now = any(it['target_type'] == 'movie' and it['relation'] in ('appraisal', 'preference') and it['valence'] in '+-' for it in pre)
    idx = next(k for k, x in enumerate(users[str(t['user'])]) if x['conv_id'] == t['conv_id'])
    past_exp = any(e['source'] == 'utterance' and e['session_idx'] < idx and e['split'] == 'history' and e.get('layer') in ('L3', 'L4')
                   for e in MEM.get(str(t['user']), []))
    h = t['n_history_sessions']; nc = len(cur)
    return {'exp_now': exp_now, 'movie_dir_now': mov_now, 'past_exp_mem': past_exp,
            'hist': '0' if h == 0 else '1' if h == 1 else '2+', 'n_cur': '0' if nc == 0 else '1-3' if nc <= 3 else '4+'}

R = {n: collections.defaultdict(list) for n in names}; TAG = collections.defaultdict(list); CAND = collections.defaultdict(list)
for t in turns:
    s = sessions[(t['user'], t['conv_id'])]; cur = list(dict.fromkeys(movies(s['messages'][:t['turn_index']])))
    G = sorted({col[r2c[g]] for g in t['targets'] if g in r2c})
    if not G: continue
    excl = {col[r2c[c]] for c in cur if c in r2c}; sl = slices(t, s, cur)
    for n in names:
        sc = to_pool(sp.norm(co_vec(co, cur)) + 1e-3 * pop / pop.max()) if S[n] is None else S[n][(t['user'], t['conv_id'], t['turn_index'])].copy()
        incand = np.isfinite(sc) & (sc > -9)
        sc = sc + np.random.default_rng(0).random(M) * 1e-7; sc[list(excl)] = -np.inf
        order = np.argsort(-sc); rk = np.empty(M, int); rk[order] = np.arange(1, M + 1)
        for g in G:
            v = 1 / math.log2(rk[g] + 1) if rk[g] <= 10 else 0.
            R[n][t['user']].append(v)
            if S[n] is not None: CAND[n].append((bool(incand[g]), v))
    for g in G: TAG[t['user']].append(sl)

users_ = sorted(R[names[0]])
def mean(n, us, f=None):
    xs = [x for u in us for x, tg in zip(R[n][u], TAG[u]) if f is None or f(tg)]
    return float(np.mean(xs)) if xs else float('nan'), len(xs)
def paired(a, b, f=None, B=2000):
    rng = random.Random(0); d = []
    for _ in range(B):
        us = rng.choices(users_, k=len(users_))
        xa = [x for u in us for x, tg in zip(R[a][u], TAG[u]) if f is None or f(tg)]
        xb = [x for u in us for x, tg in zip(R[b][u], TAG[u]) if f is None or f(tg)]
        if xa: d.append(np.mean(xb) - np.mean(xa))
    return np.round(np.percentile(d, [2.5, 97.5]), 4)

ref = names[0]
print('== overall (n_targets', sum(len(v) for v in R[ref].values()), ')')
for n in names:
    m, k = mean(n, users_)
    line = f'{n:28s} NDCG@10 {m:.4f}'
    if n != ref: line += f' | Delta vs {ref} {m - mean(ref, users_)[0]:+.4f} paired CI95 {paired(ref, n)}'
    if CAND[n]:
        rc = np.mean([c for c, _ in CAND[n]]); cond = np.mean([v for c, v in CAND[n] if c])
        line += f' | cand recall {rc:.3f} oracle NDCG@10 {rc:.3f} cond NDCG@10 {cond:.4f}'
    print(line)
SL = [('exp_now', lambda t: t['exp_now'], lambda t: not t['exp_now']),
      ('movie_dir_now', lambda t: t['movie_dir_now'], lambda t: not t['movie_dir_now']),
      ('past_exp_mem', lambda t: t['past_exp_mem'], lambda t: not t['past_exp_mem']),
      ('hist 2+ vs <2', lambda t: t['hist'] == '2+', lambda t: t['hist'] != '2+'),
      ('n_cur 0 vs >0', lambda t: t['n_cur'] == '0', lambda t: t['n_cur'] != '0')]
for n in names[1:]:
    print(f'== slices: {n} - {ref}  [yes | no | interaction(yes-no)]')
    for nm, fy, fn in SL:
        dy = mean(n, users_, fy)[0] - mean(ref, users_, fy)[0]; dn = mean(n, users_, fn)[0] - mean(ref, users_, fn)[0]
        print(f'  {nm:15s} yes {dy:+.4f} (n={mean(ref, users_, fy)[1]}, CI {paired(ref, n, fy, 1000)}) | no {dn:+.4f} (n={mean(ref, users_, fn)[1]}) | inter {dy - dn:+.4f}')
