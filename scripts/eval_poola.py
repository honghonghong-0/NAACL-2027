"""Pool-A evaluation (main protocol from 2026-10-12): canonical items = CRSLab/UniCRS movie entity ids in pool A (6,002).
Internal methods score the 6,924 ReDial ids and are converted with max over ReDial ids sharing a canonical id; external
methods (npz from *_infer.py) are already canonical. Same filters for everyone: candidates = pool A minus canonical ids of
current-dialogue movies; secondary unseen protocol also removes past-seen; targets = canonical ids of in-pool targets;
turns without an in-pool target are excluded and counted. Metrics per target (CRSLab convention), user-cluster bootstrap.
Usage: eval_poola.py <split> [--external-only] [--output path.json] [name=path.npz ...]"""
import json, sys, math, random, collections, numpy as np
sys.path.insert(0, './scripts')
import baselines_simple as bs
from baselines_simple import users, movies, IDX, N, P, B, CAT, co_vec
from baselines_simple import norm_max as _nm
class sp: norm = staticmethod(_nm)
split = sys.argv[1]
args = sys.argv[2:]
external_only = '--external-only' in args
args = [a for a in args if a != '--external-only']
output = B / f'results_poola_{split}.json'
if '--output' in args:
    i = args.index('--output')
    output = args[i + 1]
    del args[i:i + 2]
ext = dict(a.split('=', 1) for a in args)
PA = json.load(open(P / 'pool_a.json')); r2c = PA['redial2canon']; pool = PA['pool']; items = PA['item_ids']
col = {c: i for i, c in enumerate(pool)}; M = len(pool)
r_idx = np.array([IDX[r] for r in r2c]); c_idx = np.array([col[c] for c in r2c.values()])
def to_pool(sc):  # 6,924 -> pool A (max over duplicates)
    out = np.full(M, -np.inf); np.maximum.at(out, c_idx, sc[r_idx]); return out
pop, co, allowed = bs.stats(split)
sessions = {(int(u), s['conv_id']): s for u, ss in users.items() for s in ss}
def internal(method, t, u, cur):
    if method == 'pop': return pop.astype(float)
    if method == 'cooc': return sp.norm(co_vec(co, cur)) + 1e-3 * pop / pop.max()
E = {}
for name, path in ext.items():
    z = np.load(path); pos_ = {c: i for i, c in enumerate(z['item_ids'].tolist())}; sel = np.array([pos_[c] for c in pool])
    E[name] = {tuple(k): z['scores'][i][sel].astype(np.float32) for i, k in enumerate(z['keys'].tolist())}
KS = (10, 50)
def m(r): return {**{f'HR@{k}': float(r <= k) for k in KS}, **{f'MRR@{k}': (1 / r if r <= k else 0.) for k in KS}, **{f'NDCG@{k}': (1 / math.log2(r + 1) if r <= k else 0.) for k in KS}}
def rank(sc, excl):
    sc = sc + np.random.default_rng(0).random(M) * 1e-7; sc[list(excl)] = -np.inf
    order = np.argsort(-sc); r = np.empty(M, int); r[order] = np.arange(1, M + 1); return r, order
def summ(pu):
    flat = [x for v in pu.values() for x in v]; out = {k: round(float(np.mean([x[k] for x in flat])), 4) for k in flat[0]}
    ul = list(pu); rng = random.Random(0); b = [np.mean([x['NDCG@10'] for u in rng.choices(ul, k=len(ul)) for x in pu[u]]) for _ in range(1000)]
    out['NDCG@10_CI'] = [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]; out['n_targets'] = len(flat); return out
turns = [json.loads(l) for l in open(P / f'eval_turns_{split}.jsonl')]; res = {}; cnt = collections.Counter()
methods = ([] if external_only else ['pop', 'cooc']) + list(E)
for method in methods:
    std, uns, ex = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for t in turns:
        s = sessions[(t['user'], t['conv_id'])]; u = str(t['user']); cur = list(dict.fromkeys(movies(s['messages'][:t['turn_index']])))
        G = sorted({col[r2c[g]] for g in t['targets'] if g in r2c})
        if method == methods[0]: cnt['turns'] += 1; cnt['turns_no_inpool_target'] += not G
        if not G: continue
        curc = {col[r2c[x]] for x in cur if x in r2c}
        past = [s2 for s2 in users[u] if s2['split'] in allowed and s2['conv_id'] != t['conv_id']]
        seen = {col[r2c[x]] for s2 in past for x, v in s2['seeker_labels'].items() if str(v.get('seen')) == '1' and x in r2c}
        dis = {col[r2c[x]] for s2 in past for x, v in s2['seeker_labels'].items() if str(v.get('liked')) == '0' and x in r2c}
        sc = E[method][(t['user'], t['conv_id'], t['turn_index'])].copy() if method in E else to_pool(internal(method, t, u, cur))
        r, order = rank(sc.copy(), curc)
        for g in G: std[t['user']].append(m(r[g]))
        top = order[:10]; ex['SeenExposure@10'].append(np.isin(top, list(seen)).mean() if seen else 0.); ex['PastDislikedExposure@10'].append(np.isin(top, list(dis)).mean() if dis else 0.)
        Gu = [g for g in G if g not in seen]
        if Gu:
            ru, _ = rank(sc.copy(), curc | seen)
            for g in Gu: uns[t['user']].append(m(ru[g]))
    res[method] = {'standard': summ(std), 'unseen': summ(uns), 'exposure': {k: round(float(np.mean(v)), 4) for k, v in ex.items()}}
    print(method, json.dumps(res[method]), flush=True)
print('counts', dict(cnt)); json.dump({'counts': dict(cnt), **res}, open(output, 'w'), indent=1)
