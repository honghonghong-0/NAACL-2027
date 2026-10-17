"""Section 3 table: how existing recommenders treat past memories (ReDial-MS validation, Pool A).
Per turn with an in-pool target (current-dialogue movies excluded, same protocol as eval_poola.py):
  Seen@10      = share of turns whose top-10 contains >=1 movie the user reported as seen in a past session
  Disliked@10  = share of turns whose top-10 contains >=1 movie the user saw and disliked in a past session
                 (survey seen == 1 and liked == 0)
  NDCG@10      = standard per-target NDCG@10
Also: share of turns whose target was already seen in a past session (data statistic).
95% CIs by user-cluster bootstrap. Usage: exposure_table.py name=path.npz ... (pop and cooc are internal)."""
import json, sys, math, random, collections
import numpy as np
sys.path.insert(0, './scripts')
import baselines_simple as bs
from baselines_simple import users, movies, IDX, P, co_vec
from baselines_simple import norm_max as _nm
class sp: norm = staticmethod(_nm)
ext = dict(a.split('=', 1) for a in sys.argv[1:])
PA = json.load(open(P / 'pool_a.json')); r2c = PA['redial2canon']; pool = PA['pool']; col = {c: i for i, c in enumerate(pool)}; M = len(pool)
r_idx = np.array([IDX[r] for r in r2c]); c_idx = np.array([col[c] for c in r2c.values()])
def to_pool(sc):
    out = np.full(M, -np.inf); np.maximum.at(out, c_idx, sc[r_idx]); return out
pop, co, allowed = bs.stats('valid')
sessions = {(int(u), s['conv_id']): s for u, ss in users.items() for s in ss}
turns = [json.loads(l) for l in open(P / 'eval_turns_valid.jsonl')]
E = {}
for n, pth in ext.items():
    z = np.load(pth); pos = {c: i for i, c in enumerate(z['item_ids'].tolist())}; sel = np.array([pos[c] for c in pool])
    E[n] = {tuple(k): z['scores'][i][sel].astype(np.float32) for i, k in enumerate(z['keys'].tolist())}
methods = ['pop', 'cooc'] + list(E)
R = {m: collections.defaultdict(list) for m in methods}; tseen = collections.defaultdict(list)
for t in turns:
    s = sessions[(t['user'], t['conv_id'])]; u = str(t['user']); cur = list(dict.fromkeys(movies(s['messages'][:t['turn_index']])))
    G = sorted({col[r2c[g]] for g in t['targets'] if g in r2c})
    if not G: continue
    curc = {col[r2c[x]] for x in cur if x in r2c}
    past = [s2 for s2 in users[u] if s2['split'] in allowed and s2['conv_id'] != t['conv_id']]
    seen = {col[r2c[x]] for s2 in past for x, v in s2['seeker_labels'].items() if str(v.get('seen')) == '1' and x in r2c}
    dis = {col[r2c[x]] for s2 in past for x, v in s2['seeker_labels'].items() if str(v.get('seen')) == '1' and str(v.get('liked')) == '0' and x in r2c}
    tseen[u].append(float(any(g in seen for g in G)))
    for m in methods:
        if m == 'pop': sc = to_pool(pop.astype(float))
        elif m == 'cooc': sc = to_pool(sp.norm(co_vec(co, cur)) + 1e-3 * pop / pop.max())
        else: sc = E[m][(t['user'], t['conv_id'], t['turn_index'])].copy()
        sc = sc + np.random.default_rng(0).random(M) * 1e-7; sc[list(curc)] = -np.inf
        order = np.argsort(-sc); rk = np.empty(M, int); rk[order] = np.arange(1, M + 1); top = set(order[:10].tolist())
        nd = float(np.mean([1 / math.log2(rk[g] + 1) if rk[g] <= 10 else 0. for g in G]))
        R[m][u].append((float(bool(top & seen)), float(bool(top & dis)), nd))
us = sorted(tseen)
def ci(get):
    rng = random.Random(0); b = []
    for _ in range(1000):
        s_ = rng.choices(us, k=len(us)); b.append(np.mean([x for u in s_ for x in get(u)]))
    return np.percentile(b, [2.5, 97.5])
out = {'target_already_seen': [float(np.mean([x for u in us for x in tseen[u]]))] + list(ci(lambda u: tseen[u]))}
print(f"turns {sum(len(v) for v in tseen.values())} | target already seen {out['target_already_seen'][0]*100:.1f}% [{out['target_already_seen'][1]*100:.1f}, {out['target_already_seen'][2]*100:.1f}]")
for m in methods:
    row = {}
    for k, name in enumerate(['seen@10', 'disliked@10', 'ndcg@10']):
        v = float(np.mean([x[k] for u in us for x in R[m][u]])); lo, hi = ci(lambda u, k=k: [x[k] for x in R[m][u]]); row[name] = [v, lo, hi]
    out[m] = row
    print(f"{m:12s} Seen@10 {row['seen@10'][0]*100:5.1f}% [{row['seen@10'][1]*100:.1f},{row['seen@10'][2]*100:.1f}]  Disliked@10 {row['disliked@10'][0]*100:5.1f}% [{row['disliked@10'][1]*100:.1f},{row['disliked@10'][2]*100:.1f}]  NDCG@10 {row['ndcg@10'][0]:.4f}")
json.dump(out, open('./analysis/exposure_table_valid.json', 'w'), indent=1, default=float)
