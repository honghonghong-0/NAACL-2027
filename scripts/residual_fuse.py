"""M-v5 end-to-end residual integration (valid only).
  s = z_C(s_backbone) + g1 * z_C(rho_pop) + g2 * z_C(rho_pers)
  rho_pop  = q^T Wbar x_i                      (population realization through item->property->experience)
  rho_pers = q^T Delta_u x_i                   (this user's utterances: past sessions [alpha-weighted] + current-session prefix)
  z_C = query-wise z-score over the backbone's candidate set (finite scores, current movies excluded); uncovered movies get 0.
gamma chosen by CROSS-FITTING over a 2-fold user split (choose on one half, apply to the other) -> one honest score file.
Step 1 'chain': cache rho arrays per valid turn (backbone-independent).
Step 2 'fuse <name> <backbone.npz|cooc>': write fused npz for {nochain, pop, pop+pers} and paired-compare vs backbone.
"""
import json, sys, math, random, collections
import numpy as np
sys.path.insert(0, './scripts')
import alpha_train as A
from baselines_simple import users, movies, IDX, N, P, co_vec
from baselines_simple import norm_max as _nm
class sp: norm = staticmethod(_nm)

PA = json.load(open(P / 'pool_a.json')); r2c = PA['redial2canon']; pool = PA['pool']; col = {c: i for i, c in enumerate(pool)}; M = len(pool)
r_idx = np.array([IDX[r] for r in r2c]); c_idx = np.array([col[c] for c in r2c.values()])
def to_pool(sc, fill=-np.inf):
    out = np.full(M, -np.inf); np.maximum.at(out, c_idx, sc[r_idx]); return out
COVP = to_pool(A.COVER.astype(float)) > 0
turns = [json.loads(l) for l in open(P / 'eval_turns_valid.jsonl')]
sessions = {(int(u), s['conv_id']): s for u, ss in users.items() for s in ss}
import os
CACHE = P / 'item_chain' / os.environ.get('CACHE_FILE', 'rho_valid_v1.npz')
KAPPA = 5.0
try:
    AL = json.load(open(P / 'item_chain' / os.environ.get('ALPHA_FILE', 'alpha_v1.json')))
except FileNotFoundError:
    AL = None

def alpha(f):
    if AL is None: return 1.0
    x = (np.array(f) - AL['mu']) / AL['sd']; return float(1 / (1 + np.exp(-(x @ AL['coef'] + AL['intercept']))))

def chain():
    keys, RP, RU, RC, info = [], [], [], [], collections.Counter()
    for t in turns:
        u = str(t['user']); ss = users[u]; idx = next(k for k, s in enumerate(ss) if s['conv_id'] == t['conv_id']); s = ss[idx]
        i = t['turn_index']; cur, q = A.context(u, s, idx, i)
        rp = np.zeros(N, np.float32); ru = np.zeros(N, np.float32); rc = np.zeros(N, np.float32)
        if np.abs(q).sum():
            info['q'] += 1
            rp = A.EC @ q
            # past sessions, alpha-weighted
            S = np.zeros_like(A.SBAR); n = 0.; npos = 0.
            for m, (dm, ago) in A.mem_items(u, idx).items():
                if m in cur: continue
                a = alpha(A.feats(m, dm, ago, cur, q)); k = IDX[m]
                S += a * dm * np.outer(A.YN[k], A.XN[k]); n += a; npos += a * dm
            if n > 0: ru = A.XN @ (((S - npos * A.SBAR) / (n + KAPPA)).T @ q); info['past'] += 1
            # current-session prefix, explicit +-1
            d = {x: w for x, w in A.deltas(u, s, i, cur).items() if abs(w) == 1.0 and A.COVER[IDX[x]]}
            if d:
                ids = [IDX[x] for x in d]; dv = np.array(list(d.values()), np.float32)
                Sc = (A.YN[ids] * dv[:, None]).T @ A.XN[ids]
                rc = A.XN @ (((Sc - dv.sum() * A.SBAR) / (len(d) + KAPPA)).T @ q); info['cur'] += 1
        keys.append((t['user'], t['conv_id'], i)); RP.append(to_pool(rp)); RU.append(to_pool(ru)); RC.append(to_pool(rc))
    np.savez_compressed(CACHE, keys=np.array(keys), pop=np.array(RP, np.float32), past=np.array(RU, np.float32), cur=np.array(RC, np.float32))
    print('cached', dict(info), 'alpha', 'on' if AL else 'off')

def zc(v, mask):
    out = np.zeros(M, np.float32); m = mask & np.isfinite(v)
    if m.sum() > 1 and v[m].std() > 0: out[m] = (v[m] - v[m].mean()) / v[m].std()
    return out

def fuse(name, bpath):
    C = np.load(CACHE); ck = {tuple(k): j for j, k in enumerate(C['keys'].tolist())}
    if bpath == 'cooc':
        import baselines_simple as bs
        pop, co, _ = bs.stats('valid'); B = {}
        for t in turns:
            s = sessions[(t['user'], t['conv_id'])]; cur = list(dict.fromkeys(movies(s['messages'][:t['turn_index']])))
            B[(t['user'], t['conv_id'], t['turn_index'])] = to_pool(sp.norm(co_vec(co, cur)) + 1e-3 * pop / pop.max())
    else:
        z = np.load(bpath); pos = {c: i for i, c in enumerate(z['item_ids'].tolist())}; sel = np.array([pos[c] for c in pool])
        B = {tuple(k): z['scores'][i][sel].astype(np.float32) for i, k in enumerate(z['keys'].tolist())}
    rows = []
    for t in turns:
        k = (t['user'], t['conv_id'], t['turn_index']); s = sessions[(t['user'], t['conv_id'])]
        cur = list(dict.fromkeys(movies(s['messages'][:t['turn_index']])))
        excl = {col[r2c[c]] for c in cur if c in r2c}
        b = B[k].copy(); cand = np.isfinite(b) & (b > -1e8); cand[list(excl)] = False
        j = ck[k]; cm = cand & COVP
        rows.append((k, t, excl, cand, zc(b, cand), zc(C['pop'][j], cm), zc(C['past'][j] + C['cur'][j], cm)))
    def ndcg_user(g1, g2):
        pu = collections.defaultdict(list)
        for k, t, excl, cand, zb, zp, zu in rows:
            G = sorted({col[r2c[g]] for g in t['targets'] if g in r2c})
            if not G: continue
            sc = np.where(cand, zb + g1 * zp + g2 * zu, -np.inf) + np.random.default_rng(0).random(M) * 1e-7
            sc[list(excl)] = -np.inf; order = np.argsort(-sc); rk = np.empty(M, int); rk[order] = np.arange(1, M + 1)
            for g in G: pu[t['user']].append(1 / math.log2(rk[g] + 1) if rk[g] <= 10 else 0.)
        return pu
    GR = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    grids = {'nochain': [(0, 0)], 'pop': [(g, 0) for g in GR], 'pop+pers': [(g1, g2) for g1 in GR for g2 in GR]}
    res = {cfg: ndcg_user(*cfg) for cfg in set(c for v in grids.values() for c in v)}
    us = sorted(res[(0, 0)]); rr = random.Random(0); rr.shuffle(us); halves = [set(us[::2]), set(us[1::2])]
    def mean(pu, S_): return float(np.mean([x for u in S_ for x in pu[u]]))
    final = {}
    for cond, grid in grids.items():
        pu = {}; picks = []
        for h in (0, 1):
            pick = max(grid, key=lambda c: mean(res[c], halves[h])); picks.append(pick)
            for u in halves[1 - h]: pu[u] = res[pick][u]
        final[cond] = (pu, picks)
    base = final['nochain'][0]
    print(f'== backbone {name} (cross-fitted gamma; paired user bootstrap vs backbone)')
    for cond, (pu, picks) in final.items():
        m = mean(pu, us); line = f'  {cond:9s} NDCG@10 {m:.4f}  gammas(per half) {picks}'
        if cond != 'nochain':
            rng = random.Random(1); d = []
            for _ in range(2000):
                bs_ = rng.choices(us, k=len(us)); d.append(np.mean([x for u in bs_ for x in pu[u]]) - np.mean([x for u in bs_ for x in base[u]]))
            line += f'  Delta {m - mean(base, us):+.4f} CI95 {np.round(np.percentile(d, [2.5, 97.5]), 4)}'
        print(line)
    # pop+pers vs pop
    rng = random.Random(2); d = []
    for _ in range(2000):
        bs_ = rng.choices(us, k=len(us)); d.append(np.mean([x for u in bs_ for x in final['pop+pers'][0][u]]) - np.mean([x for u in bs_ for x in final['pop'][0][u]]))
    print(f'  pop+pers vs pop  Delta {mean(final["pop+pers"][0], us) - mean(final["pop"][0], us):+.4f} CI95 {np.round(np.percentile(d, [2.5, 97.5]), 4)}')

if __name__ == '__main__':
    if sys.argv[1] == 'chain': chain()
    else: fuse(sys.argv[2], sys.argv[3])
