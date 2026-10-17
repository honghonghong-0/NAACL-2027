"""D1 + D2 (minimal learned version) of the user-specific affordance.
Memory encoder (replaces the count-based Delta_u):
  h_m   = [direction * v_m (experience profile, A) , kind one-hot (3), log #evidence]
          v_m: movie -> centered population experience of the movie; experience entry -> cos on its concept;
               attribute entry -> cos * population experience profile of its property concept
  e_u   = g(n) * sum_m alpha_m MLP(h_m) / sum_m alpha_m ,  n = sum_m alpha_m ,  g(n) = n / (n + 5)   (zero evidence -> e_u = 0)
  Delta_u = U x e_u  (U: [A, P, r], r = 16, initialised at 0)  ->  rho_pers(i) = q^T Delta_u x_i
Loss = L_rank (softmax over candidates of  z(backbone) + g1 z(rho_pop) + g2 rho_pers)
     + 0.1 * L_map (hold out one past liked movie, predict its experience profile from e_u of the remaining memories)
Training turns: recommender turns in HISTORY sessions of the cohort (memory = earlier history sessions; co-mention
statistics exclude the turn's own conversation). Validation sessions are never trained on; the test split is never read.
Output: a cache (pop / past=learned rho_pers / cur=0) consumed unchanged by `residual_fuse.py fuse` (CACHE_FILE).
"""
import json, os, sys, re, math, random, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpha_train as A
import residual_fuse as RF
import affordance_read_v2 as AR
from baselines_simple import users, movies, IDX, N, CAT
torch.manual_seed(0); random.seed(0); np.random.seed(0); torch.set_num_threads(16)
NA, NP, R, KAPPA, LAM = len(AR.EXPS), len(AR.PROPS), 16, 5.0, 0.1
XN = torch.tensor(A.XN); YC = A.EC

def entry_vec(e):
    d = 1.0 if e['direction'] == '+' else -1.0; v = np.zeros(NA, np.float32); kind = [0, 0, 0]
    if e['movie_id'] and e['movie_id'] in IDX and A.COVER[IDX[e['movie_id']]]:
        v = YC[IDX[e['movie_id']]]; kind[0] = 1
    else:
        for typ, name, w in e['concepts']:
            if typ == 'experience' and name in AR.ai: v = v.copy(); v[AR.ai[name]] += w; kind[2] = 1
            if typ == 'property' and name in AR.pi and kind[1] == 0: v = v + w * AR.PROF_P[AR.pi[name]]; kind[1] = 1
        if not any(kind): return None
    return np.concatenate([d * v, kind, [math.log1p(len(e['evidence']))]]).astype(np.float32)

def build_turn(u, s, idx, i, cur, q, target_ids, own_conv):
    st = AR.STORE.get(u, {'entries': {}, 'affordance_index': {}})
    top = [AR.EXPS[k] for k in np.argsort(-q)[:AR.TOPA] if q[k] > 0]
    ids = {x for a in top for x in st['affordance_index'].get(a, [])}
    H, W, liked = [], [], []
    for kk in ids:
        e = st['entries'][kk]
        evs = [ev for ev in e['evidence'] if ev['split'] == 'history' and ev['session_idx'] < idx]
        if not evs or (e['movie_id'] and e['movie_id'] in cur): continue
        h = entry_vec(e)
        if h is None: continue
        a = 1.0
        if e['movie_id'] and e['movie_id'] in IDX:
            d = 1.0 if e['direction'] == '+' else -1.0
            a = RF.alpha(A.feats(e['movie_id'], d, max(1, idx - min(ev['session_idx'] for ev in evs)), cur, q))
            if d > 0: liked.append(len(H))
        H.append(h); W.append(a)
    return (np.array(H, np.float32) if H else np.zeros((0, NA + 4), np.float32)), np.array(W, np.float32), liked

class Model(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.mlp = torch.nn.Sequential(torch.nn.Linear(NA + 4, 64), torch.nn.ReLU(), torch.nn.Linear(64, R))
        s.U = torch.nn.Parameter(torch.zeros(R, NA, NP)); s.head = torch.nn.Linear(R, NA)
        s.g = torch.nn.Parameter(torch.tensor([0.3, 0.3]))
    def enc(s, H, Wt):
        if len(H) == 0: return torch.zeros(R)
        n = Wt.sum(); return (n / (n + KAPPA)) * (Wt[:, None] * s.mlp(H)).sum(0) / n
    def pers(s, e, q, cand):
        M = torch.einsum('r,rap->ap', e, s.U)             # Delta_u [A, P]
        return XN[cand] @ (M.T @ q)

def zs(v):
    sd = v.std(); return (v - v.mean()) / sd if sd > 0 else v * 0

def make_examples(split):
    pop, co, _ = A.pop, A.co, None
    ex = []
    for u in sorted(AR.STORE):
        ss = users[u]
        sess = [(k, s) for k, s in enumerate(ss) if (s['split'] == 'history' and split == 'train') or (s['split'] == 'valid' and split == 'valid')]
        for idx, s in sess:
            if split == 'train' and not any(ev['session_idx'] == idx for e in AR.STORE[u]['entries'].values() for ev in e['evidence']): continue
            if split == 'train': tl = A.turns_of(s)
            else: tl = [(t['turn_index'], t['targets']) for t in RF.turns if str(t['user']) == u and t['conv_id'] == s['conv_id']]
            own = set(movies(s['messages'])) if split == 'train' else None
            for i, tg in tl:
                cur, q_imp = A.context(u, s, idx, i)
                q = AR.unit(q_imp)          # q_explicit added below through the store
                st = AR.STORE[u]; q_exp = np.zeros(NA, np.float32)
                for e in st['entries'].values():
                    if e['kind'] in ('experience', 'property') and any(ev['conv_id'] == s['conv_id'] and ev['turn'] - 1 < i for ev in e['evidence']):
                        d = 1.0 if e['direction'] == '+' else -1.0
                        for typ, name, w in e['concepts']:
                            if typ in ('experience', 'experience_via_property') and name in AR.ai: q_exp[AR.ai[name]] += d * w
                q = q + AR.unit(q_exp)
                if not np.abs(q).sum(): continue
                cv = np.zeros(N, np.float32)
                for a_ in cur:
                    for b_, c_ in co[a_].items(): cv[IDX[b_]] += c_
                    if own and a_ in own:
                        for b_ in own:
                            if b_ != a_: cv[IDX[b_]] -= 1
                cv = cv + 1e-3 * pop / pop.max()
                mask = np.ones(N, bool); mask[[IDX[x] for x in cur]] = False
                cand = np.nonzero(mask)[0]; cand = cand[np.argsort(-cv[cand])[:100]]
                tgt = [IDX[g] for g in tg if g in IDX and mask[IDX[g]]]
                if split == 'train':
                    if not tgt: continue
                    cand = np.unique(np.concatenate([cand, tgt]))
                H, Wt, liked = build_turn(u, s, idx, i, cur, q, tgt, s['conv_id'])
                ex.append(dict(u=u, key=(int(u), s['conv_id'], i), cand=cand, tgt=np.isin(cand, tgt), zb=zs(cv[cand]),
                               zp=zs(A.EC[cand] @ q), q=q.astype(np.float32), H=H, W=Wt, liked=liked))
    return ex

def main():
    tr = make_examples('train'); print('train examples', len(tr), 'with memory', sum(len(x['H']) > 0 for x in tr), flush=True)
    us = sorted({x['u'] for x in tr}); hold = set(random.Random(0).sample(us, max(1, len(us) // 10)))
    T = [x for x in tr if x['u'] not in hold]; Hd = [x for x in tr if x['u'] in hold]
    m = Model(); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    def run(x, train):
        H, Wt = torch.tensor(x['H']), torch.tensor(x['W']); q = torch.tensor(x['q']); cand = torch.tensor(x['cand'])
        e = m.enc(H, Wt); rp = m.pers(e, q, cand)
        s = torch.tensor(x['zb']) + m.g[0] * torch.tensor(x['zp'], dtype=torch.float32) + m.g[1] * rp
        y = torch.tensor(x['tgt']); loss = -torch.log_softmax(s, 0)[y].mean()
        if train and len(x['liked']) >= 2:                       # mapping auxiliary loss (leave one liked movie out)
            j = random.choice(x['liked']); keep = [k for k in range(len(H)) if k != j]
            e2 = m.enc(H[keep], Wt[keep]); loss = loss + LAM * ((m.head(e2) - H[j, :NA]) ** 2).mean()
        return loss, s
    def ndcg(X):
        v = []
        with torch.no_grad():
            for x in X:
                _, s = run(x, False); r = torch.argsort(torch.argsort(-s))
                for p in np.nonzero(x['tgt'])[0]: v.append(1 / math.log2(int(r[p]) + 2) if r[p] < 10 else 0)
        return float(np.mean(v))
    best = (-1, None)
    for ep in range(6):
        random.shuffle(T)
        for x in T:
            loss, _ = run(x, True); opt.zero_grad(); loss.backward(); opt.step()
        h = ndcg(Hd); print('epoch', ep, 'held-out users NDCG@10(cand)', round(h, 4), 'g', m.g.data.numpy().round(3), flush=True)
        if h > best[0]: best = (h, {k: v.clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1])
    va = {x['key']: x for x in make_examples('valid')}
    keys, RP, RU, RC = [], [], [], []
    with torch.no_grad():
        for t in RF.turns:
            k = (t['user'], t['conv_id'], t['turn_index']); ru = np.zeros(N, np.float32); rp = np.zeros(N, np.float32)
            if k in va:
                x = va[k]; e = m.enc(torch.tensor(x['H']), torch.tensor(x['W']))
                M = torch.einsum('r,rap->ap', e, m.U); ru = (XN @ (M.T @ torch.tensor(x['q']))).numpy(); rp = A.EC @ x['q']
            keys.append(k); RP.append(RF.to_pool(rp)); RU.append(RF.to_pool(ru)); RC.append(RF.to_pool(np.zeros(N, np.float32)))
    np.savez_compressed(RF.CACHE, keys=np.array(keys), pop=np.array(RP, np.float32), past=np.array(RU, np.float32), cur=np.array(RC, np.float32))
    torch.save(m.state_dict(), RF.CACHE.with_suffix('.pt')); print('cached', RF.CACHE.name)

if __name__ == '__main__':
    main()
