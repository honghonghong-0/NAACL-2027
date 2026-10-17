"""Affordance read v2 (A1 + A2 + B1 as one patch). Writes the same cache format as residual_fuse.py chain
(keys / pop / past / cur), so `residual_fuse.py fuse` is reused unchanged (set CACHE_FILE).
B1  q = unit(q_implicit) + unit(q_explicit)
      q_implicit: experiences of movies the user liked / disliked earlier in the current session (as before)
      q_explicit: experience and property statements earlier in the current session, mapped to concepts
                  (experience: dir * cos;  property: dir * cos * population profile of that property)
A1  retrieval: past memories indexed (memory_store_v6 affordance_index) under the top-5 experiences of q
A2  user-specific affordance Delta_u [A,P] uses every retrieved memory kind:
      movie    : alpha * dir * (yhat_m (x) x_m)           minus its population expectation (as before)
      property : dir * cos * (population experience profile of p) placed in column p
      experience: dir * cos * (population property profile of a) placed in row a
    Delta_u = (sum - npos * Sbar) / (n + kappa), kappa = 5. Zero evidence -> exactly zero.
The current-session prefix gives a second Delta (cur) with the same rule (no alpha).
"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alpha_train as A
import residual_fuse as RF
from baselines_simple import users, IDX, N, P

STORE = json.load(open(P / 'memory_store_v6.json'))
VOC = json.load(open(P / 'item_chain' / 'item_chain_v1_vocab.json')); PROPS, EXPS = VOC['properties'], VOC['experiences']
pi, ai = {p: k for k, p in enumerate(PROPS)}, {a: k for k, a in enumerate(EXPS)}
W = np.load(P / 'item_chain' / 'item_chain_v1.npz')['W']            # [P, A] population property -> experience
Wp = np.clip(W, 0, None)
PROF_P = Wp / (Wp.max(1, keepdims=True) + 1e-9)                    # property p -> experience profile [P, A], max 1
PROF_A = (Wp / (Wp.max(0, keepdims=True) + 1e-9)).T                # experience a -> property profile [A, P], max 1
KAPPA = 5.0; TOPA = 5

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 0 else v

def entry_evidence(e, pred):
    return [ev for ev in e['evidence'] if pred(ev)]

def add(e, d, S, cnt, alpha_fn=None, cur=None, q=None):
    """accumulate one memory entry into S [A,P]; returns weight added"""
    if e['movie_id'] and e['movie_id'] in IDX and A.COVER[IDX[e['movie_id']]]:
        k = IDX[e['movie_id']]; a = alpha_fn(e['movie_id'], d) if alpha_fn else 1.0
        S += a * d * np.outer(A.YN[k], A.XN[k]); cnt['n'] += a; cnt['npos'] += a * d; cnt['movie'] += 1; return
    for typ, name, w in e['concepts']:
        if w <= 0: continue
        if typ == 'property' and name in pi:
            S[:, pi[name]] += d * w * PROF_P[pi[name]]; cnt['n'] += w; cnt['prop'] += 1; break   # top-1 property concept
    for typ, name, w in e['concepts']:
        if typ == 'experience' and name in ai and w > 0:
            S[ai[name], :] += d * w * PROF_A[ai[name]]; cnt['n'] += w; cnt['exp'] += 1

def main():
    keys, RP, RU, RC, info = [], [], [], [], collections.Counter()
    for t in RF.turns:
        u = str(t['user']); ss = users[u]; idx = next(k for k, s in enumerate(ss) if s['conv_id'] == t['conv_id']); s = ss[idx]
        i = t['turn_index']; cur, q_imp = A.context(u, s, idx, i)
        st = STORE.get(u, {'entries': {}, 'affordance_index': {}})
        is_cur = lambda ev: ev['conv_id'] == t['conv_id'] and ev['turn'] - 1 < i
        is_past = lambda ev: ev['split'] == 'history' and ev['session_idx'] < idx
        # ---- B1: explicit experience / property requests in the current-session prefix
        q_exp = np.zeros(len(EXPS), np.float32)
        for e in st['entries'].values():
            if e['kind'] not in ('experience', 'property') or not entry_evidence(e, is_cur): continue
            d = 1.0 if e['direction'] == '+' else -1.0
            for typ, name, w in e['concepts']:
                if typ == 'experience' and name in ai: q_exp[ai[name]] += d * w
                if typ == 'experience_via_property' and name in ai: q_exp[ai[name]] += d * w
        q = unit(q_imp) + unit(q_exp)
        info['turns'] += 1; info['q_imp'] += bool(np.abs(q_imp).sum()); info['q_exp'] += bool(np.abs(q_exp).sum())
        rp = np.zeros(N, np.float32); ru = np.zeros(N, np.float32); rc = np.zeros(N, np.float32)
        if np.abs(q).sum():
            rp = A.EC @ q
            # ---- A1: retrieve past memories through the affordance index
            top = [EXPS[k] for k in np.argsort(-q)[:TOPA] if q[k] > 0]
            ids = {x for a in top for x in st['affordance_index'].get(a, [])}
            def alpha_fn(m, d):
                return RF.alpha(A.feats(m, d, max(1, idx - min(ev['session_idx'] for ev in st['entries'][kk]['evidence'])), cur, q_imp if np.abs(q_imp).sum() else q))
            S = np.zeros_like(A.SBAR); cnt = collections.Counter()
            for kk in ids:
                e = st['entries'][kk]
                if not entry_evidence(e, is_past) or (e['movie_id'] and e['movie_id'] in cur): continue
                d = 1.0 if e['direction'] == '+' else -1.0
                add(e, d, S, cnt, alpha_fn=lambda m, dd, kk=kk: RF.alpha(A.feats(m, dd, max(1, idx - min(ev['session_idx'] for ev in st['entries'][kk]['evidence'])), cur, q)))
            if cnt['n'] > 0:
                ru = A.XN @ (((S - cnt['npos'] * A.SBAR) / (cnt['n'] + KAPPA)).T @ q); info['past'] += 1
                info['past_prop'] += cnt['prop'] > 0; info['past_exp'] += cnt['exp'] > 0
            # ---- current-session prefix (same rule, no alpha)
            S = np.zeros_like(A.SBAR); cnt = collections.Counter()
            for e in st['entries'].values():
                if not entry_evidence(e, is_cur): continue
                add(e, 1.0 if e['direction'] == '+' else -1.0, S, cnt)
            if cnt['n'] > 0:
                rc = A.XN @ (((S - cnt['npos'] * A.SBAR) / (cnt['n'] + KAPPA)).T @ q); info['cur'] += 1
        keys.append((t['user'], t['conv_id'], i)); RP.append(RF.to_pool(rp)); RU.append(RF.to_pool(ru)); RC.append(RF.to_pool(rc))
    np.savez_compressed(RF.CACHE, keys=np.array(keys), pop=np.array(RP, np.float32), past=np.array(RU, np.float32), cur=np.array(RC, np.float32))
    print('cached', RF.CACHE.name, dict(info))

if __name__ == '__main__':
    main()
