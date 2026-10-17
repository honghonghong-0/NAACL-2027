"""Our method inside PEPPER-MS (S2 sessions). Wraps the PEPPER ChatGPT CRS: its top-50 candidates are re-ranked by
  s = z(-rank) + G1 z(rho_pop) + G2 z(rho_user)
Conditions (same S1 memory for both; no extra API calls, no questions):
  allmem : every S1 memory, carryover alpha = 1, no affordance-indexed retrieval
  ours   : memories retrieved through the top-5 desired experiences, weighted by the carryover classifier (alpha_v6)
q (desired experience) = soft alignment of the seeker's utterances in the current session to experience / attribute
concepts (MiniLM cosine, top-3), attributes mapped through the population affordance. Delta_u as in Section 4.5
(closed form, kappa = 5). G1, G2 fixed before running (not tuned on PEPPER)."""
import json, os, numpy as np
from pathlib import Path
G1 = float(os.environ.get('OURS_G1', '0.3')); G2 = float(os.environ.get('OURS_G2', '0.3')); KAPPA = 5.0; TOPA = 5
R = Path('./pepper_ms/PEPPER'); D = R / 'research-notes/pepper-ms-odkg-100/ours'
IC = Path('./processed/item_chain')
EMB = 'sentence-transformers/all-MiniLM-L6-v2'
_S = {}
def _load():
    if _S: return _S
    it = np.load(D / 'items_imdb.npz'); voc = json.load(open(IC / 'item_chain_v1_vocab.json'))
    z = np.load(IC / 'item_chain_v1.npz'); W = np.clip(z['W'], 0, None)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(str(EMB), device='cpu')
    _S.update(pos={k: i for i, k in enumerate(it['imdb'].tolist())}, X=it['X'], Y=it['Y'], Yc=it['Y'] - it['ymean'],
              P=voc['properties'], A=voc['experiences'], PROF_P=W / (W.max(1, keepdims=True) + 1e-9),
              PROF_A=(W / (W.max(0, keepdims=True) + 1e-9)).T, enc=enc)
    _S['EP'] = enc.encode(_S['P'], normalize_embeddings=True); _S['EA'] = enc.encode(_S['A'], normalize_embeddings=True)
    Xg = np.clip(z['X'], 0, 1); yh = 1 / (1 + np.exp(-(((z['X'] - z['mu']) / z['sd']) @ z['W'] + z['b'])))
    _S['SBAR'] = (yh.T @ Xg) / len(Xg)
    _S['MEM'] = json.load(open(D / 'memory.json'))
    _S['AL'] = json.load(open(IC / 'alpha_v6.json')) if (IC / 'alpha_v6.json').exists() else None
    return _S

def _z(v):
    s = v.std(); return (v - v.mean()) / s if s > 0 else v * 0

class OursCRS:
    def __init__(self, base, user_id, condition, entities, title_ids, identity):
        self.base, self.cond = base, condition; S = _load(); self.S = S
        self.rev = {int(v): k for k, v in entities.items()}; self.title_ids, self.identity = title_ids, identity
        self.mem = []
        for e in S['MEM'].get(user_id, []):
            d = 1.0 if e['direction'] == '+' else -1.0; Em = None; prof = np.zeros(len(S['A']), np.float32)
            if e['imdb'] and e['imdb'] in S['pos']:
                k = S['pos'][e['imdb']]; Em = np.outer(S['Y'][k], S['X'][k]); prof = S['Yc'][k]; kind = 'movie'
            elif e['kind'] in ('property', 'attribute', 'experience'):
                v = S['enc'].encode([e['target']], normalize_embeddings=True)[0]
                if e['kind'] == 'experience':
                    s = S['EA'] @ v; top = np.argsort(-s)[:3]; Em = np.zeros_like(S['SBAR'])
                    for a in top: Em[a, :] += s[a] * S['PROF_A'][a]; prof[a] += s[a]
                else:
                    s = S['EP'] @ v; p = int(np.argmax(s)); Em = np.zeros_like(S['SBAR']); Em[:, p] += s[p] * S['PROF_P'][p]; prof = s[p] * S['PROF_P'][p]
                kind = e['kind']
            if Em is not None: self.mem.append((d, Em, prof, kind, len(e['evidence'])))
        self.log = []

    async def get_conv(self, conv_dict, candidate_list):
        return await self.base.get_conv(conv_dict, candidate_list)

    def _q(self, history):
        S = self.S; utt = [m['content'] for m in history if isinstance(m, dict) and m.get('role') == 'user' and m.get('content')]
        q = np.zeros(len(S['A']), np.float32)
        if not utt: return q
        V = S['enc'].encode(utt, normalize_embeddings=True)
        for v in V:
            sa = S['EA'] @ v
            for a in np.argsort(-sa)[:3]: q[a] += max(sa[a], 0)
            sp = S['EP'] @ v; p = int(np.argmax(sp)); q += max(sp[p], 0) * S['PROF_P'][p]
        n = np.linalg.norm(q); return q / n if n > 0 else q

    def _alpha(self, prof, d, q):
        AL = self.S['AL']
        if self.cond == 'allmem' or AL is None: return 1.0
        cosq = float(prof @ q / (np.linalg.norm(prof) * np.linalg.norm(q) + 1e-9)) * d
        f = np.array([cosq, 0.0, 1 / (1 + 1), d, np.log1p(np.linalg.norm(q)), cosq])   # no co-mention in PEPPER; one session ago
        x = (f - AL['mu']) / AL['sd']; return float(1 / (1 + np.exp(-(x @ AL['coef'] + AL['intercept']))))

    async def get_rec(self, history):
        rec, labels, cost = await self.base.get_rec(history)
        S = self.S; q = self._q(history)
        keys = []
        for eid in rec:
            name = self.rev.get(int(eid)) if str(eid).lstrip('-').isdigit() else None
            cid = self.title_ids.get(name) if name else None
            keys.append(self.identity.evaluation_key(str(cid)) if cid is not None else None)
        idx = [S['pos'].get(k) for k in keys]
        rpop = np.array([S['Yc'][i] @ q if i is not None else 0.0 for i in idx])
        top = set(np.argsort(-q)[:TOPA].tolist()) if np.abs(q).sum() else set()
        Ssum = np.zeros_like(S['SBAR']); n = 0.; npos = 0.
        for d, Em, prof, kind, nev in self.mem:
            if self.cond == 'ours' and not (set(np.argsort(-prof)[:TOPA].tolist()) & top): continue
            a = self._alpha(prof, d, q); Ssum += a * d * Em; n += a; npos += a * d
        Delta = (Ssum - npos * S['SBAR']) / (n + KAPPA) if n > 0 else np.zeros_like(S['SBAR'])
        ruser = np.array([S['X'][i] @ (Delta.T @ q) if i is not None else 0.0 for i in idx])
        base = -np.arange(len(rec), dtype=float)
        s = _z(base) + G1 * _z(rpop) + G2 * _z(ruser)
        order = np.argsort(-s, kind='stable')
        self.log.append({'n_mem_used': n, 'q_nonzero': bool(np.abs(q).sum()), 'covered': int(sum(i is not None for i in idx))})
        return [rec[k] for k in order], labels, cost
