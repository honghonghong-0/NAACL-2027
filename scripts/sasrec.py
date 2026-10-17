"""SASRec (Kang & McAuley, ICDM 2018) on ReDial-MS. Sequence = a user's movie mentions in order across sessions
(seeker+recommender lines). Training sequences use only sessions allowed for the evaluated split (valid: 'history';
test: 'history'+'valid'). At an eval turn, input = past-session movies + current-dialogue movies before the turn.
Full-softmax cross-entropy over the catalog. Ranking/metrics identical to baselines_simple (same filter, per-target)."""
import json, sys, math, random, collections, numpy as np, torch, torch.nn as nn
sys.path.insert(0, './scripts')
from baselines_simple import users, movies, IDX, N, P, B
L = 50; D = 64; DEV = 'cuda:0'
class SASRec(nn.Module):
    def __init__(s, n, d=D, layers=2, heads=1, drop=0.2):
        super().__init__(); s.emb = nn.Embedding(n + 1, d, padding_idx=0); s.pos = nn.Embedding(L, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, d * 4, drop, batch_first=True, norm_first=True), layers)
        s.drop = nn.Dropout(drop); s.ln = nn.LayerNorm(d)
    def forward(s, x):
        h = s.drop(s.emb(x) * math.sqrt(D) + s.pos.weight[None, :x.size(1)])
        mask = torch.triu(torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device), 1)
        h = s.ln(s.enc(h, mask=mask, src_key_padding_mask=(x == 0)))
        return h @ s.emb.weight[1:].T          # logits over catalog (index i -> item i)
def pad(seq): seq = seq[-L:]; return [0] * (L - len(seq)) + [IDX[m] + 1 for m in seq]
def train_seqs(allowed):
    out = []
    for u, ss in users.items():
        seq = [x for s in ss if s['split'] in allowed for x in movies(s['messages'])]
        seq = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]
        for e in range(2, len(seq) + 1, 1): out.append(seq[max(0, e - L - 1):e])   # sliding windows
        if len(seq) >= 2: out.append(seq[-L - 1:])
    return out
def eval_turns(model, split, allowed, ks=(10, 50)):
    sessions = {(int(u), s['conv_id']): s for u, ss in users.items() for s in ss}
    turns = [json.loads(l) for l in open(P / f'eval_turns_{split}.jsonl')]; per_user = collections.defaultdict(list)
    model.eval()
    with torch.no_grad():
        for t in turns:
            s = sessions[(t['user'], t['conv_id'])]; cur = list(dict.fromkeys(movies(s['messages'][:t['turn_index']])))
            hist = [x for s2 in users[str(t['user'])] if s2['split'] in allowed and s2['conv_id'] != t['conv_id'] for x in movies(s2['messages'])]
            seq = hist + movies(s['messages'][:t['turn_index']])
            sc = model(torch.tensor([pad(seq) if seq else [0] * L], device=DEV))[0, -1].float().cpu().numpy() if seq else np.zeros(N)
            for x in cur: sc[IDX[x]] = -1e9
            order = np.argsort(-sc); rank = np.empty(N, int); rank[order] = np.arange(1, N + 1)
            for g in t['targets']:
                if g in IDX:
                    r = rank[IDX[g]]; per_user[t['user']].append({f'HR@{k}': float(r <= k) for k in ks} | {f'MRR@{k}': (1 / r if r <= k else 0.) for k in ks} | {f'NDCG@{k}': (1 / math.log2(r + 1) if r <= k else 0.) for k in ks})
    flat = [x for v in per_user.values() for x in v]; res = {k: float(np.mean([x[k] for x in flat])) for k in flat[0]}
    us = list(per_user); rng = random.Random(0); bs = [np.mean([x['NDCG@10'] for u in rng.choices(us, k=len(us)) for x in per_user[u]]) for _ in range(1000)]
    res['NDCG@10_CI'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; res['n_samples'] = len(flat); return res
def run(split, epochs=40, seed=0, lr=1e-3):
    torch.manual_seed(seed); random.seed(seed)
    allowed = {'history'} if split == 'valid' else {'history', 'valid'}
    seqs = train_seqs(allowed); X = torch.tensor([pad(s[:-1]) for s in seqs]); Y = torch.tensor([pad(s[1:]) for s in seqs])
    m = SASRec(N).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr); best, hist = None, []
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]; x, y = X[b].to(DEV), Y[b].to(DEV)
            loss = nn.functional.cross_entropy(m(x).reshape(-1, N), (y - 1).clamp(min=-100).masked_fill(y == 0, -100).reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 5 == 0:
            r = eval_turns(m, split, allowed); hist.append((ep + 1, r['NDCG@10'])); print(split, 'epoch', ep + 1, 'loss', round(loss.item(), 3), 'NDCG@10', round(r['NDCG@10'], 4), 'HR@10', round(r['HR@10'], 4), flush=True)
            if best is None or r['NDCG@10'] > best[1]['NDCG@10']: best = (ep + 1, r)
    return best, len(seqs)
if __name__ == '__main__':
    best, n = run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    print('BEST epoch', best[0], json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in best[1].items()}), 'train_seqs', n)
    json.dump({'best_epoch': best[0], **best[1]}, open(B / f'results_sasrec_{sys.argv[1]}.json', 'w'), indent=1)
