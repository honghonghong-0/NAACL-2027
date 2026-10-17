"""S1-c / S2: item-side chain.
x[i,p]  = Tag Genome (TagDL) relevance of property tag p for movie i      (property evidence score, not probability)
y[i,a]  = relevance of experience tag a for movie i
Population realization map: y_a ~ sigmoid(b_a + x W_a), fit with L2 logistic loss on ALL genome movies
(external item data, no ReDial split info). Held-out 20% genome movies for fit quality.
Also prints value distributions / dynamic range before anything is scored (S2).
"""
import json, csv, collections
import numpy as np, torch

GEN = './data/tag_genome_2021'
OUT = './processed/item_chain'

# manual overrides (confirmed by the authors): judgement words, not effects on viewer
TO_EVAL = {'awesome', 'brilliant', 'cool', 'interesting', 'very interesting', 'better than expected', 'waste of time',
           'pointless', 'stunning', 'breathtaking', 'beautiful', 'stupid as hell', 'guilty pleasure',
           "so bad it's good", 'sad but good', 'suprisingly clever', 'boring!', 'cute!', 'educational'}
TO_EXP = {'atmospheric', 'cerebral', 'dreamlike', 'visceral'}

layers = json.load(open(f'{OUT}/tag_layers.json'))
for t in TO_EVAL:
    if t in layers: layers[t] = 'evaluation'
for t in TO_EXP:
    if t in layers: layers[t] = 'experience'
json.dump({'to_evaluation': sorted(TO_EVAL), 'to_experience': sorted(TO_EXP)}, open(f'{OUT}/tag_layer_overrides.json', 'w'), indent=1)

P = sorted(t for t, v in layers.items() if v == 'property')
A = sorted(t for t, v in layers.items() if v == 'experience')
pi, ai = {t: k for k, t in enumerate(P)}, {t: k for k, t in enumerate(A)}

movies = {}
rows = collections.defaultdict(dict)
with open(f'{GEN}/scores/tagdl.csv') as f:
    r = csv.reader(f); next(r)
    for tag, iid, s in r:
        rows[int(iid)][tag.lower()] = float(s)
items = sorted(rows)
X = np.zeros((len(items), len(P)), np.float32)
Y = np.zeros((len(items), len(A)), np.float32)
for n, iid in enumerate(items):
    for tag, s in rows[iid].items():
        if tag in pi: X[n, pi[tag]] = s
        elif tag in ai: Y[n, ai[tag]] = s
print('movies', len(items), 'properties', len(P), 'experiences', len(A))

# ---- S2: distribution / dynamic range
def dist(name, M):
    q = np.quantile(M, [0, .5, .9, .99, 1])
    per_item_max = M.max(1)
    print(f'{name}: quantiles(0,.5,.9,.99,1)={np.round(q,3)} | share>0.5={np.mean(M>0.5):.4f} '
          f'| per-movie #>0.5 median={np.median((M>0.5).sum(1)):.0f} | per-movie max median={np.median(per_item_max):.3f}')
dist('x(property)', X); dist('y(experience)', Y)

# ---- population realization map
rng = np.random.default_rng(0)
perm = rng.permutation(len(items)); cut = int(.8 * len(items))
tr, te = perm[:cut], perm[cut:]
Xt, Yt = torch.tensor(X), torch.tensor(Y)
mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-6
Z = (Xt - mu) / sd

def fit(lam, idx, epochs=400):
    W = torch.zeros(len(P), len(A), requires_grad=True)
    b = torch.zeros(len(A), requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=epochs, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        logit = Z[idx] @ W + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, Yt[idx]) + lam * (W ** 2).sum() / len(idx)
        loss.backward(); return loss
    opt.step(closure)
    return W.detach(), b.detach()

def r2(W, b, idx):
    pred = torch.sigmoid(Z[idx] @ W + b); y = Yt[idx]
    ss = ((y - pred) ** 2).sum(0); st = ((y - y.mean(0)) ** 2).sum(0)
    return (1 - ss / st).numpy()

# choose lambda on an inner split of train
inner = tr[: int(.8 * len(tr))]; val = tr[int(.8 * len(tr)):]
best = None
for lam in [1, 10, 100, 1000]:
    W, b = fit(lam, inner)
    s = float(np.median(r2(W, b, val)))
    print('lambda', lam, 'val median R2', round(s, 4))
    if best is None or s > best[1]: best = (lam, s)
W, b = fit(best[0], tr)
R2 = r2(W, b, te)
print('chosen lambda', best[0], '| held-out movies R2 median', round(float(np.median(R2)), 4),
      'min', round(float(R2.min()), 4), 'max', round(float(R2.max()), 4))
order = np.argsort(R2)
print('worst experiences', [(A[k], round(float(R2[k]), 3)) for k in order[:8]])
print('best experiences', [(A[k], round(float(R2[k]), 3)) for k in order[-8:]])
for a in ['thought-provoking', 'scary', 'funny', 'feel-good', 'tense', 'heartwarming']:
    if a in ai:
        k = ai[a]; top = torch.topk(W[:, k], 6).indices.tolist()
        print(f'  {a} <- ' + ', '.join(f'{P[j]}({W[j,k]:.2f})' for j in top))

np.savez_compressed(f'{OUT}/item_chain_v1.npz', items=np.array(items), X=X, Y=Y,
                    W=W.numpy(), b=b.numpy(), mu=mu.numpy(), sd=sd.numpy(), R2=R2)
json.dump({'properties': P, 'experiences': A, 'lambda': best[0]}, open(f'{OUT}/item_chain_v1_vocab.json', 'w'))
