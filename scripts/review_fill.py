"""D5: fill property evidence x for linked ReDial movies that have no Tag Genome scores, from IMDb reviews
(Tag Genome raw/reviews.json). For each target movie: share of its reviews that mention each property tag
(whole-word match), then mapped onto the TagDL scale by per-tag quantile matching against scored movies
(so filled x has the same marginal distribution as genome x; no hand-set constants).
Validation: the same procedure on 500 SCORED movies -> Spearman correlation with their genome x.
Output: processed/item_chain/review_fill_v1.npz (items, X)."""
import json, re, csv, collections, random
import numpy as np
GEN = './data/tag_genome_2021'
IC = './processed/item_chain'
z = np.load(f'{IC}/item_chain_v1.npz'); scored = z['items'].tolist(); gpos = {g: k for k, g in enumerate(scored)}
PROPS = json.load(open(f'{IC}/item_chain_v1_vocab.json'))['properties']
r2g = json.load(open(f'{IC}/redial2genome.json'))
targets = sorted({g for g in r2g.values() if g not in gpos})
random.seed(0); check = random.sample(scored, 500)
want = set(targets) | set(check)
pats = [re.compile(r'\b' + re.escape(p) + r'\b') for p in PROPS]
cnt = collections.defaultdict(lambda: np.zeros(len(PROPS), np.float32)); nrev = collections.Counter()
with open(f'{GEN}/raw/reviews.json') as f:
    for line in f:
        d = json.loads(line); g = d['item_id']
        if g not in want: continue
        txt = d['txt'].lower(); nrev[g] += 1
        v = cnt[g]
        for k, p in enumerate(pats):
            if p.search(txt): v[k] += 1
def share(g): return cnt[g] / max(nrev[g], 1)
Xs = np.clip(z['X'], 0, 1)
sorted_cols = np.sort(Xs, axis=0)
def qmap(raw):  # per-tag quantile matching against the genome distribution
    out = np.zeros_like(raw)
    for k in range(raw.shape[1]):
        r = raw[:, k]; ranks = np.argsort(np.argsort(r)) / max(len(r) - 1, 1)
        out[:, k] = np.quantile(sorted_cols[:, k], ranks)
    return out
from scipy.stats import spearmanr
C = np.array([share(g) for g in check]); Cq = qmap(C); G = Xs[[gpos[g] for g in check]]
rho = [spearmanr(Cq[i], G[i]).correlation for i in range(len(check)) if nrev[check[i]] > 0]
print('validation on scored movies: per-movie Spearman(review x, genome x) median', round(float(np.nanmedian(rho)), 3),
      '| movies with reviews', sum(nrev[g] > 0 for g in check), '/500')
T = [g for g in targets if nrev[g] > 0]
X = qmap(np.array([share(g) for g in T])) if T else np.zeros((0, len(PROPS)))
np.savez_compressed(f'{IC}/review_fill_v1.npz', items=np.array(T), X=X.astype(np.float32))
print('targets', len(targets), 'with reviews', len(T), 'median reviews', float(np.median([nrev[g] for g in T])) if T else 0)
