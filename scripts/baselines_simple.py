"""ReDial-MS common evaluation harness + simple baselines (BASELINE_PLAN §1, step 1).
Per eval turn: context = messages before the recommender turn; candidates = all ReDial movies minus movies already
mentioned in the current dialogue (MemoCRS filter); each new target counted as a separate sample (CRSLab convention).
Training statistics use only sessions before the evaluated split: valid -> 'history' sessions; test -> 'history'+'valid'.
Methods: pop (recommender-mention popularity), cooc (item co-mention with current-dialogue movies), hist (co-mention with
the user's past-session movies only), cooc+hist (sum, history weight w chosen on valid)."""
import json, re, csv, sys, math, random, collections
import numpy as np
from pathlib import Path
B = Path('.'); P = B / 'processed'
MENTION = re.compile(r'@(\d+)')
CAT = sorted({r['movieId'] for r in csv.DictReader(open(B / 'raw_unzip/movies_with_mentions.csv'))})
IDX = {m: i for i, m in enumerate(CAT)}; N = len(CAT)
users = json.load(open(P / 'users.json'))

def movies(msgs): return [x for m in msgs for x in MENTION.findall(m['text']) if x in IDX]

def stats(split):
    allowed = {'history'} if split == 'valid' else {'history', 'valid'}
    pop = np.zeros(N); co = collections.defaultdict(collections.Counter)
    for u, ss in users.items():
        for s in ss:
            if s['split'] not in allowed: continue
            for m in s['messages']:
                if m['role'] == 'recommender':
                    for x in MENTION.findall(m['text']):
                        if x in IDX: pop[IDX[x]] += 1
            ms = list(dict.fromkeys(movies(s['messages'])))
            for a in ms:
                for b in ms:
                    if a != b: co[a][b] += 1
    return pop, co, allowed

def norm_max(v):
    m = np.abs(v).max(); return v / m if m > 0 else v

def co_vec(co, src):
    v = np.zeros(N)
    for a in src:
        for b, c in co[a].items(): v[IDX[b]] += c
    return v

def evaluate(split, method, w=1.0, ks=(10, 50), seed=0, target_key="targets", cur_bin=None):
    pop, co, allowed = stats(split); sessions = {(int(u), s['conv_id']): s for u, ss in users.items() for s in ss}
    turns = [json.loads(l) for l in open(P / f'eval_turns_{split}.jsonl')]
    per_user = collections.defaultdict(list)
    for t in turns:
        s = sessions[(t['user'], t['conv_id'])]; ctx = s['messages'][:t['turn_index']]
        cur = list(dict.fromkeys(movies(ctx)))
        hist = [x for s2 in users[str(t['user'])] if s2['split'] in allowed and s2['conv_id'] != t['conv_id'] for x in movies(s2['messages'])]
        tie = np.random.default_rng(seed).random(N) * 1e-6
        if method == 'pop': sc = pop.copy()
        elif method == 'cooc': sc = co_vec(co, cur)
        elif method == 'hist': sc = co_vec(co, hist)
        elif method == "cooc+hist":
            a, h = co_vec(co, cur), co_vec(co, hist); sc = a / (a.max() + 1e-9) + w * h / (h.max() + 1e-9)
        sc = sc / (sc.max() + 1e-9) + 1e-3 * pop / pop.max() + tie   # popularity as tie-breaker
        for x in cur: sc[IDX[x]] = -1
        order = np.argsort(-sc); rank = np.empty(N, int); rank[order] = np.arange(1, N + 1)
        if cur_bin is not None and not (cur_bin[0] <= len(cur) <= cur_bin[1]): continue
        for g in t[target_key]:
            if g not in IDX: continue
            r = rank[IDX[g]]; per_user[t['user']].append({f'HR@{k}': float(r <= k) for k in ks} | {f'MRR@{k}': (1 / r if r <= k else 0.) for k in ks} | {f'NDCG@{k}': (1 / math.log2(r + 1) if r <= k else 0.) for k in ks})
    keys = list(next(iter(per_user.values()))[0]); flat = [x for v in per_user.values() for x in v]
    res = {k: float(np.mean([x[k] for x in flat])) for k in keys}
    us = list(per_user); rng = random.Random(seed); bs = []
    for _ in range(1000):
        smp = [x for u in rng.choices(us, k=len(us)) for x in per_user[u]]; bs.append(np.mean([x['NDCG@10'] for x in smp]))
    res['NDCG@10_CI'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]; res['n_samples'] = len(flat); res['n_users'] = len(us)
    return res

if __name__ == '__main__':
    split = sys.argv[1]; out = {}
    for m in ["pop", "cooc", "hist"]: out[m] = evaluate(split, m)
    if split == 'valid':
        for w in [0.03, 0.1, 0.3]: out[f'cooc+hist(w={w})'] = evaluate(split, 'cooc+hist', w)
    for k, v in out.items(): print(f"{k:22s} HR@10={v['HR@10']:.3f} HR@50={v['HR@50']:.3f} MRR@10={v['MRR@10']:.3f} NDCG@10={v['NDCG@10']:.3f} CI={np.round(v['NDCG@10_CI'],3)} NDCG@50={v['NDCG@50']:.3f} n={v['n_samples']}")
    json.dump(out, open(B / f'results_simple_{split}.json', 'w'), indent=1)
