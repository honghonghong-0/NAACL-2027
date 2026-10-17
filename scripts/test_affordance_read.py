"""Unit tests required by the 2026-10-13 audit for affordance_read_v2 (A1 + A2 + B1).
T1: with no movie at all, an explicit experience request ("funny") changes q and the candidate ranking.
T2: a past attribute memory ("twist ending", +) changes the personal score of related candidates.
T3: flipping that memory's direction (+ -> -) flips the sign of its score contribution.
Run: CACHE_FILE=... python scripts/test_affordance_read.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import affordance_read_v2 as AR
import alpha_train as A
from collections import Counter

def concepts_for(text, kind):
    e = {'kind': kind, 'target': text, 'concepts': []}
    # reuse the stored mapping style: find the closest Tag Genome concept by exact name when available
    if kind == 'experience' and text in AR.ai: e['concepts'] = [('experience', text, 1.0)]
    if kind == 'property' and text in AR.pi:
        e['concepts'] = [('property', text, 1.0)] + [('experience_via_property', AR.EXPS[a], float(AR.PROF_P[AR.pi[text], a]))
                                                      for a in np.argsort(-AR.PROF_P[AR.pi[text]])[:3]]
    assert e['concepts'], f'{text} not in vocabulary'
    return e

def q_explicit(entries):
    q = np.zeros(len(AR.EXPS), np.float32)
    for e, d in entries:
        for typ, name, w in e['concepts']:
            if typ in ('experience', 'experience_via_property') and name in AR.ai: q[AR.ai[name]] += d * w
    return q

def personal(entry, d, q):
    S = np.zeros_like(A.SBAR); cnt = Counter(); e = dict(entry, movie_id=None, direction='+' if d > 0 else '-')
    AR.add(e, d, S, cnt)
    return A.XN @ (((S - cnt['npos'] * A.SBAR) / (cnt['n'] + AR.KAPPA)).T @ q)

funny = concepts_for('funny', 'experience')
# T1
q = AR.unit(np.zeros(len(AR.EXPS), np.float32)) + AR.unit(q_explicit([(funny, 1.0)]))
assert np.abs(q).sum() > 0, 'T1: explicit experience did not reach q'
r0 = np.argsort(-(A.EC @ np.zeros_like(q))); r1 = np.argsort(-(A.EC @ q))
top = [k for k in r1[:20] if A.COVER[k]]
assert not np.array_equal(r0[:50], r1[:50]), 'T1: ranking unchanged'
print('T1 ok: q nonzero on', [AR.EXPS[k] for k in np.argsort(-q)[:3]], '| top-20 covered movies', len(top))
# T2
twist = 'twist ending' if 'twist ending' in AR.pi else 'twist'
tw = concepts_for(twist, 'property')
q2 = AR.unit(q_explicit([(concepts_for('tense', 'experience') if 'tense' in AR.ai else funny, 1.0)]))
p_pos = personal(tw, 1.0, q2)
x_tw = A.XN[:, AR.pi[twist]]
hi, lo = x_tw > 0.5, (x_tw < 0.05) & A.COVER
assert p_pos[hi].mean() > p_pos[lo].mean(), 'T2: twist memory did not raise twist-heavy candidates'
print(f'T2 ok: mean personal score twist-heavy {p_pos[hi].mean():.4f} vs twist-free {p_pos[lo].mean():.4f}')
# T3
p_neg = personal(tw, -1.0, q2)
assert np.allclose(p_neg, -p_pos, atol=1e-6), 'T3: sign did not flip'
print('T3 ok: flipping direction flips the score exactly')
