"""Section 3 figure: share of seeker utterances containing each evidence type, ReDial-MS (evaluation cohort) vs INSPIRED.
Source: v5 extraction (same schema for both datasets): extract_v4/full_redial_v5.jsonl, full_inspired_v5.jsonl.
Types: entity preference = L1 preference/appraisal; item attribute = L2; experience = L3 not rejected by the verifier;
situation = any item with a stated situation; constraint = relation constraint.
Value = (#user utterances with >=1 item of the type) / (#user utterances); 95% CI by dialog bootstrap (2,000)."""
import json, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B = '.'
TYPES = [('Entity preference', lambda it: it['layer'] == 'L1' and it['relation'] in ('preference', 'appraisal')),
         ('Item attribute', lambda it: it['layer'] == 'L2'),
         ('Experience', lambda it: it['layer'] == 'L3' and it.get('verified') is not False),
         ('Situation', lambda it: bool(it.get('situation'))),
         ('Constraint', lambda it: it['relation'] == 'constraint')]

def load(path):
    out = []
    for l in open(path):
        r = json.loads(l); n = r['n_user_lines']
        if n == 0: continue
        hits = [len({it['turn'] for it in r['items'] if f(it)}) for _, f in TYPES]
        out.append((n, hits))
    return out

res = {}
for name, path in [('ReDial-MS', f'{B}/extract_v4/full_redial_v5.jsonl'), ('INSPIRED', f'{B}/extract_v4/full_inspired_v5.jsonl')]:
    d = load(path); n = np.array([x[0] for x in d]); h = np.array([x[1] for x in d])
    est = h.sum(0) / n.sum(); rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(d), len(d)); bs.append(h[idx].sum(0) / n[idx].sum())
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    res[name] = (est * 100, lo * 100, hi * 100, len(d), int(n.sum()))
    print(name, 'dialogs', len(d), 'user utterances', int(n.sum()))
    for (t, _), e, a, b in zip(TYPES, est * 100, lo * 100, hi * 100): print(f'  {t:18s} {e:5.2f}% [{a:.2f}, {b:.2f}]')
json.dump({k: [list(map(float, v[0])), list(map(float, v[1])), list(map(float, v[2])), v[3], v[4]] for k, v in res.items()},
          open(f'{B}/analysis/fig_evidence_types.json', 'w'), indent=1)

plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': '#9d9d9d', 'axes.linewidth': 0.6})
fig, ax = plt.subplots(figsize=(3.3, 2.2))
col = {'ReDial-MS': '#2a78d6', 'INSPIRED': '#eb6834'}
y = np.arange(len(TYPES))[::-1]; hgt = 0.36
for k, (name, off) in enumerate([('ReDial-MS', hgt / 2 + 0.02), ('INSPIRED', -hgt / 2 - 0.02)]):
    e, lo, hi = res[name][:3]
    ax.barh(y + off, e, height=hgt, color=col[name], label=name, zorder=2)
    ax.errorbar(e, y + off, xerr=[e - lo, hi - e], fmt='none', ecolor='#555555', elinewidth=0.6, capsize=1.5, zorder=3)
    for yy, v in zip(y + off, e):
        ax.text(v + max(hi - e) + 0.6, yy, f'{v:.1f}', va='center', fontsize=6.5, color='#333333')
ax.set_yticks(y); ax.set_yticklabels([t for t, _ in TYPES])
ax.set_xlabel('Share of seeker utterances (%)')
ax.xaxis.grid(True, color='#e6e6e6', linewidth=0.5, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc='lower right')
ax.set_xlim(0, max(max(res[n][2]) for n in res) * 1.18)
fig.tight_layout(pad=0.3)
fig.savefig(f'{B}/analysis/fig_evidence_types.pdf'); fig.savefig(f'{B}/analysis/fig_evidence_types.png', dpi=300)
print('saved')
