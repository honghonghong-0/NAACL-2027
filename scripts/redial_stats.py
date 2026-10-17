import json, collections, sys
R = './raw_unzip/'
rows = [json.loads(l) for f in ['train_data.jsonl', 'test_data.jsonl'] for l in open(R + f)]
print('dialogues', len(rows)); print('keys', sorted(rows[0].keys()))
d = rows[0]; print('sample msg', d['messages'][0]); print('initiatorQuestions sample', list(d['initiatorQuestions'].items())[:2])
seek = collections.Counter(r['initiatorWorkerId'] for r in rows)
resp = collections.Counter(r['respondentWorkerId'] for r in rows)
dist = collections.Counter(seek.values())
print('seeker workers', len(seek), '| with >=2 sessions', sum(c >= 2 for c in seek.values()), '| >=3', sum(c >= 3 for c in seek.values()), '| >=5', sum(c >= 5 for c in seek.values()))
print('sessions-per-seeker dist (top):', sorted(dist.items())[:12], '... max', max(seek.values()))
ids = [int(r['conversationId']) for r in rows]; print('conversationId range', min(ids), max(ids))
ts = [r['messages'][0].get('timeOffset') for r in rows[:3]]; print('timeOffset sample', ts)
# acceptance label distribution (initiator = seeker answers)
lab = collections.Counter()
for r in rows:
    iq = r["initiatorQuestions"]
    if not isinstance(iq, dict): lab[("EMPTY",)] += 1; continue
    for mid, q in iq.items():
        lab[(q.get('suggested'), q.get('seen'), q.get('liked'))] += 1
print('initiatorQuestions (suggested,seen,liked) top:', lab.most_common(10))
