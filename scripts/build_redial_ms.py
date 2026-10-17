"""ReDial multi-session dataset. Follows the UCCR/MemoCRS chronological idea with explicit, reproducible rules:
- user = seeker (initiatorWorkerId); sessions of a user ordered by conversationId (no wall-clock time in ReDial);
- history = ALL earlier sessions of the same seeker (not only consecutive ones);
- users with >=2 sessions: last session = test, second-to-last = valid (only if >=3 sessions), rest = history;
- targets at each recommender turn = movies mentioned in that recommender message that were not mentioned earlier
  in the dialogue (MemoCRS repetition filter); accepted targets = those the seeker marked liked==1 (initiatorQuestions).
Output: users.json (per-user ordered sessions + split), eval_turns_{valid,test}.jsonl, stats.json."""
import json, collections, re
from pathlib import Path
R = Path('./raw_unzip'); O = Path('./processed'); O.mkdir(exist_ok=True)
rows = [json.loads(l) for f in ['train_data.jsonl', 'test_data.jsonl'] for l in open(R / f)]
MENTION = re.compile(r'@(\d+)')
by = collections.defaultdict(list)
for r in rows: by[r['initiatorWorkerId']].append(r)
users, turns = {}, {'valid': [], 'test': []}
for u, convs in by.items():
    convs.sort(key=lambda r: int(r['conversationId']))
    split = ['history'] * len(convs)
    if len(convs) >= 2: split[-1] = 'test'
    if len(convs) >= 3: split[-2] = 'valid'
    sess = []
    for r, sp in zip(convs, split):
        seeker = r['initiatorWorkerId']; iq = r['initiatorQuestions'] if isinstance(r['initiatorQuestions'], dict) else {}
        msgs = [{'role': 'seeker' if m['senderWorkerId'] == seeker else 'recommender', 'text': m['text']} for m in r['messages']]
        sess.append({'conv_id': int(r['conversationId']), 'split': sp, 'messages': msgs, 'mentions': r['movieMentions'] if isinstance(r['movieMentions'], dict) else {},
                     'seeker_labels': iq})
        if sp in ('valid', 'test'):
            seen = set()
            for i, m in enumerate(msgs):
                ids = MENTION.findall(m['text'])
                if m['role'] == 'recommender':
                    tgt = [x for x in dict.fromkeys(ids) if x not in seen]
                    if tgt and i > 0:
                        acc = [x for x in tgt if str(iq.get(x, {}).get('liked')) == '1']
                        turns[sp].append({'user': u, 'conv_id': int(r['conversationId']), 'turn_index': i, 'targets': tgt, 'accepted_targets': acc,
                                          'n_history_sessions': sum(1 for s2 in split[:split.index(sp)] if s2 == 'history') if sp == 'test' else split.index('valid')})
                seen.update(ids)
    users[u] = sess
json.dump(users, open(O / 'users.json', 'w'))
for sp in turns:
    with open(O / f'eval_turns_{sp}.jsonl', 'w') as f:
        for t in turns[sp]: f.write(json.dumps(t) + '\n')
multi = {u: s for u, s in users.items() if len(s) >= 2}
hist = [len(s) - 1 for s in multi.values()]
st = {'dialogues': len(rows), 'seekers': len(users), 'seekers_ge2': len(multi), 'seekers_ge3': sum(len(s) >= 3 for s in users.values()),
      'test_turns': len(turns['test']), 'test_turns_with_accepted': sum(bool(t['accepted_targets']) for t in turns['test']),
      'valid_turns': len(turns['valid']), 'history_sessions_per_test_user_median': sorted(hist)[len(hist) // 2], 'history_sessions_max': max(hist),
      'test_users_with_turns': len({t['user'] for t in turns['test']})}
json.dump(st, open(O / 'stats.json', 'w'), indent=1); print(json.dumps(st, indent=1))
