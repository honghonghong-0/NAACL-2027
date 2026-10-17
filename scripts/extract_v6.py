"""Memory write v6 (M-v5). Adopts LightMem's extract -> store -> update flow with our modifications
(no topic segmentation: one ReDial session = one unit; see design doc).
Per session, the LLM extracts only what the seeker states, with a reduced schema:
  kind | target | direction | condition | evidence | turn
  kind: movie, genre, person, franchise, property, experience   (chain: entity -> property -> affordance)
  status (past/current) is NOT asked: it follows from the session position.
Every raw response (first + retry + verifier) is stored. Evidence = normalized containment in the USER line.
Cohort: users with a validation session -> their last 21 history sessions + the validation session (prefix use is
restricted to turns before the recommendation turn downstream). Test sessions are never read.
Usage: extract_v6.py pilot N | full
"""
import os
import json, re, sys, csv, asyncio, collections
from pathlib import Path
B = Path('.'); O = B / 'extract_v6'; O.mkdir(exist_ok=True)
MODEL = 'gpt-4o-mini'
KINDS = {'movie', 'genre', 'person', 'franchise', 'property', 'experience'}
SYSTEM = open(B / 'scripts' / 'prompt_v6_system.txt').read()
VERIFY = open(B / 'scripts' / 'prompt_v6_verify.txt').read()


def norm(s):
    s = re.sub(r'\(\d{4}\)', '', s)
    s = re.sub(r'[\'"‘’“”]', '', s)
    return re.sub(r'\s+', ' ', s).strip().lower().strip('.!?, ')


def validate(raw, roles):
    if raw.strip().upper() == 'NONE':
        return [], []
    items, errs = [], []
    for line in raw.strip().splitlines():
        f = [x.strip() for x in line.split(' | ')]
        if len(f) != 6:
            errs.append(f'not 6 fields: {line[:60]}'); continue
        kind, tg, d, cond, ev, turn = f
        kind = kind.lower()
        if kind not in KINDS: errs.append(f'bad kind {kind}'); continue
        if d not in ('+', '-'): errs.append(f'bad direction {d}'); continue
        if not turn.isdigit() or int(turn) not in roles or roles[int(turn)][0] != 'USER':
            errs.append(f'turn {turn} not a USER line'); continue
        if len(norm(ev)) < 3 or norm(ev) not in norm(roles[int(turn)][1]):
            errs.append(f'evidence not in line {turn}: {ev[:50]}'); continue
        items.append({'kind': kind, 'target': tg, 'direction': d, 'condition': None if cond.lower() in ('null', 'none', '') else cond,
                      'evidence': ev, 'turn': int(turn)})
    return items, errs


def load(n=None):
    names = {r['movieId']: r['movieName'].strip() for r in csv.DictReader(open(B / 'raw_unzip/movies_with_mentions.csv'))}
    users = json.load(open(B / 'processed/users.json'))
    vu = sorted(u for u, ss in users.items() if any(s['split'] == 'valid' for s in ss))
    out = []
    for u in vu:
        ss = users[u]
        keep = [s for s in ss if s['split'] == 'history'][-21:] + [s for s in ss if s['split'] == 'valid']
        for s in keep:
            lines = [('USER' if m['role'] == 'seeker' else 'RECOMMENDER',
                      re.sub(r'@(\d+)', lambda x: names.get(x.group(1), 'a movie'), m['text'])) for m in s['messages']]
            out.append((f"redial:{u}:{s['conv_id']}", s['split'], lines))
    return out[:n] if n else out


async def main(mode, n=None):
    from openai import AsyncOpenAI
    cli = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=120, max_retries=3)
    sem = asyncio.Semaphore(12); use = [0, 0]; recs = []; mids = set()

    async def one(key, split, lines):
        roles = {i + 1: l for i, l in enumerate(lines)}
        text = '\n'.join(f'[{i}] {r}: {t}' for i, (r, t) in roles.items())
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': text}]
        raws, items, errs = [], [], []
        for a in range(2):
            async with sem:
                r = await cli.chat.completions.create(model=MODEL, messages=msgs, temperature=0)
            use[0] += r.usage.prompt_tokens; use[1] += r.usage.completion_tokens; mids.add(r.model)
            raw = r.choices[0].message.content or ''; raws.append(raw)
            items, errs = validate(raw, roles)
            if not errs: break
            msgs += [{'role': 'assistant', 'content': raw},
                     {'role': 'user', 'content': 'Some lines were invalid: ' + '; '.join(errs[:5]) + '. Output the full corrected list again in the exact format.'}]
        vraw = None
        exp = [it for it in items if it['kind'] == 'experience']
        if exp:
            claims = '\n'.join(f'{k+1}. claimed: {it["target"]} | evidence: "{it["evidence"]}"' for k, it in enumerate(exp))
            async with sem:
                v = await cli.chat.completions.create(model=MODEL, messages=[{'role': 'system', 'content': VERIFY}, {'role': 'user', 'content': claims}], temperature=0)
            use[0] += v.usage.prompt_tokens; use[1] += v.usage.completion_tokens
            vraw = v.choices[0].message.content or ''
            ok = {int(m.group(1)) for m in re.finditer(r'(\d+)\W+VALID', vraw)}
            for k, it in enumerate(exp): it['verified'] = (k + 1) in ok
        recs.append({'key': key, 'split': split, 'items': items, 'errors_after_retry': errs,
                     'raw_responses': raws, 'verify_raw': vraw, 'attempts': len(raws)})

    await asyncio.gather(*[one(*x) for x in load(n)]); await cli.close()
    p = O / (f'pilot_{n}_v6.jsonl' if mode == 'pilot' else 'full_redial_v6.jsonl')
    with open(p, 'w') as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    kc = collections.Counter(i['kind'] for r in recs for i in r['items'])
    print(json.dumps({'sessions': len(recs), 'items': sum(len(r['items']) for r in recs), 'kinds': dict(kc),
                      'retry_rate': round(sum(r['attempts'] > 1 for r in recs) / len(recs), 3),
                      'dropped_lines': sum(len(r['errors_after_retry']) for r in recs),
                      'verified_experience': sum(bool(i.get('verified')) for r in recs for i in r['items']),
                      'model': sorted(mids), 'usd': round(use[0] * .15e-6 + use[1] * .6e-6, 3)}))

if __name__ == '__main__':
    if sys.argv[1] == 'pilot': asyncio.run(main('pilot', int(sys.argv[2])))
    else: asyncio.run(main('full'))
