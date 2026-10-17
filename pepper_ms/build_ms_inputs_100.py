"""PEPPER-MS dev inputs. Rules are fixed before running.
Situation 'light' (emotional state): genre has Comedy/Romance/Family/Animation and not Horror/War.
S1 = original persona/targets. S2-keep = same as S1 (new session). S2-shift = persona + situation card,
targets filtered to situation-compatible ones (>=2 required). Users sampled with fixed seed."""
import json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent / 'PEPPER'
info = json.loads((ROOT / 'crs_data/opendialkg/id2info.json').read_text())
idm = json.loads((ROOT / 'research-notes/rating7-imdb-20261008/candidate-identity-map-final.json').read_text())
key2cid = {v['identity_key']: k for k, v in idm.items()}
users = json.loads((ROOT / 'research-notes/evaluation-dataset-seen-only-profiles-20261009-v1/users-id-evaluable-2626.json').read_text())
LIGHT_IN = {'Comedy', 'Romance Film', 'Romantic comedy', 'Romance', 'Family', 'Family Film', 'Family (Film genre) #395', 'Animation', 'Comedy film'}
LIGHT_OUT = {'Horror', 'Horror film', 'War film', 'War'}
def light_ok(cid):
    g = set(info[cid].get('genre', []))
    return bool(g & LIGHT_IN) and not (g & LIGHT_OUT)
CARD = ("\n\nTODAY'S SITUATION: You are exhausted after a long week and want something light and feel-good tonight. "
        "You do not want anything dark, violent, scary or emotionally heavy today. Let this come up naturally in the "
        "conversation; you do not have to say it in your first message.")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rng = random.Random(20261011)
elig = []
for uid, v in users.items():
    pairs = list(zip(v['target_imdb_ids'], v['target movies'])) if len(v['target_imdb_ids']) == len(v['target movies']) else None
    if pairs is None: continue
    ok = [(k, t) for k, t in pairs if key2cid.get(str(k)) and light_ok(key2cid[str(k)])]
    if len(ok) >= 2 and len(ok) < len(pairs):  # shift must actually remove some targets
        elig.append((uid, ok))
rng.shuffle(elig); pick = elig[:N]
out = ROOT / 'research-notes/pepper-ms-odkg-100'; out.mkdir(exist_ok=True)
s1, keep, shift = {}, {}, {}
for uid, ok in pick:
    v = users[uid]; s1[uid] = v; keep[uid] = v
    w = dict(v); w['persona'] = v['persona'] + CARD
    w['target_imdb_ids'] = [k for k, _ in ok]; w['target movies'] = [t for _, t in ok]
    keepset = {'tt' + str(k).removeprefix('tt') for k, _ in ok}
    w['target'] = [r for r in v['target'] if 'tt' + str(r['imdb_id']).removeprefix('tt') in keepset]
    assert len(w['target']) == len(ok), (uid, len(w['target']), len(ok))
    w['situation'] = 'light'
    shift[uid] = w
for name, d in [('s1', s1), ('s2keep', keep), ('s2shift', shift)]:
    (out / f'users-{name}.json').write_text(json.dumps(d, ensure_ascii=False))
meta = {'eligible_users': len(elig), 'picked': [u for u, _ in pick], 'seed': 20261011, 'card': CARD,
        'light_in': sorted(LIGHT_IN), 'light_out': sorted(LIGHT_OUT),
        'targets': {u: {'all': len(users[u]['target_imdb_ids']), 'light': len(ok)} for u, ok in pick}}
(out / 'meta.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
print(json.dumps({k: meta[k] for k in ['eligible_users', 'targets']}, indent=1))
