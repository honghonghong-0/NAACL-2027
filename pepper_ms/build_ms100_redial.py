"""PEPPER-MS on the ReDial item space, 100 users. Same rules as build_ms_inputs.py + build_ms_v1.py
(pre-registered design v0/v1), but users come from PEPPER's ReDial user samples and genres from crs_data/redial.
S1 = original persona/targets. S2keep = S1 + no-restate note. S2shift = + light-situation card, targets filtered to
light-compatible ones (>=2 kept and >=1 removed). Fixed seed."""
import json, random, ast, sys
from pathlib import Path
ROOT = Path('./pepper_ms/PEPPER')
info = json.loads((ROOT / 'crs_data/redial/id2info.json').read_text())
imdb2g = {v['imdb_id']: set(g.lower() for g in v.get('genre', [])) for v in info.values() if v.get('imdb_id')}
users = json.loads((ROOT / 'dataset/user_data/user_IMDB_ReDIal_samples.json').read_text())
LIGHT_IN = {'comedy', 'romance', 'family', 'animation'}; LIGHT_OUT = {'horror', 'war'}
NO_RESTATE = ("\n\nIMPORTANT: You already chatted with this same recommender in a previous session and told it your "
              "general movie taste; assume it remembers. Do NOT restate your general preferences or favourite genres "
              "unless the recommender explicitly asks. Keep your messages about today's request and the current suggestions.")
CARD = ("\n\nTODAY'S SITUATION: You are exhausted after a long week and want something light and feel-good tonight, "
        "nothing dark, violent, scary or emotionally heavy. Mention this ONCE, in one of your first three messages, "
        "and do not repeat it afterwards unless the recommender asks.")
def lst(x): return x if isinstance(x, list) else ast.literal_eval(x)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
elig = []
for uid, v in users.items():
    tg = lst(v['target']); ok = [r for r in tg if (g := imdb2g.get('tt' + str(r['imdb_id']).removeprefix('tt'))) and g & LIGHT_IN and not g & LIGHT_OUT]
    if len(ok) >= 2 and len(ok) < len(tg): elig.append((uid, ok))
rng = random.Random(20261011); rng.shuffle(elig); pick = elig[:N]
out = ROOT / 'research-notes/pepper-ms-redial-100'; out.mkdir(exist_ok=True)
s1, keep, shift = {}, {}, {}
for uid, ok in pick:
    v = users[uid]; s1[uid] = v
    k = dict(v); k['persona'] = v['persona'] + NO_RESTATE; keep[uid] = k
    w = dict(v); w['persona'] = v['persona'] + NO_RESTATE + CARD
    w['target'] = ok; w['target movies'] = [r['title'] for r in ok]; w['situation'] = 'light'; shift[uid] = w
for name, d in [('s1', s1), ('s2keep', keep), ('s2shift', shift)]:
    (out / f'users-{name}.json').write_text(json.dumps(d, ensure_ascii=False))
print('eligible', len(elig), 'picked', len(pick))
