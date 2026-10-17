"""S1-a: ReDial movie <-> Tag Genome 2021 movie linking (title + year).
Ambiguous matches are left unlinked. Reports link rate (not accuracy) and Pool A coverage.
Usage: python link_genome.py
"""
import json, re, csv, collections, os, unicodedata

ROOT = '.'
GEN = './data/tag_genome_2021'
OUT = f'{ROOT}/processed/item_chain'
os.makedirs(OUT, exist_ok=True)

ARTS = ('the', 'a', 'an', 'la', 'le', 'les', 'el', 'il', 'die', 'das', 'der')


def split_year(t):
    t = t.strip()
    m = re.search(r'\((\d{4})\)\s*$', t)
    if m:
        return t[:m.start()].strip(), int(m.group(1))
    return t, None


def norm(t):
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode()
    t = t.lower().strip()
    t = re.sub(r'\s*\([^)]*\)\s*$', '', t)  # trailing alt-title in parens
    m = re.match(r'^(.*),\s*(' + '|'.join(ARTS) + r')$', t)  # "matrix, the"
    if m:
        t = m.group(2) + ' ' + m.group(1)
    t = t.replace('&', 'and')
    t = re.sub(r"[^a-z0-9 ]", ' ', t)
    w = t.split()
    if w and w[0] in ARTS:
        w = w[1:]
    return ' '.join(w)


scored = set()
with open(f'{GEN}/scores/tagdl.csv') as f:
    r = csv.reader(f); next(r)
    for row in r:
        scored.add(int(row[1]))

idx = collections.defaultdict(list)  # norm title -> [(year, item_id)]
for line in open(f'{GEN}/raw/metadata_updated.json'):
    d = json.loads(line)
    base, y = split_year(d['title'])
    idx[norm(base)].append((y, d['item_id']))

redial = []
with open(f'{ROOT}/raw_unzip/movies_with_mentions.csv') as f:
    for row in csv.DictReader(f):
        redial.append((int(row['movieId']), row['movieName'], int(row['nbMentions'])))

link, status = {}, collections.Counter()
for rid, name, _ in redial:
    base, y = split_year(name)
    cands = idx.get(norm(base), [])
    if not cands:
        status['no_title'] += 1; continue
    if y is None:
        pick = cands if len(cands) == 1 else []
        if not pick:
            status['ambiguous_noyear'] += 1; continue
    else:
        pick = [c for c in cands if c[0] == y]
        if not pick:
            pick = [c for c in cands if c[0] is not None and abs(c[0] - y) == 1]
    if len(pick) != 1:
        status['ambiguous' if len(pick) > 1 else 'year_mismatch'] += 1; continue
    gid = pick[0][1]
    link[rid] = gid
    status['linked_scored' if gid in scored else 'linked_unscored'] += 1

pool = json.load(open(f'{ROOT}/processed/pool_a.json'))
r2c = {int(k): v for k, v in pool['redial2canon'].items()}
canon_scored = {}
for rid, gid in link.items():
    c = r2c.get(rid)
    if c is not None and gid in scored:
        canon_scored.setdefault(c, gid)
pool_items = set(pool['item_ids'])
cov = len(pool_items & set(canon_scored)) / len(pool_items)

ment = collections.Counter()
for rid, name, n in redial:
    ment['all'] += n
    if rid in link and link[rid] in scored:
        ment['scored'] += n

json.dump({str(k): v for k, v in link.items()}, open(f'{OUT}/redial2genome.json', 'w'))
json.dump({str(k): v for k, v in canon_scored.items()}, open(f'{OUT}/canon2genome_scored.json', 'w'))
rep = {'redial_movies': len(redial), 'status': status, 'genome_scored_movies': len(scored),
       'link_rate': round(len(link) / len(redial), 4),
       'poolA_items': len(pool_items), 'poolA_scored_coverage': round(cov, 4),
       'mention_weighted_scored_coverage': round(ment['scored'] / ment['all'], 4)}
json.dump(rep, open(f'{OUT}/link_report.json', 'w'), indent=1, default=dict)
print(json.dumps(rep, indent=1, default=dict))
