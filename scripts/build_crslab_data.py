"""Build CRSLab-format ReDial (nltk resource layout) for OUR ReDial-MS split from UniCRS's DBpedia-linked ReDial
(official CRSLab redial_nltk.zip is unreachable: SharePoint 404/403, 2026-10-12).
- dialogs: all 11,348 conversations; train = 'history', valid/test = our valid/test sessions (users.json).
- utterance: role (Seeker = initiatorWorkerId), text tokens (lowercased regex tokenization; @id -> movie title like
  UniCRS process.py), movies/entity = DBpedia URIs from UniCRS linking, word = [] (no ConceptNet; KBRD/UCCR do not need it).
- extra fields conv_id, user_id (UCCR loader uses them; official CRSLab ignores them).
- KG: UniCRS entity2id (31,161) + dbpedia_subkg + relation2id (same format as CRSLab); token2id from TRAIN only.
Output: <crslab_root>/data/dataset/redial/nltk/ + .built marker (version 0.31) so CRSLab does not try to download."""
import json, re, html, sys, collections, datetime
from pathlib import Path
R = Path('.'); U = R / 'baselines/unicrs/src/data/redial'
out = Path(sys.argv[1]) / 'data/dataset/redial/nltk'; out.mkdir(parents=True, exist_ok=True)
split = {s['conv_id']: s['split'] for ss in json.load(open(R / 'processed/users.json')).values() for s in ss}
TOK = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
def tokens(text, names):
    text = re.sub(r'@(\d+)', lambda m: ' '.join(names.get(m.group(1), 'movie').split()), text)
    return TOK.findall(html.unescape(text).lower())
data = {'train': [], 'valid': [], 'test': []}; cnt = collections.Counter()
for f in ['train', 'valid', 'test']:
    for line in open(U / f'{f}_data_dbpedia.jsonl'):
        d = json.loads(line); k = {'history': 'train', 'valid': 'valid', 'test': 'test'}.get(split.get(int(d['conversationId'])))
        if k is None: cnt['skip'] += 1; continue
        dialog = [{'utt_id': i, 'role': 'Seeker' if m['senderWorkerId'] == d['initiatorWorkerId'] else 'Recommender',
                   'text': tokens(m['text'], d['movieMentions']), 'movies': m['movie'], 'entity': m['entity'], 'word': []}
                  for i, m in enumerate(d['messages'])]
        if dialog: data[k].append({'conv_id': int(d['conversationId']), 'user_id': int(d['initiatorWorkerId']), 'dialog': dialog}); cnt[k] += 1
tf = collections.Counter(t for c in data['train'] for u in c['dialog'] for t in u['text'])
tok2id = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
for t, c in tf.most_common():
    if c >= 2: tok2id.setdefault(t, len(tok2id))
for k, v in data.items(): json.dump(v, open(out / f'{k}_data.json', 'w'))
json.dump(tok2id, open(out / 'token2id.json', 'w'))
for f in ['entity2id.json', 'dbpedia_subkg.json', 'relation2id.json']: (out / f).write_text((U / f).read_text())
(out / "movie_ids.json").write_text((R / "baselines/unicrs/src/data/redial_ms/item_ids.json").read_text())  # 6,281 movie entities (UniCRS process.py)
json.dump({"__pad__": 0}, open(out / 'concept2id.json', 'w')); (out / 'conceptnet_subkg.txt').write_text('')
(out / '.built').write_text(f'{datetime.datetime.now()}\n0.31\n')
print(dict(cnt), 'vocab', len(tok2id), '->', out)
