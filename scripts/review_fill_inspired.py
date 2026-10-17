"""D5 for Inspired: attribute evidence x from IMDb reviews (Tag Genome raw/reviews.json) for movies that appear in
Inspired dialogs (train+dev) and have no Tag Genome scores. Same procedure as review_fill.py: share of a movie's
reviews mentioning each attribute tag (whole word), per-tag quantile matching to the genome distribution.
(review_fill.py validation on 500 scored movies: per-movie Spearman with genome x, median 0.73.)"""
import csv, json, re, collections
import numpy as np
GEN = "./data/tag_genome_2021"
IC = "./processed/item_chain"
D = "./data/inspired"
z = np.load(f"{IC}/item_chain_v1.npz"); gpos = {g: k for k, g in enumerate(z["items"].tolist())}
PROPS = json.load(open(f"{IC}/item_chain_v1_vocab.json"))["properties"]
SEP = re.compile(r"[;|]")

def tn(s):
    s = re.sub(r"\(\d{4}\)", "", s.lower()); s = re.sub(r"^(the|a|an) ", "", s.strip()); s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())

tyear2imdb = {(tn(r["title"]), r["year"][:4]): r["imdb_id"].replace("tt", "").lstrip("0")
              for r in csv.DictReader(open(f"{D}/movie_database.tsv"), delimiter="\t")}
imdb2item = {}
for l in open(f"{GEN}/raw/metadata_updated.json"):
    d = json.loads(l); imdb2item[d["imdbId"].lstrip("0")] = d["item_id"]
titles = set()
for sp in ("train", "dev"):
    for r in csv.DictReader(open(f"{D}/dialog_data/{sp}.tsv"), delimiter="\t"):
        titles |= {t.strip() for t in SEP.split(r["movies"]) if t.strip()}
tmap = {}
for t in titles:
    m = re.search(r"\((\d{4})\)", t); imdb = tyear2imdb.get((tn(t), m.group(1) if m else ""))
    if imdb and imdb in imdb2item and imdb2item[imdb] not in gpos:
        tmap[t] = imdb2item[imdb]
want = set(tmap.values())
print("dialog titles", len(titles), "unscored but in genome metadata", len(want), flush=True)
pats = [re.compile(r"\b" + re.escape(p) + r"\b") for p in PROPS]
cnt = collections.defaultdict(lambda: np.zeros(len(PROPS), np.float32)); nrev = collections.Counter()
with open(f"{GEN}/raw/reviews.json") as f:
    for line in f:
        d = json.loads(line); g = d["item_id"]
        if g not in want: continue
        txt = d["txt"].lower(); nrev[g] += 1; v = cnt[g]
        for k, p in enumerate(pats):
            if p.search(txt): v[k] += 1
Xs = np.clip(z["X"], 0, 1); sc = np.sort(Xs, axis=0)
T = [g for g in sorted(want) if nrev[g] > 0]
raw = np.array([cnt[g] / nrev[g] for g in T]) if T else np.zeros((0, len(PROPS)))
X = np.zeros_like(raw)
for k in range(raw.shape[1]):
    r = raw[:, k]; ranks = np.argsort(np.argsort(r)) / max(len(r) - 1, 1); X[:, k] = np.quantile(sc[:, k], ranks)
np.savez_compressed(f"{IC}/review_fill_inspired_v1.npz", items=np.array(T), X=X.astype(np.float32))
json.dump(tmap, open(f"{IC}/inspired_title2genome_unscored.json", "w"))
print("filled", len(T), "median reviews", float(np.median([nrev[g] for g in T])) if T else 0)
