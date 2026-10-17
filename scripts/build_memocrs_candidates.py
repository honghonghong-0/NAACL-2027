#!/usr/bin/env python3
"""Build validation top-K candidate records for MemoCRS-style.

Default generator is the validation-only co-occurrence baseline for smoke tests.
An external score npz can replace it (e.g. the finished UCCR exporter) without
changing the downstream MemoCRS implementation.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import baselines_simple as bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid", choices=["valid"])
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--external-npz")
    ap.add_argument("--name", default="cooc")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    pa = json.load(open(ROOT / "processed/pool_a.json"))
    pool, r2c = pa["pool"], pa["redial2canon"]
    col = {c:i for i,c in enumerate(pool)}
    names = {r["movieId"]:r["movieName"].strip() for r in csv.DictReader(open(ROOT / "raw_unzip/movies_with_mentions.csv"))}
    canon_names, canon_rid = {}, {}
    for rid,cid in r2c.items(): canon_names.setdefault(cid, names.get(rid, rid))
    for rid,cid in r2c.items(): canon_rid.setdefault(cid, rid)
    users = json.load(open(ROOT / "processed/users.json"))
    sessions = {(int(u),s["conv_id"]):s for u,ss in users.items() for s in ss}
    turns = [json.loads(x) for x in open(ROOT / "processed/eval_turns_valid.jsonl")]
    ext = None
    if args.external_npz:
        z=np.load(args.external_npz); pos={c:i for i,c in enumerate(z["item_ids"].tolist())}; sel=np.array([pos[c] for c in pool])
        ext={tuple(k):z["scores"][i][sel].astype(float) for i,k in enumerate(z["keys"].tolist())}
    else:
        pop,co,_=bs.stats(args.split)
        ridx=np.array([bs.IDX[r] for r in r2c]); cidx=np.array([col[c] for c in r2c.values()])
    with open(args.output,"w") as out:
        for t in turns:
            s=sessions[(t["user"],t["conv_id"])]
            cur=list(dict.fromkeys(bs.movies(s["messages"][:t["turn_index"]])))
            if ext is not None: score=ext[(t["user"],t["conv_id"],t["turn_index"])].copy()
            else:
                raw=bs.co_vec(co,cur).astype(float); score=np.full(len(pool),-np.inf); np.maximum.at(score,cidx,raw[ridx])
                score[np.isneginf(score)]=0
                score += 1e-9*np.asarray([pop[bs.IDX[canon_rid[c]]] if canon_rid.get(c) in bs.IDX else 0 for c in pool])
            excluded={col[r2c[r]] for r in cur if r in r2c}
            if excluded: score[list(excluded)]=-np.inf
            order=np.argsort(-score)[:args.topk]
            rec={"key":[t["user"],t["conv_id"],t["turn_index"]],"generator":args.name,
                 "candidates":[{"item_id":pool[i],"title":canon_names.get(pool[i],str(pool[i])),"score":float(score[i])} for i in order]}
            out.write(json.dumps(rec,ensure_ascii=False)+"\n")

if __name__=="__main__": main()
