#!/usr/bin/env python3
"""Controlled MemoCRS-style reproduction (validation only).

User memory: prior-history relational records. Retrieval: embedding top-N then
LLM selects Q relevant memories and reranks a fixed top-40 candidate list.
No target/future utterance is placed in prompts; no out-of-candidate generation.
"""
import argparse, asyncio, csv, json, re, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; P=ROOT/"processed"; MODEL="gpt-4o-mini"; TEMP=0

def render_memory(m,i):
    return f"M{i}: {m.get('relation',m.get('rel'))}; {m.get('target_type')}; {m.get('target')}; valence={m.get('valence')}; situation={m.get('situation') or 'none'}; evidence={m.get('evidence','')}"

def render_dialog(s,turn,names):
    z=[]
    for m in s["messages"][:turn]:
        role="USER" if m["role"]=="seeker" else "RECOMMENDER"
        text=re.sub(r"@(\d+)",lambda x:names.get(x.group(1),"a movie"),m["text"])
        z.append(f"{role}: {text}")
    return "\n".join(z)

async def main(args):
    from openai import AsyncOpenAI
    from sentence_transformers import SentenceTransformer
    users=json.load(open(P/"users.json")); mem=json.load(open(P/"relational_memory_v5.1.json"))
    names={r["movieId"]:r["movieName"].strip() for r in csv.DictReader(open(ROOT/"raw_unzip/movies_with_mentions.csv"))}
    sessions={(int(u),s["conv_id"]):s for u,ss in users.items() for s in ss}
    cand=[json.loads(x) for x in open(args.candidates)]; cand=cand[:args.limit] if args.limit else cand
    encoder=SentenceTransformer(args.encoder,device=args.device)
    # Encode each user's memory bank once; validation has multiple turns per user.
    mem_text, mem_vec = {}, {}
    for u, rows in mem.items():
        rows=[x for x in rows if x.get("split")=="history" and x.get("source")=="utterance"]
        texts=[render_memory(x,i) for i,x in enumerate(rows)]
        mem_text[u]=texts
        if texts: mem_vec[u]=encoder.encode(texts,normalize_embeddings=True,batch_size=128)
    cli=AsyncOpenAI(api_key=Path(args.api_key).read_text().strip(),timeout=120,max_retries=3)
    outp=Path(args.output); outp.parent.mkdir(parents=True,exist_ok=True); done=set()
    if outp.exists(): done={tuple(x["key"]) for x in map(json.loads,outp.read_text().splitlines())}
    sem=asyncio.Semaphore(args.concurrency); lock=asyncio.Lock()
    async def one(row):
        key=tuple(row["key"])
        if key in done:return
        u,conv,turn=key; dialog=render_dialog(sessions[(u,conv)],turn,names)
        mt=mem_text.get(str(u),[])
        if mt:
            eq=encoder.encode([dialog],normalize_embeddings=True)[0]
            idx=np.argsort(-(mem_vec[str(u)]@eq))[:args.prefilter]; selected_pool=[mt[i] for i in idx]
        else:selected_pool=[]
        sp="\n".join(selected_pool) or "(none)"
        select_prompt=f"Select the {args.q} most relevant memory IDs for the user's current need. Return only IDs like M3,M8. If fewer are relevant, return fewer.\nCURRENT DIALOGUE:\n{dialog}\nMEMORIES:\n{sp}"
        async with sem:r1=await cli.chat.completions.create(model=MODEL,temperature=TEMP,messages=[{"role":"user","content":select_prompt}],max_tokens=100)
        ids=list(dict.fromkeys(re.findall(r"M\d+",r1.choices[0].message.content or "")))[:args.q]
        selected=[x for x in selected_pool if x.split(":",1)[0] in ids]
        choices="\n".join(f"C{i}: {x['title']}" for i,x in enumerate(row["candidates"]))
        rank_prompt=f"Rank the candidate movies for the current user. Use the relevant memories only when applicable to the current dialogue. Return every candidate ID exactly once, best first, comma-separated. Do not add movies.\nCURRENT DIALOGUE:\n{dialog}\nRELEVANT MEMORIES:\n"+("\n".join(selected) or "(none)")+f"\nCANDIDATES:\n{choices}"
        async with sem:r2=await cli.chat.completions.create(model=MODEL,temperature=TEMP,messages=[{"role":"user","content":rank_prompt}],max_tokens=400)
        order=[]
        for x in re.findall(r"C(\d+)",r2.choices[0].message.content or ""):
            i=int(x)
            if i<len(row["candidates"]) and i not in order:order.append(i)
        order += [i for i in range(len(row["candidates"])) if i not in order]
        usage={"prompt_tokens":r1.usage.prompt_tokens+r2.usage.prompt_tokens,"completion_tokens":r1.usage.completion_tokens+r2.usage.completion_tokens}
        rec={"key":list(key),"generator":row["generator"],"selected_memory_ids":ids,"selected_memories":selected,"ranking":order,"ranked_item_ids":[row["candidates"][i]["item_id"] for i in order],"raw_selection":r1.choices[0].message.content,"raw_ranking":r2.choices[0].message.content,"actual_models":[r1.model,r2.model],"usage":usage,"temperature":TEMP,"timestamp":int(time.time())}
        async with lock:
            with outp.open("a") as f:f.write(json.dumps(rec,ensure_ascii=False)+"\n")
    await asyncio.gather(*(one(x) for x in cand)); await cli.close()

if __name__=="__main__":
    a=argparse.ArgumentParser(); a.add_argument("--candidates",required=True); a.add_argument("--output",required=True); a.add_argument("--api-key",default=None); a.add_argument("--encoder",default="sentence-transformers/all-MiniLM-L6-v2"); a.add_argument("--device",default="cuda:1"); a.add_argument("--prefilter",type=int,default=20); a.add_argument("--q",type=int,default=5); a.add_argument("--limit",type=int); a.add_argument("--concurrency",type=int,default=8); args=a.parse_args(); asyncio.run(main(args))
