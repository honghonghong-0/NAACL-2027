"""E2E baseline launcher.
10 users x 20 turns x {baseline,no_update,full}, one subprocess per (condition,user).
Completed runs are never repeated; total spend capped. Usage: python launch_e2e_10x20.py RUN_TAG [--preflight-only]
"""
import os, sys, json, subprocess, hashlib, re, fcntl
from pathlib import Path
W = Path(__file__).resolve().parent
ROOT = W / 'PEPPER'
os.chdir(ROOT)
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  HF_HUB_CACHE=str(W / 'A-mem/.venv/huggingface/hub'))
PY_TEAM = str(W / 'A-mem-paper-repro/.venv-paper/bin/python')
PY_LM = 'python'
LM_ENV = dict(JUDGE_URL="http://127.0.0.1:18765", JUDGE_MODEL_NAME="Qwen2.5-7B-Instruct", HF_HOME='./.cache/hf', OMP_NUM_THREADS='4', LIGHTMEM_EMBEDDER='sentence-transformers/all-MiniLM-L6-v2')
TAG = sys.argv[1]
TURNS = 20; K = '40'; MAX_OUTPUT = '512'; TOTAL_CAP = float(os.environ.get('E2E_CAP', '1.50')); PER_RUN_CAP = 0.10
CONDS = [c for c in (os.environ.get("E2E_CONDS") or "baseline,no_update,full").split(",")]
source = ROOT / os.environ.get("E2E_INPUT", "research-notes/evaluation-dataset-seen-only-profiles-20261009-v1/pilot-users-10-id-evaluable.json")
users = json.loads(source.read_text())
assert len(users) >= 1

# ---- preflight: leakage checks ----
def norm(s): return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()
leaks = {}
for uid, v in users.items():
    assert v['persona'].strip() and v['target movies']
    assert not set(map(str, v['seen_imdb_ids'])) & set(map(str, v['target_imdb_ids']))
    p = norm(v['persona'])
    hit = [t for t in v['target movies'] if len(norm(re.sub(r'\(\d{4}\)', '', t))) > 3
           and norm(re.sub(r'\(\d{4}\)', '', t)) in p]
    if hit: leaks[uid] = hit
print("PREFLIGHT users=%d turns=%d sha256=%s" % (len(users), TURNS, hashlib.sha256(source.read_bytes()).hexdigest()), flush=True)
print('PERSONA_TARGET_TITLE_MATCHES', json.dumps(leaks, ensure_ascii=False), flush=True)
if '--preflight-only' in sys.argv: sys.exit(0)

work = ROOT / 'research-notes' / f'e2e-10x20-{TAG}'; work.mkdir(exist_ok=True)
lock = (work / 'launcher.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_API_KEY']

def spent():
    return sum(json.loads(l)['estimated_usd'] for f in work.glob('*/usage.jsonl')
               for l in f.read_text().splitlines() if l.strip())

manifest = {'source': str(source.relative_to(ROOT)), 'turns': TURNS, 'memory_retrieval_k': int(K),
            'max_output': int(MAX_OUTPUT), 'total_cap_usd': TOTAL_CAP, 'runs': {}}
for cond in CONDS:
    manifest['runs'][cond] = {}
    for uid, data in users.items():
        tag = uid.split('/')[2]; name = f'{work.name}/{cond}-{tag}'; out = ROOT / 'research-notes' / name
        if not (out / 'report.json').exists():
            if out.exists(): raise RuntimeError(f'Incomplete run preserved; inspect before retry: {name}')
            remaining = TOTAL_CAP - spent()
            if remaining < .05: raise RuntimeError(f'Total cap reached: spent {spent():.4f}')
            inp = work / f'input-{tag}.json'; inp.write_text(json.dumps({uid: data}))
            env = dict(os.environ); py = PY_TEAM
            if cond in ("lightmem", "always_rank", "need", "need_nosuff", "need_noquery"):
                env.update(LM_ENV); env.pop('HF_HUB_CACHE', None); py = PY_LM
            with (work / f'{cond}-{tag}.log').open('w') as log:
                subprocess.run([py, 'research-notes/dialogue_e2e_runner.py', '--run', '--condition', cond,
                                    '--memory-retrieval-k', K, '--input', str(inp.relative_to(ROOT)), '--users', '1',
                                    '--turns', str(TURNS), '--max-cost', f'{min(PER_RUN_CAP, remaining):.4f}',
                                    '--max-output', MAX_OUTPUT, '--output-name', name],
                                   stdout=log, stderr=subprocess.STDOUT, check=True, env=env)
        ev = json.loads((out / 'eval' / f'{tag}.json').read_text()); assert len(ev) == TURNS + 1
        manifest['runs'][cond][uid] = name
        print(f'{cond} {tag} complete cumulative ${spent():.4f}', flush=True)
        (work / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(f'ALL_COMPLETE ${spent():.4f}', flush=True)
