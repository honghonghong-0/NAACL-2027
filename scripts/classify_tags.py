"""S1-b: classify Tag Genome tags into chain layers with one LLM pass (temperature 0).
Layers: entity | property | experience | evaluation | other.
Stores every raw response (first + retry) for traceability.
"""
import json, os, asyncio, re
from openai import AsyncOpenAI

GEN = './data/tag_genome_2021'
OUT = './processed/item_chain'
MODEL = 'gpt-4o-mini-2024-07-18'
LAYERS = ('entity', 'property', 'experience', 'evaluation', 'other')

SYSTEM = """You label movie tags by the layer they describe. Output one line per tag: tag<TAB>layer

Layers:
entity = a named or categorical thing the movie belongs to or features: genre (horror, sci-fi), person (tom hanks), studio, franchise, source material (based on a book), country, era label.
property = an observable characteristic of the movie's content or form: setting, plot device, theme, visual or narrative style, pacing (time travel, space, twist ending, slow, dialogue driven, great soundtrack, dark).
experience = the effect the movie has on the viewer: feelings or mental states it produces (scary, funny, thought-provoking, feel-good, tense, depressing, heartwarming, mind-bending, boring).
evaluation = a judgement of quality or worth with no specific effect (good, overrated, masterpiece, bad acting, classic).
other = noise, typos, personal or platform notes (netflix, seen at the cinema, dvd, own it).

When a tag both describes the movie and its effect, choose experience only if the core meaning is the effect on the viewer."""


async def run(cli, batch, sem):
    async with sem:
        msgs = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': '\n'.join(batch)}]
        raws = []
        for attempt in range(2):
            r = await cli.chat.completions.create(model=MODEL, messages=msgs, temperature=0)
            raw = r.choices[0].message.content
            raws.append(raw)
            lab = {}
            for line in raw.strip().splitlines():
                parts = re.split(r'\t|\s*\|\s*', line.strip())
                if len(parts) >= 2 and parts[-1].strip().lower() in LAYERS:
                    lab[parts[0].strip().lower()] = parts[-1].strip().lower()
            missing = [t for t in batch if t not in lab]
            if not missing:
                break
            msgs += [{'role': 'assistant', 'content': raw},
                     {'role': 'user', 'content': 'Missing or invalid: ' + ', '.join(missing[:20]) + '. Output the full list again.'}]
        return lab, raws, r.usage


async def main():
    key = os.environ['OPENAI_API_KEY']
    cli = AsyncOpenAI(api_key=key)
    tags = sorted(json.loads(l)['tag'].lower() for l in open(f'{GEN}/raw/tags.json'))
    batches = [tags[i:i + 80] for i in range(0, len(tags), 80)]
    sem = asyncio.Semaphore(6)
    res = await asyncio.gather(*[run(cli, b, sem) for b in batches])
    labels, raws, pin, pout = {}, [], 0, 0
    for (lab, rr, us), b in zip(res, batches):
        for t in b:
            labels[t] = lab.get(t, 'unlabeled')
        raws.append({'batch': b, 'responses': rr})
        pin += us.prompt_tokens; pout += us.completion_tokens
    json.dump(labels, open(f'{OUT}/tag_layers.json', 'w'), indent=0)
    json.dump(raws, open(f'{OUT}/tag_layers_raw.json', 'w'))
    from collections import Counter
    print(Counter(labels.values()), 'usd~', round(pin * 0.15e-6 + pout * 0.6e-6, 4))
    for L in LAYERS:
        print(L, [t for t, v in labels.items() if v == L][:25])

asyncio.run(main())
