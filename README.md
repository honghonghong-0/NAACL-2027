# When Do Past Preferences Carry Over? User-Specific Affordances in Multi-Session Conversational Recommendation

Code for the anonymous NAACL 2027 submission. All reported numbers are on the **validation** split.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...        # memory extraction, tag classification, PEPPER-MS (gpt-4o-mini, temperature 0)
```

Run every command from the repository root. Place the public datasets under `data/` (not redistributed here):

| Path | Source |
|---|---|
| `raw_unzip/` | ReDial (`train_data.jsonl`, `test_data.jsonl`, `movies_with_mentions.csv`) |
| `data/redial/` | ReDial with DBpedia entity linking released by CRSLab/UniCRS (candidate pool) |
| `data/inspired/` | INSPIRED |
| `data/tag_genome_2021/` | MovieLens Tag Genome 2021 (`raw/`, `scores/tagdl.csv`) |

Intermediate files are written to `processed/`, `extract_v6/`, and `baselines/`.

## Pipeline (paper section → script)

| Step | Paper | Script |
|---|---|---|
| 1. ReDial-MS construction and Pool A | §3.1, App. A | `scripts/build_redial_ms.py`, `scripts/pool_a.py`, `scripts/redial_stats.py` |
| 2. Memory writing (extraction, verification, consolidation, concept alignment) | §4.2, App. B | `scripts/extract_v6.py` (prompts: `scripts/prompt_v6_*.txt`), `scripts/build_memory_v6.py` |
| 3. Population affordance (Tag Genome linking, tag layers, attribute→experience model) | §4.3, App. C | `scripts/link_genome.py`, `scripts/classify_tags.py`, `scripts/build_item_chain.py`, `scripts/review_fill.py` |
| 4. Carryover classifier | §4.4, App. D | `scripts/alpha_train.py` |
| 5. Affordance-indexed reading and user-specific affordance | §4.4–4.5 | `scripts/affordance_read_v2.py`, `scripts/learn_affordance.py` |
| 6. Residual scoring on backbones (cross-fitted weights) | §4.6, Table 3 | `scripts/residual_fuse.py` (see `runs/run_v6c.sh`) |
| 7. Evaluation and statistics | §5–6, Tables 1–2, Fig. 2 | `scripts/eval_poola.py`, `scripts/paired_eval.py`, `scripts/exposure_table.py`, `scripts/fig_evidence_types.py` |
| 8. INSPIRED analysis | App. E | `scripts/inspired_eval.py`, `scripts/inspired_eval_covered.py`, `scripts/review_fill_inspired.py` |
| 9. PEPPER-MS interactive evaluation | §5, Table 4, Fig. 4, App. G | `pepper_ms/` |

Unit tests for the reading path: `python scripts/test_affordance_read.py`.

### Baselines (App. F)

- Popularity and co-mention: `scripts/baselines_simple.py`
- SASRec: `scripts/sasrec.py`, `scripts/sasrec_poola.py`, `scripts/run_sasrec_poola_valid.sh`
- KBRD (CRSLab): `scripts/build_crslab_data.py`, `scripts/crslab_map.py`, `scripts/crslab_infer.py`, `scripts/run_kbrd_multiseed_valid.sh`
- MemoCRS (reimplementation; no official code): `scripts/build_memocrs_candidates.py`, `scripts/memocrs_style.py`

Every recommender writes an `.npz` with `keys`, `scores`, `item_ids` and is scored by the same evaluator:

```bash
python scripts/eval_poola.py valid name=baselines/scores_X_valid.npz
python scripts/residual_fuse.py fuse <name> <backbone.npz | cooc>
```

## PEPPER-MS

`pepper_ms/` extends the target-free PEPPER simulator to two sessions (S1 past session, S2 shifted situation). It requires the PEPPER code and resources from the original authors. `build_ms_inputs_100.py` / `build_ms100_redial.py` build the session inputs, `pepper_memory_prep.py` extracts S1 memories, `pepper_ours.py` implements selective carryover, and `run_ms100_*.sh` launch the three conditions.

## Licenses

Code is released under the MIT License. Data are **not** redistributed. Item-side scores derived from the Tag Genome dataset (CC BY-NC 3.0) may be used for non-commercial research only; please cite Vig et al. (2012) and Kotkov et al. (2021). ReDial, INSPIRED, CRSLab, and PEPPER resources follow their original licenses.
