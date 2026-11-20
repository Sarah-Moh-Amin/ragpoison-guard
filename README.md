# RAGPoison Guard

BSc graduation project: detecting poisoned passages in a small local RAG system using embedding anomaly scores.

## Status
Task 2 (clean RAG baseline) is complete on the development data (18 questions).

## Environment
- Python 3.14.7
- macOS on Apple M1 Pro, 16 GB RAM
- CPU only (no GPU)
- Exact package versions: `requirements-lock.txt`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```
The first run downloads about 3.4 GB of models.

## Run commands (in this order)
```bash
python src/smoke_test.py
python src/build_dataset.py
python src/validate_data.py
python src/run_retrieval.py
python src/run_baseline.py
python src/evaluate.py runs/<run_id>
python src/retrieval_depth.py
```
`<run_id>` is the baseline run folder printed by `run_baseline.py`, for example
`baseline_20261120_210939`.

## Inputs
- `data/clean/`: `draft_dataset.json` (source of truth), `passages.jsonl`, `queries.jsonl`
- `data/manifests/`: `split_manifest.json` (locked 18/6/6 split), `passage_roles.json`
- `config.json`: fixed settings (models, revisions, top_k, seed, generation settings, prompt version)

## Expected outputs of each script
| Script | Writes |
|---|---|
| `smoke_test.py` | `resource_notes.txt` (hardware, model load times, test answer) |
| `build_dataset.py` | `data/clean/passages.jsonl`, `data/clean/queries.jsonl`, `data/manifests/passage_roles.json`, `data/manifests/split_manifest.json` |
| `validate_data.py` | nothing; prints PASS or FAIL for each data check |
| `run_retrieval.py` | `cache/passage_embeddings.npy`, `cache/passage_ids.json`, `cache/cache_metadata.json` (if not cached yet); `runs/retrieval_<time>/retrieval_results.jsonl`, `run_config.json`, `retrieval_summary.json` |
| `run_baseline.py` | `runs/baseline_<time>/baseline_results.jsonl`, `timing.csv`, `run_config.json`, `prompt_template.txt`, `run_summary.json` |
| `evaluate.py` | in the run folder: `scored_results.jsonl`, `baseline_summary.csv`, `manual_review.csv`, `trace_Q001.json` |
| `retrieval_depth.py` | `runs/baseline_20261120_210939/support_hit_by_k.csv` |

## Models
- Retriever: [facebook/contriever](https://huggingface.co/facebook/contriever), licence CC BY-NC 4.0,
  revision `2bd46a25019aeea091fd42d1f0fd4801675cf699`
- Generator: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), licence Apache 2.0,
  revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Retrieval: masked mean pooling, max 256 tokens, dot-product scores, top 10.
- Generation: greedy decoding, max 64 new tokens.

## Prompt versions
- `pilot_v1`: first evidence prompt; it abstained on 17 / 18 questions, and the oracle diagnostic showed the prompt caused this.
- `pilot_v2`: narrower abstain rule plus an example answer format; 2 / 18 abstentions, but 6 answers copied the example answer "Room 4".
- `pilot_v3` (final, set in `config.json`): removed the example and asks for a short answer plus the passage ID.

## Runs
| Purpose | Run IDs |
|---|---|
| v1 baseline | `baseline_20261120_202847`, repeat `baseline_20261120_203318` |
| v2 baseline | `baseline_20261120_205655`, repeat `baseline_20261120_210040` |
| v3 baseline (final) | `baseline_20261120_210939`, repeat `baseline_20261120_211325` |
| Oracle-context diagnostic (not a metric) | `diagnostic_oracle_20261120_205614` |
| Retrieval only | `retrieval_20261120_013439` (cache build), `retrieval_20261120_013449` (cache load) |

The final results are in `runs/baseline_20261120_210939/`, including
`manual_review_decisions.csv` (the student's manual review) and `final_metrics.csv`.

## Folder layout
```
src/             Python code
data/clean/      clean passages and questions
data/manifests/  split manifest and passage roles
data/attacks/    poisoned (attack) passages (later tasks)
cache/           embedding cache (not in git)
runs/            one subfolder per run
reports/         reports (not in git)
references/      papers (not in git)
```

## Data
Data: fictional Asterford University corpus, see `data/DATA_CARD.md`.

## Reused code
Reused code: smoke test and encoding adapted from the facebook/contriever and
Qwen2.5-1.5B-Instruct model card examples.

## AI assistance
AI assistance: Claude (Anthropic) was used for planning, explanations, code drafting and drafting
the fictional dataset. The student set the rules, reviewed the outputs and
ran all experiments.
