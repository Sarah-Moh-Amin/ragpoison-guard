# Data card: Asterford University fictional pilot corpus (clean_v1)

## What it is
A small question-answering corpus about **Asterford University**, a fictional
university and its town. **All facts are invented**: no real universities,
people, companies or events. Built for Task 2 (clean RAG baseline) of RAGPoison Guard.

## How it was made
The passages and questions were written to rules set by Sarah Helal and
reviewed by her. The readable source of truth is
`data/clean/draft_dataset.json`. `src/build_dataset.py` turns it into the data
files, and `src/validate_data.py` checks them (answer leakage, word counts,
duplicates, split rules).

## Contents
| Item | Count |
|---|---|
| Passages (40 to 100 words each, English) | 200 |
| - supporting (one per question, states the answer) | 30 |
| - distractors (about the question entities, no answer) | 50 |
| - background (other aspects of the university and town) | 120 |
| Questions (2 per group, short answers) | 30 |
| Groups (one campus entity each, G01 to G15) | 15 |

Passage roles are stored only in `data/manifests/passage_roles.json`
(evaluator metadata, never a model input). Gold answers and support IDs are
only in `queries.jsonl` and are never put in the prompt.

## Split
- Rule: split **by group**, so both questions about one entity are in the same split.
- Seed 42. G01 (Aster Archive) is always development; the other 14 groups are
  shuffled: 9 groups development, 3 validation, 3 test (18 / 6 / 6 questions).
- The split is locked in `data/manifests/split_manifest.json`, with the
  sha256 of `passages.jsonl` and `queries.jsonl`.
- Task 2 uses only the development questions. Test outputs are never inspected.

## IDs
Passage IDs P001 to P200 (assigned after a seeded shuffle) and query IDs
Q001 to Q030 (in group order). IDs are never reused: if the corpus changes,
it gets a new `corpus_version`.
