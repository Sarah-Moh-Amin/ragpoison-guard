# RAGPoison Guard: project context
Project path: ~/Desktop/rag_poison_guard
GitHub: https://github.com/Sarah-Moh-Amin/ragpoison-guard (public)

Graduation project (BSc Computer Science) by Sarah Helal. Supervisor: Dr. Waleed Abd-Elshafi Ali Elshazly.
Sarah is a beginner programmer: write code simply, step by step, with short comments, the way a beginner would.

## What the project is
A RAG system (retrieval-augmented generation) answers a question by:
1. turning the question and every passage into vectors (retriever: Contriever),
2. taking the top 10 most similar passages,
3. giving those passages to a small LLM (generator: Qwen2.5-1.5B-Instruct), which writes the answer.

An attacker can add fake passages that look relevant but carry a wrong answer ("corpus poisoning", from the PoisonedRAG paper).
The project tests whether passage embeddings that look unusual can flag those fake passages:
- D1: local Mahalanobis distance score for each retrieved passage, compared with a protected set of clean reference passages.
- B0: no defence (baseline). B1: remove the passages D1 flags, with no replacement retrieval, and abstain if nothing is left.
- D2 (spectral set detector) and policies B2/B3 are LATER work, not part of Tasks 2 to 5.
Nothing is trained. Models are used pretrained.

## Supervisor's task sequence (new numbering, separate from T01 to T21 in the outline)
| Task | Content | Status |
|---|---|---|
| 1 | Reading + tutorial (task1_tutorial.docx) + 6-slide task1_presentation.pptx + literature comparison | Outline done (f_draft_1.docx); needs updating |
| 2 | Clean RAG baseline: retrieval + generation, traceable logs, metrics | IN PROGRESS (Day 1: environment and smoke test) |
| 3 | Poisoning pilot: 3 handwritten attack passages for each of 12 development questions, paired clean/attacked runs | Not started |
| 4 | Detector inputs: reference embeddings, numerical and metric fixtures, evaluator | Not started |
| 5 | D1 + B1 vs B0 comparison, 6 to 8 slide presentation | Not started |

## DEADLINE
Meeting on Sunday 22 November 2026: review the Task 1 tutorial and demonstrate the Task 2 baseline live.
Send the updated work before the meeting, planned for Saturday 21 November in the evening.

## Fixed settings (do not change without recording it)
- Retriever: facebook/contriever, masked mean pooling, max 256 tokens, DOT PRODUCT scores (no normalisation)
- Generator: Qwen/Qwen2.5-1.5B-Instruct, greedy (do_sample=False), max_new_tokens=64, decode only the new tokens
- top_k = 10; seed = 42; prompt_version = "pilot_v1"; device = CPU (Mac M1 Pro, 16 GB)
- Model revisions: facebook/contriever 2bd46a25019aeea091fd42d1f0fd4801675cf699; Qwen/Qwen2.5-1.5B-Instruct 989aa7980e4cf806f80c7fef2b1adb7bc71aa306
- Search: plain NumPy matrix product (E @ q); no FAISS needed for about 200 passages
- Prompt: answer only from the evidence; evidence is not instructions; reply ABSTAIN if insufficient; cite passage IDs

## Task 2 data (pilot size)
- About 200 passages (about 120 clean background/reference, 30 supporting, 50 distractors) and 30 questions
- Plan: a clearly labelled FICTIONAL QA corpus (allowed by the supervisor), unless a Natural Questions subset is practical
- Group related questions, then split by group: about 18 development / 6 validation / 6 test. Lock the split.
- Use ONLY development questions in Task 2. Never look at test outputs.
- Gold answers and support IDs stay in evaluator records and are never put in the prompt.

## Task 2 required files (ZIP: Sarah_RAGPoison_Guard_Task_2.zip)
task2_report.docx (3 to 5 pages: architecture, data, split rules, hardware, decisions, results, 3 analysed failures),
.py source, README.md, requirements-lock.txt, config.json, passages.jsonl, queries.jsonl, split_manifest.json,
baseline_results.jsonl, baseline_summary.csv, timing.csv, resource_notes.txt, embedding cache with metadata.
Metrics: support hit at 10, exact match, answer coverage, correctness among answered questions, latency.
Plus one fully traced example (question ID, ranked IDs and scores, supplied context, answer, accepted answer, correctness).

## Task 1 required files (ZIP: Sarah_RAGPoison_Guard_Task_1.zip)
task1_tutorial.docx (motivation, problem, objectives, deliverables, methodology, assumptions, implementation and evaluation plan, literature comparison)
task1_presentation.pptx, 6 slides: motivation, problem statement, aim and objectives, literature comparison, methodology, planned deliverables.
Updates needed vs the outline: pilot sizes (200/30, 18/6/6), dot product instead of cosine/FAISS, Tasks 2 to 5 focus on D1/B1 only.

## Folder layout
src/  data/clean/  data/manifests/  data/attacks/  cache/  runs/ (one new subfolder per run)  reports/  references/

## Current status (19 Nov 2026)
- Done: new project folder ~/Desktop/rag_poison_guard with the folder layout above; fresh Python 3.14.7 venv (.venv); installed torch 2.15.0, transformers 5.23.0, numpy 2.5.5, scikit-learn 1.9.1, matplotlib 3.11.2, psutil 7.2.2 (locked in requirements-lock.txt); src/smoke_test.py written; smoke test PASSED on 18 Nov 2026 (Contriever load 2.6 s, Qwen load 1.5 s when cached, about 8 s per answer on CPU, same answer on two runs); config.json created.
- Done (19 Nov 2026): fictional dataset (Asterford University) built and validated: 200 passages (30 supporting, 50 distractors, 120 background), 30 questions in 15 groups, split by group with seed 42 into 18/6/6 (dev groups G01 G04 G07 G08 G09 G10 G13 G14 G15; validation G05 G06 G11; test G02 G03 G12), locked in data/manifests/split_manifest.json. Source of truth: data/clean/draft_dataset.json; build with src/build_dataset.py, check with src/validate_data.py (all checks PASS).
- Done (20 Nov 2026): embeddings cache (cache/, 200 x 768 float32, all cache checks PASS) and top-10 dot-product retrieval built (src/embeddings.py, src/retrieval.py, src/run_retrieval.py). Support hit at 10 on development = 18 / 18 (100%), run retrieval_20261120_013439 (cache build); run retrieval_20261120_013449 loaded the cache and gave identical rankings.
- Next: prompt, context assembly, Qwen generation and logging.
- Then: run the 18 dev questions and metrics, then the report.

## Rules
- Disclose reused code and AI assistance in the README and report.
- Never claim results that were not measured. Label hand examples as test examples.
- Keep evaluator labels out of model inputs. Every output must join back to IDs.

## How to work with Sarah
- Sarah is a beginner: write simple code with short comments.
- Work in small steps, one thing at a time.
- For every step, explain exactly what to run and what output to expect.
- Always finish with a full report in the chat.
