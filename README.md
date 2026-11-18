# RAGPoison Guard

BSc graduation project: detecting poisoned passages in a small local RAG system using embedding anomaly scores.

## Status
Task 2: clean baseline in progress.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

## Smoke test
```bash
python src/smoke_test.py
```
It prints the hardware (OS, Python, chip, RAM, free disk, PyTorch version),
loads Contriever and prints the passage/question vector shapes (3, 768) and (1, 768)
and the dot-product retrieval scores, then loads Qwen and prints a short answer
and the generation time. The results are saved to `resource_notes.txt`.
The first run downloads about 3.4 GB of models.

## Folder layout
```
src/             Python code
data/clean/      clean passages and questions
data/manifests/  split manifests
data/attacks/    poisoned (attack) passages
cache/           embedding cache (not in git)
runs/            one subfolder per run
reports/         reports (not in git)
references/      papers (not in git)
```

## Models
- Retriever: [facebook/contriever](https://huggingface.co/facebook/contriever), licence CC BY-NC 4.0
- Generator: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), licence Apache 2.0

## Reused code
The smoke test is adapted from the Contriever and Qwen2.5 model card examples.

## AI assistance
Claude (Anthropic) was used for planning, explanations and code drafting; the student reviews and runs all code.
