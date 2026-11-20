# run_baseline.py
# Task 2: clean RAG baseline (B0) on the DEVELOPMENT questions.
# For each question: retrieve top 10 -> assemble context -> generate -> log.

import csv
import json
import os
import time
from datetime import datetime

import torch

from embeddings import load_config, load_retriever, build_or_load_cache, load_passages
from retrieval import retrieve
from generation import (get_template, load_generator, assemble_context,
                        generate_answer, is_abstention)

QUERIES_FILE = "data/clean/queries.jsonl"

# Step 1: fixed seed, run ID and a new run folder.
torch.manual_seed(42)
run_start = time.time()
now = datetime.now()
run_id = "baseline_" + now.strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_id)
os.makedirs(run_dir)
print("Run ID:", run_id)

# Step 2: load settings, the retriever and the passage cache (timed).
config = load_config()
print("\nLoading retriever and cache...")
start = time.time()
ret_tokenizer, ret_model = load_retriever(config)
E, ids, meta, cache_mode, cache_time = build_or_load_cache(config, ret_tokenizer, ret_model)
retriever_load_s = time.time() - start

# Step 3: load the generator (timed separately).
print("Loading generator...")
start = time.time()
gen_tokenizer, gen_model = load_generator(config)
generator_load_s = time.time() - start
print("   retriever + cache:", round(retriever_load_s, 2), "s, generator:",
      round(generator_load_s, 2), "s")

# Step 4: passage texts by ID (for the evidence lines).
passages_by_id = {}
for p in load_passages():
    passages_by_id[p["passage_id"]] = {"passage_id": p["passage_id"], "text": p["text"]}

# Step 5: keep ONLY query_id, question and split of development questions.
# "answers" and "support_ids" are dropped right away and never used.
dev_queries = []
with open(QUERIES_FILE) as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            if row["split"] == "development":
                dev_queries.append({"query_id": row["query_id"],
                                    "question": row["question"],
                                    "split": row["split"]})
print("\nDevelopment questions:", len(dev_queries), "\n")

# Step 6: run every question.
results = []
timing_rows = []
for query in dev_queries:
    # Retrieval.
    start = time.time()
    hits = retrieve(query["question"], E, ids, ret_tokenizer, ret_model,
                    config["retriever_max_tokens"], config["top_k"])
    retrieval_s = time.time() - start

    # Context assembly.
    start = time.time()
    context_ids, omitted_ids, prompt = assemble_context(
        query["question"], hits, passages_by_id, gen_tokenizer, config)
    context_s = time.time() - start

    # Generation (skip Qwen if no passage fits).
    start = time.time()
    if len(context_ids) == 0:
        output_text = "ABSTAIN"
        input_tokens = 0
        new_tokens = 0
        abstain_reason = "empty_context"
    else:
        output_text, input_tokens, new_tokens = generate_answer(
            prompt, gen_tokenizer, gen_model, config)
        abstain_reason = "model" if is_abstention(output_text) else None
    generation_s = time.time() - start
    abstained = abstain_reason is not None

    results.append({
        "run_id": run_id,
        "query_id": query["query_id"],
        "split": query["split"],
        "corpus_condition": "clean",
        "corpus_version": meta["corpus_version"],
        "retriever": config["retriever"],
        "retriever_revision": config["retriever_revision"],
        "generator": config["generator"],
        "generator_revision": config["generator_revision"],
        "prompt_version": config["prompt_version"],
        "ranked_ids": [hit["passage_id"] for hit in hits],
        "ranked_scores": [hit["score"] for hit in hits],
        "context_ids": context_ids,
        "omitted_ids": omitted_ids,
        "prompt_text": prompt,
        "input_tokens": input_tokens,
        "output_text": output_text,
        "new_tokens": new_tokens,
        "abstained": abstained,
        "abstain_reason": abstain_reason,
        "retrieval_s": round(retrieval_s, 4),
        "context_s": round(context_s, 4),
        "generation_s": round(generation_s, 4),
        "device": config["device"],
    })
    timing_rows.append([query["query_id"], round(retrieval_s, 4), round(context_s, 4),
                        round(generation_s, 4),
                        round(retrieval_s + context_s + generation_s, 4),
                        input_tokens, new_tokens])

    # One line per question: ID, abstained, first 80 characters.
    short = output_text.replace("\n", " ")[:80]
    print(query["query_id"], "| abstained:", "yes" if abstained else "no", "|", short)

# Step 7: write the results, one JSON line per question.
with open(os.path.join(run_dir, "baseline_results.jsonl"), "w") as f:
    for row in results:
        f.write(json.dumps(row) + "\n")

# Step 8: write the timing table.
with open(os.path.join(run_dir, "timing.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "retrieval_s", "context_s", "generation_s",
                     "total_s", "input_tokens", "new_tokens"])
    writer.writerows(timing_rows)

# Step 9: save the settings and the exact prompt template.
run_config = dict(config)
run_config["run_id"] = run_id
run_config["date"] = now.isoformat(timespec="seconds")
with open(os.path.join(run_dir, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)
with open(os.path.join(run_dir, "prompt_template.txt"), "w") as f:
    f.write(get_template(config["prompt_version"]))

# Step 10: save the summary.
gen_times = [row["generation_s"] for row in results]
summary = {
    "run_id": run_id,
    "retriever_and_cache_load_s": round(retriever_load_s, 2),
    "cache_mode": cache_mode,
    "generator_load_s": round(generator_load_s, 2),
    "num_questions": len(results),
    "num_abstentions": sum(1 for row in results if row["abstained"]),
    "mean_generation_s": round(sum(gen_times) / len(gen_times), 3),
    "max_generation_s": round(max(gen_times), 3),
    "total_run_s": round(time.time() - run_start, 2),
}
with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSummary:", json.dumps(summary, indent=2))
print("Saved results in", run_dir)
