# run_retrieval.py
# Task 2: retrieve the top 10 passages for every DEVELOPMENT question,
# save the results in a new run folder, then compute support hit at 10.

import json
import os
import time
from datetime import datetime

from embeddings import load_config, load_retriever, build_or_load_cache, check_cache
from retrieval import retrieve

QUERIES_FILE = "data/clean/queries.jsonl"

# Step 1: make a run ID and a new run folder.
now = datetime.now()
run_id = "retrieval_" + now.strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_id)
os.makedirs(run_dir)
print("Run ID:", run_id)

# Step 2: load settings, model and the passage cache.
config = load_config()
print("\nLoading retriever...")
tokenizer, model = load_retriever(config)

print("\nBuilding or loading the passage cache...")
E, ids, meta, cache_mode, cache_time = build_or_load_cache(config, tokenizer, model)

print("\nChecking the cache...")
check_cache(E, ids, tokenizer, model, config)

# Step 3: keep ONLY the development questions.
dev_queries = []
with open(QUERIES_FILE) as f:
    for line in f:
        if line.strip():
            query = json.loads(line)
            if query["split"] == "development":
                dev_queries.append(query)
print("\nDevelopment questions:", len(dev_queries))

# Step 4: retrieve the top 10 for each question and time it.
# Only the question text goes into retrieve(); no answers or support IDs.
max_tokens = config["retriever_max_tokens"]
top_k = config["top_k"]
results = {}
times = []
lines = []
for query in dev_queries:
    start = time.time()
    hits = retrieve(query["question"], E, ids, tokenizer, model, max_tokens, top_k)
    times.append(time.time() - start)
    results[query["query_id"]] = hits
    for hit in hits:
        lines.append({
            "run_id": run_id,
            "query_id": query["query_id"],
            "split": query["split"],
            "corpus_version": meta["corpus_version"],
            "rank": hit["rank"],
            "passage_id": hit["passage_id"],
            "score": hit["score"],
        })

with open(os.path.join(run_dir, "retrieval_results.jsonl"), "w") as f:
    for row in lines:
        f.write(json.dumps(row) + "\n")

# Step 5: save a copy of the settings for this run.
run_config = dict(config)
run_config["run_id"] = run_id
run_config["date"] = now.isoformat(timespec="seconds")
with open(os.path.join(run_dir, "run_config.json"), "w") as f:
    json.dump(run_config, f, indent=2)

# Step 6: AFTER retrieval, use support_ids to compute support hit at 10.
hit_count = 0
misses = []
for query in dev_queries:
    top_ids = [hit["passage_id"] for hit in results[query["query_id"]]]
    if any(pid in top_ids for pid in query["support_ids"]):
        hit_count += 1
    else:
        # Rank all 200 passages to see where the support passage really is.
        full = retrieve(query["question"], E, ids, tokenizer, model, max_tokens, len(ids))
        full_ranks = {hit["passage_id"]: hit["rank"] for hit in full}
        misses.append({
            "query_id": query["query_id"],
            "question": query["question"],
            "support_ids": query["support_ids"],
            "support_ranks_in_top_200": [full_ranks.get(pid) for pid in query["support_ids"]],
        })

total = len(dev_queries)
rate = hit_count / total
print("\nSupport hit at 10:", hit_count, "/", total, "(" + str(round(100 * rate, 1)) + "%)")
if misses:
    print("Misses:")
    for miss in misses:
        print("  ", miss["query_id"], miss["support_ids"],
              "ranks", miss["support_ranks_in_top_200"], "|", miss["question"])
else:
    print("Misses: none")

mean_time = sum(times) / len(times)
max_time = max(times)
print("Retrieval time per question: mean", round(mean_time, 4), "s, max", round(max_time, 4), "s")

# Step 7: save the summary.
summary = {
    "run_id": run_id,
    "support_hit_at_10": hit_count,
    "total": total,
    "rate": rate,
    "misses": misses,
    "cache_mode": cache_mode,
    "cache_time_seconds": round(cache_time, 3),
    "mean_retrieval_seconds": round(mean_time, 4),
    "max_retrieval_seconds": round(max_time, 4),
}
with open(os.path.join(run_dir, "retrieval_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved results in", run_dir)
