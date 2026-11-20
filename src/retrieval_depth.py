# retrieval_depth.py
# Task 2: support hit at k = 1, 3, 5, 10 for the final v3 baseline run.
# It only reads the logged support_rank; it does not rerun retrieval.
# Usage: python src/retrieval_depth.py

import csv
import json
import os
import sys

RUN_DIR = "runs/baseline_20261120_210939"
SCORED_FILE = os.path.join(RUN_DIR, "scored_results.jsonl")
OUT_FILE = os.path.join(RUN_DIR, "support_hit_by_k.csv")
K_VALUES = [1, 3, 5, 10]

# Step 1: read the scored results.
if not os.path.exists(SCORED_FILE):
    print("CHECK FAILED: missing", SCORED_FILE, "(run src/evaluate.py first)")
    sys.exit(1)
rows = []
with open(SCORED_FILE) as f:
    for line in f:
        rows.append(json.loads(line))
n = len(rows)
print("Read", n, "rows from", SCORED_FILE)

# Step 2: for each k, count hits and list the questions that would miss.
# support_rank is 1 for the top passage; None means not in the top 10.
table = []
for k in K_VALUES:
    misses = []
    for row in rows:
        rank = row["support_rank"]
        if rank is None or rank > k:
            misses.append(row["query_id"])
    hits = n - len(misses)
    table.append([k, str(hits) + "/" + str(n), " ".join(misses)])
    print("  k =", k, "hit", str(hits) + "/" + str(n), "misses:", " ".join(misses) or "none")

# Step 3: save the table.
with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "support_hit", "missed_query_ids"])
    writer.writerows(table)
print("Saved", OUT_FILE)
