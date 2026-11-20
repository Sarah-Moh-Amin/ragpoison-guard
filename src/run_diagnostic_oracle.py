# run_diagnostic_oracle.py
# Task 2 DIAGNOSTIC (not a baseline run): give Qwen ONLY the support passage
# of three development questions, to see if pilot_v1 still abstains.
# This script reads support_ids on purpose. Its results are NOT metrics.

import json
import os
from datetime import datetime

import torch

from embeddings import load_config, load_passages
from generation import load_generator, build_prompt, generate_answer, is_abstention

QUERIES_FILE = "data/clean/queries.jsonl"
DIAGNOSTIC_IDS = ["Q001", "Q013", "Q019"]

# Step 1: fixed seed, run ID and a new run folder.
torch.manual_seed(42)
run_id = "diagnostic_oracle_" + datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join("runs", run_id)
os.makedirs(run_dir)
print("Run ID:", run_id)

# Step 2: settings, generator and passage texts.
config = load_config()
print("Prompt version:", config["prompt_version"])
tokenizer, model = load_generator(config)
passages_by_id = {}
for p in load_passages():
    passages_by_id[p["passage_id"]] = {"passage_id": p["passage_id"], "text": p["text"]}

# Step 3: read the three development questions WITH their support IDs.
queries = []
with open(QUERIES_FILE) as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            if row["query_id"] in DIAGNOSTIC_IDS and row["split"] == "development":
                queries.append(row)

# Step 4: context = only the support passage(s), then generate.
results = []
for query in queries:
    context = [passages_by_id[pid] for pid in query["support_ids"]]
    prompt = build_prompt(query["question"], context)
    output_text, input_tokens, new_tokens = generate_answer(prompt, tokenizer, model, config)
    results.append({
        "query_id": query["query_id"],
        "context_ids": query["support_ids"],
        "prompt_text": prompt,
        "output_text": output_text,
        "abstained": is_abstention(output_text),
    })
    print(query["query_id"], "| abstained:", is_abstention(output_text), "|",
          output_text.replace("\n", " ")[:80])

# Step 5: save results and the README.
with open(os.path.join(run_dir, "diagnostic_results.jsonl"), "w") as f:
    for row in results:
        f.write(json.dumps(row) + "\n")
with open(os.path.join(run_dir, "README.txt"), "w") as f:
    f.write("Oracle-context diagnostic. Uses support labels. Not part of the baseline or its metrics.\n")

print("Saved results in", run_dir)
