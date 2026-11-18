# build_dataset.py
# Task 2: build the fictional pilot dataset from the draft file.
# Reads data/clean/draft_dataset.json and writes:
#   data/clean/passages.jsonl          (model input: passage ID + text)
#   data/clean/queries.jsonl           (questions, gold answers, support IDs, split)
#   data/manifests/passage_roles.json  (evaluator metadata only, never a model input)
#   data/manifests/split_manifest.json (locked split + file hashes)
# No model is loaded here.

import hashlib
import json
import random

SEED = 42
DRAFT_FILE = "data/clean/draft_dataset.json"
PASSAGES_FILE = "data/clean/passages.jsonl"
QUERIES_FILE = "data/clean/queries.jsonl"
ROLES_FILE = "data/manifests/passage_roles.json"
MANIFEST_FILE = "data/manifests/split_manifest.json"


# Small helper: the sha256 fingerprint of a file.
def sha256_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# Small helper: write a list of dictionaries as JSON Lines (one per line).
def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------
# Step 1: read the draft (the readable source of truth)
# ---------------------------------------------------------------
with open(DRAFT_FILE, encoding="utf-8") as f:
    draft = json.load(f)

groups = draft["groups"]
draft_passages = draft["passages"]
corpus_version = draft["corpus_version"]
print("Read", len(groups), "groups and", len(draft_passages), "passages from the draft")

# ---------------------------------------------------------------
# Step 2: shuffle the passages and give them IDs P001 to P200
# (shuffling means the ID tells nothing about the passage role)
# ---------------------------------------------------------------
random.seed(SEED)
random.shuffle(draft_passages)

passages = []    # rows for passages.jsonl
roles = {}       # passage_id -> role and group (evaluator only)
for i, p in enumerate(draft_passages):
    passage_id = "P" + str(i + 1).zfill(3)
    p["passage_id"] = passage_id  # remember the ID for the support lookup below

    passages.append({
        "passage_id": passage_id,
        "text": p["text"],
        "source": "fictional_pilot",
        "corpus_version": corpus_version,
    })
    roles[passage_id] = {
        "role": p["role"],
        "group_id": p.get("group"),     # None for background passages
        "question_key": p.get("question"),  # only for supporting passages
    }

# ---------------------------------------------------------------
# Step 3: split BY GROUP (9 development, 3 validation, 3 test)
# G01 is always development; the other 14 groups are shuffled.
# ---------------------------------------------------------------
random.seed(SEED)
other_groups = [g["group_id"] for g in groups if g["group_id"] != "G01"]
random.shuffle(other_groups)

dev_groups = sorted(["G01"] + other_groups[0:8])
val_groups = sorted(other_groups[8:11])
test_groups = sorted(other_groups[11:14])

split_of_group = {}
for g in dev_groups:
    split_of_group[g] = "development"
for g in val_groups:
    split_of_group[g] = "validation"
for g in test_groups:
    split_of_group[g] = "test"

# ---------------------------------------------------------------
# Step 4: build the queries Q001 to Q030 in group order
# ---------------------------------------------------------------
queries = []
query_id_of_key = {}  # "G01-1" -> "Q001"
number = 1
for g in groups:
    for q in g["questions"]:
        query_id = "Q" + str(number).zfill(3)
        query_id_of_key[q["key"]] = query_id

        # The supporting passages are the ones that point at this question.
        support_ids = []
        for p in draft_passages:
            if p["role"] == "supporting" and p.get("question") == q["key"]:
                support_ids.append(p["passage_id"])

        queries.append({
            "query_id": query_id,
            "group_id": g["group_id"],
            "question": q["question"],
            "answers": q["answers"],
            "support_ids": sorted(support_ids),
            "split": split_of_group[g["group_id"]],
        })
        number += 1

# Add the query ID to the role file (easier to join than the draft key).
for passage_id in roles:
    key = roles[passage_id].pop("question_key")
    roles[passage_id]["query_id"] = query_id_of_key.get(key)  # None if not supporting

# ---------------------------------------------------------------
# Step 5: write the data files (passages sorted by ID)
# ---------------------------------------------------------------
write_jsonl(PASSAGES_FILE, passages)
write_jsonl(QUERIES_FILE, queries)

with open(ROLES_FILE, "w", encoding="utf-8") as f:
    json.dump(roles, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------
# Step 6: write the split manifest (with file hashes, locked)
# ---------------------------------------------------------------
def query_ids_for(split_name):
    return [q["query_id"] for q in queries if q["split"] == split_name]


role_counts = {}
for passage_id in roles:
    role = roles[passage_id]["role"]
    role_counts[role] = role_counts.get(role, 0) + 1

manifest = {
    "seed": SEED,
    "rule": "split by group",
    "note": "G01 is always development; the other 14 groups are shuffled with the seed: 8 development, 3 validation, 3 test",
    "corpus_version": corpus_version,
    "groups": {
        "development": dev_groups,
        "validation": val_groups,
        "test": test_groups,
    },
    "query_ids": {
        "development": query_ids_for("development"),
        "validation": query_ids_for("validation"),
        "test": query_ids_for("test"),
    },
    "counts": {
        "passages": len(passages),
        "queries": len(queries),
        "passages_by_role": role_counts,
        "queries_by_split": {
            "development": len(query_ids_for("development")),
            "validation": len(query_ids_for("validation")),
            "test": len(query_ids_for("test")),
        },
    },
    "sha256": {
        "passages.jsonl": sha256_of_file(PASSAGES_FILE),
        "queries.jsonl": sha256_of_file(QUERIES_FILE),
    },
    "locked": True,
}

with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

# ---------------------------------------------------------------
# Step 7: print a short summary
# ---------------------------------------------------------------
print("Wrote", len(passages), "passages to", PASSAGES_FILE)
print("Wrote", len(queries), "queries to", QUERIES_FILE)
print("Passages by role:", role_counts)
print("Development groups:", dev_groups)
print("Validation groups: ", val_groups)
print("Test groups:       ", test_groups)
print("Wrote", ROLES_FILE, "and", MANIFEST_FILE)
