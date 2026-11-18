# validate_data.py
# Task 2: check the built dataset files before they are used.
# Prints PASS or FAIL for each check and exits with code 1 if any check fails.
# No model is loaded here.

import hashlib
import json
import re
import sys

PASSAGES_FILE = "data/clean/passages.jsonl"
QUERIES_FILE = "data/clean/queries.jsonl"
MANIFEST_FILE = "data/manifests/split_manifest.json"

PASSAGE_FIELDS = {"passage_id", "text", "source", "corpus_version"}
QUERY_FIELDS = {"query_id", "group_id", "question", "answers", "support_ids", "split"}

failures = 0  # how many checks failed


# Print PASS/FAIL for one check, and up to 5 example problems if it failed.
def report(name, problems):
    global failures
    if len(problems) == 0:
        print("PASS", name)
    else:
        failures += 1
        print("FAIL", name, "(" + str(len(problems)) + " problems)")
        for problem in problems[:5]:
            print("     -", problem)


# Read a JSON Lines file. Returns the rows and a list of bad lines.
def read_jsonl(path):
    rows = []
    bad = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad.append(path + " line " + str(line_number) + " is not valid JSON")
    return rows, bad


# True if the answer appears in the text as whole words (case-insensitive).
# (?<!\w) and (?!\w) mean "no letter or digit right before / after".
def contains_whole(text, answer):
    pattern = r"(?<!\w)" + re.escape(answer.lower()) + r"(?!\w)"
    return re.search(pattern, text.lower()) is not None


# Lowercase and remove punctuation, for the duplicate checks.
def normalise(text):
    return re.sub(r"[^\w\s]", "", text.lower())


# ---------------------------------------------------------------
# Check 1: every line is valid JSON with the exact fields
# ---------------------------------------------------------------
passages, bad_p = read_jsonl(PASSAGES_FILE)
queries, bad_q = read_jsonl(QUERIES_FILE)
report("every line is valid JSON", bad_p + bad_q)

problems = []
for p in passages:
    if set(p.keys()) != PASSAGE_FIELDS:
        problems.append("passage fields wrong: " + str(sorted(p.keys())))
for q in queries:
    if set(q.keys()) != QUERY_FIELDS:
        problems.append("query fields wrong: " + str(sorted(q.keys())))
report("every line has exactly the expected fields", problems)

# ---------------------------------------------------------------
# Check 2: unique IDs and the right counts
# ---------------------------------------------------------------
passage_ids = [p["passage_id"] for p in passages]
query_ids = [q["query_id"] for q in queries]

problems = []
if len(set(passage_ids)) != len(passage_ids):
    problems.append("duplicate passage IDs")
if len(set(query_ids)) != len(query_ids):
    problems.append("duplicate query IDs")
report("passage IDs and query IDs are unique", problems)

problems = []
if len(passages) != 200:
    problems.append("expected 200 passages, found " + str(len(passages)))
if len(queries) != 30:
    problems.append("expected 30 queries, found " + str(len(queries)))
report("200 passages and 30 queries", problems)

# ---------------------------------------------------------------
# Check 3: no empty text, every passage is 40 to 100 words
# ---------------------------------------------------------------
problems = []
for p in passages:
    if p["text"].strip() == "":
        problems.append(p["passage_id"] + " has empty text")
for q in queries:
    if q["question"].strip() == "":
        problems.append(q["query_id"] + " has an empty question")
report("no empty text", problems)

problems = []
for p in passages:
    n_words = len(p["text"].split())
    if n_words < 40 or n_words > 100:
        problems.append(p["passage_id"] + " has " + str(n_words) + " words")
report("every passage is 40 to 100 words", problems)

# ---------------------------------------------------------------
# Check 4: support IDs exist and every query has an answer
# ---------------------------------------------------------------
text_of = {}
for p in passages:
    text_of[p["passage_id"]] = p["text"]

problems = []
for q in queries:
    if len(q["support_ids"]) == 0:
        problems.append(q["query_id"] + " has no support IDs")
    for sid in q["support_ids"]:
        if sid not in text_of:
            problems.append(q["query_id"] + " support ID " + sid + " does not exist")
report("every support_id exists", problems)

problems = []
for q in queries:
    if len(q["answers"]) == 0:
        problems.append(q["query_id"] + " has no answers")
report("every query has at least one answer", problems)

# ---------------------------------------------------------------
# Check 5: each supporting passage contains one of its answers
# ---------------------------------------------------------------
problems = []
for q in queries:
    for sid in q["support_ids"]:
        text = text_of.get(sid, "").lower()
        found = False
        for answer in q["answers"]:
            if answer.lower() in text:
                found = True
        if not found:
            problems.append(sid + " does not contain an answer of " + q["query_id"])
report("each supporting passage contains one of its answers", problems)

# ---------------------------------------------------------------
# Check 6: no OTHER passage contains any accepted answer of any question
# ---------------------------------------------------------------
problems = []
for q in queries:
    for p in passages:
        if p["passage_id"] in q["support_ids"]:
            continue  # its own supporting passage is allowed
        for answer in q["answers"]:
            if contains_whole(p["text"], answer):
                problems.append(p["passage_id"] + " contains '" + answer + "' (answer of " + q["query_id"] + ")")
report("no other passage contains any accepted answer", problems)

# ---------------------------------------------------------------
# Check 7: no duplicate or near-duplicate passages
# ---------------------------------------------------------------
problems = []
seen = {}
for p in passages:
    key = normalise(p["text"])
    if key in seen:
        problems.append(p["passage_id"] + " duplicates " + seen[key])
    seen[key] = p["passage_id"]
report("no duplicate texts", problems)

problems = []
word_sets = [set(normalise(p["text"]).split()) for p in passages]
highest = 0.0
for i in range(len(passages)):
    for j in range(i + 1, len(passages)):
        a = word_sets[i]
        b = word_sets[j]
        jaccard = len(a & b) / len(a | b)
        highest = max(highest, jaccard)
        if jaccard > 0.8:
            problems.append(passage_ids[i] + " and " + passage_ids[j] + " overlap " + str(round(jaccard, 2)))
report("no passage pair with word overlap (Jaccard) above 0.8 (highest = " + str(round(highest, 2)) + ")", problems)

# ---------------------------------------------------------------
# Check 8: split by group, counts 18 / 6 / 6
# ---------------------------------------------------------------
problems = []
splits_of_group = {}
for q in queries:
    if q["split"] not in ("development", "validation", "test"):
        problems.append(q["query_id"] + " has unknown split " + q["split"])
    splits_of_group.setdefault(q["group_id"], set()).add(q["split"])
for group_id in splits_of_group:
    if len(splits_of_group[group_id]) != 1:
        problems.append(group_id + " is in more than one split")
report("all questions in a group have the same split", problems)

counts = {"development": 0, "validation": 0, "test": 0}
for q in queries:
    if q["split"] in counts:
        counts[q["split"]] += 1
problems = []
if counts != {"development": 18, "validation": 6, "test": 6}:
    problems.append("split counts are " + str(counts))
report("split counts are 18 / 6 / 6", problems)

# ---------------------------------------------------------------
# Check 9: the locked manifest matches the data files
# ---------------------------------------------------------------
with open(MANIFEST_FILE) as f:
    manifest = json.load(f)

problems = []
for name, path in [("passages.jsonl", PASSAGES_FILE), ("queries.jsonl", QUERIES_FILE)]:
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if manifest["sha256"][name] != digest:
        problems.append(name + " hash does not match the manifest")
if manifest.get("locked") is not True:
    problems.append("manifest is not locked")
report("split manifest is locked and its hashes match", problems)

# ---------------------------------------------------------------
# Final result
# ---------------------------------------------------------------
print()
if failures == 0:
    print("ALL CHECKS PASSED")
    sys.exit(0)
else:
    print(str(failures) + " CHECK(S) FAILED")
    sys.exit(1)
