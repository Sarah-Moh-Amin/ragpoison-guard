# evaluate.py
# Task 2: score one baseline run against the gold answers.
# This is the ONLY file allowed to read "answers" and "support_ids".
# Usage: python src/evaluate.py runs/<run_id>

import csv
import json
import os
import re
import statistics
import sys

QUERIES_FILE = "data/clean/queries.jsonl"
PASSAGES_FILE = "data/clean/passages.jsonl"
TRACE_QUERY = "Q001"
EXPECTED_ROWS = 18


def stop(message):
    # Print a clear message and stop the program.
    print("CHECK FAILED:", message)
    sys.exit(1)


def normalise(text):
    # Same rules for model answers and accepted answers.
    text = text.lower()
    # Remove passage IDs like [p123], (p123) or p123.
    text = re.sub(r"[\[\(]?\bp\d+\b[\]\)]?", " ", text)
    # Keep only letters, digits, ":" and "."; everything else becomes a space.
    text = re.sub(r"[^a-z0-9:.]", " ", text)
    # Remove a trailing ".".
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    # Remove the words "a", "an", "the" (split also collapses spaces).
    words = [w for w in text.split() if w not in ("a", "an", "the")]
    return " ".join(words)


def self_test():
    # Check normalise() on fixed examples; stop if any fails.
    tests = [("18:00 [P056]", "18:00"),
             ("Glaciology (P059)", "glaciology"),
             ("Room 4 [P000]", "room 4"),
             ("Cedar [P087]", "cedar"),
             ("Four days [P103]", "four days")]
    all_ok = True
    for text, expected in tests:
        got = normalise(text)
        ok = got == expected
        all_ok = all_ok and ok
        print("  ", "PASS" if ok else "FAIL", repr(text), "->", repr(got))
    if not all_ok:
        stop("normalise() self-test")


def contains_words(answer, accepted):
    # True if "accepted" appears as whole words inside "answer".
    return (" " + accepted + " ") in (" " + answer + " ")


def count(n, d):
    # Show a count with its denominator, for example "12/18".
    return str(n) + "/" + str(d)


# Step 1: which run to score.
if len(sys.argv) != 2:
    stop("usage: python src/evaluate.py runs/<run_id>")
run_dir = sys.argv[1].rstrip("/")
run_id = os.path.basename(run_dir)
print("Scoring run:", run_id)

print("\nnormalise() self-test:")
self_test()

# Step 2: load the run rows and the questions.
rows = []
with open(os.path.join(run_dir, "baseline_results.jsonl")) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
with open(os.path.join(run_dir, "run_summary.json")) as f:
    run_summary = json.load(f)
queries = {}
with open(QUERIES_FILE) as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            queries[q["query_id"]] = q

# Step 3: checks before scoring.
run_ids = [row["query_id"] for row in rows]
for qid in run_ids:
    if qid not in queries:
        stop(qid + " is not in " + QUERIES_FILE)
    if queries[qid]["split"] != "development":
        stop(qid + " is not a development question")
if len(set(run_ids)) != len(run_ids):
    stop("duplicate query IDs in the run")
if len(rows) != EXPECTED_ROWS:
    stop("expected " + str(EXPECTED_ROWS) + " rows, found " + str(len(rows)))
print("\nChecks: all", len(rows), "rows are unique development questions - PASS")

# Step 4: score every question.
scored = []
for row in rows:
    q = queries[row["query_id"]]
    support = q["support_ids"]
    accepted = [normalise(a) for a in q["answers"]]
    answer = normalise(row["output_text"])
    abstained = row["abstained"]

    # Abstained answers score 0.
    exact = (not abstained) and answer in accepted
    contains = (not abstained) and any(a and contains_words(answer, a) for a in accepted)

    # Rank (1 = top) of the best-ranked support passage, or None.
    support_rank = None
    for i, pid in enumerate(row["ranked_ids"]):
        if pid in support:
            support_rank = i + 1
            break
    in_context = any(pid in support for pid in row["context_ids"])

    # Passage IDs the model wrote (unique, in order).
    cited = []
    for pid in re.findall(r"\bP\d+\b", row["output_text"]):
        if pid not in cited:
            cited.append(pid)
    cited_support = any(pid in support for pid in cited)

    # Where did it go wrong?
    if support_rank is None:
        failure = "retrieval"
    elif not in_context:
        failure = "context"
    elif abstained or not contains:
        failure = "generation"
    else:
        failure = "none"

    scored.append({
        "query_id": row["query_id"],
        "output_text": row["output_text"],
        "normalised_answer": answer,
        "accepted_answers": q["answers"],
        "abstained": abstained,
        "exact_match": exact,
        "contains_match": contains,
        "support_rank": support_rank,
        "support_in_context": in_context,
        "cited_ids": cited,
        "cited_support": cited_support,
        "failure_type": failure,
    })

# Step 5: scored_results.jsonl.
with open(os.path.join(run_dir, "scored_results.jsonl"), "w") as f:
    for s in scored:
        f.write(json.dumps(s) + "\n")

# Step 6: baseline_summary.csv (counts with denominators).
n = len(scored)
answered = sum(1 for s in scored if not s["abstained"])
n_exact = sum(1 for s in scored if s["exact_match"])
n_contains = sum(1 for s in scored if s["contains_match"])
n_hit = sum(1 for s in scored if s["support_rank"] is not None)
gen_times = [row["generation_s"] for row in rows]
if answered == 0:
    correctness = "undefined"
else:
    correctness = count(n_contains, answered)
summary = {
    "run_id": run_id,
    "prompt_version": rows[0]["prompt_version"],
    "n": n,
    "support_hit_at_10": count(n_hit, n),
    "exact_match": count(n_exact, n),
    "contains_match": count(n_contains, n),
    "coverage": count(answered, n),
    "correctness_among_answered": correctness,
    "abstentions": count(n - answered, n),
    "mean_generation_s": round(statistics.mean(gen_times), 3),
    "median_generation_s": round(statistics.median(gen_times), 3),
    "max_generation_s": round(max(gen_times), 3),
    "generator_load_s": run_summary["generator_load_s"],
}
with open(os.path.join(run_dir, "baseline_summary.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(list(summary.keys()))
    writer.writerow(list(summary.values()))

# Step 7: manual_review.csv (answered, exact and contains disagree).
with open(os.path.join(run_dir, "manual_review.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["query_id", "question", "output_text", "normalised_answer",
                     "accepted_answers", "exact_match", "contains_match"])
    for s in scored:
        if not s["abstained"] and s["exact_match"] != s["contains_match"]:
            writer.writerow([s["query_id"], queries[s["query_id"]]["question"],
                             s["output_text"], s["normalised_answer"],
                             " | ".join(s["accepted_answers"]),
                             s["exact_match"], s["contains_match"]])

# Step 8: trace_Q001.json (one fully traced example).
passage_text = {}
with open(PASSAGES_FILE) as f:
    for line in f:
        if line.strip():
            p = json.loads(line)
            passage_text[p["passage_id"]] = p["text"]
row = [r for r in rows if r["query_id"] == TRACE_QUERY][0]
s = [x for x in scored if x["query_id"] == TRACE_QUERY][0]
# The context text exactly as supplied in the prompt (one line per passage).
context_text = "\n".join("[" + pid + "] " + passage_text[pid] for pid in row["context_ids"])
if context_text not in row["prompt_text"]:
    stop("trace context text does not match the logged prompt")
trace = {
    "run_id": run_id,
    "query_id": TRACE_QUERY,
    "question": queries[TRACE_QUERY]["question"],
    "ranked": [{"rank": i + 1, "passage_id": pid, "score": score}
               for i, (pid, score) in enumerate(zip(row["ranked_ids"], row["ranked_scores"]))],
    "context_ids": row["context_ids"],
    "context_text": context_text,
    "output_text": row["output_text"],
    "accepted_answers": s["accepted_answers"],
    "exact_match": s["exact_match"],
    "contains_match": s["contains_match"],
    "failure_type": s["failure_type"],
}
with open(os.path.join(run_dir, "trace_" + TRACE_QUERY + ".json"), "w") as f:
    json.dump(trace, f, indent=2)

# Step 9: print the results.
print("\nPer question:")
for s in scored:
    print(" ", s["query_id"], "| abstained:", s["abstained"], "| exact:", s["exact_match"],
          "| contains:", s["contains_match"], "| rank:", s["support_rank"],
          "| failure:", s["failure_type"])
print("\nSummary:")
for key, value in summary.items():
    print(" ", key + ":", value)
print("\nSaved scored_results.jsonl, baseline_summary.csv, manual_review.csv,",
      "trace_" + TRACE_QUERY + ".json in", run_dir)
