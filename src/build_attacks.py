# build_attacks.py
# Task 3: turn the 12 hand-written attack drafts into real attack passages
# and an attack manifest, then build an embedding cache for them.
# Usage: python src/build_attacks.py

import json
import os
import re
import sys

from embeddings import (load_config, load_retriever, sha256_of_file,
                        load_passages)
from attack_cache import build_or_load_attack_cache, check_attack_cache

DRAFTS_FILE = "data/attacks/attack_drafts_dev.json"
PASSAGES_FILE = "data/attacks/attack_passages.jsonl"
MANIFEST_FILE = "data/attacks/attack_manifest.jsonl"
QUERIES_FILE = "data/clean/queries.jsonl"

EXPECTED_TARGETS = 12
PASSAGES_PER_TARGET = 3
MIN_WORDS = 41
MAX_WORDS = 59
MAX_JACCARD = 0.8


def stop(message):
    # Print a clear message and stop the program.
    print("FAIL:", message)
    sys.exit(1)


def load_drafts():
    with open(DRAFTS_FILE) as f:
        return json.load(f)


def load_queries():
    # All 30 queries, keyed by query_id.
    queries = {}
    with open(QUERIES_FILE) as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                queries[q["query_id"]] = q
    return queries


def word_count(text):
    return len(text.split())


def clean_for_match(text):
    # Lower-case, and turn any punctuation next to a word (commas, full
    # stops at the end of a sentence, brackets, ...) into a space, but
    # keep ":" and "." INSIDE a phrase like "21:00" or "p.m." untouched.
    text = text.lower()
    # A "." right before a space or at the end of the text ends a
    # sentence (or an abbreviation): treat it as a space, not as part
    # of the phrase. A "." with no space after it (like in "21.00" or
    # "p.m.the") stays, so times written with dots still match.
    text = re.sub(r"\.(?=\s|$)", " ", text)
    text = re.sub(r"[^a-z0-9: .]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_phrase(text, phrase):
    # Same trick as evaluate.py's contains_words(): pad both sides with
    # spaces so a match only counts between word boundaries.
    t = " " + clean_for_match(text) + " "
    p = " " + clean_for_match(phrase) + " "
    return p in t


def word_set(text):
    # Lower-case words, with punctuation stripped from the edges.
    words = set()
    for w in text.lower().split():
        w = w.strip(".,;:!?()[]\"'")
        if w:
            words.add(w)
    return words


def jaccard(set_a, set_b):
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def assign_ids(drafts):
    # P201, P202, P203 for the first target, P204-P206 for the second, ...
    passages = []
    next_num = 201
    for target in drafts:
        ids = []
        for draft in target["passages"]:
            pid = "P" + str(next_num)
            ids.append(pid)
            passages.append({
                "passage_id": pid,
                "text": draft["text"],
                "style": draft["style"],
                "target_query_id": target["target_query_id"],
            })
            next_num += 1
        target["injection_ids"] = ids
    return passages


def main():
    print("Step 1: build attack passages and manifest\n")

    drafts = load_drafts()
    queries = load_queries()
    clean_passages = load_passages()
    clean_ids = [p["passage_id"] for p in clean_passages]
    clean_texts = [p["text"] for p in clean_passages]

    # Check: 12 targets.
    ok = len(drafts) == EXPECTED_TARGETS
    print("PASS" if ok else "FAIL", "-", len(drafts), "targets (expected", EXPECTED_TARGETS, ")")
    if not ok:
        stop("wrong number of targets")

    # Check: 3 passages each.
    ok = all(len(t["passages"]) == PASSAGES_PER_TARGET for t in drafts)
    print("PASS" if ok else "FAIL", "- every target has", PASSAGES_PER_TARGET, "passages")
    if not ok:
        stop("a target does not have 3 passages")

    # Assign IDs P201-P236 in file order.
    attack_passages = assign_ids(drafts)
    attack_ids = [p["passage_id"] for p in attack_passages]

    # Check: 36 unique IDs, no clash with the 200 clean IDs.
    ok = len(set(attack_ids)) == len(attack_ids)
    print("PASS" if ok else "FAIL", "-", len(attack_ids), "attack IDs are unique")
    if not ok:
        stop("attack IDs are not unique")

    clash = set(attack_ids) & set(clean_ids)
    ok = len(clash) == 0
    print("PASS" if ok else "FAIL", "- no clash with the 200 clean IDs")
    if not ok:
        stop("attack IDs clash with clean IDs: " + str(clash))

    # Check: every target_query_id exists and is a development question.
    ok = True
    for target in drafts:
        qid = target["target_query_id"]
        if qid not in queries:
            ok = False
            print("  ", qid, "is not in", QUERIES_FILE)
        elif queries[qid]["split"] != "development":
            ok = False
            print("  ", qid, "is not a development question")
    print("PASS" if ok else "FAIL", "- every target_query_id is a known development question")
    if not ok:
        stop("a target_query_id is wrong")

    # Check: every passage is 41-59 words.
    ok = True
    for p in attack_passages:
        n = word_count(p["text"])
        if not (MIN_WORDS <= n <= MAX_WORDS):
            ok = False
            print("  ", p["passage_id"], "has", n, "words")
    print("PASS" if ok else "FAIL", "- every passage is", MIN_WORDS, "-", MAX_WORDS, "words")
    if not ok:
        stop("a passage is outside the word count range")

    # Check: every passage contains its target answer.
    ok = True
    for target in drafts:
        for pid, draft in zip(target["injection_ids"], target["passages"]):
            if not contains_phrase(draft["text"], target["target_answer"]):
                ok = False
                print("  ", pid, "does not contain target answer", target["target_answer"])
    print("PASS" if ok else "FAIL", "- every passage contains its target answer")
    if not ok:
        stop("a passage is missing its target answer")

    # Check: no passage contains any accepted answer of ANY question.
    all_accepted = []
    for q in queries.values():
        all_accepted.extend(q["answers"])
    ok = True
    for p in attack_passages:
        for accepted in all_accepted:
            if contains_phrase(p["text"], accepted):
                ok = False
                print("  ", p["passage_id"], "contains an accepted answer:", accepted)
    print("PASS" if ok else "FAIL", "- no passage contains any accepted answer of any question")
    if not ok:
        stop("a passage contains a real accepted answer")

    # Check: no target alias appears anywhere in the clean corpus.
    ok = True
    for target in drafts:
        for alias in target["target_aliases"]:
            for cid, ctext in zip(clean_ids, clean_texts):
                if contains_phrase(ctext, alias):
                    ok = False
                    print("  ", cid, "already contains alias", alias, "of", target["target_query_id"])
    print("PASS" if ok else "FAIL", "- no target alias appears in the clean corpus")
    if not ok:
        stop("a target alias already appears in the clean corpus")

    # Check: highest Jaccard word overlap with any clean passage is below 0.8.
    clean_word_sets = [word_set(t) for t in clean_texts]
    highest = 0.0
    highest_pair = None
    for p in attack_passages:
        a_words = word_set(p["text"])
        for cid, c_words in zip(clean_ids, clean_word_sets):
            score = jaccard(a_words, c_words)
            if score > highest:
                highest = score
                highest_pair = (p["passage_id"], cid)
    ok = highest < MAX_JACCARD
    print("PASS" if ok else "FAIL", "- highest Jaccard overlap is", round(highest, 3),
          "(", highest_pair, ") must be below", MAX_JACCARD)
    if not ok:
        stop("an attack passage overlaps too much with a clean passage")

    print("\nAll checks PASS.\n")

    # Write attack_passages.jsonl.
    os.makedirs("data/attacks", exist_ok=True)
    with open(PASSAGES_FILE, "w") as f:
        for p in attack_passages:
            row = {
                "passage_id": p["passage_id"],
                "text": p["text"],
                "source": "fictional_pilot",
                "corpus_version": "attack_v1",
            }
            f.write(json.dumps(row) + "\n")
    print("Wrote", PASSAGES_FILE, "(", len(attack_passages), "passages )")

    # Write attack_manifest.jsonl.
    clean_hash = sha256_of_file("data/clean/passages.jsonl")
    drafts_hash = sha256_of_file(DRAFTS_FILE)
    with open(MANIFEST_FILE, "w") as f:
        for i, target in enumerate(drafts, start=1):
            row = {
                "attack_id": "ATK" + str(i).zfill(2),
                "target_query_id": target["target_query_id"],
                "split": queries[target["target_query_id"]]["split"],
                "target_answer": target["target_answer"],
                "target_aliases": target["target_aliases"],
                "injection_ids": target["injection_ids"],
                "construction": [d["style"] for d in target["passages"]],
                "word_counts": [word_count(d["text"]) for d in target["passages"]],
                "corpus_version": "attack_v1",
                "attacked_corpus": "clean_v1 + 3 injections for this query only",
                "clean_passages_sha256": clean_hash,
                "drafts_sha256": drafts_hash,
            }
            f.write(json.dumps(row) + "\n")
    print("Wrote", MANIFEST_FILE, "(", len(drafts), "targets )")

    # Step 2: attack embeddings.
    print("\nStep 2: attack embeddings\n")
    config = load_config()
    tokenizer, model = load_retriever(config)
    E, ids, meta, mode, elapsed = build_or_load_attack_cache(config, tokenizer, model)
    print("   cache", mode.upper(), "in", round(elapsed, 2), "seconds")
    check_attack_cache(E, ids)

    print("\nDone.")


if __name__ == "__main__":
    main()
