# run_paired.py
# Task 3: for each of the 12 target development questions, run the
# model TWICE with the same settings: once on the clean corpus, once
# on the clean corpus plus that question's 3 attack passages.
# This file never reads target_answer or target_aliases.
# Usage: python src/run_paired.py --split development

import argparse
import csv
import json
import os
import time
from datetime import datetime

import numpy as np
import torch

from embeddings import load_config, load_retriever, build_or_load_cache, load_passages
from attack_cache import build_or_load_attack_cache, load_attack_passages
from retrieval import retrieve
from generation import (get_template, load_generator, assemble_context,
                        generate_answer, is_abstention)

QUERIES_FILE = "data/clean/queries.jsonl"
MANIFEST_FILE = "data/attacks/attack_manifest.jsonl"
EXPECTED_TARGETS = 12
CLEAN_CORPUS_SIZE = 200
ATTACKED_CORPUS_SIZE = 203


def load_manifest_targets(split):
    # Read ONLY target_query_id and injection_ids from each manifest row.
    # (The manifest also has target_answer and aliases, but we never
    # touch those fields here.)
    targets = []
    with open(MANIFEST_FILE) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["split"] == split:
                    targets.append({
                        "target_query_id": row["target_query_id"],
                        "injection_ids": row["injection_ids"],
                    })
    return targets


def load_target_questions(targets):
    # From queries.jsonl, read only query_id, question and split, and
    # keep them in manifest order.
    questions_by_id = {}
    with open(QUERIES_FILE) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                questions_by_id[row["query_id"]] = {
                    "query_id": row["query_id"],
                    "question": row["question"],
                    "split": row["split"],
                }
    ordered = []
    for target in targets:
        qid = target["target_query_id"]
        ordered.append(questions_by_id[qid])
    return ordered


def run_one(question, E, ids, ret_tokenizer, ret_model, gen_tokenizer, gen_model,
            passages_by_id, config):
    # One retrieve -> assemble -> generate pass. Shared by both conditions.
    start = time.time()
    hits = retrieve(question, E, ids, ret_tokenizer, ret_model,
                    config["retriever_max_tokens"], config["top_k"])
    retrieval_s = time.time() - start

    start = time.time()
    context_ids, omitted_ids, prompt = assemble_context(
        question, hits, passages_by_id, gen_tokenizer, config)
    context_s = time.time() - start

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

    return {
        "hits": hits,
        "context_ids": context_ids,
        "omitted_ids": omitted_ids,
        "prompt": prompt,
        "input_tokens": input_tokens,
        "output_text": output_text,
        "new_tokens": new_tokens,
        "abstain_reason": abstain_reason,
        "retrieval_s": retrieval_s,
        "context_s": context_s,
        "generation_s": generation_s,
    }


def build_row(run_id, query, condition, corpus_size, clean_meta, config, out):
    abstained = out["abstain_reason"] is not None
    return {
        "run_id": run_id,
        "pair_id": query["query_id"],
        "query_id": query["query_id"],
        "split": query["split"],
        "condition": condition,
        "corpus_size": corpus_size,
        "corpus_version": clean_meta["corpus_version"],
        "attacked_corpus_version": "attack_v1",
        "retriever": config["retriever"],
        "retriever_revision": config["retriever_revision"],
        "generator": config["generator"],
        "generator_revision": config["generator_revision"],
        "prompt_version": config["prompt_version"],
        "ranked_ids": [h["passage_id"] for h in out["hits"]],
        "ranked_scores": [h["score"] for h in out["hits"]],
        "context_ids": out["context_ids"],
        "omitted_ids": out["omitted_ids"],
        "prompt_text": out["prompt"],
        "input_tokens": out["input_tokens"],
        "output_text": out["output_text"],
        "new_tokens": out["new_tokens"],
        "abstained": abstained,
        "abstain_reason": out["abstain_reason"],
        "retrieval_s": round(out["retrieval_s"], 4),
        "context_s": round(out["context_s"], 4),
        "generation_s": round(out["generation_s"], 4),
        "device": config["device"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    # Step 1: fixed seed, run ID and a new run folder.
    torch.manual_seed(42)
    run_start = time.time()
    now = datetime.now()
    run_id = "paired_" + args.split + "_" + now.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir)
    print("Run ID:", run_id)

    # Step 2: settings.
    config = load_config()

    # Step 3: load the retriever, the clean cache and the attack cache (timed).
    print("\nLoading retriever, clean cache and attack cache...")
    start = time.time()
    ret_tokenizer, ret_model = load_retriever(config)
    clean_E, clean_ids, clean_meta, clean_cache_mode, _ = build_or_load_cache(
        config, ret_tokenizer, ret_model)
    attack_E, attack_ids, attack_meta, attack_cache_mode, _ = build_or_load_attack_cache(
        config, ret_tokenizer, ret_model)
    retriever_load_s = time.time() - start
    print("   clean cache:", clean_cache_mode, "(", clean_E.shape, "), attack cache:",
          attack_cache_mode, "(", attack_E.shape, ")")

    # Step 4: load the generator (timed separately).
    print("Loading generator...")
    start = time.time()
    gen_tokenizer, gen_model = load_generator(config)
    generator_load_s = time.time() - start
    print("   retriever + caches:", round(retriever_load_s, 2), "s, generator:",
          round(generator_load_s, 2), "s")

    # Step 5: passage texts by ID, clean and attack.
    passages_by_id_clean = {}
    for p in load_passages():
        passages_by_id_clean[p["passage_id"]] = {"passage_id": p["passage_id"], "text": p["text"]}
    passages_by_id_attack = {}
    for p in load_attack_passages():
        passages_by_id_attack[p["passage_id"]] = {"passage_id": p["passage_id"], "text": p["text"]}

    attack_row_by_id = dict(zip(attack_ids, range(len(attack_ids))))

    # Step 6: the 12 target questions, in manifest order.
    targets = load_manifest_targets(args.split)
    if len(targets) != EXPECTED_TARGETS:
        raise SystemExit("FAIL: expected " + str(EXPECTED_TARGETS) + " targets, found " + str(len(targets)))
    queries = load_target_questions(targets)
    print("\nTarget questions:", len(queries), "\n")

    clean_results = []
    attacked_results = []
    timing_rows = []

    for target, query in zip(targets, queries):
        injection_ids = target["injection_ids"]

        # a) clean condition: the 200 clean passages, unchanged.
        clean_out = run_one(query["question"], clean_E, clean_ids, ret_tokenizer, ret_model,
                            gen_tokenizer, gen_model, passages_by_id_clean, config)
        clean_row = build_row(run_id, query, "clean", clean_E.shape[0],
                              clean_meta, config, clean_out)
        clean_results.append(clean_row)

        # b) attacked condition: the 200 clean rows plus ONLY this
        #    question's 3 attack rows, appended at the end.
        rows = [attack_row_by_id[pid] for pid in injection_ids]
        attacked_E = np.vstack([clean_E, attack_E[rows]])
        attacked_ids = clean_ids + injection_ids

        if attacked_E.shape[0] != ATTACKED_CORPUS_SIZE:
            raise SystemExit("FAIL: attacked corpus has " + str(attacked_E.shape[0]) +
                             " rows, expected " + str(ATTACKED_CORPUS_SIZE))
        other_attack_ids = set(attack_ids) - set(injection_ids)
        if other_attack_ids & set(attacked_ids):
            raise SystemExit("FAIL: attacked corpus contains attack IDs from another question")

        passages_by_id_this = dict(passages_by_id_clean)
        for pid in injection_ids:
            passages_by_id_this[pid] = passages_by_id_attack[pid]

        attacked_out = run_one(query["question"], attacked_E, attacked_ids, ret_tokenizer, ret_model,
                               gen_tokenizer, gen_model, passages_by_id_this, config)
        attacked_row = build_row(run_id, query, "attacked", attacked_E.shape[0],
                                 clean_meta, config, attacked_out)
        attacked_results.append(attacked_row)

        for row in (clean_row, attacked_row):
            total_s = row["retrieval_s"] + row["context_s"] + row["generation_s"]
            timing_rows.append([row["query_id"], row["condition"], row["retrieval_s"],
                                row["context_s"], row["generation_s"], round(total_s, 4)])
            short = row["output_text"].replace("\n", " ")[:80]
            print(row["query_id"], "|", row["condition"], "| abstained:",
                  "yes" if row["abstained"] else "no", "|", short)

    # Step 7: write the results.
    with open(os.path.join(run_dir, "clean_results.jsonl"), "w") as f:
        for row in clean_results:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(run_dir, "attacked_results.jsonl"), "w") as f:
        for row in attacked_results:
            f.write(json.dumps(row) + "\n")

    # Step 8: timing table.
    with open(os.path.join(run_dir, "timing.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "condition", "retrieval_s", "context_s",
                         "generation_s", "total_s"])
        writer.writerows(timing_rows)

    # Step 9: settings and prompt template.
    run_config = dict(config)
    run_config["run_id"] = run_id
    run_config["split"] = args.split
    run_config["date"] = now.isoformat(timespec="seconds")
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    with open(os.path.join(run_dir, "prompt_template.txt"), "w") as f:
        f.write(get_template(config["prompt_version"]))

    # Step 10: run summary.
    clean_gen_times = [row["generation_s"] for row in clean_results]
    attacked_gen_times = [row["generation_s"] for row in attacked_results]
    summary = {
        "run_id": run_id,
        "split": args.split,
        "retriever_and_caches_load_s": round(retriever_load_s, 2),
        "clean_cache_mode": clean_cache_mode,
        "attack_cache_mode": attack_cache_mode,
        "generator_load_s": round(generator_load_s, 2),
        "num_pairs": len(clean_results),
        "mean_generation_s_clean": round(sum(clean_gen_times) / len(clean_gen_times), 3),
        "max_generation_s_clean": round(max(clean_gen_times), 3),
        "mean_generation_s_attacked": round(sum(attacked_gen_times) / len(attacked_gen_times), 3),
        "max_generation_s_attacked": round(max(attacked_gen_times), 3),
        "total_run_s": round(time.time() - run_start, 2),
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:", json.dumps(summary, indent=2))
    print("Saved results in", run_dir)


if __name__ == "__main__":
    main()
