# embeddings.py
# Task 2: turn passages and questions into vectors with Contriever,
# and keep the passage vectors in a cache so we only compute them once.

import hashlib
import json
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

PASSAGES_FILE = "data/clean/passages.jsonl"
CACHE_DIR = "cache"
EMB_FILE = os.path.join(CACHE_DIR, "passage_embeddings.npy")
IDS_FILE = os.path.join(CACHE_DIR, "passage_ids.json")
META_FILE = os.path.join(CACHE_DIR, "cache_metadata.json")

EXPECTED_SHAPE = (200, 768)
BATCH_SIZE = 16


def load_config():
    # Read the fixed settings from config.json.
    with open("config.json") as f:
        return json.load(f)


def load_retriever(config):
    # Load the Contriever tokenizer and model at the pinned revision.
    name = config["retriever"]
    revision = config["retriever_revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModel.from_pretrained(name, revision=revision).eval()
    return tokenizer, model


def encode(texts, tokenizer, model, max_tokens):
    # Turn a list of texts into vectors (one row per text).
    # The SAME function is used for questions and passages.
    all_vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=max_tokens, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
        # Mean pooling: average the word vectors, ignoring padding.
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        all_vectors.append(pooled.numpy())
    return np.concatenate(all_vectors, axis=0).astype(np.float32)


def sha256_of_file(path):
    # Fingerprint of a file: changes if even one byte changes.
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_passages():
    # Read passages.jsonl in file order.
    passages = []
    with open(PASSAGES_FILE) as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
    return passages


def build_or_load_cache(config, tokenizer, model):
    # Load the cached passage vectors if they are still valid,
    # otherwise compute them again and save them.
    os.makedirs(CACHE_DIR, exist_ok=True)
    passages_hash = sha256_of_file(PASSAGES_FILE)

    # Step 1: try to load the cache.
    if os.path.exists(EMB_FILE) and os.path.exists(IDS_FILE) and os.path.exists(META_FILE):
        with open(META_FILE) as f:
            meta = json.load(f)
        if (meta.get("revision") == config["retriever_revision"]
                and meta.get("passages_sha256") == passages_hash):
            start = time.time()
            E = np.load(EMB_FILE)
            with open(IDS_FILE) as f:
                ids = json.load(f)
            load_time = time.time() - start
            print("   cache LOADED in", round(load_time, 3), "seconds")
            return E, ids, meta, "load", load_time
        print("   cache metadata does not match, rebuilding...")

    # Step 2: build the cache.
    passages = load_passages()
    ids = [p["passage_id"] for p in passages]
    texts = [p["text"] for p in passages]

    start = time.time()
    E = encode(texts, tokenizer, model, config["retriever_max_tokens"])
    build_time = time.time() - start

    np.save(EMB_FILE, E)
    with open(IDS_FILE, "w") as f:
        json.dump(ids, f, indent=2)

    meta = {
        "model": config["retriever"],
        "revision": config["retriever_revision"],
        "pooling": config["pooling"],
        "max_tokens": config["retriever_max_tokens"],
        "shape": list(E.shape),
        "dtype": str(E.dtype),
        "corpus_version": passages[0]["corpus_version"],
        "passages_sha256": passages_hash,
        "passage_ids_sha256": sha256_of_file(IDS_FILE),
        "build_time_seconds": round(build_time, 2),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print("   cache BUILT in", round(build_time, 2), "seconds")
    return E, ids, meta, "build", build_time


def check_cache(E, ids, tokenizer, model, config):
    # Stop with a clear error if anything is wrong with the cache.
    print("   check shape is", EXPECTED_SHAPE, ":", E.shape)
    if E.shape != EXPECTED_SHAPE:
        raise SystemExit("FAIL: cache shape is " + str(E.shape))
    print("   PASS shape")

    if not np.isfinite(E).all():
        raise SystemExit("FAIL: cache has values that are not finite")
    print("   PASS all values finite")

    norms = np.linalg.norm(E, axis=1)
    if (norms == 0).any():
        raise SystemExit("FAIL: some rows have zero norm")
    print("   PASS no zero-norm rows (min norm", round(float(norms.min()), 3), ")")

    if len(set(ids)) != len(ids):
        raise SystemExit("FAIL: passage IDs are not unique")
    print("   PASS IDs unique")

    if len(ids) != E.shape[0]:
        raise SystemExit("FAIL: " + str(len(ids)) + " IDs but " + str(E.shape[0]) + " rows")
    print("   PASS ID count equals row count (", len(ids), ")")

    # Re-encode 3 fixed passages (first, middle, last) and compare.
    passages = load_passages()
    rows = [0, len(passages) // 2, len(passages) - 1]
    for row in rows:
        if passages[row]["passage_id"] != ids[row]:
            raise SystemExit("FAIL: ID order does not match passages.jsonl at row " + str(row))
        fresh = encode([passages[row]["text"]], tokenizer, model, config["retriever_max_tokens"])[0]
        diff = float(np.abs(fresh - E[row]).max())
        if not np.allclose(fresh, E[row], atol=1e-5):
            raise SystemExit("FAIL: re-encoded " + ids[row] + " differs (max diff " + str(diff) + ")")
        print("   PASS re-encode row", row, ids[row], "(max diff", diff, ")")
