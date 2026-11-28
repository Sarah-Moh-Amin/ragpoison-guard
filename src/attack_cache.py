# attack_cache.py
# Task 3: embedding cache for the 36 attack passages.
# This mirrors build_or_load_cache() in embeddings.py, but for attack passages.

import json
import os
import time

import numpy as np

from embeddings import encode, sha256_of_file

PASSAGES_FILE = "data/attacks/attack_passages.jsonl"
CACHE_DIR = "cache"
EMB_FILE = os.path.join(CACHE_DIR, "attack_embeddings.npy")
IDS_FILE = os.path.join(CACHE_DIR, "attack_ids.json")
META_FILE = os.path.join(CACHE_DIR, "attack_cache_metadata.json")

EXPECTED_SHAPE = (36, 768)


def load_attack_passages():
    # Read attack_passages.jsonl in file order.
    passages = []
    with open(PASSAGES_FILE) as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
    return passages


def build_or_load_attack_cache(config, tokenizer, model):
    # Load the cached attack vectors if they are still valid,
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
            return E, ids, meta, "load", load_time
        print("   attack cache metadata does not match, rebuilding...")

    # Step 2: build the cache.
    passages = load_attack_passages()
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
        "passages_sha256": passages_hash,
        "build_time_seconds": round(build_time, 2),
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    return E, ids, meta, "build", build_time


def check_attack_cache(E, ids):
    # Stop with a clear error if anything is wrong with the attack cache.
    print("   check shape is", EXPECTED_SHAPE, ":", E.shape)
    if E.shape != EXPECTED_SHAPE:
        raise SystemExit("FAIL: attack cache shape is " + str(E.shape))
    print("   PASS shape")

    if not np.isfinite(E).all():
        raise SystemExit("FAIL: attack cache has values that are not finite")
    print("   PASS all values finite")

    norms = np.linalg.norm(E, axis=1)
    if (norms == 0).any():
        raise SystemExit("FAIL: some attack rows have zero norm")
    print("   PASS no zero-norm rows (min norm", round(float(norms.min()), 3), ")")

    passages = load_attack_passages()
    file_ids = [p["passage_id"] for p in passages]
    ok = file_ids == ids
    print("   PASS" if ok else "   FAIL", "- IDs aligned with attack_passages.jsonl")
    if not ok:
        raise SystemExit("FAIL: attack cache IDs do not match attack_passages.jsonl order")
