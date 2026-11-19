# retrieval.py
# Task 2: find the top-k passages for a question with a dot product.

import numpy as np

from embeddings import encode


def retrieve(question, E, ids, tokenizer, model, max_tokens, top_k):
    # Encode the question with the same encode() used for passages.
    q = encode([question], tokenizer, model, max_tokens)[0]
    # Dot product score for every passage (no normalisation).
    scores = E @ q
    # Highest scores first; "stable" keeps file order for exact ties.
    order = np.argsort(-scores, kind="stable")[:top_k]
    hits = []
    for rank, row in enumerate(order, start=1):
        hits.append({"rank": rank, "passage_id": ids[row], "score": float(scores[row])})
    return hits
