"""
Query embedding + retrieval wrapper
"""

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

from src.corpus import load_index


# cache the embedding model so it loads only once
_MODEL = None


def get_model(name="all-MiniLM-L6-v2"):
    global _MODEL

    if _MODEL is None:
        _MODEL = SentenceTransformer(name)

    return _MODEL



def retrieve(query, index, embedding_model="all-MiniLM-L6-v2", top_k=5, search_k=20):

    model = get_model(embedding_model)

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")[0]

    # retrieve more candidates
    candidates = index.retrieve(query_embedding, top_k=search_k)

    # sort by score (already cosine similarity)
    candidates.sort(key=lambda x: -x[1])

    return candidates[:top_k]