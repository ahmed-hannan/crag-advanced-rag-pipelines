"""
Graph-RAG starter: expand a seed set by adding nearest neighbors
from the index embedding matrix (k-NN expansion).
"""
from typing import List, Tuple
import numpy as np

from src.retrieval import retrieve
from src.generation import generate_answer



def run_pipeline(query: str, index, embedding_model: str = "all-MiniLM-L6-v2", top_k: int = 5, expand_k: int = 5):
    # seed retrieval
    seed = retrieve(query, index, embedding_model=embedding_model, top_k=top_k)
    # find neighbors of seeds by scanning full index embeddings (cheap implementation)
    # we will treat index.embeddings and index.chunks as accessible
    seeds_idx = []
    selected_texts = []
    if not seed:
        return {"answer": "I don't know.", "sources": [], "retrieved": []}
    # map seed text to index positions (simple scan)
    seed_texts = [t for t, s in seed]
    for t in seed_texts:
        try:
            idx = index.chunks.index(t)
            seeds_idx.append(idx)
        except ValueError:
            continue
    # collect neighbor indices
    all_neighbors = set(seeds_idx)
    if hasattr(index, "embeddings"):
        emb = index.embeddings  # (N,D)
        for sidx in seeds_idx:
            vec = emb[sidx : sidx + 1]
            sims = (emb @ vec.T).squeeze() / (np.linalg.norm(emb, axis=1) * np.linalg.norm(vec))
            sims = np.nan_to_num(sims)
            nn = np.argsort(-sims)[:expand_k]
            for n in nn:
                all_neighbors.add(int(n))
    # prepare final retrieved list
    final = [(index.chunks[i], float(0.0)) for i in list(all_neighbors)[:top_k]]
    answer, sources = generate_answer(query, final)
    return {"answer": answer, "sources": sources, "retrieved": final}