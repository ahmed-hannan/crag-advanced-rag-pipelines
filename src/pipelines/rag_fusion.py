"""
RAG-Fusion simple implementation (Reciprocal Rank Fusion over simple paraphrases).
This is a basic, fully offline variant suitable as a starter.
"""
from typing import List, Tuple
from collections import defaultdict

from src.retrieval import retrieve
from src.generation import generate_answer

PARAPHRASE_TEMPLATES = [
    "{q}",
    "Short answer to: {q}",
    "Briefly explain: {q}",
    "One-sentence response for: {q}",
]


def run_pipeline(query: str, index, embedding_model: str = "all-MiniLM-L6-v2", top_k: int = 5):
    # gather ranked lists
    ranked_lists = []
    for tmpl in PARAPHRASE_TEMPLATES:
        q = tmpl.format(q=query)
        ranked = retrieve(q, index, embedding_model=embedding_model, top_k=top_k)
        ranked_lists.append(ranked)

    # RRF fusion: score each doc by sum(1 / (k + rank))
    agg = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (text, score) in enumerate(ranked, start=1):
            agg[text] += 1.0 / (60 + rank)  # k=60 reduces dominance of short lists

    fused = sorted(agg.items(), key=lambda x: -x[1])
    fused_top = [(t, s) for t, s in fused[:top_k]]
    answer, sources = generate_answer(query, fused_top)
    return {"answer": answer, "sources": sources, "retrieved": fused_top}