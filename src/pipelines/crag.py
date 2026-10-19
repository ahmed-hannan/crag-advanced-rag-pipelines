"""
CRAG (Corrective RAG) simple fallback logic:
  1) retrieve normally
  2) if top similarity < threshold then use HyDE fallback
"""
from src.retrieval import retrieve
from src.generation import generate_answer
from src.pipelines.hyde import run_pipeline as hyde_pipeline


def run_pipeline(query: str, index, embedding_model: str = "all-MiniLM-L6-v2", top_k: int = 5, threshold: float = 0.45):
    retrieved = retrieve(query, index, embedding_model=embedding_model, top_k=top_k)
    top_score = retrieved[0][1] if retrieved else 0.0
    if top_score < threshold:
        # fallback to hyde
        return hyde_pipeline(query, index, embedding_model=embedding_model, top_k=top_k)
    answer, sources = generate_answer(query, retrieved)
    return {"answer": answer, "sources": sources, "retrieved": retrieved}