from src.retrieval import retrieve, get_model
from src.generation import generate_answer
import faiss


def run_pipeline(query: str, index, embedding_model="all-MiniLM-L6-v2", top_k=5):

    # initial retrieval
    initial = retrieve(query, index, embedding_model=embedding_model, top_k=top_k)

    # build hypothetical doc
    hypo_doc = "This document answers: " + query + " " + " ".join([t for t, _ in initial[:2]])

    embedder = get_model(embedding_model)

    h_emb = embedder.encode([hypo_doc], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(h_emb)

    retrieved = index.retrieve(h_emb[0], top_k=top_k)

    answer, sources = generate_answer(query, retrieved)

    return {"answer": answer, "sources": sources, "retrieved": retrieved}