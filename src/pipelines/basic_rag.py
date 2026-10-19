from src.retrieval import retrieve
from src.generation import generate_answer


def run_pipeline(query, index, embedding_model="all-MiniLM-L6-v2", top_k=5):

    retrieved = retrieve(query, index, embedding_model, top_k)

    answer, sources = generate_answer(query, retrieved)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved": retrieved
    }