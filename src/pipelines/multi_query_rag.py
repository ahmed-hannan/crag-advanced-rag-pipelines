from src.retrieval import retrieve
from src.generation import generate_answer


def expand_queries(query):

    return [
        query,
        f"Explain {query}",
        f"Information about {query}",
        f"Facts about {query}"
    ]


def run_pipeline(query, index, embedding_model="all-MiniLM-L6-v2", top_k=5):

    queries = expand_queries(query)

    results = []

    for q in queries:

        r = retrieve(q, index, embedding_model, top_k)

        results.extend(r)

    # remove duplicates
    seen = set()
    merged = []

    for text, score in results:

        if text not in seen:
            merged.append((text, score))
            seen.add(text)

    answer, sources = generate_answer(query, merged[:top_k])

    return {
        "answer": answer,
        "sources": sources,
        "retrieved": merged[:top_k]
    }