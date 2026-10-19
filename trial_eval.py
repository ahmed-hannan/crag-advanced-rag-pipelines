from src.corpus import load_index
from src.retrieval import retrieve
from src.generation import generate_answer

index = load_index()

query = "Who directed Inception?"

retrieved = retrieve(query, index, top_k=3,rerank_k=20)

print("Retrieved:")
for r in retrieved:
    print(r[0][:120], "...")

answer, _ = generate_answer(query, retrieved)

print("\nGenerated answer:", answer)