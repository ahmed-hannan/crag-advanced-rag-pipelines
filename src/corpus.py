import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.data_loader import load_examples, get_passages_for_retrieval


DEFAULT_INDEX_PATH = Path("data/faiss.index")
DEFAULT_CHUNKS_PATH = Path("data/crag_chunks.json")


class FaissIndex:

    def __init__(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def retrieve(self, query_embedding, top_k=5):

        query_embedding = query_embedding.astype("float32").reshape(1, -1)

        scores, ids = self.index.search(query_embedding, top_k)

        results = []

        for i, score in zip(ids[0], scores[0]):
            results.append((self.chunks[i], float(score)))

        return results


def build_index(dataset_path, embedding_model="all-MiniLM-L6-v2"):

    snippets = []

    for ex in tqdm(load_examples(dataset_path)):
        passages = get_passages_for_retrieval(ex)

        for p in passages:
            if p:
                snippets.append(p.strip())

    snippets = list(dict.fromkeys(snippets))

    print("Unique snippets:", len(snippets))

    model = SentenceTransformer(embedding_model)

    embeddings = model.encode(snippets, batch_size=512, show_progress_bar=True)

    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(embeddings)

    index.add(embeddings)

    faiss.write_index(index, str(DEFAULT_INDEX_PATH))

    with open(DEFAULT_CHUNKS_PATH, "w") as f:
        json.dump(snippets, f)

    print("Index built")


def load_index(index_path=DEFAULT_INDEX_PATH, chunks_path=DEFAULT_CHUNKS_PATH):

    index = faiss.read_index(str(index_path))

    with open(chunks_path) as f:
        chunks = json.load(f)

    return FaissIndex(index, chunks)

if __name__ == "__main__":
    print("Building FAISS index...")

    build_index(
        dataset_path="dataset/crag_task_1_and_2_dev_v4.jsonl",
        embedding_model="all-MiniLM-L6-v2"
    )