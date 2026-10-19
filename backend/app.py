"""
Tiny Flask backend to run pipelines against a loaded index.
Endpoint: POST /query { "query": "...", "pipeline": "hyde" }
"""
import os
from flask import Flask, request, jsonify
from src.corpus import load_index, DEFAULT_INDEX_PATH, DEFAULT_CHUNKS_PATH
from src.pipelines import rag_fusion, hyde, crag, graph_rag

app = Flask(__name__)

INDEX = None
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.environ.get("TOP_K", "3"))


@app.before_first_request
def load_resources():
    global INDEX
    try:
        INDEX = load_index(DEFAULT_INDEX_PATH, DEFAULT_CHUNKS_PATH)
    except FileNotFoundError:
        raise RuntimeError("Index not found. Run src.corpus.build_index first.")


PIPELINE_MAP = {
    "rag_fusion": rag_fusion.run_pipeline,
    "hyde": hyde.run_pipeline,
    "crag": crag.run_pipeline,
    "graph_rag": graph_rag.run_pipeline,
}


@app.route("/query", methods=["POST"])
def query():
    payload = request.get_json(force=True)
    q = payload.get("query", "")
    pipeline = payload.get("pipeline", "hyde")
    if pipeline not in PIPELINE_MAP:
        return jsonify({"error": f"unknown pipeline {pipeline}"}), 400
    fn = PIPELINE_MAP[pipeline]
    out = fn(q, INDEX, embedding_model=EMBED_MODEL, top_k=TOP_K)
    return jsonify(out)


if __name__ == "__main__":
    app.run(port=5000, debug=True)