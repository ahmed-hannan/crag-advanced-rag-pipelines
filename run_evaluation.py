"""
Run evaluation across the 4 pipelines on the dev set.

Improvements:
- Progress bar for dataset iteration
- Periodic progress logs
- Timing per pipeline
- Final summary table
"""

import argparse
import json
import time
from collections import defaultdict

from tqdm import tqdm

from src.data_loader import load_examples
from src.corpus import build_index, load_index, DEFAULT_CHUNKS_PATH, DEFAULT_INDEX_PATH
from src.evaluation import exact_match
from src.pipelines import basic_rag, multi_query_rag, rrr, rag_fusion, hyde, crag, graph_rag


CONFIG_DATASET = "dataset/crag_task_1_and_2_dev_v4.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

INDEX_PATH = DEFAULT_INDEX_PATH
CHUNKS_PATH = DEFAULT_CHUNKS_PATH


def run(limit=None):

    print("\nLoading index...")

    try:
        index = load_index(INDEX_PATH, CHUNKS_PATH)
    except FileNotFoundError:
        print("\nIndex not found — building now (this may take time)...")
        build_index(
            CONFIG_DATASET,
            embedding_model=EMBED_MODEL,
            limit=None,
        )
        index = load_index(INDEX_PATH, CHUNKS_PATH)

    

    pipelines = {
        "basic_rag": basic_rag.run_pipeline,
        "multi_query_rag": multi_query_rag.run_pipeline,
        "rrr": rrr.run_pipeline,
        "rag_fusion": rag_fusion.run_pipeline,
        "hyde": hyde.run_pipeline,
        "crag": crag.run_pipeline,
        "graph_rag": graph_rag.run_pipeline,
    }

    stats = {k: {"correct": 0, "total": 0} for k in pipelines.keys()}
    times = defaultdict(float)

    print("\nStarting evaluation...\n")

    examples = list(load_examples(path=CONFIG_DATASET, limit=limit))

    for ex in tqdm(examples, desc="Evaluating queries"):

        q = ex["query"] or ""
        gold = ex["answer"] or ""
        alt = ex.get("alt_ans") or []

        for name, fn in pipelines.items():

            start = time.time()

            out = fn(q, index, embedding_model=EMBED_MODEL, top_k=TOP_K)

            elapsed = time.time() - start
            times[name] += elapsed

            pred = out.get("answer", "")

            is_ok = exact_match(pred, gold, alt)

            stats[name]["total"] += 1
            if is_ok:
                stats[name]["correct"] += 1

    print("\nEvaluation finished\n")

    print("Results:")
    print("-" * 40)

    summary = {}

    for name, s in stats.items():

        total = s["total"]
        correct = s["correct"]
        acc = correct / total if total else 0

        avg_time = times[name] / total if total else 0

        print(
            f"{name:10} | accuracy: {correct}/{total} = {acc:.4f} | avg_time: {avg_time:.3f}s"
        )

        summary[name] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "avg_time": avg_time,
        }

    print("-" * 40)

    with open("results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved results_summary.json\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit examples for quick tests",
    )

    args = parser.parse_args()

    run(limit=args.limit)