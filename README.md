# Advanced RAG Pipelines & Evaluation Benchmark

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61dafb.svg)](https://vitejs.dev/)
[![Flask](https://img.shields.io/badge/Backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end framework and empirical case study evaluating **Advanced Retrieval-Augmented Generation (RAG) Architectures** under realistic, noisy web search conditions. Built on the Meta **CRAG (Comprehensive RAG)** benchmark.

**Author**: **Ahmed Hannan** ([@ahmed-hannan](https://github.com/ahmed-hannan))

---

## Overview

Traditional RAG systems perform well on clean, curated documentation, but often degrade when applied to real-world, uncurated web corpora where documents contain advertisements, ambiguous text fragments, or superficial keyword matches.

This project implements a unified global corpus and vector index, benchmarking seven retrieval paradigms across diverse query categories (simple lookups, entity comparisons, multi-hop reasoning, and conditional queries):

1. **Basic Dense RAG**: Baseline single-vector nearest-neighbor search.
2. **Multi-Query RAG**: Query decomposition and perspective expansion.
3. **Rewrite-Retrieve-Read (RRR)**: Intent-preserving query reformulation prior to retrieval.
4. **RAG Fusion**: Multi-query generation paired with Reciprocal Rank Fusion (RRF).
5. **HyDE (Hypothetical Document Embeddings)**: Zero-shot hypothetical document generation for semantic vector search.
6. **CRAG (Corrective RAG)**: Confidence-gated evidence verification and attribution citations.
7. **Graph RAG**: Entity-relationship and graph-augmented context retrieval.

---

## Architecture

```
                                 ┌──────────────────────┐
                                 │      User Query      │
                                 └──────────┬───────────┘
                                            │
               ┌────────────────────────────┼────────────────────────────┐
               │                            │                            │
               ▼                            ▼                            ▼
      ┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
      │  RAG Fusion      │        │      HyDE        │        │   Corrective RAG │
      │(Multi-Query+RRF) │        │ (Hypothetical)   │        │(Confidence Gate) │
      └────────┬─────────┘        └─────────┬────────┘        └─────────┬────────┘
               │                            │                            │
               └────────────────────────────┼────────────────────────────┘
                                            ▼
                               ┌──────────────────────────┐
                               │ Global FAISS Vector Index│
                               │(Sentence-Transformers L2)│
                               └────────────┬─────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │  Answer Extraction & LLM │
                               │(spaCy NER + Heuristics)  │
                               └────────────┬─────────────┘
                                            │
                                            ▼
                               ┌──────────────────────────┐
                               │ React + Vite Interactive │
                               │     Web Interface        │
                               └──────────────────────────┘
```

---

## Retrieval Strategies Implemented

### 1. RAG Fusion (`src/pipelines/rag_fusion.py`)
Generates multiple linguistic variants of the user prompt, retrieves top-$k$ ranked lists for each from the global FAISS index, and merges candidate rankings using **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF Score}(d) = \sum_{q \in Q} \frac{1}{k + \text{rank}(q, d)}$$

### 2. HyDE: Hypothetical Document Embeddings (`src/pipelines/hyde.py`)
Rather than embedding short, keyword-sparse queries directly, HyDE prompts a language model to synthesize a hypothetical answer document. The hypothetical passage is mapped into vector space, capturing dense semantic proximity to actual grounded facts.

### 3. Corrective RAG: CRAG (`src/pipelines/crag.py`)
Computes confidence and semantic consistency scores over retrieved passages:
- **High Confidence**: Passes validated chunks to the generator with formal source attribution.
- **Low Confidence / Ambiguity**: Falls back to query reformulation or conservative answering to prevent hallucination.

### 4. Graph RAG (`src/pipelines/graph_rag.py`)
Extracts entity-relation subgraphs and document co-occurrence neighborhoods from retrieved passages, providing structured topological context for multi-hop queries.

### 5. Rewrite-Retrieve-Read: RRR (`src/pipelines/rrr.py`)
Employs an upstream rewriting step that strips colloquialisms and resolves pronouns, adapting ambiguous input into optimal information-seeking syntax.

---

## Empirical Benchmark & Results

Evaluated on $N = 100$ held-out questions from the **Meta CRAG (Comprehensive RAG)** benchmark spanning entertainment, sports, finance, and factual knowledge:

| Pipeline Architecture | Exact Match Accuracy | Average Retrieval Latency (s) | Strengths / Best For |
| :--- | :---: | :---: | :--- |
| **Basic RAG** | **15.0%** | 0.296s | Fast baseline for direct lexical lookups |
| **Multi-Query RAG** | **15.0%** | 0.245s | Covers varied viewpoints of the prompt |
| **Rewrite-Retrieve-Read (RRR)** | **15.0%** | 0.243s | Resolves ambiguous and colloquial inputs |
| **HyDE** | **15.0%** | 0.281s | Effective for open-domain conceptual queries |
| **Corrective RAG (CRAG)** | **15.0%** | **0.189s** | Fastest execution with citation grounding |
| **RAG Fusion** | 13.0% | 0.248s | Deep re-ranking across multi-variant searches |
| **Graph RAG** | 13.0% | **0.187s** | Structured entity and relationship exploration |

*Evaluation metrics generated via `run_evaluation.py` and serialized to `results_summary.json`.*

---

## Repository Structure

```
├── README.md                      # Comprehensive system documentation
├── .gitignore                     # Rules ignoring caches, node_modules, and raw data
├── report_A2_22i-0943.pdf         # Formal benchmark evaluation report
├── requirements.txt               # Python package dependencies
├── run_evaluation.py              # Automated batch evaluation harness
├── trial_eval.py                  # Single-query debugging script
├── results_summary.json           # Raw benchmark metrics across all pipelines
│
├── backend/                       # REST API
│   └── app.py                     # Flask service exposing pipeline endpoints
│
├── frontend/                      # Interactive Web UI
│   ├── package.json               # Node.js dependencies
│   ├── vite.config.js             # Vite development server configuration
│   ├── index.html                 # App shell
│   └── src/
│       ├── App.jsx                # Interactive dashboard (pipeline selector & results)
│       └── main.jsx               # React entrypoint
│
├── src/                           # Core RAG Modules
│   ├── corpus.py                  # Global corpus extractor & FAISS indexer
│   ├── data_loader.py             # CRAG dataset loader & parser
│   ├── retrieval.py               # Vector retrieval primitives
│   ├── generation.py              # LLM synthesis & spaCy NER answer extractor
│   ├── evaluation.py              # Normalized exact-match scoring
│   └── pipelines/                 # Advanced Pipeline Implementations
│       ├── basic_rag.py
│       ├── crag.py
│       ├── graph_rag.py
│       ├── hyde.py
│       ├── multi_query_rag.py
│       ├── rag_fusion.py
│       └── rrr.py
│
├── config/
│   ├── config.example.yaml        # Template configuration
│   └── config.yaml                # Active parameters (top_k, model, paths)
│
├── data/                          # Pre-built Index Artifacts
│   ├── faiss.index                # Serialized FAISS L2 index
│   ├── crag_chunks.json           # Indexed snippet chunks
│   └── crag_index.npz             # Chunk embeddings array
│
├── dataset/                       # Raw CRAG dataset destination
│   └── .gitkeep
└── docs/
    └── dataset.md                 # CRAG dataset schema and specifications
```

---

## Getting Started

### 1. Prerequisites & Environment Setup

Clone the repository:

```bash
git clone https://github.com/ahmed-hannan/crag-advanced-rag-pipelines.git
cd crag-advanced-rag-pipelines
```

Install Python packages:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Pre-computed Indices

The repository includes pre-computed embeddings and FAISS indices in `data/` (`faiss.index`, `crag_chunks.json`, `crag_index.npz`). You can run evaluations and the web demo immediately.

If you wish to rebuild the index from scratch using the raw CRAG dataset:

```bash
# Download CRAG Task 1 & 2 dev set into dataset/
python -m src.corpus
```

### 3. Running Automated Evaluation

Run the benchmark harness over the dev set:

```bash
python run_evaluation.py --limit 100
```
Summary statistics and per-pipeline exact-match scores will be displayed and saved to `results_summary.json`.

### 4. Running the Interactive Web Application

#### Start the Flask Backend:
```bash
python backend/app.py
```
*(Runs on `http://localhost:5000`)*

#### Start the React Frontend:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interactively submit queries, switch between RAG Fusion, HyDE, CRAG, and Graph RAG, and view retrieved chunks and confidence scores.

---

## Author

- **Ahmed Hannan**
  - Department of Computer Science, FAST National University of Computer and Emerging Sciences
  - GitHub: [@ahmed-hannan](https://github.com/ahmed-hannan)

---

## License

This project is licensed under the [MIT License](LICENSE).
