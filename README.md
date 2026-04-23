# CodeSearchNet BM25 (lexical retrieval)

Lexical BM25 retrieval on the CodeSearchNet Python split: download raw data, preprocess, build a SQLite inverted index, then evaluate on the test split.

## Setup

Python 3.12+. Install dependencies (e.g. `uv sync` or `pip install -e .`). The download step needs network access to fetch the dataset from Hugging Face.

## Pipeline

Run from the project root, in order:

```bash
python data/download_CodeSearchNet.py
python preprocessing.py
python index.py
python retrieve_only_BM25.py
```

1. **download** — Writes `data/raw/CodeSearchNet_python/{train,valid,test}.jsonl`.
2. **preprocessing** — Tokenizes and writes `data/processed/CodeSearchNet_python/` plus `corpus_stats.json`.
3. **index** — Builds `data/indexes/CodeSearchNet_python_bm25/bm25_index.sqlite` and `index_summary.json`.
4. **retrieve** — Evaluates BM25 on test queries and writes metrics to `data/results/bm25_test_metrics.json` (see `--help` for options).

Optional: `python sanity_check.py` to validate processed files.

## Demo (Gradio)

After running the pipeline (including `python build_dense_index.py` for the CodeBERT reranker), launch the web UI:

```bash
uv sync
python app.py
```

Open http://127.0.0.1:7860. The app has two tabs:

- **Search** — type a natural-language query (or pick a preset example from the test split) and see BM25 vs BM25+CodeBERT rerank results side-by-side, with code snippets, docstrings, scores, and links to the source on GitHub.
- **Evaluation metrics** — MRR@10 / Recall@10 / Recall@50 / nDCG@10 for both pipelines pulled from `data/results/*.json`, plus sample rankings for each.

The first rerank call lazily loads CodeBERT (`microsoft/codebert-base`) and the FAISS index; subsequent queries are fast.

### Recording a demo

Use QuickTime (`Cmd+Shift+5` → "Record Selected Portion") or Loom. Suggested 60–90s script:

- 0:00–0:10 open the app, show the two-column Search layout.
- 0:10–0:35 run an example query where BM25 already wins (e.g. "read a json file and return a dict"); point out that both columns return the same top hit.
- 0:35–1:00 run a query where lexical overlap is low and rerank helps (e.g. "establish an ssh connection"); highlight how the rerank column surfaces semantically relevant code BM25 ranked lower.
- 1:00–1:30 switch to the Evaluation tab, compare the metric table, expand the sample-rankings accordions.
