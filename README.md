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
