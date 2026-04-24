# CodeSearchNet Python Search: BM25 and CodeBERT Reranking

This repository contains a course project for Information Retrieval. The goal is
to search Python functions from CodeSearchNet with a natural language query.

The project has two retrieval modes:

- BM25 only, as a lexical sparse retrieval baseline.
- BM25 plus CodeBERT reranking, where BM25 first returns candidates and CodeBERT
  changes their order by dense cosine similarity.

The current full experiment was run on the Python part of CodeSearchNet:

| Split | Documents |
| --- | ---: |
| train | 412178 |
| valid | 23107 |
| test | 22176 |
| total | 457461 |

The evaluation uses test docstrings as queries. For each query, the paired test
function is treated as the relevant document.

## 1. Setup

Use Python 3.12 or newer. The easiest setup is with `uv`:

```bash
uv sync
```

Then run commands with:

```bash
uv run python <script_name>.py
```

You can also use a normal virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The main dependencies are `datasets`, `numpy`, `torch`, `transformers`,
`faiss-cpu`, `tqdm`, and `gradio`.

The data and indexes are large. In the current run they take about:

- `data/raw/CodeSearchNet_python`: 1.2 GB
- `data/processed/CodeSearchNet_python`: 1.8 GB
- `data/indexes/CodeSearchNet_python_bm25`: 2.2 GB
- `data/indexes/CodeSearchNet_python_codebert`: 1.4 GB

The first data download needs internet access. The first CodeBERT run also needs
internet access to download `microsoft/codebert-base` from Hugging Face.

## 2. Data Loading and Preprocessing

Download the Python subset of CodeSearchNet:

```bash
python data/download_CodeSearchNet.py
```

This creates:

```text
data/raw/CodeSearchNet_python/train.jsonl
data/raw/CodeSearchNet_python/valid.jsonl
data/raw/CodeSearchNet_python/test.jsonl
```

Preprocess the raw files:

```bash
python preprocessing.py
```

The preprocessing script does the following work:

- keeps the function name, docstring, code, split name, repository name, file
  path, and source URL;
- creates one document text from function name, docstring, and code;
- tokenizes text for BM25;
- splits `snake_case` and `camelCase` identifiers;
- lowercases lexical tokens;
- writes processed documents and metadata.

Output files:

```text
data/processed/CodeSearchNet_python/train_documents.jsonl
data/processed/CodeSearchNet_python/train_metadata.jsonl
data/processed/CodeSearchNet_python/valid_documents.jsonl
data/processed/CodeSearchNet_python/valid_metadata.jsonl
data/processed/CodeSearchNet_python/test_documents.jsonl
data/processed/CodeSearchNet_python/test_metadata.jsonl
data/processed/CodeSearchNet_python/corpus_stats.json
```

You can check that the processed files are valid:

```bash
python sanity_check.py
```

## 3. Indexing and Retrieval Using Only BM25

Build the sparse BM25 index:

```bash
python index.py
```

By default, the script indexes `train`, `valid`, and `test`. It creates a SQLite
inverted index:

```text
data/indexes/CodeSearchNet_python_bm25/bm25_index.sqlite
data/indexes/CodeSearchNet_python_bm25/index_summary.json
```

The index contains:

- 457461 documents;
- 296688 vocabulary terms;
- average document length 178.0416 tokens;
- BM25 parameters `k1 = 1.5` and `b = 0.75`.

Run BM25 evaluation:

```bash
python retrieve_only_BM25.py
```

For a quick test run:

```bash
python retrieve_only_BM25.py --max-queries 100
```

The full run writes:

```text
data/results/bm25_test_metrics.json
```

Current BM25 results on the full test split:

| Metric | Value |
| --- | ---: |
| queries | 22176 |
| MRR@10 | 0.939881 |
| Recall@10 | 0.983451 |
| Recall@50 | 0.992109 |
| nDCG@10 | 0.950830 |

BM25 is very strong here because the query is the original docstring of the same
function. Many words from the docstring also appear in the indexed document.

## 4. Indexing and Retrieval Using BM25 + CodeBERT Dense Index

Build the dense CodeBERT index:

```bash
python build_dense_index.py
```

This script uses `microsoft/codebert-base`. It encodes the document text with
mean pooling, normalizes vectors, and stores them in a FAISS inner product index.
Because the vectors are normalized, inner product is cosine similarity.

Output files:

```text
data/indexes/CodeSearchNet_python_codebert/dense_index.faiss
data/indexes/CodeSearchNet_python_codebert/document_ids.jsonl
data/indexes/CodeSearchNet_python_codebert/test_queries.jsonl
data/indexes/CodeSearchNet_python_codebert/test_query_embeddings.npy
data/indexes/CodeSearchNet_python_codebert/dense_index_summary.json
```

The dense index contains:

- 457461 document vectors;
- embedding size 768;
- model `microsoft/codebert-base`;
- maximum input length 256 tokens.

Run BM25 + CodeBERT reranking:

```bash
python retrieve_bm25_codebert_rerank.py
```

For a quick test run:

```bash
python retrieve_bm25_codebert_rerank.py --max-queries 100
```

The full run writes:

```text
data/results/bm25_codebert_test_metrics.json
```

Current BM25 + CodeBERT reranking results:

| Metric | Value |
| --- | ---: |
| queries | 22176 |
| MRR@10 | 0.195510 |
| Recall@10 | 0.340458 |
| Recall@50 | 0.992109 |
| nDCG@10 | 0.229310 |

The dense stage reranks only the top 50 BM25 candidates. This is why Recall@50 is
the same as BM25: the same candidate set is used. However, MRR@10 and nDCG@10 are
lower. In this experiment, the simple CodeBERT mean-pooling representation is not
good enough to improve the top ranks. It often moves semantically similar but not
paired functions above the exact relevant function.

This result is useful because it shows that adding a neural model is not always
better. For this dataset and evaluation setup, BM25 is the stronger method.

## 5. Gradio App for Demo

After the indexes and results are ready, start the demo:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:7860
```

The app has two tabs.

The `Search` tab lets you enter a natural language query. It shows two result
lists side by side:

- BM25 lexical retrieval;
- BM25 + CodeBERT dense reranking.

Each result includes the function name, repository, score, source link,
docstring, and code snippet. You can change `Top K` and the number of BM25
candidates used for reranking.

The `Evaluation metrics` tab shows the metrics from:

```text
data/results/bm25_test_metrics.json
data/results/bm25_codebert_test_metrics.json
```

It also shows sample rankings and raw JSON results.

The first dense search can be slow because the app loads CodeBERT and the FAISS
index lazily. After this, the next queries are faster.

## Full Reproduction Order

Run this sequence from the repository root:

```bash
python data/download_CodeSearchNet.py
python preprocessing.py
python sanity_check.py
python index.py
python retrieve_only_BM25.py
python build_dense_index.py
python retrieve_bm25_codebert_rerank.py
python app.py
```

If you only want to test that scripts work, add `--max-queries 100` to the two
retrieval scripts. Full indexing and full evaluation can take several hours on a
laptop.