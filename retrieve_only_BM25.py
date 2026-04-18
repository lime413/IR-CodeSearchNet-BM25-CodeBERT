import argparse
import json
import math
import sqlite3
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from preprocessing import tokenize_for_bm25

import heapq


DEFAULT_INDEX_DIR = Path("data/indexes/CodeSearchNet_python_bm25")
DEFAULT_PROCESSED_DIR = Path("data/processed/CodeSearchNet_python")
DEFAULT_RESULTS_PATH = Path("data/results/bm25_test_metrics.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BM25-only retrieval on the CodeSearchNet Python test split."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing bm25_index.sqlite and index_summary.json.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing test_documents.jsonl.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path where the evaluation metrics JSON will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Maximum ranking depth to score and store per query.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional limit for quick debugging runs.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=2048,
        help="LRU cache size for postings lists.",
    )
    return parser.parse_args()


def iter_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield line_number, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc


def load_index_statistics(connection):
    rows = connection.execute("SELECT key, value FROM statistics").fetchall()
    stats = {key: json.loads(value) for key, value in rows}
    return stats


def load_document_lengths(connection):
    rows = connection.execute(
        "SELECT doc_id, lexical_token_count FROM documents"
    ).fetchall()
    return {doc_id: length for doc_id, length in rows}


def load_vocabulary(connection):
    rows = connection.execute("SELECT term, idf FROM vocabulary").fetchall()
    return {term: idf for term, idf in rows}


def make_postings_fetcher(connection, cache_size):
    @lru_cache(maxsize=cache_size)
    def get_postings(term):
        rows = connection.execute(
            "SELECT doc_id, term_frequency FROM postings WHERE term = ?",
            (term,),
        ).fetchall()
        return rows

    return get_postings


def bm25_score_query(query_text, vocabulary, doc_lengths, avg_doc_len, get_postings, top_k):
    query_tokens = tokenize_for_bm25(query_text)
    if not query_tokens:
        return []

    scores = {}
    k1 = 1.5
    b = 0.75

    for term in set(query_tokens):
        idf = vocabulary.get(term)
        if idf is None:
            continue

        for doc_id, term_frequency in get_postings(term):
            doc_length = doc_lengths[doc_id]
            denominator = term_frequency + k1 * (1 - b + b * doc_length / avg_doc_len)
            score = idf * (term_frequency * (k1 + 1)) / denominator
            scores[doc_id] = scores.get(doc_id, 0.0) + score

    ranked = heapq.nlargest(top_k, scores.items(), key=lambda item: item[1])
    return ranked[:top_k]


def reciprocal_rank(ranked_doc_ids, relevant_doc_id, k):
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id == relevant_doc_id:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_doc_ids, relevant_doc_id, k):
    return 1.0 if relevant_doc_id in ranked_doc_ids[:k] else 0.0


def ndcg_at_k(ranked_doc_ids, relevant_doc_id, k):
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id == relevant_doc_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def evaluate(index_dir, processed_dir, top_k, max_queries, cache_size):
    db_path = index_dir / "bm25_index.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing BM25 index: {db_path}")

    test_docs_path = processed_dir / "test_documents.jsonl"
    if not test_docs_path.exists():
        raise FileNotFoundError(f"Missing processed test documents: {test_docs_path}")

    connection = sqlite3.connect(db_path)
    index_stats = load_index_statistics(connection)
    doc_lengths = load_document_lengths(connection)
    vocabulary = load_vocabulary(connection)
    get_postings = make_postings_fetcher(connection, cache_size)

    avg_doc_len = index_stats["average_document_length"]
    queries = []
    for _, record in iter_jsonl(test_docs_path):
        query_text = (record.get("docstring") or "").strip()
        if not query_text:
            continue
        queries.append(
            {
                "query_id": record["id"],
                "query_text": query_text,
                "relevant_doc_id": record["id"],
                "func_name": record["func_name"],
            }
        )
        if max_queries is not None and len(queries) >= max_queries:
            break

    metrics = {
        "query_count": len(queries),
        "MRR@10": 0.0,
        "Recall@10": 0.0,
        "Recall@50": 0.0,
        "nDCG@10": 0.0,
    }
    samples = []

    for query in tqdm(queries, desc="Evaluating BM25", unit="query"):
        ranked = bm25_score_query(
            query_text=query["query_text"],
            vocabulary=vocabulary,
            doc_lengths=doc_lengths,
            avg_doc_len=avg_doc_len,
            get_postings=get_postings,
            top_k=max(top_k, 50),
        )
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        relevant_doc_id = query["relevant_doc_id"]

        metrics["MRR@10"] += reciprocal_rank(ranked_doc_ids, relevant_doc_id, 10)
        metrics["Recall@10"] += recall_at_k(ranked_doc_ids, relevant_doc_id, 10)
        metrics["Recall@50"] += recall_at_k(ranked_doc_ids, relevant_doc_id, 50)
        metrics["nDCG@10"] += ndcg_at_k(ranked_doc_ids, relevant_doc_id, 10)

        if len(samples) < 5:
            samples.append(
                {
                    "query_id": query["query_id"],
                    "func_name": query["func_name"],
                    "query_text": query["query_text"],
                    "top_results": [
                        {"doc_id": doc_id, "score": round(score, 6)}
                        for doc_id, score in ranked[: min(5, len(ranked))]
                    ],
                }
            )

    query_count = metrics["query_count"] or 1
    metrics["MRR@10"] = round(metrics["MRR@10"] / query_count, 6)
    metrics["Recall@10"] = round(metrics["Recall@10"] / query_count, 6)
    metrics["Recall@50"] = round(metrics["Recall@50"] / query_count, 6)
    metrics["nDCG@10"] = round(metrics["nDCG@10"] / query_count, 6)

    connection.close()

    return {
        "evaluation_setup": {
            "index_dir": str(index_dir),
            "processed_dir": str(processed_dir),
            "query_split": "test",
            "retrieval_depth": max(top_k, 50),
            "relevance_assumption": (
                "Each test docstring is used as a query and its paired test function "
                "is treated as the relevant document."
            ),
        },
        "index_statistics": index_stats,
        "metrics": metrics,
        "sample_rankings": samples,
    }


def main():
    args = parse_args()
    results = evaluate(
        index_dir=args.index_dir,
        processed_dir=args.processed_dir,
        top_k=args.top_k,
        max_queries=args.max_queries,
        cache_size=args.cache_size,
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    metrics = results["metrics"]
    print(f"Results written to: {args.results_path}")
    print(f"Queries evaluated: {metrics['query_count']}")
    print(f"MRR@10: {metrics['MRR@10']}")
    print(f"Recall@10: {metrics['Recall@10']}")
    print(f"Recall@50: {metrics['Recall@50']}")
    print(f"nDCG@10: {metrics['nDCG@10']}")


if __name__ == "__main__":
    main()
