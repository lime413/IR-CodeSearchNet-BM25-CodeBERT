import argparse
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import dense_runtime  # noqa: F401
import faiss
import numpy as np
from tqdm import tqdm

from dense_utils import (
    DEFAULT_DENSE_INDEX_DIR,
    DEFAULT_MODEL_NAME,
    build_query_records,
    encode_texts,
    get_query_file_paths,
    load_doc_id_to_row_map,
    load_jsonl_records,
    load_model_and_tokenizer,
    reconstruct_vectors,
    resolve_device,
)
from retrieve_only_BM25 import (
    DEFAULT_INDEX_DIR,
    DEFAULT_PROCESSED_DIR,
    bm25_score_query,
    configure_logging,
    format_elapsed_time,
    load_document_lengths,
    load_index_statistics,
    load_vocabulary,
    make_postings_fetcher,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


DEFAULT_RESULTS_PATH = Path("data/results/bm25_codebert_test_metrics.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run BM25 retrieval first and rerank the candidates with CodeBERT cosine similarity."
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing bm25_index.sqlite and index_summary.json.",
    )
    parser.add_argument(
        "--dense-index-dir",
        type=Path,
        default=DEFAULT_DENSE_INDEX_DIR,
        help="Directory containing dense_index.faiss, document_ids.jsonl, and query embeddings.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing processed *_documents.jsonl files.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path where the evaluation metrics JSON will be written.",
    )
    parser.add_argument(
        "--query-split",
        type=str,
        default="test",
        help="Split used for evaluation queries.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Final ranking depth to score and store per query.",
    )
    parser.add_argument(
        "--bm25-candidates",
        type=int,
        default=50,
        help="How many BM25 candidates to rerank with CodeBERT.",
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
        help="LRU cache size for BM25 postings lists.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to log retrieval progress, in processed queries.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model name used only if query embeddings need to be encoded on the fly.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Query encoding batch size when embeddings are not precomputed.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length used for CodeBERT encoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override, for example cpu, cuda, or mps.",
    )
    return parser.parse_args()


def load_queries_and_embeddings(dense_index_dir, processed_dir, query_split, max_queries, logger, args):
    metadata_path, embeddings_path = get_query_file_paths(dense_index_dir, query_split)
    if metadata_path.exists() and embeddings_path.exists():
        queries = load_jsonl_records(metadata_path)
        query_embeddings = np.load(embeddings_path)
        logger.info("Loaded precomputed query embeddings from %s", embeddings_path)
    else:
        queries = build_query_records(processed_dir, query_split)
        if not queries:
            raise ValueError(f"No queries found for split: {query_split}")

        device = resolve_device(args.device)
        tokenizer, model = load_model_and_tokenizer(args.model_name, device)
        logger.info("Precomputed query embeddings not found. Encoding queries with %s", args.model_name)

        batches = []
        progress = tqdm(
            range(0, len(queries), args.batch_size),
            desc="Encoding queries",
            unit="batch",
        )
        for start in progress:
            batch = queries[start : start + args.batch_size]
            texts = [item["query_text"] for item in batch]
            batches.append(encode_texts(texts, tokenizer, model, device, args.max_length))

        query_embeddings = batches[0] if len(batches) == 1 else np.vstack(batches)

    if len(queries) != len(query_embeddings):
        raise ValueError("Query metadata and query embeddings have different lengths.")

    if max_queries is not None:
        queries = queries[:max_queries]
        query_embeddings = query_embeddings[:max_queries]

    return queries, query_embeddings.astype("float32")


def rerank_candidates(query_embedding, ranked_candidates, doc_id_to_row, dense_index):
    dense_scores = {}
    candidate_rows = []
    candidate_doc_ids = []

    for doc_id, _ in ranked_candidates:
        row_id = doc_id_to_row.get(doc_id)
        if row_id is None:
            continue
        candidate_rows.append(row_id)
        candidate_doc_ids.append(doc_id)

    if candidate_rows:
        candidate_vectors = reconstruct_vectors(dense_index, candidate_rows)
        candidate_scores = candidate_vectors @ query_embedding
        dense_scores = {
            doc_id: float(score)
            for doc_id, score in zip(candidate_doc_ids, candidate_scores.tolist())
        }

    reranked = []
    for doc_id, bm25_score in ranked_candidates:
        reranked.append(
            {
                "doc_id": doc_id,
                "score": dense_scores.get(doc_id, -2.0),
                "bm25_score": float(bm25_score),
            }
        )

    reranked.sort(key=lambda item: (item["score"], item["bm25_score"]), reverse=True)
    return reranked


def evaluate(args, logger):
    db_path = args.index_dir / "bm25_index.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing BM25 index: {db_path}")

    dense_index_path = args.dense_index_dir / "dense_index.faiss"
    doc_ids_path = args.dense_index_dir / "document_ids.jsonl"
    if not dense_index_path.exists():
        raise FileNotFoundError(f"Missing dense FAISS index: {dense_index_path}")
    if not doc_ids_path.exists():
        raise FileNotFoundError(f"Missing dense document id file: {doc_ids_path}")

    connection = sqlite3.connect(db_path)
    index_stats = load_index_statistics(connection)
    doc_lengths = load_document_lengths(connection)
    vocabulary = load_vocabulary(connection)
    get_postings = make_postings_fetcher(connection, args.cache_size)
    dense_index = faiss.read_index(str(dense_index_path))
    doc_id_to_row = load_doc_id_to_row_map(doc_ids_path)
    queries, query_embeddings = load_queries_and_embeddings(
        dense_index_dir=args.dense_index_dir,
        processed_dir=args.processed_dir,
        query_split=args.query_split,
        max_queries=args.max_queries,
        logger=logger,
        args=args,
    )

    logger.info("Loaded BM25 index from %s", db_path)
    logger.info("Loaded dense index from %s", dense_index_path)
    logger.info("Loaded %s query embeddings", len(queries))

    metrics = {
        "query_count": len(queries),
        "MRR@10": 0.0,
        "Recall@10": 0.0,
        "Recall@50": 0.0,
        "nDCG@10": 0.0,
    }
    samples = []
    retrieval_depth = max(args.top_k, 50)
    bm25_depth = max(args.bm25_candidates, retrieval_depth)
    avg_doc_len = index_stats["average_document_length"]

    for query_index, (query, query_embedding) in enumerate(
        tqdm(zip(queries, query_embeddings), total=len(queries), desc="Evaluating rerank", unit="query"),
        start=1,
    ):
        bm25_ranked = bm25_score_query(
            query_text=query["query_text"],
            vocabulary=vocabulary,
            doc_lengths=doc_lengths,
            avg_doc_len=avg_doc_len,
            get_postings=get_postings,
            top_k=bm25_depth,
        )
        reranked = rerank_candidates(
            query_embedding=query_embedding,
            ranked_candidates=bm25_ranked,
            doc_id_to_row=doc_id_to_row,
            dense_index=dense_index,
        )
        reranked_doc_ids = [item["doc_id"] for item in reranked[:retrieval_depth]]
        relevant_doc_id = query["relevant_doc_id"]

        metrics["MRR@10"] += reciprocal_rank(reranked_doc_ids, relevant_doc_id, 10)
        metrics["Recall@10"] += recall_at_k(reranked_doc_ids, relevant_doc_id, 10)
        metrics["Recall@50"] += recall_at_k(reranked_doc_ids, relevant_doc_id, 50)
        metrics["nDCG@10"] += ndcg_at_k(reranked_doc_ids, relevant_doc_id, 10)

        if len(samples) < 5:
            samples.append(
                {
                    "query_id": query["query_id"],
                    "func_name": query["func_name"],
                    "query_text": query["query_text"],
                    "top_results": [
                        {
                            "doc_id": item["doc_id"],
                            "score": round(item["score"], 6),
                            "bm25_score": round(item["bm25_score"], 6),
                        }
                        for item in reranked[: min(5, len(reranked))]
                    ],
                }
            )

        if query_index % args.log_interval == 0 or query_index == len(queries):
            logger.info(
                "Progress: %s/%s queries processed (%.2f%%)",
                query_index,
                len(queries),
                (query_index / len(queries)) * 100 if queries else 100.0,
            )

    query_count = metrics["query_count"] or 1
    metrics["MRR@10"] = round(metrics["MRR@10"] / query_count, 6)
    metrics["Recall@10"] = round(metrics["Recall@10"] / query_count, 6)
    metrics["Recall@50"] = round(metrics["Recall@50"] / query_count, 6)
    metrics["nDCG@10"] = round(metrics["nDCG@10"] / query_count, 6)

    connection.close()

    return {
        "evaluation_setup": {
            "index_dir": str(args.index_dir),
            "dense_index_dir": str(args.dense_index_dir),
            "processed_dir": str(args.processed_dir),
            "query_split": args.query_split,
            "retrieval_depth": retrieval_depth,
            "bm25_candidate_count": bm25_depth,
            "reranker_model": args.model_name,
            "relevance_assumption": (
                "Each query docstring is used as a query and its paired function "
                "is treated as the relevant document."
            ),
        },
        "index_statistics": index_stats,
        "metrics": metrics,
        "sample_rankings": samples,
    }


def main():
    args = parse_args()
    logger = configure_logging()
    started_at = datetime.now().astimezone()
    started_perf = time.perf_counter()
    logger.info("BM25 + CodeBERT rerank started")
    results = evaluate(args, logger)
    finished_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started_perf

    results["execution"] = {
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_hms": format_elapsed_time(elapsed_seconds),
    }

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    with args.results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    metrics = results["metrics"]
    logger.info("Results written to: %s", args.results_path)
    logger.info("Queries evaluated: %s", metrics["query_count"])
    logger.info("MRR@10: %s", metrics["MRR@10"])
    logger.info("Recall@10: %s", metrics["Recall@10"])
    logger.info("Recall@50: %s", metrics["Recall@50"])
    logger.info("nDCG@10: %s", metrics["nDCG@10"])
    logger.info("Elapsed time: %s", results["execution"]["elapsed_hms"])
    logger.info("BM25 + CodeBERT rerank done")


if __name__ == "__main__":
    main()
