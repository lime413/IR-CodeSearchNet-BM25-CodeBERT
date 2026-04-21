import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import dense_runtime  # noqa: F401
import faiss
import numpy as np
from tqdm import tqdm

from dense_utils import (
    AVAILABLE_SPLITS,
    DEFAULT_DENSE_INDEX_DIR,
    DEFAULT_MODEL_NAME,
    build_query_records,
    encode_texts,
    get_query_file_paths,
    iter_documents,
    load_model_and_tokenizer,
    resolve_device,
    write_jsonl,
)
from index import load_split_totals
from retrieve_only_BM25 import configure_logging, format_elapsed_time


DEFAULT_PROCESSED_DIR = Path("data/processed/CodeSearchNet_python")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build CodeBERT document embeddings and store them in a FAISS index."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing *_documents.jsonl files created by preprocessing.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DENSE_INDEX_DIR,
        help="Directory where the FAISS index and query embeddings will be written.",
    )
    parser.add_argument(
        "--doc-splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=AVAILABLE_SPLITS,
        help="Document splits to encode and add to the FAISS index.",
    )
    parser.add_argument(
        "--query-splits",
        nargs="+",
        default=["test"],
        choices=AVAILABLE_SPLITS,
        help="Splits whose docstrings will be encoded and saved as query embeddings.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name used to encode text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size.",
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
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help="How often to log document encoding progress.",
    )
    return parser.parse_args()


def flush_document_batch(batch_records, tokenizer, model, device, max_length, index, ids_handle):
    texts = [record["document_text"] for record in batch_records]
    embeddings = encode_texts(texts, tokenizer, model, device, max_length)
    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    for record in batch_records:
        ids_handle.write(f"{record['id']}\n")
    batch_records.clear()
    return index


def save_query_embeddings(query_split, queries, tokenizer, model, device, max_length, batch_size, output_dir):
    metadata_path, embeddings_path = get_query_file_paths(output_dir, query_split)
    write_jsonl(metadata_path, queries)

    batches = []
    progress = tqdm(
        range(0, len(queries), batch_size),
        desc=f"Encoding {query_split} queries",
        unit="batch",
    )
    for start in progress:
        batch = queries[start : start + batch_size]
        texts = [item["query_text"] for item in batch]
        batches.append(encode_texts(texts, tokenizer, model, device, max_length))

    if batches:
        query_embeddings = batches[0] if len(batches) == 1 else np.vstack(batches)
    else:
        query_embeddings = np.empty((0, 0), dtype="float32")
    np.save(embeddings_path, query_embeddings)
    return metadata_path, embeddings_path, len(queries)


def build_dense_index(args, logger):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = args.output_dir / "dense_index.faiss"
    doc_ids_path = args.output_dir / "document_ids.jsonl"
    summary_path = args.output_dir / "dense_index_summary.json"

    for path in (faiss_path, doc_ids_path, summary_path):
        if path.exists():
            path.unlink()

    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args.model_name, device)
    split_totals = load_split_totals(args.processed_dir)
    logger.info("Using device: %s", device)
    logger.info("Loading model: %s", args.model_name)

    index = None
    encoded_documents = 0
    batch_records = []
    split_counts = {}

    with doc_ids_path.open("w", encoding="utf-8") as ids_handle:
        for split_name in args.doc_splits:
            total = split_totals.get(split_name)
            progress = tqdm(
                iter_documents(args.processed_dir, [split_name]),
                total=total,
                desc=f"Encoding {split_name} docs",
                unit="doc",
            )
            split_counts[split_name] = 0

            for _, record in progress:
                batch_records.append(record)
                split_counts[split_name] += 1
                if len(batch_records) >= args.batch_size:
                    index = flush_document_batch(
                        batch_records=batch_records,
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        max_length=args.max_length,
                        index=index,
                        ids_handle=ids_handle,
                    )
                    encoded_documents += args.batch_size
                    if encoded_documents % args.log_interval == 0:
                        logger.info("Encoded %s documents", encoded_documents)

            if batch_records:
                batch_size = len(batch_records)
                index = flush_document_batch(
                    batch_records=batch_records,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=args.max_length,
                    index=index,
                    ids_handle=ids_handle,
                )
                encoded_documents += batch_size

    if index is None:
        raise ValueError("No documents were encoded. Check the processed input files.")

    faiss.write_index(index, str(faiss_path))
    logger.info("FAISS index written to %s", faiss_path)

    query_counts = {}
    for query_split in args.query_splits:
        queries = build_query_records(args.processed_dir, query_split)
        metadata_path, embeddings_path, query_count = save_query_embeddings(
            query_split=query_split,
            queries=queries,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
        logger.info("Saved %s query embeddings to %s", query_count, embeddings_path)
        logger.info("Saved query metadata to %s", metadata_path)
        query_counts[query_split] = query_count

    summary = {
        "processed_dir": str(args.processed_dir),
        "output_dir": str(args.output_dir),
        "model_name": args.model_name,
        "device": str(device),
        "document_count": index.ntotal,
        "embedding_dimension": index.d,
        "doc_splits": args.doc_splits,
        "query_splits": args.query_splits,
        "query_counts": query_counts,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "normalized_embeddings": True,
        "faiss_metric": "inner_product",
        "similarity": "cosine",
        "document_counts_by_split": split_counts,
        "faiss_index_path": str(faiss_path),
        "document_ids_path": str(doc_ids_path),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary_path, summary


def main():
    args = parse_args()
    logger = configure_logging()
    started_at = datetime.now().astimezone()
    started_perf = time.perf_counter()
    logger.info("Build dense index started")
    summary_path, summary = build_dense_index(args, logger)
    finished_at = datetime.now().astimezone()
    elapsed_seconds = time.perf_counter() - started_perf
    logger.info("Dense summary written to: %s", summary_path)
    logger.info("Indexed documents: %s", summary["document_count"])
    logger.info("Embedding dimension: %s", summary["embedding_dimension"])
    logger.info("Elapsed time: %s", format_elapsed_time(elapsed_seconds))
    logger.info("Started at: %s", started_at.isoformat(timespec="seconds"))
    logger.info("Finished at: %s", finished_at.isoformat(timespec="seconds"))
    logger.info("Build dense index done")


if __name__ == "__main__":
    main()
