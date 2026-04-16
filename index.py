import argparse
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from tqdm import tqdm


PROCESSED_DIR = Path("data/processed/CodeSearchNet_python")
INDEX_DIR = Path("data/indexes/CodeSearchNet_python_bm25")
AVAILABLE_SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a BM25 inverted index from preprocessed CodeSearchNet documents."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing *_documents.jsonl files created by preprocessing.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INDEX_DIR,
        help="Directory where the SQLite index and summary files will be written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=AVAILABLE_SPLITS,
        help="Document splits to include in the retrieval index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="How many posting rows to buffer before writing to SQLite.",
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


def load_split_totals(processed_dir):
    stats_path = processed_dir / "corpus_stats.json"
    if not stats_path.exists():
        return {}

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    totals = {}
    for split_name, split_stats in stats.get("splits", {}).items():
        totals[split_name] = split_stats.get("kept_records")
    return totals


def ensure_index_schema(connection):
    connection.executescript(
        """
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;

        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            func_name TEXT NOT NULL,
            lexical_token_count INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS postings (
            term TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            term_frequency INTEGER NOT NULL,
            PRIMARY KEY (term, doc_id)
        );

        CREATE TABLE IF NOT EXISTS vocabulary (
            term TEXT PRIMARY KEY,
            document_frequency INTEGER NOT NULL,
            collection_frequency INTEGER NOT NULL,
            idf REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS statistics (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    connection.commit()


def reset_index_schema(connection):
    connection.executescript(
        """
        DROP TABLE IF EXISTS postings;
        DROP TABLE IF EXISTS documents;
        DROP TABLE IF EXISTS vocabulary;
        DROP TABLE IF EXISTS statistics;
        """
    )
    connection.commit()


def flush_postings(connection, buffered_postings):
    if not buffered_postings:
        return
    connection.executemany(
        """
        INSERT INTO postings (term, doc_id, term_frequency)
        VALUES (?, ?, ?)
        """,
        buffered_postings,
    )
    connection.commit()
    buffered_postings.clear()


def create_sql_indexes(connection):
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);
        CREATE INDEX IF NOT EXISTS idx_postings_doc_id ON postings(doc_id);
        CREATE INDEX IF NOT EXISTS idx_documents_split ON documents(split);
        """
    )
    connection.commit()


def build_index(processed_dir, output_dir, splits, batch_size):
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "bm25_index.sqlite"
    if db_path.exists():
        db_path.unlink()

    connection = sqlite3.connect(db_path)
    reset_index_schema(connection)
    ensure_index_schema(connection)

    document_count = 0
    total_document_length = 0
    vocabulary_document_frequency = Counter()
    vocabulary_collection_frequency = Counter()
    buffered_postings = []
    split_totals = load_split_totals(processed_dir)

    for split_name in splits:
        docs_path = processed_dir / f"{split_name}_documents.jsonl"
        if not docs_path.exists():
            raise FileNotFoundError(f"Missing processed documents file: {docs_path}")

        total = split_totals.get(split_name)
        progress = tqdm(
            iter_jsonl(docs_path),
            total=total,
            desc=f"Indexing {split_name}",
            unit="doc",
        )

        for _, record in progress:
            doc_id = record["id"]
            func_name = record["func_name"]
            lexical_text = record["lexical_document"]
            tokens = lexical_text.split()
            if not tokens:
                continue

            document_length = len(tokens)
            term_counts = Counter(tokens)

            connection.execute(
                """
                INSERT INTO documents (doc_id, split, func_name, lexical_token_count)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, split_name, func_name, document_length),
            )

            for term, term_frequency in term_counts.items():
                buffered_postings.append((term, doc_id, term_frequency))
                vocabulary_document_frequency[term] += 1
                vocabulary_collection_frequency[term] += term_frequency

            if len(buffered_postings) >= batch_size:
                flush_postings(connection, buffered_postings)

            document_count += 1
            total_document_length += document_length

            if document_count % 1000 == 0:
                progress.set_postfix(
                    docs_indexed=document_count,
                    vocab=len(vocabulary_document_frequency),
                )

    flush_postings(connection, buffered_postings)

    average_document_length = (
        total_document_length / document_count if document_count else 0.0
    )

    vocabulary_rows = []
    for term, document_frequency in vocabulary_document_frequency.items():
        idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        vocabulary_rows.append(
            (
                term,
                document_frequency,
                vocabulary_collection_frequency[term],
                idf,
            )
        )

    connection.executemany(
        """
        INSERT INTO vocabulary (term, document_frequency, collection_frequency, idf)
        VALUES (?, ?, ?, ?)
        """,
        vocabulary_rows,
    )

    stats = {
        "document_count": document_count,
        "average_document_length": round(average_document_length, 4),
        "vocabulary_size": len(vocabulary_document_frequency),
        "total_token_count": total_document_length,
        "splits": list(splits),
        "bm25_parameters": {"k1": 1.5, "b": 0.75},
    }

    connection.executemany(
        "INSERT INTO statistics (key, value) VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in stats.items()],
    )

    create_sql_indexes(connection)
    connection.close()

    summary_path = output_dir / "index_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

    return db_path, summary_path, stats


def main():
    args = parse_args()
    db_path, summary_path, stats = build_index(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        batch_size=args.batch_size,
    )

    print(f"BM25 index written to: {db_path}")
    print(f"Index summary written to: {summary_path}")
    print(f"Indexed documents: {stats['document_count']}")
    print(f"Vocabulary size: {stats['vocabulary_size']}")
    print(f"Average document length: {stats['average_document_length']}")


if __name__ == "__main__":
    main()
