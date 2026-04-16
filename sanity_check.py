import argparse
import json
import random
from pathlib import Path


PROCESSED_DIR = Path("data/processed/CodeSearchNet_python")
SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sanity checks on preprocessed CodeSearchNet Python files."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing processed documents, metadata, and corpus stats.",
    )
    parser.add_argument(
        "--samples-per-split",
        type=int,
        default=2,
        help="How many sample examples to print from each split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible example sampling.",
    )
    return parser.parse_args()


def load_jsonl(path):
    items = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                items.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
    return items


def preview(text, limit=160):
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3] + "..."


def check_split(processed_dir, split_name, samples_per_split, rng):
    docs_path = processed_dir / f"{split_name}_documents.jsonl"
    metadata_path = processed_dir / f"{split_name}_metadata.jsonl"
    docs = load_jsonl(docs_path)
    metadata = load_jsonl(metadata_path)

    if len(docs) != len(metadata):
        raise ValueError(
            f"Mismatch in {split_name}: {len(docs)} docs vs {len(metadata)} metadata rows"
        )

    doc_ids = {item["id"] for item in docs}
    metadata_ids = {item["id"] for item in metadata}
    if doc_ids != metadata_ids:
        raise ValueError(f"ID mismatch between document and metadata files for {split_name}")

    for item in docs:
        if not item["func_name"].strip():
            raise ValueError(f"Empty function name in {split_name}: {item['id']}")
        if not item["code"].strip():
            raise ValueError(f"Empty code in {split_name}: {item['id']}")
        if not item["lexical_document"].strip():
            raise ValueError(f"Empty lexical document in {split_name}: {item['id']}")

    sample_count = min(samples_per_split, len(docs))
    samples = rng.sample(docs, sample_count) if sample_count else []

    print(f"\n[{split_name}] documents={len(docs)}")
    for sample in samples:
        print(f"  id: {sample['id']}")
        print(f"  func_name: {sample['func_name']}")
        print(f"  has_docstring: {sample['has_docstring']}")
        print(f"  lexical_token_count: {sample['lexical_token_count']}")
        print(f"  docstring_preview: {preview(sample['docstring'])}")
        print(f"  code_preview: {preview(sample['code'])}")
        print(f"  lexical_preview: {preview(sample['lexical_document'])}")


def main():
    args = parse_args()
    stats_path = args.processed_dir / "corpus_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing {stats_path}. Run preprocessing.py before sanity_check.py."
        )

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)

    rng = random.Random(args.seed)
    print(f"Loaded stats from: {stats_path}")
    print(f"Total kept records: {stats['total_kept_records']}")

    for split_name in SPLITS:
        split_stats = stats["splits"].get(split_name)
        if split_stats is None:
            raise ValueError(f"Missing split stats for {split_name}")
        print(
            f"{split_name}: kept={split_stats['kept_records']}, "
            f"missing_docstring_rate={split_stats['missing_docstring_rate']}, "
            f"avg_lexical_tokens={split_stats['lexical_token_avg']}"
        )
        check_split(args.processed_dir, split_name, args.samples_per_split, rng)

    print("\nSanity checks passed.")


if __name__ == "__main__":
    main()
