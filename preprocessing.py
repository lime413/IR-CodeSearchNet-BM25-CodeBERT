import argparse
import json
import re
from collections import Counter
from pathlib import Path


RAW_DIR = Path("data/raw/CodeSearchNet_python")
PROCESSED_DIR = Path("data/processed/CodeSearchNet_python")
SPLITS = ("train", "valid", "test")
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
CAMEL_CASE_PATTERN = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
WHITESPACE_PATTERN = re.compile(r"\s+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess the CodeSearchNet Python subset for lexical retrieval."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory with train.jsonl, valid.jsonl, and test.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory where processed documents, metadata, and stats will be written.",
    )
    return parser.parse_args()


def normalize_whitespace(text):
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def split_identifier(text):
    pieces = []
    for chunk in re.split(r"[^A-Za-z0-9_]+", text):
        if not chunk:
            continue
        for snake_part in chunk.split("_"):
            if not snake_part:
                continue
            pieces.extend(part for part in CAMEL_CASE_PATTERN.split(snake_part) if part)
    return pieces


def tokenize_for_bm25(*parts):
    tokens = []
    for part in parts:
        for match in TOKEN_PATTERN.findall(part):
            subtokens = split_identifier(match)
            if subtokens:
                tokens.extend(token.lower() for token in subtokens)
            else:
                tokens.append(match.lower())
    return tokens


def clean_docstring(docstring):
    return normalize_whitespace(docstring or "")


def clean_code(code):
    return (code or "").strip()


def build_document_text(func_name, docstring, code):
    sections = [func_name.strip(), clean_docstring(docstring), clean_code(code)]
    return "\n\n".join(section for section in sections if section)


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


def update_length_histogram(counter, length):
    bucket_start = (length // 50) * 50
    bucket_label = f"{bucket_start:04d}-{bucket_start + 49:04d}"
    counter[bucket_label] += 1


def preprocess_record(record, split_name, record_id):
    func_name = (record.get("func_name") or "").strip()
    code = clean_code(record.get("func_code_string") or record.get("whole_func_string") or "")
    docstring = clean_docstring(record.get("func_documentation_string") or "")

    if not func_name or not code:
        return None

    document_text = build_document_text(func_name, docstring, code)
    lexical_tokens = tokenize_for_bm25(func_name, docstring, code)
    if not lexical_tokens:
        return None

    doc = {
        "id": record_id,
        "split": split_name,
        "func_name": func_name,
        "docstring": docstring,
        "code": code,
        "document_text": document_text,
        "lexical_document": " ".join(lexical_tokens),
        "lexical_token_count": len(lexical_tokens),
        "has_docstring": bool(docstring),
    }

    metadata = {
        "id": record_id,
        "split": split_name,
        "language": record.get("language", "python"),
        "repository_name": record.get("repository_name", ""),
        "func_path_in_repository": record.get("func_path_in_repository", ""),
        "func_name": func_name,
        "func_code_url": record.get("func_code_url", ""),
    }
    return doc, metadata


def process_split(raw_path, output_dir, split_name):
    docs_path = output_dir / f"{split_name}_documents.jsonl"
    metadata_path = output_dir / f"{split_name}_metadata.jsonl"

    stats = {
        "raw_records": 0,
        "kept_records": 0,
        "dropped_missing_name": 0,
        "dropped_missing_code": 0,
        "dropped_empty_lexical": 0,
        "records_with_docstring": 0,
        "records_without_docstring": 0,
        "document_char_total": 0,
        "document_char_avg": 0.0,
        "lexical_token_total": 0,
        "lexical_token_avg": 0.0,
        "docstring_char_total": 0,
        "docstring_char_avg": 0.0,
        "length_histogram": Counter(),
    }

    with docs_path.open("w", encoding="utf-8") as docs_handle, metadata_path.open(
        "w", encoding="utf-8"
    ) as metadata_handle:
        for line_number, record in iter_jsonl(raw_path):
            stats["raw_records"] += 1
            func_name = (record.get("func_name") or "").strip()
            code = clean_code(record.get("func_code_string") or record.get("whole_func_string") or "")
            if not func_name:
                stats["dropped_missing_name"] += 1
                continue
            if not code:
                stats["dropped_missing_code"] += 1
                continue

            record_id = f"{split_name}-{line_number}"
            processed = preprocess_record(record, split_name, record_id)
            if processed is None:
                stats["dropped_empty_lexical"] += 1
                continue

            doc, metadata = processed
            json.dump(doc, docs_handle, ensure_ascii=False)
            docs_handle.write("\n")
            json.dump(metadata, metadata_handle, ensure_ascii=False)
            metadata_handle.write("\n")

            stats["kept_records"] += 1
            stats["document_char_total"] += len(doc["document_text"])
            stats["lexical_token_total"] += doc["lexical_token_count"]
            stats["docstring_char_total"] += len(doc["docstring"])
            update_length_histogram(stats["length_histogram"], doc["lexical_token_count"])

            if doc["has_docstring"]:
                stats["records_with_docstring"] += 1
            else:
                stats["records_without_docstring"] += 1

    kept = stats["kept_records"] or 1
    stats["document_char_avg"] = round(stats["document_char_total"] / kept, 2)
    stats["lexical_token_avg"] = round(stats["lexical_token_total"] / kept, 2)
    stats["docstring_char_avg"] = round(stats["docstring_char_total"] / kept, 2)
    stats["missing_docstring_rate"] = round(
        stats["records_without_docstring"] / kept, 4
    )
    stats["length_histogram"] = dict(sorted(stats["length_histogram"].items()))
    return stats


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {"raw_dir": str(args.raw_dir), "output_dir": str(args.output_dir), "splits": {}}
    total_kept = 0

    for split_name in SPLITS:
        raw_path = args.raw_dir / f"{split_name}.jsonl"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing input file: {raw_path}")
        split_stats = process_split(raw_path, args.output_dir, split_name)
        all_stats["splits"][split_name] = split_stats
        total_kept += split_stats["kept_records"]

    all_stats["total_kept_records"] = total_kept
    stats_path = args.output_dir / "corpus_stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(all_stats, handle, indent=2, ensure_ascii=False)

    print(f"Processed dataset written to: {args.output_dir}")
    print(f"Corpus statistics written to: {stats_path}")


if __name__ == "__main__":
    main()
