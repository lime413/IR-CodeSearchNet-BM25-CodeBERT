import json
from pathlib import Path

import dense_runtime  # noqa: F401
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from retrieve_only_BM25 import iter_jsonl


DEFAULT_DENSE_INDEX_DIR = Path("data/indexes/CodeSearchNet_python_codebert")
DEFAULT_MODEL_NAME = "microsoft/codebert-base"
AVAILABLE_SPLITS = ("train", "valid", "test")


def resolve_device(device_name=None):
    if device_name:
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def mean_pool_embeddings(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_hidden = last_hidden_state * mask
    token_sums = masked_hidden.sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)
    return token_sums / token_counts


@torch.inference_mode()
def encode_texts(texts, tokenizer, model, device, max_length):
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded)
    embeddings = mean_pool_embeddings(outputs.last_hidden_state, encoded["attention_mask"])
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype("float32")


def iter_documents(processed_dir, splits):
    for split_name in splits:
        docs_path = processed_dir / f"{split_name}_documents.jsonl"
        if not docs_path.exists():
            raise FileNotFoundError(f"Missing processed documents file: {docs_path}")

        for _, record in iter_jsonl(docs_path):
            document_text = (record.get("document_text") or "").strip()
            if not document_text:
                continue
            yield split_name, record


def build_query_records(processed_dir, query_split):
    docs_path = processed_dir / f"{query_split}_documents.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"Missing processed documents file: {docs_path}")

    queries = []
    for _, record in iter_jsonl(docs_path):
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
    return queries


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")


def load_jsonl_records(path):
    records = []
    for _, record in iter_jsonl(path):
        records.append(record)
    return records


def load_doc_id_to_row_map(path):
    mapping = {}
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            doc_id = line.strip()
            if doc_id:
                mapping[doc_id] = row_index
    return mapping


def get_query_file_paths(dense_index_dir, query_split):
    metadata_path = dense_index_dir / f"{query_split}_queries.jsonl"
    embeddings_path = dense_index_dir / f"{query_split}_query_embeddings.npy"
    return metadata_path, embeddings_path


def reconstruct_vectors(index, row_ids):
    if not row_ids:
        return np.empty((0, index.d), dtype="float32")

    vector_ids = np.asarray(row_ids, dtype="int64")
    if hasattr(index, "reconstruct_batch"):
        return np.asarray(index.reconstruct_batch(vector_ids), dtype="float32")

    vectors = np.empty((len(row_ids), index.d), dtype="float32")
    for position, row_id in enumerate(row_ids):
        vectors[position] = index.reconstruct(int(row_id))
    return vectors
