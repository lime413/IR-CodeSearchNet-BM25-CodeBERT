from __future__ import annotations

import html
import json
import pickle
import sqlite3
import sys
from pathlib import Path
from typing import Iterable

import gradio as gr

from dense_utils import (
    DEFAULT_DENSE_INDEX_DIR,
    DEFAULT_MODEL_NAME,
    encode_texts,
    load_doc_id_to_row_map,
    load_model_and_tokenizer,
    resolve_device,
)
from retrieve_bm25_codebert_rerank import rerank_candidates
from retrieve_only_BM25 import (
    DEFAULT_INDEX_DIR,
    DEFAULT_PROCESSED_DIR,
    bm25_score_query,
    configure_logging,
    iter_jsonl,
    load_document_lengths,
    load_index_statistics,
    load_vocabulary,
    make_postings_fetcher,
)


RESULTS_DIR = Path("data/results")
BM25_METRICS_PATH = RESULTS_DIR / "bm25_test_metrics.json"
RERANK_METRICS_PATH = RESULTS_DIR / "bm25_codebert_test_metrics.json"

DEMO_CACHE_DIR = Path("data/indexes")
DOC_OFFSETS_CACHE = DEMO_CACHE_DIR / "demo_doc_offsets.pkl"
EXAMPLES_CACHE = DEMO_CACHE_DIR / "demo_examples.json"

SPLITS = ("train", "valid", "test")
MAX_CODE_CHARS = 1800
EXAMPLE_COUNT = 8


logger = configure_logging()


# ----------------------------------------------------------------------------
# Startup validation
# ----------------------------------------------------------------------------


class StartupError(Exception):
    pass


def validate_artifacts(
    index_dir: Path,
    processed_dir: Path,
    dense_index_dir: Path,
) -> None:
    missing: list[tuple[str, str]] = []

    bm25_db = index_dir / "bm25_index.sqlite"
    if not bm25_db.exists():
        missing.append((str(bm25_db), "python index.py"))

    for split_name in SPLITS:
        docs_path = processed_dir / f"{split_name}_documents.jsonl"
        meta_path = processed_dir / f"{split_name}_metadata.jsonl"
        if not docs_path.exists():
            missing.append((str(docs_path), "python preprocessing.py"))
        if not meta_path.exists():
            missing.append((str(meta_path), "python preprocessing.py"))

    dense_index_path = dense_index_dir / "dense_index.faiss"
    doc_ids_path = dense_index_dir / "document_ids.jsonl"
    if not dense_index_path.exists():
        missing.append((str(dense_index_path), "python build_dense_index.py"))
    if not doc_ids_path.exists():
        missing.append((str(doc_ids_path), "python build_dense_index.py"))

    if not missing:
        return

    lines = ["Missing required artifacts:"]
    for path, _ in missing:
        lines.append(f"  - {path}")
    lines.append("")
    lines.append("Run the pipeline from the project root, in order:")
    for command in (
        "python data/download_CodeSearchNet.py",
        "python preprocessing.py",
        "python index.py",
        "python build_dense_index.py",
    ):
        lines.append(f"  $ {command}")
    raise StartupError("\n".join(lines))


# ----------------------------------------------------------------------------
# Document offset index + lookup
# ----------------------------------------------------------------------------


def _scan_jsonl_offsets(path: Path) -> dict[str, int]:
    offsets: dict[str, int] = {}
    with path.open("rb") as handle:
        offset = handle.tell()
        line = handle.readline()
        while line:
            stripped = line.strip()
            if stripped:
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    offset = handle.tell()
                    line = handle.readline()
                    continue
                doc_id = record.get("id")
                if doc_id:
                    offsets[doc_id] = offset
            offset = handle.tell()
            line = handle.readline()
    return offsets


def build_offset_index(processed_dir: Path, cache_path: Path) -> dict:
    files: list[tuple[str, Path]] = []
    for split_name in SPLITS:
        files.append(("docs", processed_dir / f"{split_name}_documents.jsonl"))
        files.append(("meta", processed_dir / f"{split_name}_metadata.jsonl"))

    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            cached_signature = cached.get("signature")
            signature = {
                str(path): path.stat().st_mtime_ns for _, path in files if path.exists()
            }
            if cached_signature == signature:
                logger.info("Loaded cached offset index from %s", cache_path)
                return cached
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            logger.warning("Failed to read offset cache %s: %s", cache_path, exc)

    logger.info("Building offset index over processed jsonl files...")
    docs_offsets: dict[str, tuple[str, int]] = {}
    meta_offsets: dict[str, tuple[str, int]] = {}
    for kind, path in files:
        if not path.exists():
            continue
        for doc_id, offset in _scan_jsonl_offsets(path).items():
            target = docs_offsets if kind == "docs" else meta_offsets
            target[doc_id] = (str(path), offset)

    signature = {str(path): path.stat().st_mtime_ns for _, path in files if path.exists()}
    result = {"signature": signature, "docs": docs_offsets, "meta": meta_offsets}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(result, handle)
    logger.info(
        "Offset index built: %s docs / %s metadata entries",
        len(docs_offsets),
        len(meta_offsets),
    )
    return result


class DocumentStore:
    def __init__(self, offsets: dict) -> None:
        self._docs = offsets["docs"]
        self._meta = offsets["meta"]
        self._handles: dict[str, "object"] = {}

    def _read_line(self, path: str, offset: int) -> dict | None:
        handle = self._handles.get(path)
        if handle is None:
            handle = open(path, "rb")
            self._handles[path] = handle
        handle.seek(offset)
        line = handle.readline()
        if not line:
            return None
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def get(self, doc_id: str) -> dict:
        result: dict = {"id": doc_id}
        doc_loc = self._docs.get(doc_id)
        if doc_loc is not None:
            record = self._read_line(*doc_loc)
            if record is not None:
                result.update(record)
        meta_loc = self._meta.get(doc_id)
        if meta_loc is not None:
            record = self._read_line(*meta_loc)
            if record is not None:
                for key in ("repository_name", "func_path_in_repository", "func_code_url", "language"):
                    if record.get(key):
                        result[key] = record[key]
        return result


# ----------------------------------------------------------------------------
# BM25 engine
# ----------------------------------------------------------------------------


class BM25Engine:
    def __init__(self, index_dir: Path, cache_size: int = 4096) -> None:
        db_path = index_dir / "bm25_index.sqlite"
        self._connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self._stats = load_index_statistics(self._connection)
        self._doc_lengths = load_document_lengths(self._connection)
        self._vocabulary = load_vocabulary(self._connection)
        self._get_postings = make_postings_fetcher(self._connection, cache_size)
        self._avg_doc_len = self._stats["average_document_length"]
        logger.info(
            "BM25 index loaded from %s (docs=%s, vocab=%s)",
            db_path,
            self._stats.get("document_count"),
            self._stats.get("vocabulary_size"),
        )

    @property
    def stats(self) -> dict:
        return self._stats

    def search(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        return bm25_score_query(
            query_text=query_text,
            vocabulary=self._vocabulary,
            doc_lengths=self._doc_lengths,
            avg_doc_len=self._avg_doc_len,
            get_postings=self._get_postings,
            top_k=top_k,
        )


# ----------------------------------------------------------------------------
# Rerank engine (lazy)
# ----------------------------------------------------------------------------


class RerankEngine:
    def __init__(
        self,
        dense_index_dir: Path,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 256,
    ) -> None:
        self._dense_index_dir = dense_index_dir
        self._model_name = model_name
        self._max_length = max_length
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._device = None
        self._dense_index = None
        self._doc_id_to_row: dict[str, int] = {}

    @property
    def loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        import faiss  # imported lazily

        logger.info("Loading CodeBERT (%s) and FAISS index...", self._model_name)
        self._device = resolve_device(None)
        self._tokenizer, self._model = load_model_and_tokenizer(
            self._model_name, self._device
        )
        dense_index_path = self._dense_index_dir / "dense_index.faiss"
        doc_ids_path = self._dense_index_dir / "document_ids.jsonl"
        try:
            self._dense_index = faiss.read_index(str(dense_index_path))
        except RuntimeError as error:
            file_size = (
                dense_index_path.stat().st_size
                if dense_index_path.exists()
                else 0
            )
            raise RuntimeError(
                f"Failed to read FAISS index at {dense_index_path} "
                f"(on-disk size: {file_size} bytes). The file is likely truncated "
                "or corrupt. Rebuild it with: python build_dense_index.py\n"
                f"Original error: {error}"
            ) from error
        self._doc_id_to_row = load_doc_id_to_row_map(doc_ids_path)
        self._loaded = True
        logger.info(
            "CodeBERT reranker ready (device=%s, vectors=%s)",
            self._device,
            self._dense_index.ntotal,
        )

    def rerank(
        self,
        query_text: str,
        bm25_candidates: list[tuple[str, float]],
        top_k: int,
    ) -> list[dict]:
        self.ensure_loaded()
        if not bm25_candidates:
            return []
        query_embedding = encode_texts(
            [query_text],
            self._tokenizer,
            self._model,
            self._device,
            self._max_length,
        )[0]
        reranked = rerank_candidates(
            query_embedding=query_embedding,
            ranked_candidates=bm25_candidates,
            doc_id_to_row=self._doc_id_to_row,
            dense_index=self._dense_index,
        )
        return reranked[:top_k]


# ----------------------------------------------------------------------------
# Example queries
# ----------------------------------------------------------------------------


def load_examples(
    processed_dir: Path,
    bm25_engine: BM25Engine,
    cache_path: Path,
    count: int = EXAMPLE_COUNT,
) -> list[str]:
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if isinstance(cached, list) and cached:
                return [str(item) for item in cached[:count]]
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read examples cache %s: %s", cache_path, exc)

    logger.info("Selecting preset example queries...")
    test_docs = processed_dir / "test_documents.jsonl"
    chosen: list[str] = []
    seen_texts: set[str] = set()

    for line_number, record in iter_jsonl(test_docs):
        if line_number > 5000:
            break
        docstring = (record.get("docstring") or "").strip()
        if not docstring or len(docstring) > 150 or "\n" in docstring:
            continue
        if docstring in seen_texts:
            continue
        doc_id = record.get("id")
        if not doc_id:
            continue
        ranked = bm25_engine.search(docstring, top_k=5)
        if not ranked:
            continue
        top_id, _ = ranked[0]
        if top_id == doc_id:
            chosen.append(docstring)
            seen_texts.add(docstring)
            if len(chosen) >= count:
                break

    if not chosen:
        chosen = [
            "read a json file and return a dict",
            "establish an ssh connection",
            "compute the sha256 hash of a string",
            "split a list into chunks of size n",
            "download a url and save to disk",
            "parse an iso 8601 timestamp",
        ]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(chosen, handle, indent=2, ensure_ascii=False)
    logger.info("Selected %s example queries", len(chosen))
    return chosen


# ----------------------------------------------------------------------------
# Result rendering
# ----------------------------------------------------------------------------


def _truncate_code(code: str) -> str:
    if len(code) <= MAX_CODE_CHARS:
        return code
    return code[:MAX_CODE_CHARS] + "\n# ... truncated ..."


def _result_header(rank: int, doc_id: str, record: dict, score_label: str, score_value: float, secondary: str | None) -> str:
    func_name = record.get("func_name") or "(unknown function)"
    repo = record.get("repository_name") or ""
    url = record.get("func_code_url") or ""
    title_parts = [f"**#{rank}** `{func_name}`"]
    if repo:
        title_parts.append(f"in `{repo}`")
    title = " ".join(title_parts)
    score_bits = [f"{score_label}={score_value:.4f}"]
    if secondary:
        score_bits.append(secondary)
    score_bits.append(f"id=`{doc_id}`")
    score_line = " · ".join(score_bits)
    link_line = f"[source]({url})" if url else ""
    header = f"{title}\n\n{score_line}"
    if link_line:
        header += f"\n\n{link_line}"
    return header


def render_results(
    results: Iterable[dict],
    documents: DocumentStore,
    score_label: str,
) -> str:
    blocks: list[str] = []
    any_results = False
    for index, item in enumerate(results, start=1):
        any_results = True
        doc_id = item["doc_id"]
        record = documents.get(doc_id)
        secondary = None
        if "bm25_score" in item and score_label != "bm25":
            secondary = f"bm25={item['bm25_score']:.4f}"
        header = _result_header(
            rank=index,
            doc_id=doc_id,
            record=record,
            score_label=score_label,
            score_value=item["score"],
            secondary=secondary,
        )
        docstring = (record.get("docstring") or "").strip()
        code = _truncate_code((record.get("code") or "").strip())
        code_block = f"```python\n{code}\n```" if code else "_(code body not found)_"
        docstring_block = ""
        if docstring:
            docstring_block = f"> {html.escape(docstring)}\n\n"
        blocks.append(f"{header}\n\n{docstring_block}{code_block}")
    if not any_results:
        return "_No matches found. Try a different query._"
    return "\n\n---\n\n".join(blocks)


# ----------------------------------------------------------------------------
# Metrics tab
# ----------------------------------------------------------------------------


def _load_metrics_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load metrics %s: %s", path, exc)
        return None


def _format_metric(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_metrics_table(bm25_doc: dict | None, rerank_doc: dict | None) -> str:
    rows = [
        ("Queries evaluated", "query_count"),
        ("MRR@10", "MRR@10"),
        ("Recall@10", "Recall@10"),
        ("Recall@50", "Recall@50"),
        ("nDCG@10", "nDCG@10"),
    ]
    bm25_metrics = (bm25_doc or {}).get("metrics", {})
    rerank_metrics = (rerank_doc or {}).get("metrics", {})
    bm25_exec = (bm25_doc or {}).get("execution", {})
    rerank_exec = (rerank_doc or {}).get("execution", {})

    lines = ["| Metric | BM25 | BM25 + CodeBERT rerank |", "| --- | --- | --- |"]
    for label, key in rows:
        lines.append(
            f"| {label} | {_format_metric(bm25_metrics.get(key))} | {_format_metric(rerank_metrics.get(key))} |"
        )
    lines.append(
        "| Elapsed | "
        f"{_format_metric(bm25_exec.get('elapsed_hms'))} | "
        f"{_format_metric(rerank_exec.get('elapsed_hms'))} |"
    )
    if bm25_doc is None:
        lines.append("")
        lines.append(f"_{BM25_METRICS_PATH} not found. Run `python retrieve_only_BM25.py` to produce it._")
    if rerank_doc is None:
        lines.append("")
        lines.append(
            f"_{RERANK_METRICS_PATH} not found. "
            "Run `python retrieve_bm25_codebert_rerank.py` to produce it._"
        )
    return "\n".join(lines)


def build_sample_rankings_markdown(doc: dict | None, pipeline_label: str) -> str:
    if doc is None:
        return f"_No metrics file available for {pipeline_label}._"
    samples = doc.get("sample_rankings", [])
    if not samples:
        return f"_No sample rankings recorded for {pipeline_label}._"
    blocks: list[str] = []
    for sample in samples:
        query_text = (sample.get("query_text") or "").strip()
        func_name = sample.get("func_name") or ""
        query_id = sample.get("query_id") or ""
        top_results = sample.get("top_results", [])
        lines = [f"**Query** (`{query_id}` · `{func_name}`): {query_text}"]
        if top_results:
            lines.append("")
            lines.append("| Rank | doc_id | score | bm25_score |")
            lines.append("| --- | --- | --- | --- |")
            for rank, row in enumerate(top_results, start=1):
                lines.append(
                    f"| {rank} | `{row.get('doc_id')}` | "
                    f"{_format_metric(row.get('score'))} | "
                    f"{_format_metric(row.get('bm25_score'))} |"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ----------------------------------------------------------------------------
# Gradio UI
# ----------------------------------------------------------------------------


def build_ui(
    bm25_engine: BM25Engine,
    rerank_engine: RerankEngine,
    documents: DocumentStore,
    examples: list[str],
) -> gr.Blocks:
    bm25_metrics_doc = _load_metrics_json(BM25_METRICS_PATH)
    rerank_metrics_doc = _load_metrics_json(RERANK_METRICS_PATH)

    def run_search(query_text: str, top_k: int, bm25_candidates: int):
        query_text = (query_text or "").strip()
        if not query_text:
            empty = "_Type a natural-language query and press **Search**._"
            return empty, empty, "_Waiting for a query._"

        top_k = int(top_k)
        bm25_candidates = max(int(bm25_candidates), top_k)

        bm25_depth = max(bm25_candidates, top_k)
        bm25_ranked = bm25_engine.search(query_text, top_k=bm25_depth)

        bm25_display = [
            {"doc_id": doc_id, "score": float(score)}
            for doc_id, score in bm25_ranked[:top_k]
        ]
        bm25_markdown = render_results(bm25_display, documents, score_label="bm25")

        if not rerank_engine.loaded:
            gr.Info("Loading CodeBERT and FAISS index (first call only)...")
        try:
            reranked = rerank_engine.rerank(
                query_text=query_text,
                bm25_candidates=bm25_ranked,
                top_k=top_k,
            )
        except Exception as error:
            logger.exception("Rerank failed")
            rerank_markdown = (
                "### Rerank unavailable\n\n"
                f"```\n{error}\n```\n\n"
                "BM25 results are still shown on the left."
            )
            stats_markdown = (
                f"query: `{query_text}` · top_k: {top_k} · bm25 hits: {len(bm25_ranked)} · "
                "rerank: **error** (see left panel unaffected)"
            )
            return bm25_markdown, rerank_markdown, stats_markdown

        rerank_markdown = render_results(reranked, documents, score_label="cosine")

        stats_parts = [
            f"query: `{query_text}`",
            f"top_k: {top_k}",
            f"bm25 candidates: {bm25_candidates}",
            f"bm25 hits: {len(bm25_ranked)}",
            f"reranked: {len(reranked)}",
        ]
        stats_markdown = " · ".join(stats_parts)
        return bm25_markdown, rerank_markdown, stats_markdown

    with gr.Blocks(title="CodeSearchNet · BM25 vs CodeBERT rerank") as demo:
        gr.Markdown(
            "# CodeSearchNet retrieval demo\n"
            "Compare **BM25** (lexical) retrieval with **BM25 + CodeBERT rerank** "
            "(dense reranking) on the Python split of CodeSearchNet."
        )

        with gr.Tabs():
            with gr.Tab("Search"):
                with gr.Row():
                    query_box = gr.Textbox(
                        label="Query",
                        placeholder="e.g. parse json from a file and return a dict",
                        lines=2,
                        scale=4,
                    )
                    run_button = gr.Button("Search", variant="primary", scale=1)

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Top K",
                    )
                    candidates_slider = gr.Slider(
                        minimum=50,
                        maximum=200,
                        value=50,
                        step=10,
                        label="BM25 candidates for rerank",
                    )

                gr.Examples(
                    examples=[[example] for example in examples],
                    inputs=[query_box],
                    label="Example queries (from the test split)",
                )

                stats_md = gr.Markdown("_Waiting for a query._")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### BM25 (lexical)")
                        bm25_output = gr.Markdown("_No results yet._")
                    with gr.Column():
                        gr.Markdown("### BM25 + CodeBERT rerank (dense)")
                        rerank_output = gr.Markdown("_No results yet._")

                run_button.click(
                    fn=run_search,
                    inputs=[query_box, top_k_slider, candidates_slider],
                    outputs=[bm25_output, rerank_output, stats_md],
                )
                query_box.submit(
                    fn=run_search,
                    inputs=[query_box, top_k_slider, candidates_slider],
                    outputs=[bm25_output, rerank_output, stats_md],
                )

            with gr.Tab("Evaluation metrics"):
                gr.Markdown(
                    "Precomputed metrics from the evaluation scripts over the "
                    "CodeSearchNet Python test split."
                )
                gr.Markdown(build_metrics_table(bm25_metrics_doc, rerank_metrics_doc))

                with gr.Accordion("Sample rankings — BM25", open=False):
                    gr.Markdown(
                        build_sample_rankings_markdown(bm25_metrics_doc, "BM25")
                    )
                with gr.Accordion("Sample rankings — BM25 + CodeBERT rerank", open=False):
                    gr.Markdown(
                        build_sample_rankings_markdown(
                            rerank_metrics_doc, "BM25 + CodeBERT rerank"
                        )
                    )

                with gr.Accordion("Raw JSON · bm25_test_metrics.json", open=False):
                    gr.JSON(bm25_metrics_doc or {})
                with gr.Accordion("Raw JSON · bm25_codebert_test_metrics.json", open=False):
                    gr.JSON(rerank_metrics_doc or {})

        gr.Markdown(
            f"Indexed documents: **{bm25_engine.stats.get('document_count', '—')}** · "
            f"vocabulary: **{bm25_engine.stats.get('vocabulary_size', '—')}** · "
            f"BM25 k1=1.5, b=0.75"
        )

    return demo


# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------


def main() -> None:
    index_dir = DEFAULT_INDEX_DIR
    processed_dir = DEFAULT_PROCESSED_DIR
    dense_index_dir = DEFAULT_DENSE_INDEX_DIR

    try:
        validate_artifacts(index_dir, processed_dir, dense_index_dir)
    except StartupError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

    bm25_engine = BM25Engine(index_dir)
    offsets = build_offset_index(processed_dir, DOC_OFFSETS_CACHE)
    documents = DocumentStore(offsets)
    examples = load_examples(processed_dir, bm25_engine, EXAMPLES_CACHE)
    rerank_engine = RerankEngine(dense_index_dir)

    demo = build_ui(bm25_engine, rerank_engine, documents, examples)
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
