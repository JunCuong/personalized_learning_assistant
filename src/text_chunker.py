"""Character-based chunking with short-chunk merge and noise filtering."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    DOCUMENTS_PATH,
    MERGE_SHORT_CHUNKS,
    MIN_CHUNK_CHARS,
)
from src.utils import load_json, save_json

logger = logging.getLogger(__name__)

_MIN_ALPHA_CHARS = 20
_PAGE_NUMBER_ONLY = re.compile(r"^[\d\s\-\.]+$")


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def _alpha_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def _is_meaningful_chunk(text: str) -> bool:
    if _alpha_count(text) < _MIN_ALPHA_CHARS:
        return False
    if _PAGE_NUMBER_ONLY.match(text.strip()):
        return False
    return True


def _postprocess_chunks(
    raw_chunks: list[str],
    merge_short: bool = MERGE_SHORT_CHUNKS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[str]:
    if not raw_chunks:
        return []

    processed: list[str] = []
    for chunk in raw_chunks:
        if merge_short and processed and len(chunk) < min_chars:
            merged = (processed[-1] + " " + chunk).strip()
            if len(merged) >= min_chars or _is_meaningful_chunk(merged):
                processed[-1] = merged
            elif _is_meaningful_chunk(chunk):
                processed.append(chunk)
            continue

        if len(chunk) < min_chars and not _is_meaningful_chunk(chunk):
            if merge_short and processed:
                processed[-1] = (processed[-1] + " " + chunk).strip()
            continue

        processed.append(chunk)

    return [c for c in processed if _is_meaningful_chunk(c)]


def chunk_documents(
    documents: list[dict] | None = None,
    documents_path: Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    if documents is None:
        path = documents_path or DOCUMENTS_PATH
        documents = load_json(path)

    all_chunks: list[dict] = []

    for doc in documents:
        page_text = doc.get("text", "")
        if not page_text:
            continue

        doc_id = doc["doc_id"]
        source = doc["source"]
        page = doc["page"]
        extraction_method = doc.get("extraction_method", "pypdf")

        raw = chunk_text(page_text, chunk_size, chunk_overlap)
        final_texts = _postprocess_chunks(raw)

        for chunk_index, chunk_text_value in enumerate(final_texts):
            chunk_id = f"{doc_id}_p{page}_c{chunk_index}"
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source": source,
                    "page": page,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text_value,
                    "chunk_length": len(chunk_text_value),
                    "extraction_method": extraction_method,
                }
            )

    logger.info(
        "Created %d chunks from %d document pages", len(all_chunks), len(documents)
    )
    return all_chunks


def save_chunks(chunks: list[dict], output_path: Path | None = None) -> Path:
    output_path = output_path or CHUNKS_PATH
    save_json(output_path, chunks)
    return output_path


def run_chunking(
    documents_path: Path | None = None, save: bool = True
) -> tuple[list[dict], dict]:
    chunks = chunk_documents(documents_path=documents_path)
    under_100 = sum(1 for c in chunks if c["chunk_length"] < 100)
    under_200 = sum(1 for c in chunks if c["chunk_length"] < 200)
    summary = {
        "chunk_count": len(chunks),
        "chunks_under_100": under_100,
        "chunks_under_200": under_200,
    }
    if save:
        path = save_chunks(chunks)
        summary["saved_to"] = str(path)
    if not chunks:
        logger.warning(
            "No chunks created. Check documents.json and OCR/Poppler setup."
        )
    return chunks, summary
