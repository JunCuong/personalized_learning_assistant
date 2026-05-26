"""End-to-end pipeline: load PDFs -> chunk -> build FAISS index."""

from __future__ import annotations

import logging

from src.pdf_loader import run_pdf_loading
from src.text_chunker import run_chunking
from src.vector_db import build_index

logger = logging.getLogger(__name__)


def build_knowledge_base() -> dict:
    """Run full ingestion and indexing pipeline."""
    _, pdf_summary = run_pdf_loading(save=True)
    chunks, chunk_summary = run_chunking(save=True)

    if not chunks:
        return {
            "status": "error",
            "message": (
                "No text chunks created. Most PDFs may be scanned images "
                "without extractable text."
            ),
            "pdf_summary": pdf_summary,
            "chunk_summary": chunk_summary,
        }

    index_summary = build_index()
    return {
        "status": "ok",
        "pdf_summary": pdf_summary,
        "chunk_summary": chunk_summary,
        "index_summary": index_summary,
    }
