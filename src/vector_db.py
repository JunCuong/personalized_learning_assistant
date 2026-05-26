"""FAISS vector index build/load with cosine similarity (normalized + IndexFlatIP)."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

from src.config import CHUNKS_PATH, FAISS_INDEX_PATH, METADATA_PATH, VECTOR_STORE_DIR
from src.embedder import Embedder
from src.utils import load_json, save_json

logger = logging.getLogger(__name__)


def index_exists(
    index_path: Path | None = None, metadata_path: Path | None = None
) -> bool:
    index_path = index_path or FAISS_INDEX_PATH
    metadata_path = metadata_path or METADATA_PATH
    return index_path.exists() and metadata_path.exists()


def build_index(
    chunks_path: Path | None = None,
    index_path: Path | None = None,
    metadata_path: Path | None = None,
    embedder: Embedder | None = None,
) -> dict:
    """Embed chunks and build FAISS index + metadata mapping."""
    chunks_path = chunks_path or CHUNKS_PATH
    index_path = index_path or FAISS_INDEX_PATH
    metadata_path = metadata_path or METADATA_PATH

    chunks = load_json(chunks_path)
    if not chunks:
        raise ValueError(
            f"No chunks found at {chunks_path}. Run PDF loading and chunking first."
        )

    embedder = embedder or Embedder()
    texts = [c["chunk_text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)

    dim = embeddings.shape[1]
    # Embeddings already L2-normalized by embedder; inner product = cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append(
            {
                "faiss_id": i,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "chunk_text": chunk["chunk_text"],
            }
        )
    save_json(metadata_path, metadata)

    summary = {
        "num_vectors": index.ntotal,
        "dimension": dim,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
    }
    logger.info("FAISS index built: %s", summary)
    return summary


def load_index(
    index_path: Path | None = None, metadata_path: Path | None = None
) -> tuple[faiss.Index, list[dict]]:
    index_path = index_path or FAISS_INDEX_PATH
    metadata_path = metadata_path or METADATA_PATH

    if not index_exists(index_path, metadata_path):
        raise FileNotFoundError(
            f"FAISS index or metadata missing. Build index first.\n"
            f"  index: {index_path}\n  metadata: {metadata_path}"
        )

    index = faiss.read_index(str(index_path))
    metadata = load_json(metadata_path)
    return index, metadata
