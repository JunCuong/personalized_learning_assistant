"""FAISS-based semantic retriever with optional lightweight reranking."""

from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np

from src.config import DEFAULT_TOP_K
from src.embedder import Embedder
from src.reranker import rerank_results
from src.vector_db import index_exists, load_index

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self._index: faiss.Index | None = None
        self._metadata: list[dict] | None = None

    def load(self) -> None:
        if not index_exists():
            raise FileNotFoundError(
                "Vector index not found. Run build_index() or POST /build-index first."
            )
        self._index, self._metadata = load_index()

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and self._metadata is not None

    def _search_faiss(self, query_vec: np.ndarray, k: int) -> list[dict[str, Any]]:
        assert self._index is not None
        assert self._metadata is not None

        scores, indices = self._index.search(query_vec, k)
        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(
            zip(indices[0].tolist(), scores[0].tolist()), start=1
        ):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "chunk_id": meta["chunk_id"],
                    "doc_id": meta["doc_id"],
                    "source": meta["source"],
                    "page": meta["page"],
                    "chunk_index": meta["chunk_index"],
                    "chunk_text": meta["chunk_text"],
                }
            )
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        use_reranking: bool = False,
        candidate_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not self.is_loaded:
            self.load()

        assert self._index is not None

        query_vec = self.embedder.embed_query(query).astype(np.float32)
        query_vec = query_vec.reshape(1, -1)

        if use_reranking:
            k = min(max(candidate_k, top_k), self._index.ntotal)
            candidates = self._search_faiss(query_vec, k)
            reranked = rerank_results(query, candidates)
            return reranked[:top_k]

        k = min(top_k, self._index.ntotal)
        return self._search_faiss(query_vec, k)
