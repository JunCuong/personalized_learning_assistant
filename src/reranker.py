"""Lightweight keyword + semantic score fusion reranking (no extra dependencies)."""

from __future__ import annotations

import re
from typing import Any


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z0-9]+\b", text.lower())
    return {t for t in tokens if len(t) >= 3}


def _keyword_overlap_score(query: str, chunk_text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenize(chunk_text)
    overlap = sum(1 for t in query_tokens if t in chunk_tokens)
    return overlap / len(query_tokens)


def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    semantic_weight: float = 0.75,
    keyword_weight: float = 0.25,
) -> list[dict[str, Any]]:
    """
    Rerank retrieval candidates by blending FAISS score and query-token overlap.
    Adds semantic_score, keyword_overlap_score, rerank_score; preserves score.
    """
    if not results:
        return []

    reranked: list[dict[str, Any]] = []
    for item in results:
        row = dict(item)
        semantic = float(row.get("score", 0.0))
        keyword = _keyword_overlap_score(query, row.get("chunk_text", ""))
        row["semantic_score"] = semantic
        row["keyword_overlap_score"] = round(keyword, 4)
        row["rerank_score"] = round(
            semantic_weight * semantic + keyword_weight * keyword, 4
        )
        reranked.append(row)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    for rank, row in enumerate(reranked, start=1):
        row["rank"] = rank
    return reranked
