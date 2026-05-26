"""Run retrieval diagnostics and export ranked results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_TOP_K, DIAGNOSTICS_DIR
from src.diagnostics_io import save_csv, save_json_report
from src.retriever import Retriever
from src.vector_db import index_exists

CSV_PATH = DIAGNOSTICS_DIR / "retrieval_diagnostics.csv"
JSON_PATH = DIAGNOSTICS_DIR / "retrieval_diagnostics_summary.json"
PREVIEW_LEN = 300

DEFAULT_QUERIES = [
    "What is retrieval augmented generation?",
    "What are the main components of RAG?",
    "What is multimodal machine learning?",
    "What is federated learning?",
    "What is MLOps?",
    "What is Kolmogorov-Arnold Network?",
]


def run_queries(queries: list[str], top_k: int = DEFAULT_TOP_K) -> tuple[list[dict], dict]:
    if not index_exists():
        raise FileNotFoundError(
            "FAISS index not found. Build knowledge base first:\n"
            "  python -c \"from src.pipeline import build_knowledge_base; build_knowledge_base()\""
        )

    retriever = Retriever()
    retriever.load()

    csv_rows: list[dict] = []
    json_summary: dict = {}

    for query in queries:
        hits = retriever.retrieve(query, top_k=top_k)
        top3_sources = []
        for h in hits[:3]:
            top3_sources.append(f"{h['source']} p{h['page']}")

        if hits:
            comment = f"Retrieved {len(hits)} chunk(s); top source: {hits[0]['source']} page {hits[0]['page']}"
        else:
            comment = "No results returned (empty index or no matches)"

        json_summary[query] = {
            "top_1_source": hits[0]["source"] if hits else None,
            "top_1_page": hits[0]["page"] if hits else None,
            "top_3_sources": top3_sources,
            "whether_results_exist": bool(hits),
            "comment": comment,
        }

        for h in hits:
            text = h.get("chunk_text", "")
            csv_rows.append(
                {
                    "query": query,
                    "rank": h["rank"],
                    "score": round(h["score"], 4),
                    "source": h["source"],
                    "page": h["page"],
                    "chunk_id": h["chunk_id"],
                    "chunk_text_preview": text[:PREVIEW_LEN],
                }
            )

    overall = {
        "num_queries": len(queries),
        "top_k": top_k,
        "queries_with_results": sum(
            1 for q in queries if json_summary[q]["whether_results_exist"]
        ),
        "queries_without_results": sum(
            1 for q in queries if not json_summary[q]["whether_results_exist"]
        ),
    }

    save_csv(
        csv_rows,
        CSV_PATH,
        fieldnames=[
            "query",
            "rank",
            "score",
            "source",
            "page",
            "chunk_id",
            "chunk_text_preview",
        ],
    )
    save_json_report({"overall": overall, "per_query": json_summary}, JSON_PATH)

    print(f"Queries run: {len(queries)} | With results: {overall['queries_with_results']}")
    for q, info in json_summary.items():
        print(f"  - {q[:60]}... -> {info['comment']}")
    print(f"Saved: {CSV_PATH}")
    print(f"Saved: {JSON_PATH}")
    return {"overall": overall, "per_query": json_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval diagnostics")
    parser.add_argument("--query", type=str, default=None, help="Single query to test")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    args = parser.parse_args()

    queries = [args.query] if args.query else DEFAULT_QUERIES
    try:
        run_queries(queries, top_k=args.top_k)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
