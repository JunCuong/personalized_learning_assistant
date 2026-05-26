"""Inspect chunk statistics from data/processed/chunks.json."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import CHUNKS_PATH, DIAGNOSTICS_DIR
from src.diagnostics_io import save_csv, save_json_report
from src.utils import load_json

STATS_CSV = DIAGNOSTICS_DIR / "chunk_statistics.csv"
STATS_JSON = DIAGNOSTICS_DIR / "chunk_statistics_summary.json"
PREVIEW_CSV = DIAGNOSTICS_DIR / "chunk_preview.csv"
PREVIEW_LEN = 300


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"chunks.json not found at {CHUNKS_PATH}\n"
            "Run build first: python -c \"from src.pipeline import build_knowledge_base; "
            "build_knowledge_base()\""
        )
    return load_json(CHUNKS_PATH)


def build_statistics(chunks: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for c in chunks:
        by_source[c["source"]].append(c)

    stats_rows: list[dict] = []
    lengths = [len(c.get("chunk_text", "")) for c in chunks]
    under_100 = sum(1 for L in lengths if L < 100)
    under_200 = sum(1 for L in lengths if L < 200)

    for source, items in sorted(by_source.items()):
        lens = [len(i.get("chunk_text", "")) for i in items]
        pages = sorted({i["page"] for i in items})
        u100 = sum(1 for L in lens if L < 100)
        u200 = sum(1 for L in lens if L < 200)
        stats_rows.append(
            {
                "source": source,
                "number_of_chunks": len(items),
                "min_chunk_length": min(lens) if lens else 0,
                "max_chunk_length": max(lens) if lens else 0,
                "avg_chunk_length": round(sum(lens) / len(lens), 1) if lens else 0,
                "chunks_under_100_chars": u100,
                "chunks_under_200_chars": u200,
                "pages_covered": "|".join(str(p) for p in pages),
            }
        )

    avg_len = sum(lengths) / len(lengths) if lengths else 0
    recommendation = "Chunk statistics look reasonable."
    if not chunks:
        recommendation = "No chunks found; rebuild knowledge base after fixing PDF text extraction."
    elif under_100 > len(chunks) * 0.2:
        recommendation = (
            "Many chunks are very short (<100 chars); consider increasing CHUNK_SIZE "
            "or checking source PDF text quality."
        )
    elif avg_len < 200:
        recommendation = "Average chunk length is low; review PDF extraction quality."

    summary = {
        "total_chunks": len(chunks),
        "total_sources": len(by_source),
        "avg_chunk_length": round(avg_len, 1),
        "min_chunk_length": min(lengths) if lengths else 0,
        "max_chunk_length": max(lengths) if lengths else 0,
        "chunks_under_100_chars": under_100,
        "chunks_under_200_chars": under_200,
        "recommendation": recommendation,
    }

    preview_rows = []
    for c in chunks:
        text = c.get("chunk_text", "")
        preview_rows.append(
            {
                "chunk_id": c.get("chunk_id", ""),
                "source": c.get("source", ""),
                "page": c.get("page", ""),
                "chunk_index": c.get("chunk_index", ""),
                "chunk_length": len(text),
                "chunk_text_preview": text[:PREVIEW_LEN],
            }
        )

    return stats_rows, summary, preview_rows


def run_inspection(
    source_filter: str | None = None, preview_limit: int | None = None
) -> dict:
    chunks = load_chunks()
    stats_rows, summary, preview_rows = build_statistics(chunks)

    if source_filter:
        stats_rows = [r for r in stats_rows if r["source"] == source_filter]
        preview_rows = [r for r in preview_rows if r["source"] == source_filter]
        summary["filtered_source"] = source_filter

    if preview_limit is not None:
        preview_rows = preview_rows[:preview_limit]
        summary["preview_limit"] = preview_limit

    save_csv(stats_rows, STATS_CSV)
    save_json_report(summary, STATS_JSON)
    save_csv(preview_rows, PREVIEW_CSV)

    print(f"Total chunks: {summary['total_chunks']} | Sources: {summary['total_sources']}")
    print(f"Recommendation: {summary['recommendation']}")
    print(f"Saved: {STATS_CSV}")
    print(f"Saved: {STATS_JSON}")
    print(f"Saved: {PREVIEW_CSV} ({len(preview_rows)} rows)")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect processed chunks")
    parser.add_argument("--source", type=str, default=None, help="Filter by source PDF filename")
    parser.add_argument("--limit", type=int, default=None, help="Limit preview rows")
    args = parser.parse_args()

    try:
        run_inspection(source_filter=args.source, preview_limit=args.limit)
    except FileNotFoundError as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
