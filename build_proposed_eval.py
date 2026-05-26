"""Build proposed eval CSV files from chunks.json (no LLM, does not overwrite eval_questions.csv)."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import CHUNKS_PATH, EVALUATION_DIR
from src.utils import load_json

PROPOSED = EVALUATION_DIR / "eval_questions_proposed.csv"
PROPOSED_OCR = EVALUATION_DIR / "eval_questions_proposed_with_ocr.csv"

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "been", "will", "can", "may", "also", "into", "such", "than",
}


def _keywords(text: str, n: int = 5) -> str:
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    top = [w for w, _ in Counter(w for w in words if w not in STOPWORDS).most_common(n)]
    return "|".join(top)


def _question_from_chunk(source: str, page: int, text: str, qtype: str) -> str:
    kws = _keywords(text, 3).replace("|", ", ")
    if qtype == "definition":
        return f"What is described about {kws} in the course materials ({source}, page {page})?"
    if qtype == "architecture":
        return f"What are the main components or steps related to {kws} ({source}, page {page})?"
    return f"What information is provided about {kws} on page {page} of {source}?"


def _pick_chunks(
    chunks: list[dict],
    source: str,
    n: int,
    min_len: int = 200,
    prefer_ocr: bool = False,
) -> list[dict]:
    pool = [
        c
        for c in chunks
        if c["source"] == source
        and c.get("chunk_length", len(c.get("chunk_text", ""))) >= min_len
    ]
    if prefer_ocr:
        ocr_pool = [c for c in pool if c.get("extraction_method") == "ocr"]
        if ocr_pool:
            pool = ocr_pool
    pool.sort(key=lambda c: c.get("chunk_length", 0), reverse=True)
    seen_pages: set[int] = set()
    picked: list[dict] = []
    for c in pool:
        if c["page"] in seen_pages and len(picked) >= n:
            continue
        picked.append(c)
        seen_pages.add(c["page"])
        if len(picked) >= n:
            break
    return picked[:n]


def _build_rows(chunks: list[dict], spec: dict[str, int], prefer_ocr: bool = False) -> list[dict]:
    rows = []
    for source, count in spec.items():
        for c in _pick_chunks(chunks, source, count, prefer_ocr=prefer_ocr):
            text = c["chunk_text"]
            rows.append(
                {
                    "question": _question_from_chunk(
                        c["source"], c["page"], text, "factual"
                    ),
                    "expected_source": c["source"],
                    "expected_page": c["page"],
                    "expected_keywords": _keywords(text, 6),
                }
            )
    return rows


def main() -> None:
    if not CHUNKS_PATH.exists():
        print("Run build_knowledge_base first.")
        sys.exit(1)

    chunks = load_json(CHUNKS_PATH)
    sources = {c["source"] for c in chunks}
    has_kan = "Kolmogorov-Arnold Networks (KAN).pdf.pdf" in sources
    has_mlops = "Scalable_MLOps_Architecture.pptx.pdf" in sources
    kan_usable = any(
        c["source"] == "Kolmogorov-Arnold Networks (KAN).pdf.pdf"
        and c.get("extraction_method") == "ocr"
        and c.get("chunk_length", 0) >= 200
        for c in chunks
    )
    mlops_usable = any(
        c["source"] == "Scalable_MLOps_Architecture.pptx.pdf"
        and c.get("extraction_method") == "ocr"
        and c.get("chunk_length", 0) >= 200
        for c in chunks
    )

    spec_base = {
        "RAG.pdf": 4,
        "Federated Learning AI.pdf": 4,
        "Multimodal ML.pdf": 4,
        "Module 1 Guidance Notebook.pdf": 3,
    }
    rows = _build_rows(chunks, spec_base)
    while len(rows) < 15:
        extra = _pick_chunks(chunks, "RAG.pdf", 1)
        if not extra:
            break
        c = extra[0]
        rows.append(
            {
                "question": _question_from_chunk(c["source"], c["page"], c["chunk_text"], "factual"),
                "expected_source": c["source"],
                "expected_page": c["page"],
                "expected_keywords": _keywords(c["chunk_text"], 6),
            }
        )
    rows = rows[:15]
    pd.DataFrame(rows).to_csv(PROPOSED, index=False, encoding="utf-8")
    print(f"Wrote {len(rows)} questions -> {PROPOSED}")

    if has_kan or has_mlops:
        spec_ocr = {
            "RAG.pdf": 3,
            "Federated Learning AI.pdf": 3,
            "Multimodal ML.pdf": 3,
            "Module 1 Guidance Notebook.pdf": 2,
        }
        if kan_usable:
            spec_ocr["Kolmogorov-Arnold Networks (KAN).pdf.pdf"] = 2
        if mlops_usable:
            spec_ocr["Scalable_MLOps_Architecture.pptx.pdf"] = 2
        rows_ocr = _build_rows(chunks, spec_ocr, prefer_ocr=False)
        while len(rows_ocr) < 15:
            extra = _pick_chunks(chunks, "RAG.pdf", 1)
            if not extra:
                break
            c = extra[0]
            rows_ocr.append(
                {
                    "question": _question_from_chunk(
                        c["source"], c["page"], c["chunk_text"], "factual"
                    ),
                    "expected_source": c["source"],
                    "expected_page": c["page"],
                    "expected_keywords": _keywords(c["chunk_text"], 6),
                }
            )
        rows_ocr = rows_ocr[:15]
        pd.DataFrame(rows_ocr).to_csv(PROPOSED_OCR, index=False, encoding="utf-8")
        print(f"Wrote {len(rows_ocr)} questions -> {PROPOSED_OCR}")
    else:
        print(f"Skipped {PROPOSED_OCR} (no KAN/MLOps chunks in index)")


if __name__ == "__main__":
    main()
