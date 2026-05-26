"""Export chunk-based eval candidates for human review (no LLM)."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import CHUNKS_PATH, DIAGNOSTICS_DIR
from src.diagnostics_io import save_csv
from src.utils import load_json

OUTPUT_CSV = DIAGNOSTICS_DIR / "eval_candidate_chunks.csv"
MIN_CHUNK_LEN = 200
PREVIEW_LEN = 400

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
    "have", "has", "been", "will", "can", "may", "also", "into", "such", "than",
    "their", "they", "them", "these", "those", "which", "when", "where", "what",
    "about", "using", "used", "use", "over", "under", "between", "through",
}


def _keywords_from_chunk(text: str, n: int = 6) -> str:
    words = re.findall(r"\b[A-Za-z]{4,}\b", text.lower())
    filtered = [w for w in words if w not in STOPWORDS]
    counts = Counter(filtered)
    top = [w for w, _ in counts.most_common(n)]
    return "|".join(top)


def _suggested_question_type(text: str) -> str:
    t = text.lower()
    if "definition" in t or " is " in t[:80]:
        return "definition"
    if "component" in t or "architecture" in t or "pipeline" in t:
        return "architecture"
    if "example" in t or "case study" in t:
        return "example"
    if "compare" in t or "versus" in t or " vs " in t:
        return "comparison"
    return "factual"


def main() -> None:
    if not CHUNKS_PATH.exists():
        print(
            f"Missing {CHUNKS_PATH}. Run build_knowledge_base first:\n"
            '  python -c "from src.pipeline import build_knowledge_base; build_knowledge_base()"'
        )
        sys.exit(1)

    chunks = load_json(CHUNKS_PATH)
    rows = []
    for c in chunks:
        length = c.get("chunk_length", len(c.get("chunk_text", "")))
        if length < MIN_CHUNK_LEN:
            continue
        text = c.get("chunk_text", "")
        rows.append(
            {
                "source": c["source"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
                "chunk_length": length,
                "extraction_method": c.get("extraction_method", "pypdf"),
                "suggested_question_type": _suggested_question_type(text),
                "chunk_text_preview": text[:PREVIEW_LEN],
                "suggested_expected_keywords": _keywords_from_chunk(text),
            }
        )

    df = pd.DataFrame(rows)
    save_csv(rows, OUTPUT_CSV)
    print(f"Exported {len(df)} candidate chunks (length >= {MIN_CHUNK_LEN})")
    print(f"Saved: {OUTPUT_CSV}")
    print("Review this file before editing eval_questions.csv manually.")


if __name__ == "__main__":
    main()
