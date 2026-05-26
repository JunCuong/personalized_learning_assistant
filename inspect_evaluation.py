"""Inspect eval_questions.csv against retrieval without modifying the CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DIAGNOSTICS_DIR, EVALUATION_DIR
from src.diagnostics_io import save_csv, save_json_report
from src.retriever import Retriever
from src.vector_db import index_exists

DEFAULT_EVAL_PATH = EVALUATION_DIR / "eval_questions.csv"


def _output_paths(eval_path: Path, use_reranking: bool = False) -> tuple[Path, Path, Path]:
    stem = eval_path.stem
    suffix = "_rerank" if use_reranking else ""
    return (
        DIAGNOSTICS_DIR / f"{stem}_inspection{suffix}.csv",
        DIAGNOSTICS_DIR / f"{stem}_suggestions{suffix}.csv",
        DIAGNOSTICS_DIR / f"{stem}_inspection_summary{suffix}.json",
    )


def _parse_page(value) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _hit_source(retrieved: list[dict], expected_source: str, k: int) -> bool:
    return any(r["source"] == expected_source for r in retrieved[:k])


def _hit_source_page(
    retrieved: list[dict], expected_source: str, expected_page: int | None, k: int
) -> bool | str:
    if expected_page is None:
        return "SKIPPED"
    return any(
        r["source"] == expected_source and int(r["page"]) == expected_page
        for r in retrieved[:k]
    )


def _format_sources_pages(retrieved: list[dict], k: int) -> str:
    parts = [f"{r['source']} p{r['page']}" for r in retrieved[:k]]
    return "; ".join(parts)


def _suggest(
    question: str,
    expected_source: str,
    expected_page: int | None,
    retrieved: list[dict],
    source_hit_3: bool,
    page_hit_3: bool | str,
) -> str:
    if not retrieved:
        return "no relevant chunk found"

    top = retrieved[0]
    if page_hit_3 == "SKIPPED":
        if not source_hit_3:
            return "expected source may be wrong"
        return "OK (page check skipped)"

    if source_hit_3 and page_hit_3 is True:
        return "OK"

    if source_hit_3 and page_hit_3 is False:
        return "expected page may be wrong"

    if not source_hit_3:
        # Check if top result is semantically related source
        if top["source"] != expected_source:
            q_lower = question.lower()
            if len(q_lower.split()) <= 4:
                return "question too vague"
            return "expected source may be wrong"

    return "no relevant chunk found"


def run_inspection(
    eval_path: Path | None = None,
    top_k: int = 5,
    use_reranking: bool = False,
    candidate_k: int = 10,
) -> dict:
    eval_path = eval_path or DEFAULT_EVAL_PATH
    inspect_csv, suggest_csv, summary_json = _output_paths(eval_path, use_reranking)

    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    if not index_exists():
        raise FileNotFoundError(
            "FAISS index not found. Build knowledge base before inspect_evaluation."
        )

    df = pd.read_csv(eval_path, encoding="utf-8")
    required = {"question", "expected_source", "expected_page", "expected_keywords"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"eval_questions.csv missing columns: {missing}")

    df = df[df["question"].astype(str).str.strip() != ""].copy()
    retriever = Retriever()
    retriever.load()

    inspect_rows: list[dict] = []
    suggest_rows: list[dict] = []

    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        expected_source = str(row["expected_source"]).strip()
        expected_page = _parse_page(row["expected_page"])
        expected_keywords = str(row.get("expected_keywords", ""))

        retrieved = retriever.retrieve(
            question,
            top_k=top_k,
            use_reranking=use_reranking,
            candidate_k=candidate_k,
        )
        top1 = retrieved[0] if retrieved else None

        sh3 = _hit_source(retrieved, expected_source, 3)
        sh5 = _hit_source(retrieved, expected_source, 5)
        ph3 = _hit_source_page(retrieved, expected_source, expected_page, 3)
        ph5 = _hit_source_page(retrieved, expected_source, expected_page, 5)

        comment_parts = []
        if not retrieved:
            comment_parts.append("No retrieval results")
        elif not sh3:
            comment_parts.append(f"Expected source not in top 3 (got {top1['source']})")
        elif ph3 == "SKIPPED":
            comment_parts.append("Page check skipped")
        elif ph3 is False:
            comment_parts.append(
                f"Source in top 3 but page mismatch (top1 page {top1['page']}, expected {expected_page})"
            )
        else:
            comment_parts.append("Source and page match in top 3")

        suggestion = _suggest(
            question, expected_source, expected_page, retrieved, sh3, ph3
        )

        inspect_rows.append(
            {
                "question": question,
                "expected_source": expected_source,
                "expected_page": expected_page if expected_page is not None else "",
                "expected_keywords": expected_keywords,
                "top1_source": top1["source"] if top1 else "",
                "top1_page": top1["page"] if top1 else "",
                "top1_score": round(top1["score"], 4) if top1 else "",
                "top3_sources_pages": _format_sources_pages(retrieved, 3),
                "top5_sources_pages": _format_sources_pages(retrieved, 5),
                "source_hit_at_3": sh3,
                "source_page_hit_at_3": ph3,
                "source_hit_at_5": sh5,
                "source_page_hit_at_5": ph5,
                "comment": "; ".join(comment_parts),
            }
        )
        suggest_rows.append(
            {
                "question": question,
                "expected_source": expected_source,
                "expected_page": expected_page if expected_page is not None else "",
                "suggestion": suggestion,
            }
        )

    ok_count = sum(1 for r in suggest_rows if r["suggestion"] == "OK" or r["suggestion"].startswith("OK"))
    summary = {
        "num_questions": len(inspect_rows),
        "source_hit_at_3_count": sum(1 for r in inspect_rows if r["source_hit_at_3"]),
        "source_page_hit_at_3_count": sum(
            1 for r in inspect_rows if r["source_page_hit_at_3"] is True
        ),
        "ok_suggestions": ok_count,
        "needs_review": len(inspect_rows) - ok_count,
        "note": "eval_questions.csv was not modified",
    }

    summary["eval_file"] = str(eval_path)
    summary["use_reranking"] = use_reranking
    summary["candidate_k"] = candidate_k if use_reranking else None
    summary["note"] = "eval CSV was not modified"

    save_csv(inspect_rows, inspect_csv)
    save_csv(suggest_rows, suggest_csv)
    save_json_report(summary, summary_json)

    print(f"Eval file: {eval_path.name}")
    print(f"Questions inspected: {summary['num_questions']}")
    print(f"Source hit@3: {summary['source_hit_at_3_count']}/{summary['num_questions']}")
    print(f"Source+page hit@3: {summary['source_page_hit_at_3_count']}/{summary['num_questions']}")
    print(f"Saved: {inspect_csv}")
    print(f"Saved: {suggest_csv}")
    print(f"Saved: {summary_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect evaluation questions")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=DEFAULT_EVAL_PATH,
        help="Path to eval questions CSV",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-reranking", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=10)
    args = parser.parse_args()
    try:
        run_inspection(
            eval_path=args.eval_file,
            top_k=args.top_k,
            use_reranking=args.use_reranking,
            candidate_k=args.candidate_k,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
