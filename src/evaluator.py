"""Retrieval and keyword-based QA evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import EVALUATION_DIR
from src.qa_generator import QAGenerator
from src.retriever import Retriever
from src.utils import save_json

logger = logging.getLogger(__name__)

EVAL_QUESTIONS_PATH = EVALUATION_DIR / "eval_questions.csv"


def _normalize_keywords(value: str) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return []
    return [k.strip().lower() for k in str(value).split("|") if k.strip()]


def _hit_at_k(
    retrieved: list[dict], expected_source: str, expected_page: int, k: int
) -> bool:
    top = retrieved[:k]
    for r in top:
        if r["source"] == expected_source and int(r["page"]) == int(expected_page):
            return True
    return False


def _reciprocal_rank(
    retrieved: list[dict], expected_source: str, expected_page: int
) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r["source"] == expected_source and int(r["page"]) == int(expected_page):
            return 1.0 / i
    return 0.0


def run_retrieval_evaluation(
    df: pd.DataFrame,
    retriever: Retriever | None = None,
    top_k: int = 5,
    use_reranking: bool = False,
    candidate_k: int = 10,
) -> tuple[pd.DataFrame, dict]:
    retriever = retriever or Retriever()
    retriever.load()

    rows = []
    hits3, hits5, rrs = [], [], []

    for _, row in df.iterrows():
        question = str(row["question"])
        expected_source = str(row["expected_source"])
        expected_page = int(row["expected_page"])
        keywords = _normalize_keywords(row.get("expected_keywords", ""))

        retrieved = retriever.retrieve(
            question,
            top_k=top_k,
            use_reranking=use_reranking,
            candidate_k=candidate_k,
        )
        h3 = _hit_at_k(retrieved, expected_source, expected_page, 3)
        h5 = _hit_at_k(retrieved, expected_source, expected_page, 5)
        rr = _reciprocal_rank(retrieved, expected_source, expected_page)

        hits3.append(h3)
        hits5.append(h5)
        rrs.append(rr)

        top_sources = [
            f"{r['source']} p{r['page']} (score={r['score']:.3f})"
            for r in retrieved[:3]
        ]

        rows.append(
            {
                "question": question,
                "expected_source": expected_source,
                "expected_page": expected_page,
                "expected_keywords": "|".join(keywords),
                "hit_at_3": h3,
                "hit_at_5": h5,
                "reciprocal_rank": rr,
                "top_retrieved": "; ".join(top_sources),
            }
        )

    results_df = pd.DataFrame(rows)
    summary = {
        "num_questions": len(df),
        "hit_rate_at_3": float(sum(hits3) / len(hits3)) if hits3 else 0.0,
        "hit_rate_at_5": float(sum(hits5) / len(hits5)) if hits5 else 0.0,
        "mean_reciprocal_rank": float(sum(rrs) / len(rrs)) if rrs else 0.0,
    }
    return results_df, summary


def run_qa_keyword_evaluation(
    df: pd.DataFrame,
    qa: QAGenerator | None = None,
    top_k: int = 5,
    max_questions: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    qa = qa or QAGenerator()
    rows = []
    scores = []
    subset = df.head(max_questions) if max_questions else df

    for _, row in subset.iterrows():
        question = str(row["question"])
        keywords = _normalize_keywords(row.get("expected_keywords", ""))
        try:
            result = qa.ask(question, top_k=top_k)
            answer_lower = result["answer"].lower()
            if answer_lower.startswith("error:"):
                matched = []
                score = float("nan")
                status = "generation_error"
            else:
                matched = [k for k in keywords if k in answer_lower]
                score = len(matched) / len(keywords) if keywords else float("nan")
                status = "ok"
        except Exception as exc:
            matched = []
            score = float("nan")
            status = f"skipped:{exc}"
            result = {"answer": ""}

        scores.append(score)
        rows.append(
            {
                "question": question,
                "expected_keywords": "|".join(keywords),
                "matched_keywords": "|".join(matched),
                "keyword_score": score,
                "status": status,
                "answer_preview": result.get("answer", "")[:300],
            }
        )

    results_df = pd.DataFrame(rows)
    valid_scores = [s for s in scores if s == s]
    summary = {
        "num_questions": len(subset),
        "mean_keyword_score": float(sum(valid_scores) / len(valid_scores))
        if valid_scores
        else None,
        "questions_with_keywords": len(valid_scores),
    }
    return results_df, summary


def run_full_evaluation(
    eval_path: Path | None = None,
    mode: str = "retrieval",
    top_k: int = 5,
    max_questions: int | None = None,
    skip_qa: bool | None = None,
    use_reranking: bool = False,
    candidate_k: int = 10,
) -> dict:
    eval_path = eval_path or EVAL_QUESTIONS_PATH
    stem = eval_path.stem
    suffix = "_rerank" if use_reranking else ""
    results_path = EVALUATION_DIR / f"{stem}_results{suffix}.csv"
    summary_path = EVALUATION_DIR / f"{stem}_summary{suffix}.json"
    qa_results_path = EVALUATION_DIR / f"{stem}_qa_results{suffix}.csv"

    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_path}")

    df = pd.read_csv(eval_path, encoding="utf-8")
    required = {"question", "expected_source", "expected_page", "expected_keywords"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[df["question"].astype(str).str.strip() != ""].copy()
    if df.empty:
        raise ValueError("No evaluation questions found.")

    if max_questions:
        df = df.head(max_questions)

    retrieval_df, retrieval_summary = run_retrieval_evaluation(
        df,
        top_k=top_k,
        use_reranking=use_reranking,
        candidate_k=candidate_k,
    )
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    retrieval_df.to_csv(results_path, index=False, encoding="utf-8")

    combined_summary: dict = {
        "eval_file": str(eval_path),
        "mode": mode,
        "use_reranking": use_reranking,
        "candidate_k": candidate_k if use_reranking else None,
        "retrieval": retrieval_summary,
        "retrieval_results_path": str(results_path),
    }

    run_qa = mode in ("full", "qa") and not (skip_qa is True)
    if run_qa:
        try:
            qa_df, qa_summary = run_qa_keyword_evaluation(
                df, top_k=top_k, max_questions=max_questions
            )
            qa_df.to_csv(qa_results_path, index=False, encoding="utf-8")
            combined_summary["qa_keyword"] = qa_summary
            combined_summary["qa_results_path"] = str(qa_results_path)
        except Exception as exc:
            combined_summary["qa_keyword"] = {
                "status": "skipped",
                "reason": str(exc),
            }
            logger.warning("QA evaluation skipped: %s", exc)
    else:
        combined_summary["qa_keyword"] = {"status": "skipped", "reason": f"mode={mode}"}

    save_json(summary_path, combined_summary)
    return combined_summary
