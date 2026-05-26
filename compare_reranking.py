"""Compare baseline FAISS retrieval vs lightweight reranking on eval set."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION_DIR
from src.evaluator import run_full_evaluation
from src.utils import save_json
from src.vector_db import index_exists

EVAL_FILE = EVALUATION_DIR / "eval_questions_revised.csv"
JSON_OUT = EVALUATION_DIR / "reranking_comparison.json"
MD_OUT = EVALUATION_DIR / "reranking_comparison.md"


def _metrics(summary: dict) -> dict:
    r = summary.get("retrieval", {})
    return {
        "hit_rate_at_3": r.get("hit_rate_at_3"),
        "hit_rate_at_5": r.get("hit_rate_at_5"),
        "mean_reciprocal_rank": r.get("mean_reciprocal_rank"),
        "num_questions": r.get("num_questions"),
    }


def _reranking_improved(baseline: dict, rerank: dict) -> bool:
    """True if rerank strictly beats baseline on at least one metric with no regressions."""
    keys = ("hit_rate_at_3", "hit_rate_at_5", "mean_reciprocal_rank")
    improved_any = any(rerank[k] > baseline[k] for k in keys)
    worse_any = any(rerank[k] < baseline[k] for k in keys)
    return improved_any and not worse_any


def _recommendation(improved: bool) -> str:
    if improved:
        return (
            "enable_reranking_optional: Reranking improved metrics without regressions. "
            "Keep disabled by default in app; users may opt in via checkbox or --use-reranking."
        )
    return (
        "keep_reranking_disabled: Reranking did not improve all metrics vs baseline. "
        "Leave use_reranking=False as default."
    )


def main() -> None:
    if not index_exists():
        print("FAISS index not found. Build knowledge base first.")
        sys.exit(1)
    if not EVAL_FILE.exists():
        print(f"Missing eval file: {EVAL_FILE}")
        sys.exit(1)

    errors: list[str] = []

    print("Running baseline evaluation...")
    try:
        baseline_summary = run_full_evaluation(
            eval_path=EVAL_FILE, mode="retrieval", top_k=5, use_reranking=False
        )
    except Exception as exc:
        errors.append(f"baseline: {exc}")
        baseline_summary = {}

    print("Running rerank evaluation (candidate_k=10)...")
    try:
        rerank_summary = run_full_evaluation(
            eval_path=EVAL_FILE,
            mode="retrieval",
            top_k=5,
            use_reranking=True,
            candidate_k=10,
        )
    except Exception as exc:
        errors.append(f"rerank: {exc}")
        rerank_summary = {}

    baseline_m = _metrics(baseline_summary) if baseline_summary else {}
    rerank_m = _metrics(rerank_summary) if rerank_summary else {}

    improved = False
    if baseline_m and rerank_m:
        improved = _reranking_improved(baseline_m, rerank_m)

    comparison = {
        "eval_file": str(EVAL_FILE),
        "baseline": {
            "use_reranking": False,
            "metrics": baseline_m,
            "summary_path": str(EVALUATION_DIR / "eval_questions_revised_summary.json"),
            "results_path": str(EVALUATION_DIR / "eval_questions_revised_results.csv"),
        },
        "rerank": {
            "use_reranking": True,
            "candidate_k": 10,
            "metrics": rerank_m,
            "summary_path": str(
                EVALUATION_DIR / "eval_questions_revised_summary_rerank.json"
            ),
            "results_path": str(
                EVALUATION_DIR / "eval_questions_revised_results_rerank.csv"
            ),
        },
        "reranking_improved": improved,
        "recommendation": _recommendation(improved),
        "errors": errors,
    }

    save_json(JSON_OUT, comparison)

    md_lines = [
        "# Reranking comparison",
        "",
        f"Eval file: `{EVAL_FILE.name}`",
        "",
        "## Baseline (FAISS only)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Hit@3 | {baseline_m.get('hit_rate_at_3', 'N/A')} |",
        f"| Hit@5 | {baseline_m.get('hit_rate_at_5', 'N/A')} |",
        f"| MRR | {baseline_m.get('mean_reciprocal_rank', 'N/A')} |",
        "",
        "## Lightweight rerank (candidate_k=10, semantic 0.75 + keyword 0.25)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Hit@3 | {rerank_m.get('hit_rate_at_3', 'N/A')} |",
        f"| Hit@5 | {rerank_m.get('hit_rate_at_5', 'N/A')} |",
        f"| MRR | {rerank_m.get('mean_reciprocal_rank', 'N/A')} |",
        "",
        f"**Reranking improved:** {improved}",
        "",
        f"**Recommendation:** {comparison['recommendation']}",
        "",
    ]
    if errors:
        md_lines.extend(["## Errors", ""] + [f"- {e}" for e in errors])

    MD_OUT.write_text("\n".join(md_lines), encoding="utf-8")

    print("\n=== Comparison ===")
    print(f"Baseline Hit@3: {baseline_m.get('hit_rate_at_3')}")
    print(f"Baseline Hit@5: {baseline_m.get('hit_rate_at_5')}")
    print(f"Baseline MRR:   {baseline_m.get('mean_reciprocal_rank')}")
    print(f"Rerank Hit@3:   {rerank_m.get('hit_rate_at_3')}")
    print(f"Rerank Hit@5:   {rerank_m.get('hit_rate_at_5')}")
    print(f"Rerank MRR:     {rerank_m.get('mean_reciprocal_rank')}")
    print(f"Improved: {improved}")
    print(f"Saved: {JSON_OUT}")
    print(f"Saved: {MD_OUT}")


if __name__ == "__main__":
    main()
