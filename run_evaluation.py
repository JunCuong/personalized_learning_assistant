"""Run retrieval and QA keyword evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logging

from src.config import EVALUATION_DIR
from src.evaluator import EVAL_QUESTIONS_PATH, run_full_evaluation
from src.vector_db import index_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=EVAL_QUESTIONS_PATH,
        help="Eval questions CSV (default: eval_questions.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["retrieval", "qa", "full"],
        default="retrieval",
        help="retrieval=metrics only; full=retrieval+QA keyword",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limit questions (useful for full/QA mode)",
    )
    parser.add_argument(
        "--use-reranking",
        action="store_true",
        help="Enable lightweight keyword+semantic reranking",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=10,
        help="FAISS candidates before reranking (when --use-reranking)",
    )
    args = parser.parse_args()

    if not index_exists():
        logger.error(
            "FAISS index not found. Build knowledge base first:\n"
            '  python -c "from src.pipeline import build_knowledge_base; build_knowledge_base()"'
        )
        sys.exit(1)

    if not args.eval_file.exists():
        logger.error("Missing eval file: %s", args.eval_file)
        sys.exit(1)

    summary = run_full_evaluation(
        eval_path=args.eval_file,
        mode=args.mode,
        top_k=args.top_k,
        max_questions=args.max_questions,
        use_reranking=args.use_reranking,
        candidate_k=args.candidate_k,
    )
    stem = args.eval_file.stem
    suffix = "_rerank" if args.use_reranking else ""
    print("Evaluation complete.")
    print(f"Eval file: {args.eval_file}")
    print(f"Mode: {args.mode}")
    print(f"Reranking: {args.use_reranking}")
    print(f"Results: {EVALUATION_DIR / f'{stem}_results{suffix}.csv'}")
    print(f"Summary: {EVALUATION_DIR / f'{stem}_summary{suffix}.json'}")
    print(summary)


if __name__ == "__main__":
    main()
