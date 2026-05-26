"""Final benchmark report for demo readiness (no Streamlit, no OCR)."""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import DIAGNOSTICS_DIR, GEMINI_MODEL_NAME
from src.llm_client import LLMClient, is_gemini_quota_response
from src.mcq_generator import build_mcq_prompt, parse_mcq_all
from src.qa_generator import _sources_from_chunks, build_qa_prompt
from src.retriever import Retriever
from src.summarizer import build_summary_prompt
from src.embedder import Embedder
from src.utils import save_json

JSON_OUT = DIAGNOSTICS_DIR / "final_benchmark_report.json"
MD_OUT = DIAGNOSTICS_DIR / "final_benchmark_report.md"

RETRIEVAL_QUERIES = [
    "What is retrieval augmented generation?",
    "What is MLOps?",
    "What is Kolmogorov-Arnold Network?",
]

ASK_QUERY = RETRIEVAL_QUERIES[0]
SUMMARY_TOPIC = "Summarize federated learning."
MCQ_TOPIC = "retrieval augmented generation"
MCQ_N = 3
TOP_K = 5
CANDIDATE_K = 10


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _top1(chunks: list[dict]) -> dict:
    if not chunks:
        return {"source": None, "page": None, "score": None}
    c = chunks[0]
    return {"source": c.get("source"), "page": c.get("page"), "score": round(float(c.get("score", 0)), 4)}


def _gen_status(text: str, llm: LLMClient) -> str:
    if llm.gemini_quota_exhausted or is_gemini_quota_response(text):
        return "quota_error"
    if text.startswith("Error:") or "quota is exhausted" in text.lower():
        return "quota_error"
    return "success"


def _timed_retrieve(
    retriever: Retriever,
    query: str,
    use_reranking: bool,
) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    chunks = retriever.retrieve(
        query,
        top_k=TOP_K,
        use_reranking=use_reranking,
        candidate_k=CANDIDATE_K,
    )
    return chunks, time.perf_counter() - t0


def _run_generation_case(
    retriever: Retriever,
    llm: LLMClient,
    mode: str,
    use_reranking: bool,
) -> dict:
    record = {
        "mode": mode,
        "retrieve_sec": None,
        "generate_sec": None,
        "total_sec": None,
        "status": "skipped",
        "output_length": 0,
        "top1_source": None,
        "top1_page": None,
        "error": None,
    }

    try:
        if mode == "ask":
            query = ASK_QUERY
            t_total = time.perf_counter()
            chunks, record["retrieve_sec"] = _timed_retrieve(
                retriever, query, use_reranking
            )
            t1 = time.perf_counter()
            text = llm.generate(build_qa_prompt(query, chunks), max_tokens=448)
            record["generate_sec"] = time.perf_counter() - t1
            record["total_sec"] = time.perf_counter() - t_total
            record["output_length"] = len(text)
            record["status"] = _gen_status(text, llm)

        elif mode == "summarize":
            topic = SUMMARY_TOPIC
            t_total = time.perf_counter()
            chunks, record["retrieve_sec"] = _timed_retrieve(
                retriever, topic, use_reranking
            )
            t1 = time.perf_counter()
            text = llm.generate(
                build_summary_prompt(topic, chunks), max_tokens=640
            )
            record["generate_sec"] = time.perf_counter() - t1
            record["total_sec"] = time.perf_counter() - t_total
            record["output_length"] = len(text)
            record["status"] = _gen_status(text, llm)

        elif mode == "mcq":
            topic = MCQ_TOPIC
            t_total = time.perf_counter()
            chunks, record["retrieve_sec"] = _timed_retrieve(
                retriever, topic, use_reranking
            )
            t1 = time.perf_counter()
            raw = llm.generate(
                build_mcq_prompt(topic, MCQ_N, chunks), max_tokens=1024
            )
            record["generate_sec"] = time.perf_counter() - t1
            record["total_sec"] = time.perf_counter() - t_total
            mcqs, _ = parse_mcq_all(raw)
            record["output_length"] = len(raw)
            record["status"] = _gen_status(raw, llm)
            if record["status"] == "success" and not mcqs:
                record["status"] = "failed"
                record["error"] = "parse_failed"

        top = _top1(chunks if mode != "skipped" else [])
        record["top1_source"] = top["source"]
        record["top1_page"] = top["page"]

    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)[:300]
        if record["total_sec"] is None:
            record["total_sec"] = 0.0

    for k in ("retrieve_sec", "generate_sec", "total_sec"):
        if record[k] is not None:
            record[k] = round(record[k], 3)

    return record


def _load_resources_cold(
    allow_local_fallback: bool,
) -> tuple[dict, Retriever, LLMClient]:
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    embedder = Embedder()
    timings["embedder_load_cold_sec"] = time.perf_counter() - t0

    retriever = Retriever(embedder=embedder)
    t1 = time.perf_counter()
    retriever.load()
    timings["retriever_faiss_load_cold_sec"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    llm = LLMClient(allow_local_fallback=allow_local_fallback)
    timings["llm_client_load_sec"] = time.perf_counter() - t2

    return timings, retriever, llm


def _load_resources_warm(retriever: Retriever) -> dict[str, float]:
    """Warm embedder = first encode on loaded model; warm FAISS = reload index sharing embedder."""
    t0 = time.perf_counter()
    retriever.embedder.embed_query("benchmark warm probe")
    embedder_warm = time.perf_counter() - t0

    t1 = time.perf_counter()
    warm_retriever = Retriever(embedder=retriever.embedder)
    warm_retriever.load()
    faiss_warm = time.perf_counter() - t1

    return {
        "embedder_load_warm_sec": round(embedder_warm, 4),
        "retriever_faiss_load_warm_sec": round(faiss_warm, 4),
    }


def _avg_retrieval(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _demo_readiness(
    generation_rows: list[dict],
    quota_exhausted: bool,
    local_fallback_used: bool,
    retrieval_only: bool,
) -> dict:
    ask_row = next((r for r in generation_rows if r["mode"] == "ask"), None)
    sum_row = next((r for r in generation_rows if r["mode"] == "summarize"), None)
    mcq_row = next((r for r in generation_rows if r["mode"] == "mcq"), None)

    def _target(row, limit, label):
        if retrieval_only or quota_exhausted or local_fallback_used:
            return "unknown_due_quota" if quota_exhausted else "not_run"
        if not row or row.get("status") != "success":
            return "unknown_due_quota" if quota_exhausted else "not_met"
        total = row.get("total_sec")
        if total is None:
            return "unknown_due_quota"
        return "met" if total <= limit else "not_met"

    bottleneck = "none"
    if quota_exhausted and not local_fallback_used:
        bottleneck = "gemini_quota"
    elif local_fallback_used:
        bottleneck = "local_flan_t5_fallback"
    elif ask_row and ask_row.get("generate_sec"):
        if ask_row["generate_sec"] > (ask_row.get("retrieve_sec") or 0):
            bottleneck = "gemini_generate"
        else:
            bottleneck = "retrieval_embed_faiss"

    rec = (
        "Demo: warm caches in Streamlit; use retrieval-only inspect; confirm Gemini quota before live Q&A."
    )
    if quota_exhausted and not local_fallback_used:
        rec = (
            "Generation latency could not be measured reliably because all Gemini keys "
            "were quota-exhausted. Retrieval benchmarks are still valid. Add quota or "
            "use --allow-local-fallback only for offline tests (slow)."
        )
    elif ask_row and ask_row.get("status") == "success" and ask_row.get("total_sec", 99) <= 10:
        rec = "Ready for demo: Ask path within 10s target on warm backend run."

    return {
        "expected_mode_switching": "fast (Streamlit @st.cache_resource; not measured here)",
        "ask_target_10s": _target(ask_row, 10, "ask"),
        "summary_target_15s": _target(sum_row, 15, "summary"),
        "mcq_target_20s": _target(mcq_row, 20, "mcq"),
        "main_bottleneck": bottleneck,
        "recommendation_for_demo": rec,
    }


def _build_markdown(report: dict) -> str:
    sys_info = report["system_status"]
    resources = report["resource_loading"]
    retrieval = report["retrieval_latency"]
    generation = report["generation_latency"]
    demo = report["demo_readiness"]

    lines = [
        "# Final benchmark report",
        "",
        f"Generated: {sys_info['timestamp']}",
        "",
        "## 1. System status",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Timestamp | {sys_info['timestamp']} |",
        f"| Python | {sys_info['python_version']} |",
        f"| Gemini model | {sys_info['gemini_model_name']} |",
        f"| Gemini keys configured | {sys_info['gemini_keys_configured']} |",
        f"| Local fallback allowed | {sys_info['local_fallback_allowed']} |",
        f"| Gemini quota exhausted | {sys_info['gemini_quota_exhausted']} |",
        f"| Local fallback used | {sys_info['local_fallback_used']} |",
        f"| Reranking enabled in run | {sys_info['reranking_enabled']} |",
        f"| Retrieval-only mode | {sys_info['retrieval_only']} |",
        "",
    ]

    if sys_info.get("generation_note"):
        lines.extend([f"**Note:** {sys_info['generation_note']}", ""])

    lines.extend(
        [
            "## 2. Resource loading times",
            "",
            "| Resource | Cold (s) | Warm (s) |",
            "|----------|----------|----------|",
            f"| Embedder | {resources.get('embedder_load_cold_sec', 'N/A')} | {resources.get('embedder_load_warm_sec', 'N/A')} |",
            f"| Retriever / FAISS | {resources.get('retriever_faiss_load_cold_sec', 'N/A')} | {resources.get('retriever_faiss_load_warm_sec', 'N/A')} |",
            f"| LLM client init | {resources.get('llm_client_load_sec', 'N/A')} | — |",
            "",
            "## 3. Retrieval latency",
            "",
            f"Average without reranking: **{retrieval.get('avg_without_rerank_sec', 'N/A')}s**  ",
            f"Average with reranking: **{retrieval.get('avg_with_rerank_sec', 'N/A')}s**",
            "",
            "| Query | No rerank (s) | With rerank (s) | Top1 (no rerank) | Top1 (rerank) |",
            "|-------|---------------|-----------------|------------------|---------------|",
        ]
    )

    for row in retrieval.get("per_query", []):
        nr = row.get("top1_no_rerank", {})
        wr = row.get("top1_with_rerank", {})
        lines.append(
            f"| {row['query'][:45]} | {row.get('time_no_rerank_sec')} | "
            f"{row.get('time_with_rerank_sec')} | "
            f"{nr.get('source', '')} p{nr.get('page', '')} | "
            f"{wr.get('source', '')} p{wr.get('page', '')} |"
        )

    lines.extend(["", "## 4. Generation latency", ""])

    if generation.get("skipped"):
        lines.append("_Generation tests skipped (--retrieval-only)._")
    else:
        lines.extend(
            [
                "| Mode | Retrieve (s) | Generate (s) | Total (s) | Status | Output len | Top1 source |",
                "|------|------------|--------------|-----------|--------|------------|-------------|",
            ]
        )
        for row in generation.get("rows", []):
            lines.append(
                f"| {row['mode']} | {row.get('retrieve_sec')} | {row.get('generate_sec')} | "
                f"{row.get('total_sec')} | {row.get('status')} | {row.get('output_length')} | "
                f"{row.get('top1_source')} p{row.get('top1_page')} |"
            )

    lines.extend(
        [
            "",
            "## 5. Demo readiness summary",
            "",
            "| Check | Result |",
            "|-------|--------|",
            f"| Mode switching (expected) | {demo['expected_mode_switching']} |",
            f"| Ask ≤ 10s | {demo['ask_target_10s']} |",
            f"| Summary ≤ 15s | {demo['summary_target_15s']} |",
            f"| MCQ ≤ 20s | {demo['mcq_target_20s']} |",
            f"| Main bottleneck | {demo['main_bottleneck']} |",
            "",
            "## 6. Recommendations",
            "",
            demo["recommendation_for_demo"],
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final app benchmark report")
    parser.add_argument(
        "--allow-local-fallback",
        action="store_true",
        help="Allow flan-t5 when Gemini quota exhausted (slow)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip all Gemini generation calls",
    )
    parser.add_argument(
        "--with-reranking",
        action="store_true",
        help="Include reranking comparison in retrieval tests",
    )
    args = parser.parse_args()

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    resource_timings, retriever, llm = _load_resources_cold(
        allow_local_fallback=args.allow_local_fallback
    )
    resource_timings.update(_load_resources_warm(retriever))

    retrieval_rows: list[dict] = []
    for query in RETRIEVAL_QUERIES:
        row: dict = {"query": query}
        chunks_nr, t_nr = _timed_retrieve(retriever, query, use_reranking=False)
        row["time_no_rerank_sec"] = round(t_nr, 3)
        row["top1_no_rerank"] = _top1(chunks_nr)

        if args.with_reranking:
            chunks_wr, t_wr = _timed_retrieve(retriever, query, use_reranking=True)
            row["time_with_rerank_sec"] = round(t_wr, 3)
            row["top1_with_rerank"] = _top1(chunks_wr)
        else:
            row["time_with_rerank_sec"] = None
            row["top1_with_rerank"] = None

        retrieval_rows.append(row)

    generation_rows: list[dict] = []
    quota_exhausted = False
    local_fallback_used = False

    if not args.retrieval_only:
        for mode in ("ask", "summarize", "mcq"):
            row = _run_generation_case(
                retriever,
                llm,
                mode,
                use_reranking=args.with_reranking,
            )
            generation_rows.append(row)
            if row.get("status") == "quota_error":
                quota_exhausted = True
        quota_exhausted = quota_exhausted or llm.gemini_quota_exhausted
        local_fallback_used = llm.local_fallback_used

    gen_note = ""
    if args.retrieval_only:
        gen_note = "Generation tests skipped (--retrieval-only)."
    elif quota_exhausted and not local_fallback_used:
        gen_note = (
            "Generation latency could not be measured reliably because all Gemini "
            "keys were quota-exhausted."
        )

    report = {
        "system_status": {
            "timestamp": _now_iso(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "gemini_model_name": os.getenv("GEMINI_MODEL_NAME", GEMINI_MODEL_NAME),
            "gemini_keys_configured": llm.gemini_key_count(),
            "local_fallback_allowed": args.allow_local_fallback,
            "gemini_quota_exhausted": quota_exhausted,
            "local_fallback_used": local_fallback_used,
            "reranking_enabled": args.with_reranking,
            "retrieval_only": args.retrieval_only,
            "generation_note": gen_note,
        },
        "resource_loading": {k: round(v, 3) for k, v in resource_timings.items()},
        "retrieval_latency": {
            "per_query": retrieval_rows,
            "avg_without_rerank_sec": _avg_retrieval(
                retrieval_rows, "time_no_rerank_sec"
            ),
            "avg_with_rerank_sec": _avg_retrieval(
                retrieval_rows, "time_with_rerank_sec"
            ),
        },
        "generation_latency": {
            "skipped": args.retrieval_only,
            "rows": generation_rows,
        },
        "demo_readiness": _demo_readiness(
            generation_rows,
            quota_exhausted,
            local_fallback_used,
            args.retrieval_only,
        ),
    }

    save_json(JSON_OUT, report)
    MD_OUT.write_text(_build_markdown(report), encoding="utf-8")

    print("=== Final benchmark ===")
    print("JSON:", JSON_OUT)
    print("MD:  ", MD_OUT)
    print("Retrieval avg (no rerank):", report["retrieval_latency"]["avg_without_rerank_sec"], "s")
    if args.with_reranking:
        print("Retrieval avg (rerank):", report["retrieval_latency"]["avg_with_rerank_sec"], "s")
    print("Gemini quota exhausted:", quota_exhausted)
    print("Local fallback used:", local_fallback_used)
    if not args.retrieval_only:
        for row in generation_rows:
            print(
                f"  {row['mode']}: total={row.get('total_sec')}s status={row.get('status')}"
            )
    print("Demo:", report["demo_readiness"]["recommendation_for_demo"][:120], "...")


if __name__ == "__main__":
    main()
