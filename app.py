"""Streamlit frontend — Modern AI Learning Workspace demo UI."""

from __future__ import annotations

import html
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from pypdf import PdfReader

from src.config import DATASET_DIR, DEFAULT_TOP_K, DIAGNOSTICS_DIR
from src.llm_client import is_gemini_quota_response
from src.mcq_generator import MCQGenerator, parse_mcq_all
from src.pipeline import build_knowledge_base
from src.qa_generator import _sources_from_chunks, build_qa_prompt, clean_generated_answer
from src.summary_utils import clean_summary_text
from src.summarizer import Summarizer
from src.vector_db import index_exists

DATASET_CSV = DIAGNOSTICS_DIR / "dataset_inspection.csv"
DATASET_JSON = DIAGNOSTICS_DIR / "dataset_inspection_summary.json"
CHUNK_JSON = DIAGNOSTICS_DIR / "chunk_statistics_summary.json"

st.set_page_config(
    page_title="Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(165deg, #0B1020 0%, #0f172a 45%, #0B1020 100%);
        color: #F8FAFC;
    }
    .block-container {
        padding-top: 0.75rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    div[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] span {
        color: #E2E8F0;
    }
    h1, h2, h3, h4, .stSubheader { color: #F8FAFC !important; }
    .stTextInput label, .stNumberInput label, .stSlider label,
    .stCheckbox label, .stTabs [data-baseweb="tab"] {
        color: #CBD5E1 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12px;
        padding: 0.5rem 0.75rem;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: rgba(17, 24, 39, 0.5);
        border-radius: 12px;
        padding: 0.35rem;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, rgba(56, 189, 248, 0.25), rgba(129, 140, 248, 0.2)) !important;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] {
        background: rgba(17, 24, 39, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 10px;
    }
    .hero-wrap {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.12) 0%, rgba(129, 140, 248, 0.14) 50%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 18px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0 0 0.4rem 0;
        line-height: 1.2;
    }
    .hero-sub {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0 0 0.65rem 0;
        line-height: 1.45;
    }
    .hero-desc {
        color: #CBD5E1;
        font-size: 0.88rem;
        margin: 0 0 0.75rem 0;
    }
    .hero-icon {
        font-size: 3.2rem;
        line-height: 1;
        opacity: 0.95;
        flex-shrink: 0;
    }
    .badge-row { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .pill-badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #E0F2FE;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.35);
    }
    .pill-badge-purple {
        background: rgba(129, 140, 248, 0.15);
        border-color: rgba(129, 140, 248, 0.4);
        color: #E0E7FF;
    }
    .metric-card {
        background: rgba(17, 24, 39, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        min-height: 88px;
        border-top: 3px solid #38BDF8;
    }
    .metric-card.purple { border-top-color: #818CF8; }
    .metric-card.green { border-top-color: #22C55E; }
    .metric-icon { font-size: 1.35rem; margin-bottom: 0.25rem; }
    .metric-label {
        color: #94A3B8;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .ui-card-title {
        color: #38BDF8;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.65rem;
    }
    .ui-card-heading {
        color: #F8FAFC;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    .error-card {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.5rem 0 1rem 0;
    }
    .error-card-title {
        color: #FCA5A5;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }
    .error-card-msg { color: #FECACA; font-size: 0.9rem; line-height: 1.45; }
    .warn-banner {
        background: rgba(249, 115, 22, 0.12);
        border: 1px solid rgba(249, 115, 22, 0.4);
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
        color: #FDBA74;
        font-size: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .quiz-card {
        background: rgba(17, 24, 39, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.85rem;
    }
    .quiz-qnum {
        color: #38BDF8;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .quiz-question {
        color: #F8FAFC;
        font-size: 1rem;
        font-weight: 600;
        margin: 0.35rem 0 0.65rem 0;
        line-height: 1.4;
    }
    .quiz-opt {
        color: #CBD5E1;
        font-size: 0.92rem;
        margin: 0.28rem 0;
        padding-left: 0.15rem;
    }
    .quiz-answer {
        color: #86EFAC;
        font-weight: 600;
        margin-top: 0.65rem;
        font-size: 0.9rem;
    }
    .quiz-expl {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 0.35rem;
        line-height: 1.4;
    }
    .top-source-line {
        color: #94A3B8;
        font-size: 0.85rem;
        margin-bottom: 0.35rem;
    }
    .top-source-line strong { color: #38BDF8; }
    .sidebar-section {
        color: #38BDF8;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0.75rem 0 0.4rem 0;
    }
    .status-line {
        color: #94A3B8;
        font-size: 0.82rem;
        margin: 0.2rem 0;
    }
    .status-line strong { color: #F8FAFC; }
    .status-ok { color: #22C55E; font-weight: 600; }
    .status-warn { color: #F97316; font-weight: 600; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _cache_version() -> int:
    return int(st.session_state.get("kb_cache_version", 0))


@st.cache_resource
def cached_embedder(cache_version: int):
    from src.embedder import Embedder

    return Embedder()


@st.cache_resource
def cached_retriever(cache_version: int):
    from src.retriever import Retriever

    r = Retriever(embedder=cached_embedder(cache_version))
    r.load()
    return r


@st.cache_resource
def cached_llm_client(allow_local_fallback: bool):
    from src.llm_client import LLMClient

    return LLMClient(allow_local_fallback=allow_local_fallback)


@st.cache_data(ttl=600)
def cached_dataset_inspection():
    if not DATASET_CSV.exists():
        return None
    df = pd.read_csv(DATASET_CSV, encoding="utf-8")
    summary = {}
    if DATASET_JSON.exists():
        summary = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
    return df.to_dict(orient="records"), summary


@st.cache_data(ttl=600)
def cached_dashboard_metrics():
    metrics = {
        "pdfs_ok": "6",
        "pdfs_total": "6",
        "chunks": "137",
        "sources": "—",
        "ocr_ready": True,
    }
    if DATASET_JSON.exists():
        ds = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
        metrics["pdfs_ok"] = str(ds.get("ok_pdfs", metrics["pdfs_ok"]))
        metrics["pdfs_total"] = str(ds.get("total_pdfs", metrics["pdfs_total"]))
        ocr = ds.get("ocr_dependency_status", {})
        metrics["ocr_ready"] = ocr.get("ready", True)
    if CHUNK_JSON.exists():
        ch = json.loads(CHUNK_JSON.read_text(encoding="utf-8"))
        metrics["chunks"] = str(ch.get("total_chunks", metrics["chunks"]))
        metrics["sources"] = str(ch.get("total_sources", metrics["sources"]))
    return metrics


def format_sources_df(sources: list[dict]) -> pd.DataFrame:
    records = []
    for s in sources or []:
        if not isinstance(s, dict):
            continue
        score_val = s.get("score", "")
        try:
            score_display = round(float(score_val), 4)
        except (TypeError, ValueError):
            score_display = score_val if score_val != "" else ""
        records.append(
            {
                "Source": s.get("source", ""),
                "Page": s.get("page", ""),
                "Rank": s.get("rank", ""),
                "Score": score_display,
            }
        )
    return pd.DataFrame(records)


def _is_quota_message(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return (
        is_gemini_quota_response(text)
        or "quota-exhausted" in lower
        or "quota is exhausted" in lower
        or ("all gemini" in lower and "quota" in lower)
    )


def _render_quota_error_card(allow_local_fallback: bool) -> None:
    msg = (
        "All configured Gemini API keys are out of quota. "
        "Add a new key or enable local fallback in the sidebar (Advanced)."
    )
    st.markdown(
        f'<div class="error-card">'
        f'<div class="error-card-title">⚠️ Gemini quota exhausted</div>'
        f'<div class="error-card-msg">{msg}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    if not allow_local_fallback:
        st.caption("Local fallback is off. Enable it under **Advanced** only if needed.")


def _render_hero() -> None:
    st.markdown(
        """
<div class="hero-wrap">
  <div>
    <div class="hero-title">Personalized Learning Assistant</div>
    <div class="hero-sub">Ask, summarize, and generate quizzes from course PDFs using RAG + OCR</div>
    <div class="hero-desc">Upload lecture PDFs, build a searchable knowledge base, and study with grounded AI answers.</div>
    <div class="badge-row">
      <span class="pill-badge">FAISS Vector Search</span>
      <span class="pill-badge">OCR Enabled</span>
      <span class="pill-badge pill-badge-purple">Gemini LLM</span>
      <span class="pill-badge pill-badge-purple">Optional Reranking</span>
    </div>
  </div>
  <div class="hero-icon">🧠</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_metric_cards(dm: dict, index_ready: bool) -> None:
    pdfs_val = f"{dm.get('pdfs_ok', '6')} / {dm.get('pdfs_total', '6')}"
    chunks_val = str(dm.get("chunks", "137"))
    vector_val = "FAISS" if index_ready else "Not built"
    llm_val = "Gemini"

    cards = [
        ("📄", "PDFs indexed", pdfs_val, ""),
        ("🧩", "Text chunks", chunks_val, "purple"),
        ("⚡", "Vector store", vector_val, "green"),
        ("🧠", "LLM", llm_val, ""),
    ]
    cols = st.columns(4)
    for col, (icon, label, value, accent) in zip(cols, cards):
        accent_cls = f" {accent}" if accent else ""
        with col:
            st.markdown(
                f'<div class="metric-card{accent_cls}">'
                f'<div class="metric-icon">{icon}</div>'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


@contextmanager
def _input_card(title: str, heading: str):
    with st.container(border=True):
        st.markdown(f'<div class="ui-card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ui-card-heading">{heading}</div>', unsafe_allow_html=True)
        yield


@contextmanager
def _output_card(title: str, heading: str):
    with st.container(border=True):
        st.markdown(f'<div class="ui-card-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ui-card-heading">{heading}</div>', unsafe_allow_html=True)
        yield


def _show_performance_expander(total: float, details: dict) -> None:
    with st.expander("Performance details", expanded=False):
        if details:
            for k, v in details.items():
                st.write(f"**{k.replace('_', ' ').title()}:** {v:.2f}s")
        st.write(f"**Total:** {total:.2f}s")


def _show_llm_answer(answer: str, llm, allow_local_fallback: bool) -> bool:
    """Returns True if quota/error was shown (caller skips normal render)."""
    if _is_quota_message(answer):
        _render_quota_error_card(allow_local_fallback)
        return True
    if answer.startswith("Error:"):
        st.error(answer)
        return True
    st.markdown(answer)
    return False


def _format_sources_display(sources: list[dict]) -> None:
    df = format_sources_df(sources)
    with st.expander("View retrieved sources", expanded=False):
        if df.empty:
            st.caption("No sources returned.")
            return
        top = sources[0] if sources else {}
        src_name = html.escape(str(top.get("source", "—")))
        page = html.escape(str(top.get("page", "—")))
        st.markdown(
            f'<p class="top-source-line">Top source: <strong>{src_name}</strong>, page {page}</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


def _render_mcq_quiz_card(index: int, mcq: dict) -> None:
    q = html.escape(str(mcq.get("question", "Question")))
    opts = mcq.get("options", {}) if isinstance(mcq.get("options"), dict) else {}
    answer = html.escape(str(mcq.get("answer", "?")))
    expl = html.escape(str(mcq.get("explanation", "")))

    opts_html = ""
    for letter in ["A", "B", "C", "D"]:
        if letter in opts:
            opt_text = html.escape(str(opts[letter]))
            opts_html += f'<div class="quiz-opt"><strong>{letter}.</strong> {opt_text}</div>'

    expl_html = (
        f'<div class="quiz-expl"><strong>Explanation:</strong> {expl}</div>' if expl else ""
    )
    st.markdown(
        f'<div class="quiz-card">'
        f'<div class="quiz-qnum">Question {index}</div>'
        f'<div class="quiz-question">{q}</div>'
        f"{opts_html}"
        f'<div class="quiz-answer">✓ Correct answer: {answer}</div>'
        f"{expl_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _display_mcqs(mcqs: list[dict], parse_method: str) -> None:
    if parse_method == "fallback":
        st.caption("Parsed from model text format.")
    elif parse_method in ("repair", "json_repair"):
        st.caption("Recovered via JSON repair pass.")
    for i, mcq in enumerate(mcqs, 1):
        _render_mcq_quiz_card(i, mcq)


def _display_mcq_result(result: dict, allow_local_fallback: bool) -> None:
    mcqs = result.get("mcqs") or []
    method = result.get("parse_method", "json")
    raw = result.get("raw_text", "")
    requested_count = int(result.get("requested_count") or result.get("num_questions") or len(mcqs))
    parsed_count = int(result.get("parsed_count") or len(mcqs))

    if not mcqs and raw:
        parsed, method = parse_mcq_all(raw)
        if parsed:
            mcqs = parsed[:requested_count]
            parsed_count = len(mcqs)

    if mcqs:
        parsed_count = len(mcqs)
        if parsed_count < requested_count:
            st.warning(f"{parsed_count}/{requested_count} questions structured.")
        else:
            st.success(f"{parsed_count}/{requested_count} questions ready.")
        if result.get("warning"):
            st.caption(result["warning"])
        _display_mcqs(mcqs, method)
        if parsed_count < requested_count and raw:
            with st.expander("View raw MCQ output", expanded=False):
                st.text(raw)
        return

    if requested_count:
        st.warning(f"{parsed_count}/{requested_count} questions structured.")
    st.info("MCQs were generated, but could not be fully structured.")
    if result.get("warning"):
        st.caption(result["warning"])
    if raw:
        with st.expander("View raw MCQ output", expanded=False):
            st.text(raw)


def _quick_pypdf_inspect():
    reports = []
    for pdf_path in sorted(DATASET_DIR.glob("*.pdf")):
        entry = {
            "filename": pdf_path.name,
            "page_count": 0,
            "total_extracted_char_count": 0,
            "status": "ERROR",
        }
        try:
            reader = PdfReader(str(pdf_path))
            entry["page_count"] = len(reader.pages)
            chars = sum(len((p.extract_text() or "").strip()) for p in reader.pages)
            entry["total_extracted_char_count"] = chars
            entry["status"] = "OK" if chars > 500 else ("WARNING" if chars > 0 else "WARNING")
        except Exception:
            pass
        reports.append(entry)
    ok = sum(1 for r in reports if r["status"] == "OK")
    summary = {
        "total_pdfs": len(reports),
        "ok_pdfs": ok,
        "warning_pdfs": len(reports) - ok,
        "error_pdfs": 0,
        "total_extracted_chars": sum(r["total_extracted_char_count"] for r in reports),
        "conclusion": f"Quick pypdf scan: {ok} OK / {len(reports)} PDFs",
    }
    return reports, summary


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📚 Learning Assistant")
    st.caption("Build your knowledge base, then study with AI.")

    st.markdown('<p class="sidebar-section">Knowledge Base</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded and st.button("Save uploaded PDFs", use_container_width=True):
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            (DATASET_DIR / f.name).write_bytes(f.getvalue())
        st.success(f"Saved {len(uploaded)} file(s)")
        st.rerun()

    if st.button("Inspect dataset PDFs", use_container_width=True):
        t0 = time.perf_counter()
        with st.spinner("Loading dataset inspection…"):
            cached = cached_dataset_inspection()
            if cached:
                reports, summary = cached
                label = "From cached diagnostics (fast)"
            else:
                reports, summary = _quick_pypdf_inspect()
                label = "Quick pypdf scan (no OCR)"
            st.session_state["inspect_result"] = (
                reports,
                summary,
                label,
                time.perf_counter() - t0,
            )

    if st.button("Build / Rebuild Knowledge Base", type="primary", use_container_width=True):
        t0 = time.perf_counter()
        with st.spinner("Building index…"):
            try:
                result = build_knowledge_base()
                st.cache_resource.clear()
                st.cache_data.clear()
                st.session_state["kb_cache_version"] = _cache_version() + 1
                st.session_state["build_timing"] = time.perf_counter() - t0
                if result.get("status") == "error":
                    st.error(result.get("message", "Build failed"))
                else:
                    st.success(f"Built in {st.session_state['build_timing']:.1f}s")
            except Exception as exc:
                st.error(str(exc))

    if "inspect_result" in st.session_state:
        reports, summary, label, elapsed = st.session_state["inspect_result"]
        with st.expander("Inspection results", expanded=False):
            st.caption(f"{label} · {elapsed:.1f}s")
            c1, c2 = st.columns(2)
            c1.metric("PDFs", summary.get("total_pdfs", len(reports)))
            c2.metric("OK", summary.get("ok_pdfs", 0))
            st.dataframe(pd.DataFrame(reports), use_container_width=True, hide_index=True)

    allow_local_fallback = False
    with st.expander("⚙️ Advanced", expanded=False):
        allow_local_fallback = st.checkbox(
            "Allow local fallback (slow)",
            value=False,
            help="Uses flan-t5 only when all Gemini keys fail.",
        )
        st.caption("Local fallback is slower and may take longer.")
        if st.button("Run full OCR inspection (slow)", use_container_width=True):
            st.warning("May take several minutes.")
            t0 = time.perf_counter()
            with st.spinner("Running full OCR inspection…"):
                import inspect_dataset as ds_mod

                reports, summary = ds_mod.run_inspection()
                st.session_state["inspect_result"] = (
                    reports,
                    summary,
                    "Full inspection saved to data/diagnostics/",
                    time.perf_counter() - t0,
                )
                st.rerun()

    st.markdown('<p class="sidebar-section">Retrieval Settings</p>', unsafe_allow_html=True)
    top_k = st.slider("top_k", 1, 15, DEFAULT_TOP_K)
    use_reranking = st.checkbox("Use lightweight reranking", value=False)
    st.caption("Reranking combines FAISS similarity with keyword overlap.")

    st.markdown('<p class="sidebar-section">Status</p>', unsafe_allow_html=True)
    idx_ok = index_exists()
    idx_cls = "status-ok" if idx_ok else "status-warn"
    idx_txt = "Ready" if idx_ok else "Not built"
    fb_txt = "On" if allow_local_fallback else "Off"
    st.markdown(
        f'<p class="status-line">Index: <span class="{idx_cls}">{idx_txt}</span></p>'
        f'<p class="status-line">OCR: <strong>Enabled</strong></p>'
        f'<p class="status-line">LLM: <strong>Gemini</strong></p>'
        f'<p class="status-line">Fallback: <strong>{fb_txt}</strong></p>',
        unsafe_allow_html=True,
    )

# --- Main ---
_render_hero()
dm = cached_dashboard_metrics()
idx_ready = index_exists()
_render_metric_cards(dm, idx_ready)

if allow_local_fallback:
    st.markdown(
        '<div class="warn-banner">⚡ Local fallback is enabled — responses may be slower.</div>',
        unsafe_allow_html=True,
    )

if not idx_ready:
    st.warning("Build the knowledge base from the sidebar to start.")
else:
    ver = _cache_version()
    tab_ask, tab_sum, tab_mcq = st.tabs(["🔎 Ask", "📝 Summarize", "🧠 Generate MCQ"])

    with tab_ask:
        with _input_card("Input", "Ask your documents"):
            question = st.text_input(
                "Question",
                placeholder="e.g. What is retrieval augmented generation?",
                label_visibility="collapsed",
            )
            run_ask = st.button("Get Answer", type="primary", use_container_width=True)

        if run_ask and question:
            details: dict[str, float] = {}
            t_total = time.perf_counter()
            retriever = cached_retriever(ver)
            llm = cached_llm_client(allow_local_fallback)
            t0 = time.perf_counter()
            chunks = retriever.retrieve(
                question,
                top_k=top_k,
                use_reranking=use_reranking,
                candidate_k=10,
            )
            details["retrieve"] = time.perf_counter() - t0
            t1 = time.perf_counter()
            answer = clean_generated_answer(
                llm.generate(build_qa_prompt(question, chunks), max_tokens=800)
            )
            details["generate"] = time.perf_counter() - t1
            sources = _sources_from_chunks(chunks)
            st.session_state["qa_result"] = {
                "question": question,
                "answer": answer,
                "sources": sources,
            }
            st.session_state["qa_timing"] = (time.perf_counter() - t_total, details)

        if "qa_result" in st.session_state:
            total, details = st.session_state.get("qa_timing", (0, {}))
            with _output_card("Output", "Generated Answer"):
                answer = st.session_state["qa_result"]["answer"]
                if not _show_llm_answer(
                    answer,
                    cached_llm_client(allow_local_fallback),
                    allow_local_fallback,
                ):
                    pass
                _format_sources_display(st.session_state["qa_result"].get("sources", []))
            _show_performance_expander(total, details)

    with tab_sum:
        with _input_card("Input", "Study Summary"):
            topic = st.text_input(
                "Topic",
                placeholder="e.g. Summarize federated learning",
                label_visibility="collapsed",
            )
            run_sum = st.button("Generate Summary", type="primary", use_container_width=True)

        if run_sum and topic:
            t_total = time.perf_counter()
            retriever = cached_retriever(ver)
            llm = cached_llm_client(allow_local_fallback)
            summarizer = Summarizer(retriever=retriever, llm=llm)
            result = summarizer.summarize_topic(
                topic,
                top_k=max(top_k, 5),
                use_reranking=use_reranking,
                candidate_k=10,
            )
            st.session_state["sum_result"] = clean_summary_text(result.get("summary", ""))
            st.session_state["sum_sources"] = result.get("sources", [])
            st.session_state["sum_timing"] = (time.perf_counter() - t_total, {})

        if "sum_result" in st.session_state:
            total, details = st.session_state.get("sum_timing", (0, {}))
            with _output_card("Output", "Study Summary"):
                summary_text = st.session_state["sum_result"]
                if not _show_llm_answer(
                    summary_text,
                    cached_llm_client(allow_local_fallback),
                    allow_local_fallback,
                ):
                    st.markdown(summary_text)
                _format_sources_display(st.session_state.get("sum_sources", []))
            _show_performance_expander(total, details)

    with tab_mcq:
        with _input_card("Input", "Quiz Questions"):
            topic = st.text_input(
                "Topic",
                placeholder="e.g. retrieval augmented generation",
                label_visibility="collapsed",
            )
            num_q = st.number_input("Number of questions", 1, 10, 3)
            run_mcq = st.button("Generate MCQs", type="primary", use_container_width=True)

        if run_mcq and topic:
            t_total = time.perf_counter()
            mcq_gen = MCQGenerator(
                retriever=cached_retriever(ver),
                llm=cached_llm_client(allow_local_fallback),
            )
            result = mcq_gen.generate(
                topic,
                num_questions=int(num_q),
                top_k=max(top_k, 5),
                use_reranking=use_reranking,
                candidate_k=10,
            )
            st.session_state["mcq_result"] = result
            st.session_state["mcq_timing"] = (
                time.perf_counter() - t_total,
                {"total": time.perf_counter() - t_total},
            )

        if "mcq_result" in st.session_state:
            result = st.session_state["mcq_result"]
            total, details = st.session_state.get("mcq_timing", (0, {}))
            raw = result.get("raw_text", "")
            with _output_card("Output", "Quiz Questions"):
                if _is_quota_message(raw):
                    _render_quota_error_card(allow_local_fallback)
                else:
                    _display_mcq_result(result, allow_local_fallback)
                _format_sources_display(result.get("sources", []))
            _show_performance_expander(total, details)
