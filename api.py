"""FastAPI backend for the Personalized Learning Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.llm_client import LLMClient
from src.mcq_generator import MCQGenerator
from src.pipeline import build_knowledge_base
from src.qa_generator import QAGenerator
from src.summarizer import Summarizer
from src.vector_db import index_exists

app = FastAPI(
    title="Personalized Learning Assistant API",
    description="RAG-based study assistant API",
    version="1.0.0",
)

_llm: LLMClient | None = None
_qa: QAGenerator | None = None
_summarizer: Summarizer | None = None
_mcq: MCQGenerator | None = None


def _ensure_index() -> None:
    if not index_exists():
        raise HTTPException(
            status_code=400,
            detail="Knowledge base not built. Call POST /build-index first.",
        )


def _get_qa() -> QAGenerator:
    global _qa, _llm
    _ensure_index()
    if _qa is None:
        _llm = LLMClient()
        _qa = QAGenerator(llm=_llm)
    return _qa


def _get_summarizer() -> Summarizer:
    global _summarizer, _llm
    _ensure_index()
    if _summarizer is None:
        _llm = _llm or LLMClient()
        _summarizer = Summarizer(llm=_llm)
    return _summarizer


def _get_mcq() -> MCQGenerator:
    global _mcq, _llm
    _ensure_index()
    if _mcq is None:
        _llm = _llm or LLMClient()
        _mcq = MCQGenerator(llm=_llm)
    return _mcq


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1, le=20)


class SummarizeRequest(BaseModel):
    topic: str
    top_k: int = Field(default=8, ge=1, le=20)


class MCQRequest(BaseModel):
    topic: str
    num_questions: int = Field(default=5, ge=1, le=15)
    top_k: int = Field(default=10, ge=1, le=20)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "index_built": index_exists(),
        "message": "Personalized Learning Assistant API is running",
    }


@app.post("/build-index")
def build_index_endpoint():
    try:
        result = build_knowledge_base()
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result)
        global _qa, _summarizer, _mcq
        _qa = _summarizer = _mcq = None
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask_endpoint(body: AskRequest):
    try:
        return _get_qa().ask(body.question, top_k=body.top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/summarize")
def summarize_endpoint(body: SummarizeRequest):
    try:
        return _get_summarizer().summarize_topic(body.topic, top_k=body.top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate-mcq")
def generate_mcq_endpoint(body: MCQRequest):
    try:
        return _get_mcq().generate(
            body.topic, num_questions=body.num_questions, top_k=body.top_k
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
