"""Grounded question answering from retrieved chunks."""

from __future__ import annotations

import re
from typing import Any

from src.llm_client import LLMClient
from src.retriever import Retriever


def _format_context(chunks: list[dict], max_chunk_chars: int = 600) -> str:
    parts = []
    for c in chunks:
        text = c["chunk_text"]
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars] + "..."
        parts.append(
            f"[Source: {c['source']}, Page: {c['page']}, Rank: {c['rank']}]\n"
            f"{text}\n"
        )
    return "\n---\n".join(parts)


def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for c in chunks:
        key = (c["source"], c["page"], c["rank"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": c["source"],
                "page": c["page"],
                "rank": c["rank"],
                "score": c["score"],
            }
        )
    return sources


def build_qa_prompt(question: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    return f"""You are a study assistant. Answer the question using ONLY the context below.

Requirements:
- Answer in 2–4 complete sentences.
- Do not end mid-sentence.
- Use only the provided context.
- Cite sources using PDF filename and page (e.g. RAG.pdf p.6).
- Synthesize across pages briefly; do not list every chunk.
- If the answer is not in the context, reply exactly: "I cannot find this information in the uploaded documents."

Context:
{context}

Question: {question}

Answer:"""


def _last_complete_sentence_end(text: str) -> int:
    matches = list(re.finditer(r'[.!?]["\')\]]*(?=\s|$)', text))
    return matches[-1].end() if matches else -1


def clean_generated_answer(text: str) -> str:
    if not text:
        return text

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    original = cleaned

    while cleaned:
        trimmed = re.sub(r"\s*(?:\(|\[|\{)[^\)\]\}]*$", "", cleaned).rstrip()
        if trimmed != cleaned:
            cleaned = trimmed
            continue

        trimmed = re.sub(r"[\s,;:]+$", "", cleaned).rstrip()
        if trimmed != cleaned:
            cleaned = trimmed
            continue

        break

    if not cleaned:
        fallback = original.rstrip(" \t\r\n([{,;:")
        return fallback or original

    if not re.search(r'[.!?]["\')\]]*\s*$', cleaned):
        end = _last_complete_sentence_end(cleaned)
        if end > 0:
            complete = cleaned[:end].strip()
            if complete:
                return complete

    return cleaned


class QAGenerator:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    def ask(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = False,
        candidate_k: int = 10,
    ) -> dict[str, Any]:
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            use_reranking=use_reranking,
            candidate_k=candidate_k,
        )
        if not chunks:
            return {
                "question": question,
                "answer": "No relevant content found in the knowledge base.",
                "sources": [],
            }

        prompt = build_qa_prompt(question, chunks)
        answer = clean_generated_answer(self.llm.generate(prompt, max_tokens=800))
        return {
            "question": question,
            "answer": answer,
            "sources": _sources_from_chunks(chunks),
        }
