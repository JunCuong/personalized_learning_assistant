"""Topic- or document-grounded summarization via retrieval."""

from __future__ import annotations

import re
from typing import Any

from src.llm_client import LLMClient, is_gemini_quota_response
from src.qa_generator import _format_context, _sources_from_chunks
from src.retriever import Retriever
from src.summary_utils import clean_summary_text, count_bullets


_FALLBACK_SENTENCE_RE = re.compile(r'[^.!?]+[.!?]["\')\]]*')


def _normalize_fallback_sentence(text: str) -> str:
    normalized = text.casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_bad_fallback_sentence(sentence: str) -> bool:
    normalized = _normalize_fallback_sentence(sentence)
    if not normalized:
        return True
    if re.search(
        r"\b(vnuk|institute|research and executive education|footer|page|slide)\b",
        normalized,
    ):
        return True
    if re.search(r"\b(what is|types of)\b", normalized) and len(normalized) < 90:
        return True

    letters = [ch for ch in sentence if ch.isalpha()]
    if letters:
        upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
        if upper_ratio > 0.65 and len(sentence) < 140:
            return True
    return False


def _fallback_summary_from_chunks(chunks: list[dict], max_bullets: int = 5) -> str:
    selected: list[str] = []
    seen: list[str] = []

    def add_from_chunk(chunk: dict) -> None:
        raw_text = str(chunk.get("chunk_text", "")).strip()
        text = re.sub(r"\s+", " ", raw_text).strip()
        if not text:
            return

        sentences = [m.group(0).strip() for m in _FALLBACK_SENTENCE_RE.finditer(text)]
        if len(sentences) < 2:
            sentences.extend(part.strip() for part in re.split(r"[;\n]", raw_text))

        for sentence in sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip(" -*\u2022\u2013\u2014")
            sentence = re.sub(r"^RAG Components\s+", "", sentence, flags=re.IGNORECASE)
            sentence = re.split(
                r"\bwhat is the prompt\b", sentence, maxsplit=1, flags=re.IGNORECASE
            )[0].strip(" :;-")
            if len(sentence) < 45 or _is_bad_fallback_sentence(sentence):
                continue

            normalized = _normalize_fallback_sentence(sentence)
            key = normalized[:80]
            if not normalized or any(
                normalized == prev or key == prev[:80] for prev in seen
            ):
                continue

            seen.append(normalized)
            if not re.search(r'[.!?)]["\')\]]*\s*$', sentence):
                sentence = f"{sentence}."
            selected.append(sentence)
            if len(selected) >= max_bullets:
                break

    for chunk in chunks[:3]:
        add_from_chunk(chunk)
        if len(selected) >= max_bullets:
            break

    if len(selected) < 3:
        for chunk in chunks[3:5]:
            add_from_chunk(chunk)
            if len(selected) >= max_bullets:
                break

    if len(selected) < 3:
        for chunk in chunks[:5]:
            text = re.sub(r"\s+", " ", str(chunk.get("chunk_text", ""))).strip()
            if len(text) < 45 or _is_bad_fallback_sentence(text):
                continue
            snippet = text[:220].rsplit(" ", 1)[0].strip()
            if not snippet:
                continue
            normalized = _normalize_fallback_sentence(snippet)
            key = normalized[:80]
            if any(normalized == prev or key == prev[:80] for prev in seen):
                continue
            seen.append(normalized)
            selected.append(f"{snippet}.")
            if len(selected) >= max_bullets:
                break

    return "\n".join(f"- {sentence}" for sentence in selected[:max_bullets])


def _is_llm_error_text(text: str) -> bool:
    stripped = str(text).strip()
    return stripped.startswith("Error:") or is_gemini_quota_response(stripped)


def build_summary_prompt(topic: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    return f"""You are a study assistant. Summarize the following topic using ONLY the context below.

Requirements:
- Return exactly 5 bullet points.
- Do not include an introduction.
- Do not write "Here are..."
- Each bullet must be a complete sentence.
- Use only the provided context.
- Start each bullet with "- ".
- Do not output empty bullets or placeholder dashes.
- Reference source filenames or pages when helpful.

Topic: {topic}

Context:
{context}

Output exactly 5 bullets:"""


def _retry_summary_prompt(topic: str, chunks: list[dict]) -> str:
    context = _format_context(chunks, max_chunk_chars=350)
    return f"""Your previous summary was invalid or too sparse.
Return exactly 5 bullet points about "{topic}".
Do not include an introduction.
Do not write "Here are..."
Each bullet must be a complete sentence.
Use only the provided context.
Start each bullet with "- ".
No empty bullets or placeholder dashes.

Context:
{context}

Exactly 5 bullets:"""


class Summarizer:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    def summarize_topic(
        self,
        topic: str,
        top_k: int = 8,
        use_reranking: bool = False,
        candidate_k: int = 10,
    ) -> dict[str, Any]:
        chunks = self.retriever.retrieve(
            topic,
            top_k=top_k,
            use_reranking=use_reranking,
            candidate_k=candidate_k,
        )
        if not chunks:
            return {
                "topic": topic,
                "summary": "No relevant content found in the knowledge base.",
                "sources": [],
            }

        prompt = build_summary_prompt(topic, chunks)
        raw_summary = self.llm.generate(prompt, max_tokens=800)
        summary = "" if _is_llm_error_text(raw_summary) else clean_summary_text(raw_summary)

        if count_bullets(summary) < 3:
            retry_prompt = _retry_summary_prompt(topic, chunks)
            retry_text = self.llm.generate(retry_prompt, max_tokens=800)
            retry_clean = "" if _is_llm_error_text(retry_text) else clean_summary_text(retry_text)
            if count_bullets(retry_clean) > count_bullets(summary):
                summary = retry_clean

        if count_bullets(summary) < 3:
            fallback = clean_summary_text(_fallback_summary_from_chunks(chunks))
            if count_bullets(fallback) >= 3:
                summary = fallback

        return {
            "topic": topic,
            "summary": summary,
            "sources": _sources_from_chunks(chunks),
        }

    def summarize_document(
        self, source_filename: str, top_k: int = 15
    ) -> dict[str, Any]:
        query = f"summary of document {source_filename}"
        chunks = self.retriever.retrieve(query, top_k=top_k)
        chunks = [c for c in chunks if c["source"] == source_filename] or chunks

        if not chunks:
            return {
                "topic": source_filename,
                "summary": f"No indexed content found for {source_filename}.",
                "sources": [],
            }

        prompt = build_summary_prompt(f"entire document: {source_filename}", chunks)
        raw_summary = self.llm.generate(prompt, max_tokens=800)
        summary = "" if _is_llm_error_text(raw_summary) else clean_summary_text(raw_summary)
        if count_bullets(summary) < 3:
            fallback = clean_summary_text(_fallback_summary_from_chunks(chunks))
            if count_bullets(fallback) >= 3:
                summary = fallback
        return {
            "topic": source_filename,
            "summary": summary,
            "sources": _sources_from_chunks(chunks),
        }
