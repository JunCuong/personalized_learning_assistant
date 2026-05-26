"""Multiple-choice question generation from retrieved context."""

from __future__ import annotations

import json
import re
from typing import Any

from src.llm_client import LLMClient, is_gemini_quota_response
from src.qa_generator import _sources_from_chunks
from src.retriever import Retriever


def _format_mcq_context(
    chunks: list[dict], max_chunks: int = 3, max_chunk_chars: int = 850
) -> str:
    parts = []
    for idx, chunk in enumerate(chunks[:max_chunks], 1):
        text = re.sub(r"\s+", " ", str(chunk.get("chunk_text", ""))).strip()
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars].rsplit(" ", 1)[0].rstrip() + "..."
        if text:
            parts.append(f"Context {idx}:\n{text}")
    return "\n\n".join(parts)


def build_mcq_prompt(topic: str, num_questions: int, chunks: list[dict]) -> str:
    context = _format_mcq_context(chunks)
    return f"""You are a study assistant generating multiple-choice questions from retrieved study context.

Rules:
- Return exactly {num_questions} MCQs.
- Return ONLY a valid JSON array.
- Do not use markdown fences.
- Do not include prose before or after JSON.
- Do not truncate the JSON.
- Generate questions only from the provided context.
- Make questions concise.
- Avoid overlong explanations.
- Do not mention "according to the provided context" inside every question.
- Do not include underscores like according_to_the_provided_context.
- Do not return an empty array when context is provided.
- If context is limited, write questions about definitions, components, benefits, or process details stated in the context.
- Every MCQ must include:
  - question: string
  - options: object with exactly keys A, B, C, D
  - answer: one of "A", "B", "C", "D"
  - explanation: string

Use this exact schema:
[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "answer": "A",
    "explanation": "Why A is correct."
  }}
]

Topic: {topic}

Context:
{context}

JSON array:"""


def _mcq_repair_prompt(raw_text: str) -> str:
    return f"""Convert the following MCQ content into a valid JSON array only.
No markdown fences.
No prose before or after JSON.
Do not truncate the JSON.
Every item must include question, options with exactly A/B/C/D, answer, and explanation.
Must be parseable by json.loads.

Content:
{raw_text[:4000]}

JSON array:"""


def _mcq_retry_prompt(topic: str, num_questions: int, chunks: list[dict], raw_text: str) -> str:
    context = _format_mcq_context(chunks)
    return f"""Your previous response was invalid or incomplete. Regenerate exactly {num_questions} complete MCQs as valid JSON only. Do not truncate.

Rules:
- Return exactly {num_questions} MCQs.
- Return ONLY a valid JSON array.
- Do not use markdown fences.
- Do not include prose before or after JSON.
- Generate questions only from the provided context.
- Make questions concise.
- Avoid overlong explanations.
- Do not mention "according to the provided context" inside every question.
- Do not include underscores like according_to_the_provided_context.
- Every MCQ must include question, options, answer, and explanation.
- The options field must be an object with exactly keys A, B, C, D.
- The answer field must be one of "A", "B", "C", "D".
- Do not return an empty array when context is provided.
- If context is limited, write questions about definitions, components, benefits, or process details stated in the context.

Use this exact schema:
[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "answer": "A",
    "explanation": "Why A is correct."
  }}
]

Previous response:
{raw_text[:1200]}

Topic: {topic}

Context:
{context}

JSON array:"""


def build_single_mcq_prompt(
    topic: str, context: str, used_questions: list[str], attempt: int = 1
) -> str:
    used = "\n".join(f"- {question}" for question in used_questions) or "- None yet."
    retry_line = ""
    if attempt > 1:
        retry_line = (
            "Your previous response for this question was invalid, duplicated, "
            "or about metadata. Return a complete valid JSON object only.\n"
        )

    return f"""{retry_line}Return exactly ONE multiple-choice question as a valid JSON object.
Do not return a JSON array.
Do not use markdown fences.
No prose outside JSON.

Schema:
{{
  "question": "...",
  "options": {{
    "A": "...",
    "B": "...",
    "C": "...",
    "D": "..."
  }},
  "answer": "A",
  "explanation": "..."
}}

Rules:
- Focus on concepts, definitions, mechanisms, benefits, limitations, or comparisons.
- Do not ask about document title, institution name, author, page number, footer, or source metadata.
- Do not ask "what is the primary topic of the context?"
- Do not use weird phrases like "according_to_the_provided_context".
- Do not mention "provided context" or "retrieved context" in the question.
- Make the question useful for studying.
- Make all four options plausible but only one correct.
- Keep the explanation to one concise sentence.
- Avoid duplicating previous questions:
{used}
- Use only the provided context.

Topic: {topic}

Context:
{context}

JSON object:"""


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    if "```" in text:
        match = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if match:
            text = match.group(1).strip()
    return text.strip()


def _remove_trailing_commas(json_str: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", json_str)


def parse_mcq_output(raw_text: str) -> list[dict] | None:
    if not raw_text or not str(raw_text).strip():
        return None

    text = _strip_markdown_fences(str(raw_text).strip())
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    json_candidate = text[start : end + 1]
    for candidate in (json_candidate, _remove_trailing_commas(json_candidate)):
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and data:
                normalized = _normalize_mcq_list(data)
                return normalized if normalized else None
        except json.JSONDecodeError:
            continue
    return None


def _clean_field(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalize_options(options: Any) -> dict[str, str] | None:
    normalized: dict[str, str] = {}

    if isinstance(options, dict):
        for key, value in options.items():
            key_text = str(key).strip().upper()
            match = re.search(r"(?:^|[^A-Z])([ABCD])(?:[^A-Z]|$)", key_text)
            if not match:
                continue
            letter = match.group(1)
            option_text = _clean_field(value)
            if option_text:
                normalized[letter] = option_text
    elif isinstance(options, list):
        for letter, value in zip(["A", "B", "C", "D"], options):
            option_text = _clean_field(value)
            if option_text:
                normalized[letter] = option_text

    if all(normalized.get(letter) for letter in ["A", "B", "C", "D"]):
        return {letter: normalized[letter] for letter in ["A", "B", "C", "D"]}
    return None


def parse_single_mcq_output(raw_text: str) -> dict | None:
    if not raw_text or not str(raw_text).strip():
        return None

    text = _strip_markdown_fences(str(raw_text).strip())
    candidates: list[str] = []

    candidates.append(text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        for json_candidate in (candidate, _remove_trailing_commas(candidate)):
            try:
                data = json.loads(json_candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict):
                return _normalize_mcq_item(data)
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                return _normalize_mcq_item(data[0])

    return None


def _normalize_answer(answer: Any) -> str | None:
    match = re.search(r"\b([ABCD])\b", str(answer).upper())
    return match.group(1) if match else None


def _normalize_question_text(question: str) -> str:
    normalized = question.casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_metadata_question(question: str) -> bool:
    normalized = _normalize_question_text(question)
    metadata_patterns = [
        r"\b(document title|title of the document|paper title|slide title)\b",
        r"\b(institution|university|vnuk|author|authors|page number|footer)\b",
        r"\b(source metadata|source file|pdf file|file name|filename)\b",
        r"\b(provided context|retrieved context|context above)\b",
        r"\b(primary topic of the context|main topic of the context)\b",
        r"\b(primary subject discussed|topic is mainly discussed)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in metadata_patterns)


def _normalize_mcq_item(item: Any) -> dict | None:
    if not isinstance(item, dict):
        return None

    question = _clean_field(item.get("question", ""))
    options = _normalize_options(item.get("options", {}))
    answer = _normalize_answer(item.get("answer", ""))
    explanation = _clean_field(item.get("explanation", "")) or "See retrieved context."

    if (
        not question
        or not options
        or answer not in {"A", "B", "C", "D"}
        or not explanation
        or "_" in question
        or _is_metadata_question(question)
    ):
        return None

    return {
        "question": question,
        "options": options,
        "answer": answer,
        "explanation": explanation,
    }


def _normalize_mcq_list(items: list) -> list[dict]:
    normalized: list[dict] = []
    for item in items:
        mcq = _normalize_mcq_item(item)
        if mcq:
            normalized.append(mcq)
    return normalized


def parse_mcq_fallback_text(raw_text: str) -> list[dict] | None:
    """Parse plain-text MCQ blocks when JSON fails."""
    if not raw_text or not str(raw_text).strip():
        return None

    text = str(raw_text).strip()
    text = re.sub(r"(?m)^Question\s*:\s*", "Question: ", text, flags=re.IGNORECASE)
    text = re.sub(r"(?m)^Options\s*:\s*", "\nOptions:\n", text, flags=re.IGNORECASE)
    for letter in ["A", "B", "C", "D"]:
        text = re.sub(
            rf"(?m)^{letter}\s*:\s*",
            f"\n{letter}: ",
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"(?m)^Answer\s*:\s*", "\nAnswer: ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?m)^Explanation\s*:\s*", "\nExplanation: ", text, flags=re.IGNORECASE
    )
    blocks = re.split(
        r"(?=(?:Question\s*\d*\s*[:.]?\s*|Question\s*:|Q\d+\s*[:.]?\s*|###\s*Question))",
        text,
        flags=re.IGNORECASE,
    )

    mcqs: list[dict] = []
    for block in blocks:
        block = block.strip()
        if len(block) < 20:
            continue

        q_match = re.search(
            r"(?:Question\s*\d*\s*[:.]?\s*|Q\d+\s*[:.]?\s*)(.+?)(?=Options?:|(?=\n[A-D][\.\):])|$)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not q_match:
            q_match = re.search(r"^(.+?)(?=Options?:|(?=\n[A-D][\.\):]))", block, re.DOTALL)
        if not q_match:
            continue

        question = re.sub(r"\s+", " ", q_match.group(1).strip())
        options: dict[str, str] = {}
        for letter in ["A", "B", "C", "D"]:
            opt = re.search(
                rf"{letter}[\.\):]\s*(.+?)(?=\n[A-D][\.\):]|Answer:|Explanation:|$)",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if opt:
                options[letter] = opt.group(1).strip()

        if len(options) < 2:
            inline = re.findall(r"([A-D])[\.\):]\s*([^A-D]+?)(?=[A-D][\.\):]|Answer:|$)", block)
            for letter, val in inline:
                options[letter.upper()] = val.strip()

        ans_match = re.search(r"Answer\s*[:.]?\s*([A-D])", block, re.IGNORECASE)
        exp_match = re.search(
            r"Explanation\s*[:.]?\s*(.+)$", block, re.IGNORECASE | re.DOTALL
        )

        if question and len(options) >= 2:
            mcqs.append(
                {
                    "question": question,
                    "options": options,
                    "answer": ans_match.group(1).upper() if ans_match else "",
                    "explanation": exp_match.group(1).strip() if exp_match else "",
                }
            )

    normalized = _normalize_mcq_list(mcqs)
    return normalized if normalized else None


def parse_mcq_all(raw_text: str) -> tuple[list[dict] | None, str]:
    """Returns (mcqs, parse_method) where method is json, fallback, repair, or none."""
    parsed = parse_mcq_output(raw_text)
    if parsed:
        return parsed, "json"

    fallback = parse_mcq_fallback_text(raw_text)
    if fallback:
        return fallback, "fallback"

    return None, "none"


def parse_mcq_json(raw: str) -> list[dict] | None:
    mcqs, _ = parse_mcq_all(raw)
    return mcqs


def _is_generation_error(raw_text: str) -> bool:
    return str(raw_text).strip().startswith("Error:")


_FALLBACK_SENTENCE_RE = re.compile(r'[^.!?]+[.!?]["\')\]]*')
_FALLBACK_STEMS = [
    "Which statement best describes {topic}?",
    "Which point is associated with {topic}?",
    "What is a useful study takeaway about {topic}?",
    "Which statement is accurate for {topic}?",
]
_FALLBACK_DISTRACTORS = [
    "It is mainly a database normalization rule.",
    "It is primarily an operating system scheduling method.",
    "It is a file compression format for images.",
    "It is a hardware protocol for display cables.",
]


def _fallback_sentence_candidates(chunks: list[dict]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for chunk in chunks[:3]:
        text = re.sub(r"\s+", " ", str(chunk.get("chunk_text", ""))).strip()
        if not text:
            continue

        sentences = [m.group(0).strip() for m in _FALLBACK_SENTENCE_RE.finditer(text)]
        if len(sentences) < 2:
            sentences.extend(part.strip() for part in re.split(r"[;\n]", text))

        for sentence in sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip(" -*\u2022\u2013\u2014")
            if len(sentence) < 45:
                continue
            if not re.search(r'[.!?)]["\')\]]*\s*$', sentence):
                sentence = f"{sentence}."

            normalized = _normalize_question_text(sentence)
            key = normalized[:80]
            if not normalized or key in seen:
                continue
            seen.add(key)
            candidates.append(sentence)

    if len(candidates) < 3:
        for chunk in chunks[:3]:
            text = re.sub(r"\s+", " ", str(chunk.get("chunk_text", ""))).strip()
            if len(text) < 45:
                continue
            snippet = text[:180].rsplit(" ", 1)[0].strip()
            if not snippet:
                continue
            if not re.search(r'[.!?)]["\')\]]*\s*$', snippet):
                snippet = f"{snippet}."
            normalized = _normalize_question_text(snippet)
            key = normalized[:80]
            if key in seen:
                continue
            seen.add(key)
            candidates.append(snippet)

    return candidates


def _short_option(text: str, limit: int = 170) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "."


def _fallback_mcqs_from_context(
    topic: str,
    chunks: list[dict],
    needed: int,
    used_normalized_questions: set[str],
) -> list[dict]:
    fallback_mcqs: list[dict] = []
    sentences = _fallback_sentence_candidates(chunks)

    for idx, sentence in enumerate(sentences):
        if len(fallback_mcqs) >= needed:
            break

        stem = _FALLBACK_STEMS[idx % len(_FALLBACK_STEMS)].format(topic=topic)
        normalized_question = _normalize_question_text(stem)
        if normalized_question in used_normalized_questions:
            continue

        answer = ["A", "B", "C", "D"][idx % 4]
        options: dict[str, str] = {}
        distractors = iter(_FALLBACK_DISTRACTORS)
        for letter in ["A", "B", "C", "D"]:
            if letter == answer:
                options[letter] = _short_option(sentence)
            else:
                options[letter] = next(distractors)

        mcq = _normalize_mcq_item(
            {
                "question": stem,
                "options": options,
                "answer": answer,
                "explanation": f"The study material states: {_short_option(sentence, 190)}",
            }
        )
        if not mcq:
            continue

        used_normalized_questions.add(normalized_question)
        fallback_mcqs.append(mcq)

    return fallback_mcqs


class MCQGenerator:
    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMClient | None = None,
    ):
        self.retriever = retriever or Retriever()
        self.llm = llm or LLMClient()

    def _generate_single_mcq_with_raw(
        self,
        topic: str,
        context: str,
        used_questions: list[str],
        used_normalized_questions: set[str],
        attempt: int,
    ) -> tuple[dict | None, str]:
        prompt = build_single_mcq_prompt(topic, context, used_questions, attempt=attempt)
        raw = self.llm.generate(prompt, max_tokens=1200)
        mcq = parse_single_mcq_output(raw)
        if not mcq:
            return None, raw

        normalized_question = _normalize_question_text(mcq["question"])
        if not normalized_question or normalized_question in used_normalized_questions:
            return None, raw

        return mcq, raw

    def generate_single_mcq(
        self, topic: str, context: str, used_questions: list[str]
    ) -> dict | None:
        used_normalized = {
            _normalize_question_text(question) for question in used_questions
        }
        mcq, _ = self._generate_single_mcq_with_raw(
            topic, context, used_questions, used_normalized, attempt=1
        )
        return mcq

    def generate(
        self,
        topic: str,
        num_questions: int = 5,
        top_k: int = 10,
        use_reranking: bool = False,
        candidate_k: int = 10,
    ) -> dict[str, Any]:
        requested_count = max(1, int(num_questions))
        chunks = self.retriever.retrieve(
            topic,
            top_k=top_k,
            use_reranking=use_reranking,
            candidate_k=candidate_k,
        )
        sources = _sources_from_chunks(chunks)

        if not chunks:
            return {
                "topic": topic,
                "num_questions": requested_count,
                "requested_count": requested_count,
                "parsed_count": 0,
                "mcqs": [],
                "raw_text": "",
                "sources": [],
                "parse_ok": False,
                "parse_method": "none",
                "warning": f"Only 0/{requested_count} MCQs could be generated.",
            }

        prompt_chunks = chunks[:3]
        context = _format_mcq_context(prompt_chunks)
        mcqs: list[dict] = []
        raw_attempts: list[str] = []
        used_questions: list[str] = []
        used_normalized_questions: set[str] = set()
        used_retry = False
        used_fallback = False
        quota_seen = False

        for _ in range(requested_count):
            if quota_seen:
                break

            for attempt in range(1, 4):
                if attempt > 1:
                    used_retry = True
                mcq, raw = self._generate_single_mcq_with_raw(
                    topic,
                    context,
                    used_questions,
                    used_normalized_questions,
                    attempt=attempt,
                )
                raw_attempts.append(f"Attempt {len(raw_attempts) + 1}:\n{raw}")

                if is_gemini_quota_response(raw):
                    quota_seen = True
                    break

                if not mcq:
                    continue

                normalized_question = _normalize_question_text(mcq["question"])
                used_normalized_questions.add(normalized_question)
                used_questions.append(mcq["question"])
                mcqs.append(mcq)
                break

        if len(mcqs) < requested_count:
            fallback_mcqs = _fallback_mcqs_from_context(
                topic,
                prompt_chunks,
                requested_count - len(mcqs),
                used_normalized_questions,
            )
            if fallback_mcqs:
                used_fallback = True
                mcqs.extend(fallback_mcqs)

        mcqs = mcqs[:requested_count]
        parsed_count = len(mcqs)
        warning = None
        if parsed_count < requested_count:
            warning = f"Only {parsed_count}/{requested_count} MCQs could be generated."

        parse_method = "none"
        if parsed_count:
            if used_fallback:
                parse_method = "single_fallback"
            elif used_retry:
                parse_method = "single_retry"
            else:
                parse_method = "single_json"

        result = {
            "topic": topic,
            "num_questions": requested_count,
            "requested_count": requested_count,
            "parsed_count": parsed_count,
            "mcqs": mcqs,
            "raw_text": "" if parsed_count == requested_count else "\n\n".join(raw_attempts),
            "sources": sources,
            "parse_ok": parsed_count > 0,
            "parse_method": parse_method,
            "warning": warning,
        }
        return result
