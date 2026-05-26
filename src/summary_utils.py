"""Summary post-processing helpers."""

from __future__ import annotations

import re


_BULLET_PREFIX_RE = re.compile(r"^(?:[-*]|\u2022|\d+[\).])\s+")
_INTRO_RE = re.compile(
    r"^(?:here are|here is|below are|the following)\b", re.IGNORECASE
)
_SENTENCE_END_RE = re.compile(r'[.!?]["\')\]]*\s*$')
_SENTENCE_SPLIT_RE = re.compile(r'[^.!?]+(?:[.!?]["\')\]]*|\)(?=\s|$))')


def _strip_bullet_prefix(line: str) -> tuple[bool, str]:
    match = _BULLET_PREFIX_RE.match(line)
    if not match:
        return False, line.strip()
    return True, line[match.end() :].strip()


def _is_punctuation_only(text: str) -> bool:
    return bool(re.fullmatch(r"[\W_]+", text.strip()))


def _is_clearly_incomplete(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if re.search(r"[\(\[\{,;:]\s*$", stripped):
        return True
    if (
        stripped.count("(") > stripped.count(")")
        or stripped.count("[") > stripped.count("]")
        or stripped.count("{") > stripped.count("}")
    ):
        return True
    if re.search(
        r"\b(?:and|or|but|of|to|for|with|by|in|on|at|from|as|because|while|including|such as)\s*$",
        stripped,
        flags=re.IGNORECASE,
    ):
        return True
    if not _SENTENCE_END_RE.search(stripped) and re.search(
        r"\b(?:where|which|that|who|whose|when|because)\s+\S{1,24}\s*$",
        stripped,
        flags=re.IGNORECASE,
    ):
        return True
    return False


def _complete_sentence(text: str) -> str:
    stripped = text.strip()
    if _SENTENCE_END_RE.search(stripped):
        return stripped
    return f"{stripped}."


def _trim_paragraph_to_sentence(text: str) -> str:
    matches = list(re.finditer(r'[.!?]["\')\]]*(?=\s|$)', text))
    if not matches:
        return text.strip()
    return text[: matches[-1].end()].strip()


def _normalize_for_dedupe(text: str) -> str:
    normalized = text.casefold()
    normalized = re.sub(r"\([^)]*(?:pdf|page|p\.)[^)]*\)", " ", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _is_near_duplicate(normalized: str, seen: list[str]) -> bool:
    if not normalized:
        return True

    key = normalized[:80]
    for previous in seen:
        previous_key = previous[:80]
        if normalized == previous or key == previous_key:
            return True
        if len(normalized) >= 40 and len(previous) >= 40:
            if normalized.startswith(previous_key) or previous.startswith(key):
                return True
    return False


def _split_candidate_text(text: str, was_bullet: bool) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if was_bullet:
        return [text]

    matches = []
    last_end = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        matches.append(match.group(0).strip())
        last_end = match.end()
    remainder = text[last_end:].strip()
    if remainder:
        matches.append(remainder)
    return matches or [text]


def clean_summary_text(text: str) -> str:
    if not text:
        return text

    candidates: list[str] = []
    seen: list[str] = []

    for raw_line in str(text).splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.startswith("```"):
            continue

        if _INTRO_RE.match(stripped):
            continue

        is_bullet, content = _strip_bullet_prefix(stripped)
        content = re.sub(r"\s+", " ", content).strip(" -*\u2022\u2013\u2014")
        if not content or _is_punctuation_only(content):
            continue

        for candidate in _split_candidate_text(content, is_bullet):
            candidate = candidate.strip(" -*\u2022\u2013\u2014")
            if not candidate or _is_punctuation_only(candidate):
                continue

            normalized = _normalize_for_dedupe(candidate)
            if _is_near_duplicate(normalized, seen):
                continue

            seen.append(normalized)
            candidates.append(candidate)

    while candidates and (
        _is_clearly_incomplete(candidates[-1])
        or not re.search(r'[.!?)]["\')\]]*\s*$', candidates[-1])
    ):
        candidates.pop()

    if not candidates:
        return ""

    normalized = [_complete_sentence(candidate) for candidate in candidates[:5]]
    return "\n".join(f"- {candidate}" for candidate in normalized)


def count_bullets(text: str) -> int:
    if not text:
        return 0
    return sum(
        1
        for line in text.splitlines()
        if _BULLET_PREFIX_RE.match(line.strip())
    )
