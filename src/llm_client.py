"""LLM client: Gemini with quota-based API key rotation and optional local fallback."""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

from src.config import GEMINI_MODEL_NAME, LOCAL_FALLBACK_MODEL_NAME

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_QUOTA_ERROR_MSG = (
    "Error: All Gemini API keys are quota-exhausted. "
    "Enable local fallback in the sidebar or add another key."
)


def _is_placeholder_key(value: str | None) -> bool:
    if value is None:
        return True
    v = value.strip().lower()
    if not v:
        return True
    if "your" in v or "placeholder" in v:
        return True
    placeholders = {
        "your_google_api_key",
        "your_gemini_api_key",
        "your_gemini_api_key_here",
        "your_google_ai_studio_api_key",
        "your_api_key_here",
    }
    return v in placeholders or v.startswith("your_")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _load_gemini_keys() -> list[str]:
    """Load GEMINI_API_KEY1..10 (and optional GEMINI_API_KEY_1..10), then legacy keys."""

    keys: list[str] = []
    for i in range(1, 11):
        k = os.getenv(f"GEMINI_API_KEY{i}") or os.getenv(f"GEMINI_API_KEY_{i}")
        if not _is_placeholder_key(k):
            keys.append(k.strip())

    legacy = os.getenv("GEMINI_API_KEY")
    if not _is_placeholder_key(legacy):
        keys.append(legacy.strip())

    google = os.getenv("GOOGLE_API_KEY")
    if not _is_placeholder_key(google):
        keys.append(google.strip())

    return _dedupe_keep_order(keys)


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "resource_exhausted" in msg
        or "quota" in msg
        or "too many requests" in msg
    )


def is_gemini_quota_response(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return (
        "all gemini api keys are quota-exhausted" in lower
        or "gemini quota is exhausted" in lower
    )


class LLMClient:
    def __init__(self, allow_local_fallback: bool = False):
        self.allow_local_fallback = allow_local_fallback
        self._provider: str = "none"
        self._gemini_client = None
        self._local_pipeline = None
        self._gemini_model = os.getenv("GEMINI_MODEL_NAME", GEMINI_MODEL_NAME)
        self._local_model_name = os.getenv(
            "LOCAL_FALLBACK_MODEL_NAME", LOCAL_FALLBACK_MODEL_NAME
        )

        self._api_keys = _load_gemini_keys()
        self._active_key_index = 0
        self.gemini_quota_exhausted = False
        self.local_fallback_used = False

        if self._api_keys:
            if self._init_gemini_client(self._api_keys[0]):
                self._provider = "gemini"
                self._active_key_index = 0
        elif not self.allow_local_fallback:
            self._provider = "no_gemini_key"

    def _init_gemini_client(self, api_key: str) -> bool:
        try:
            from google import genai

            self._gemini_client = genai.Client(api_key=api_key)
            return True
        except Exception as exc:
            logger.warning("Gemini initialization failed: %s", exc)
            self._gemini_client = None
            return False

    def _switch_to_key_index(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self._api_keys):
            return False
        self._active_key_index = idx
        return self._init_gemini_client(self._api_keys[idx])

    def _init_local_pipeline(self) -> None:
        if not self.allow_local_fallback:
            self._local_pipeline = None
            return

        if self._local_pipeline is not None:
            return

        try:
            from transformers import pipeline

            self._local_pipeline = pipeline(
                "text2text-generation",
                model=self._local_model_name,
            )
            self._provider = "local_flan_t5"
            logger.info("Using local fallback model: %s", self._local_model_name)
        except Exception as exc:
            logger.error("Local model initialization failed: %s", exc)
            self._local_pipeline = None

    def provider_name(self) -> str:
        if self._provider == "gemini":
            return "gemini"
        if self._provider == "local_flan_t5":
            return "local_flan_t5"
        if not self._api_keys:
            return "no_gemini_key"
        return self._provider

    def gemini_key_count(self) -> int:
        return len(self._api_keys)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        self.gemini_quota_exhausted = False
        self.local_fallback_used = False

        if self._api_keys:
            result = self._generate_gemini_with_rotation(prompt, max_tokens)
            if not is_gemini_quota_response(result):
                return result
            self.gemini_quota_exhausted = True
            if self.allow_local_fallback:
                return self._try_local_fallback(prompt, max_tokens)
            return (
                "Gemini quota is exhausted for all configured keys. "
                "Enable local fallback in the sidebar or add another API key."
            )

        if not self.allow_local_fallback:
            return (
                "Error: No Gemini API key found. Add GEMINI_API_KEY1, "
                "GEMINI_API_KEY2, ... to .env."
            )

        return self._try_local_fallback(prompt, max_tokens)

    def _try_local_fallback(self, prompt: str, max_tokens: int) -> str:
        if not self.allow_local_fallback:
            return GEMINI_QUOTA_ERROR_MSG

        if self._local_pipeline is None:
            self._init_local_pipeline()

        if self._local_pipeline is None:
            return (
                "Error: Local fallback is not available. "
                "Install transformers/torch or check the model name."
            )

        self.local_fallback_used = True
        self._provider = "local_flan_t5"
        return self._generate_local(prompt, max_tokens)

    def _generate_gemini_with_rotation(self, prompt: str, max_tokens: int) -> str:
        last_error: Optional[Exception] = None
        total_keys = len(self._api_keys)
        quota_failures = 0

        start = self._active_key_index
        indices = list(range(start, total_keys)) + list(range(0, start))

        for idx in indices:
            if idx != self._active_key_index or self._gemini_client is None:
                if not self._switch_to_key_index(idx):
                    continue

            try:
                response = self._gemini_client.models.generate_content(
                    model=self._gemini_model,
                    contents=prompt,
                    config={"max_output_tokens": max_tokens},
                )

                text = getattr(response, "text", None)
                if text:
                    self._provider = "gemini"
                    return text.strip()

                if hasattr(response, "candidates") and response.candidates:
                    parts = response.candidates[0].content.parts
                    self._provider = "gemini"
                    return "".join(
                        getattr(p, "text", str(p)) for p in parts
                    ).strip()

                self._provider = "gemini"
                return str(response)

            except Exception as exc:
                last_error = exc

                if _is_quota_error(exc):
                    quota_failures += 1
                    logger.warning("Gemini quota exceeded; trying next key.")
                    continue

                logger.warning("Gemini generation failed: %s", exc)
                break

        if quota_failures >= total_keys and total_keys > 0:
            logger.warning("All Gemini keys exhausted.")
            return GEMINI_QUOTA_ERROR_MSG

        if last_error and _is_quota_error(last_error):
            logger.warning("All Gemini keys exhausted.")
            return GEMINI_QUOTA_ERROR_MSG

        return (
            f"Error: Gemini generation failed ({last_error}). "
            "Check API keys and network."
        )

    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        if self._local_pipeline is None:
            return "Error: Local fallback is not available."

        truncated = prompt[:1200]
        result = self._local_pipeline(
            truncated,
            max_new_tokens=min(max_tokens, 256),
            do_sample=False,
        )
        return result[0]["generated_text"].strip()
