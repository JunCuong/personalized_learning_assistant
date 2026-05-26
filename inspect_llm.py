"""Inspect LLM / Gemini configuration without exposing full API keys."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

from src.config import DIAGNOSTICS_DIR, GEMINI_MODEL_NAME, LOCAL_FALLBACK_MODEL_NAME, PROJECT_ROOT
from src.diagnostics_io import mask_secret, save_json_report
from src.llm_client import _get_api_key, _is_placeholder_key

JSON_PATH = DIAGNOSTICS_DIR / "llm_diagnostics.json"
ENV_PATH = PROJECT_ROOT / ".env"


def _classify_gemini_error(exc: Exception) -> tuple[str, str]:
    msg = str(exc).lower()
    if "429" in msg or "quota" in msg or "resource_exhausted" in msg:
        return (
            "QUOTA_EXCEEDED",
            "Check API key quota/billing or use a different Gemini model/key",
        )
    if "401" in msg or "403" in msg or "invalid" in msg or "api key" in msg:
        return ("AUTH_ERROR", "Verify GEMINI_API_KEY is valid in .env")
    return ("GENERATION_ERROR", str(exc)[:200])


def run_inspection() -> dict:
    load_dotenv(ENV_PATH if ENV_PATH.exists() else None)

    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    effective_key = _get_api_key()
    warnings: list[str] = []

    if google_key and _is_placeholder_key(google_key):
        warnings.append(
            "GOOGLE_API_KEY appears to be a placeholder. Remove it or replace it with a real key."
        )

    report: dict = {
        "warnings": warnings,
        "env_file_exists": ENV_PATH.exists(),
        "env_file_path": str(ENV_PATH),
        "GEMINI_API_KEY_set": bool(gemini_key),
        "GEMINI_API_KEY_masked": mask_secret(gemini_key),
        "GOOGLE_API_KEY_set": bool(google_key),
        "GOOGLE_API_KEY_masked": mask_secret(google_key),
        "effective_key_source": (
            "GEMINI_API_KEY"
            if gemini_key and not _is_placeholder_key(gemini_key)
            else (
                "GOOGLE_API_KEY"
                if google_key and not _is_placeholder_key(google_key)
                else None
            )
        ),
        "gemini_model_configured": os.getenv("GEMINI_MODEL_NAME", GEMINI_MODEL_NAME),
        "local_fallback_model": os.getenv(
            "LOCAL_FALLBACK_MODEL_NAME", LOCAL_FALLBACK_MODEL_NAME
        ),
    }

    if not effective_key:
        report["status"] = "NO_API_KEY"
        report["recommendation"] = "Add GEMINI_API_KEY to .env"
        report["provider_selected"] = None
        report["gemini_test"] = {"ran": False, "reason": "No API key"}
    else:
        report["status"] = "API_KEY_PRESENT"
        # Test Gemini tiny prompt
        gemini_test: dict = {"ran": True}
        try:
            from google import genai

            client = genai.Client(api_key=effective_key)
            model = report["gemini_model_configured"]
            response = client.models.generate_content(
                model=model,
                contents="Reply with exactly: OK",
                config={"max_output_tokens": 16},
            )
            text = getattr(response, "text", None) or str(response)
            gemini_test["status"] = "SUCCESS"
            gemini_test["response_preview"] = str(text)[:80]
            report["status"] = "GEMINI_OK"
            report["recommendation"] = "Gemini API responded successfully"
        except Exception as exc:
            status, recommendation = _classify_gemini_error(exc)
            gemini_test["status"] = status
            gemini_test["error_preview"] = str(exc)[:300]
            report["status"] = status
            report["recommendation"] = recommendation
        report["gemini_test"] = gemini_test

    # LLMClient provider (may init local pipeline)
    try:
        from src.llm_client import LLMClient

        client = LLMClient()
        report["provider_selected_by_llm_client"] = client.provider_name()
    except Exception as exc:
        report["provider_selected_by_llm_client"] = None
        report["llm_client_init_error"] = str(exc)[:200]

    # Local fallback load check only (no heavy generation)
    local_check: dict = {"model": report["local_fallback_model"]}
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(report["local_fallback_model"])
        local_check["tokenizer_load"] = "OK"
        local_check["model_load_skipped"] = (
            "Heavy weights not loaded in diagnostics; tokenizer OK"
        )
        del tok
    except Exception as exc:
        local_check["tokenizer_load"] = "FAILED"
        local_check["error"] = str(exc)[:200]
    report["local_fallback"] = local_check
    report["local_fallback_enabled"] = report.get(
        "provider_selected_by_llm_client"
    ) == "local_flan_t5" or not effective_key

    save_json_report(report, JSON_PATH)
    print(f"LLM status: {report['status']}")
    print(f"GEMINI_API_KEY: {report['GEMINI_API_KEY_masked']}")
    for w in report.get("warnings", []):
        print(f"WARNING: {w}")
    print(f"Provider (LLMClient): {report.get('provider_selected_by_llm_client')}")
    print(f"Saved: {JSON_PATH}")
    return report


def main() -> None:
    run_inspection()


if __name__ == "__main__":
    main()
