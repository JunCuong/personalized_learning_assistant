"""Load PDFs from dataset/ with pypdf first, OCR fallback for sparse pages."""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from src.config import (
    DATASET_DIR,
    DOCUMENTS_PATH,
    ENABLE_OCR,
    OCR_MIN_TEXT_CHARS,
)
from src.ocr_utils import check_ocr_dependencies, ocr_pdf_page
from src.utils import save_json, slugify_filename

logger = logging.getLogger(__name__)


def _needs_ocr(pypdf_text: str) -> bool:
    return len(pypdf_text.strip()) < OCR_MIN_TEXT_CHARS


def _extract_page(
    pdf_path: Path,
    page_num: int,
    pypdf_text: str,
    ocr_ready: bool,
) -> dict:
    """Extract one page; OCR only when pypdf text is too short and OCR enabled."""
    text = pypdf_text.strip()
    method = "pypdf"
    ocr_char_count = 0

    if _needs_ocr(text) and ENABLE_OCR and ocr_ready:
        try:
            ocr_text = ocr_pdf_page(pdf_path, page_num)
            ocr_char_count = len(ocr_text)
            if len(ocr_text) >= OCR_MIN_TEXT_CHARS:
                text = ocr_text
                method = "ocr"
            elif ocr_text:
                text = ocr_text
                method = "ocr"
            else:
                method = "empty"
        except Exception as exc:
            logger.warning(
                "OCR failed for %s page %s: %s", pdf_path.name, page_num, exc
            )
            method = "empty" if not text else "pypdf"
    elif not text:
        method = "empty"

    return {
        "text": text,
        "extraction_method": method,
        "char_count": len(text),
        "ocr_char_count": ocr_char_count if method == "ocr" else 0,
        "pypdf_char_count": len(pypdf_text.strip()),
    }


def load_pdfs(dataset_dir: Path | None = None) -> list[dict]:
    """
    Extract text page by page using pypdf, with OCR fallback for sparse pages.
    """
    dataset_dir = dataset_dir or DATASET_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    pdf_files = sorted(dataset_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", dataset_dir)
        return []

    ocr_deps = check_ocr_dependencies()
    ocr_ready = ocr_deps.get("ready", False)
    if ENABLE_OCR and not ocr_ready:
        logger.warning(
            "OCR enabled but dependencies not ready: %s", ocr_deps.get("message")
        )

    documents: list[dict] = []
    skipped: list[str] = []

    for pdf_path in pdf_files:
        doc_id = slugify_filename(pdf_path.name)
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:
            logger.warning("Skipping corrupt PDF %s: %s", pdf_path.name, exc)
            skipped.append(pdf_path.name)
            continue

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                pypdf_text = page.extract_text() or ""
            except Exception as exc:
                logger.warning(
                    "pypdf failed on %s page %s: %s", pdf_path.name, page_num, exc
                )
                pypdf_text = ""

            extracted = _extract_page(pdf_path, page_num, pypdf_text, ocr_ready)

            documents.append(
                {
                    "doc_id": doc_id,
                    "source": pdf_path.name,
                    "path": str(pdf_path.resolve()),
                    "page": page_num,
                    "text": extracted["text"],
                    "extraction_method": extracted["extraction_method"],
                    "char_count": extracted["char_count"],
                    "pypdf_char_count": extracted["pypdf_char_count"],
                    "ocr_char_count": extracted["ocr_char_count"],
                }
            )

    pages_ocr = sum(1 for d in documents if d["extraction_method"] == "ocr")
    pages_pypdf = sum(1 for d in documents if d["extraction_method"] == "pypdf")
    summary = {
        "pdf_count": len(pdf_files),
        "page_count": len(documents),
        "pages_with_text": sum(1 for d in documents if d["text"]),
        "pages_ocr": pages_ocr,
        "pages_pypdf": pages_pypdf,
        "ocr_ready": ocr_ready,
        "skipped_pdfs": skipped,
    }
    logger.info("PDF load summary: %s", summary)
    return documents


def save_documents(
    documents: list[dict], output_path: Path | None = None
) -> Path:
    output_path = output_path or DOCUMENTS_PATH
    save_json(output_path, documents)
    return output_path


def run_pdf_loading(
    dataset_dir: Path | None = None, save: bool = True
) -> tuple[list[dict], dict]:
    documents = load_pdfs(dataset_dir)
    summary = {
        "total_pages": len(documents),
        "pages_with_text": sum(1 for d in documents if d["text"]),
        "pages_ocr": sum(1 for d in documents if d.get("extraction_method") == "ocr"),
        "pages_pypdf": sum(
            1 for d in documents if d.get("extraction_method") == "pypdf"
        ),
        "unique_sources": len({d["source"] for d in documents}),
    }
    if save:
        path = save_documents(documents)
        summary["saved_to"] = str(path)
    return documents, summary
