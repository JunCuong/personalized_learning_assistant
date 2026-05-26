"""Dataset inspection with pypdf + OCR extraction stats."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.config import DATASET_DIR, ENABLE_OCR, TESSERACT_CMD
from src.ocr_utils import check_ocr_dependencies
from src.pdf_loader import load_pdfs

OK_CHAR_THRESHOLD = 500


def _status_from_counts(total_chars: int, pages_with_text: int, page_count: int) -> tuple[str, str]:
    if page_count == 0:
        return "ERROR", "PDF has no pages"
    if total_chars == 0 or pages_with_text == 0:
        return "WARNING", "No extractable text after pypdf and OCR"
    if total_chars <= OK_CHAR_THRESHOLD:
        avg = total_chars / max(page_count, 1)
        if avg < 50:
            return (
                "WARNING",
                f"Low text density: {total_chars} chars / {page_count} pages",
            )
        return "WARNING", f"Total chars ({total_chars}) below OK threshold ({OK_CHAR_THRESHOLD})"
    return "OK", ""


def inspect_all_pdfs(dataset_dir: Path | None = None) -> list[dict]:
    """Run full extraction (pypdf + OCR fallback) and aggregate per PDF."""
    dataset_dir = dataset_dir or DATASET_DIR
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    documents = load_pdfs(dataset_dir)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        by_source[doc["source"]].append(doc)

    reports: list[dict] = []
    for source in sorted(by_source.keys()):
        pages = by_source[source]
        pypdf_chars = sum(d.get("pypdf_char_count", 0) for d in pages)
        ocr_chars = sum(d.get("ocr_char_count", 0) for d in pages)
        total_chars = sum(d.get("char_count", 0) for d in pages)
        pages_pypdf = sum(1 for d in pages if d.get("extraction_method") == "pypdf")
        pages_ocr = sum(1 for d in pages if d.get("extraction_method") == "ocr")
        pages_with_text = sum(1 for d in pages if d.get("text"))
        pages_without = len(pages) - pages_with_text
        ocr_used = pages_ocr > 0

        status, warning = _status_from_counts(total_chars, pages_with_text, len(pages))

        reports.append(
            {
                "filename": source,
                "path": pages[0]["path"] if pages else "",
                "page_count": len(pages),
                "extracted_char_count_pypdf": pypdf_chars,
                "extracted_char_count_ocr": ocr_chars,
                "total_extracted_char_count": total_chars,
                "pages_with_pypdf_text": pages_pypdf,
                "pages_with_ocr_text": pages_ocr,
                "pages_without_text": pages_without,
                "ocr_used": ocr_used,
                "status": status,
                "warning_reason": warning,
                "pages": len(pages),
                "char_count": total_chars,
                "pages_with_text": pages_with_text,
                "message": warning,
            }
        )

    if not reports and not list(dataset_dir.glob("*.pdf")):
        return []
    return reports


def build_dataset_summary(reports: list[dict]) -> dict:
    ocr_deps = check_ocr_dependencies()
    ok = [r for r in reports if r["status"] == "OK"]
    warn = [r for r in reports if r["status"] == "WARNING"]
    err = [r for r in reports if r["status"] == "ERROR"]

    if not reports:
        conclusion = "No PDF files found in dataset/."
    elif err:
        conclusion = f"{len(err)} PDF(s) failed; fix before indexing."
    elif warn and ok:
        conclusion = (
            f"{len(ok)} PDF(s) OK, {len(warn)} with warnings. "
            "Review dataset_inspection.csv."
        )
    elif warn:
        conclusion = "All PDFs have warnings; check OCR/Poppler if scanned."
    else:
        conclusion = "All PDFs passed extraction checks (pypdf and/or OCR)."

    if ENABLE_OCR and not ocr_deps.get("ready"):
        conclusion += f" OCR not fully ready: {ocr_deps.get('message', '')}"

    return {
        "total_pdfs": len(reports),
        "ok_pdfs": len(ok),
        "warning_pdfs": len(warn),
        "error_pdfs": len(err),
        "total_pages": sum(r["page_count"] for r in reports),
        "pages_with_pypdf_text": sum(r["pages_with_pypdf_text"] for r in reports),
        "pages_with_ocr_text": sum(r["pages_with_ocr_text"] for r in reports),
        "pages_without_text": sum(r["pages_without_text"] for r in reports),
        "total_pypdf_chars": sum(r["extracted_char_count_pypdf"] for r in reports),
        "total_ocr_chars": sum(r["extracted_char_count_ocr"] for r in reports),
        "total_extracted_chars": sum(r["total_extracted_char_count"] for r in reports),
        "ocr_enabled": ENABLE_OCR,
        "tesseract_path": TESSERACT_CMD,
        "ocr_dependency_status": ocr_deps,
        "conclusion": conclusion.strip(),
    }
