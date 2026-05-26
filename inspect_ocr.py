"""OCR dependency check and quick/full OCR diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pypdf import PdfReader

from src.config import DATASET_DIR, DIAGNOSTICS_DIR, OCR_MIN_TEXT_CHARS
from src.dataset_inspection import inspect_all_pdfs
from src.diagnostics_io import save_csv, save_json_report
from src.ocr_utils import check_ocr_dependencies, ocr_pdf_page

JSON_PATH = DIAGNOSTICS_DIR / "ocr_diagnostics.json"
PREVIEW_CSV = DIAGNOSTICS_DIR / "ocr_page_preview.csv"
PREVIEW_LEN = 300


def _pypdf_chars(pdf_path: Path) -> list[tuple[int, int]]:
    reader = PdfReader(str(pdf_path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        out.append((i, len(text)))
    return out


def _needs_ocr_pdf(pdf_path: Path) -> bool:
    return sum(c for _, c in _pypdf_chars(pdf_path)) < OCR_MIN_TEXT_CHARS * 2


def run_inspection(
    full: bool = False,
    source_filter: str | None = None,
    max_pages_quick: int = 2,
) -> dict:
    deps = check_ocr_dependencies()
    reports = inspect_all_pdfs()
    poor = [r for r in reports if r["total_extracted_char_count"] < 500 or r["pages_with_pypdf_text"] == 0]

    preview_rows: list[dict] = []
    ocr_test_results: list[dict] = []

    targets = sorted(DATASET_DIR.glob("*.pdf"))
    if source_filter:
        targets = [p for p in targets if p.name == source_filter]

    for pdf_path in targets:
        if not _needs_ocr_pdf(pdf_path) and not source_filter:
            continue

        page_chars = _pypdf_chars(pdf_path)
        pages_to_test = (
            [p for p, _ in page_chars]
            if full
            else [p for p, _ in page_chars[:max_pages_quick]]
        )

        pdf_ocr_chars = 0
        for page_num in pages_to_test:
            method = "pypdf"
            ocr_count = 0
            text_preview = ""
            pypdf_len = next((c for p, c in page_chars if p == page_num), 0)

            if deps.get("ready") and pypdf_len < OCR_MIN_TEXT_CHARS:
                try:
                    text = ocr_pdf_page(pdf_path, page_num)
                    ocr_count = len(text)
                    text_preview = text[:PREVIEW_LEN]
                    method = "ocr" if ocr_count >= OCR_MIN_TEXT_CHARS else "ocr_partial"
                    pdf_ocr_chars += ocr_count
                except Exception as exc:
                    method = "ocr_failed"
                    text_preview = str(exc)[:PREVIEW_LEN]
            else:
                reader = PdfReader(str(pdf_path))
                text_preview = (reader.pages[page_num - 1].extract_text() or "")[:PREVIEW_LEN]

            preview_rows.append(
                {
                    "source": pdf_path.name,
                    "page": page_num,
                    "extraction_method": method,
                    "ocr_char_count": ocr_count,
                    "text_preview": text_preview,
                }
            )

        ocr_test_results.append(
            {
                "source": pdf_path.name,
                "pages_tested": len(pages_to_test),
                "full_scan": full,
                "total_ocr_chars_sample": pdf_ocr_chars,
                "ocr_usable": pdf_ocr_chars >= OCR_MIN_TEXT_CHARS,
            }
        )

    summary = {
        "ocr_dependencies": deps,
        "mode": "full" if full else "quick",
        "poor_pypdf_pdfs": [r["filename"] for r in poor],
        "ocr_test_results": ocr_test_results,
        "preview_rows": len(preview_rows),
    }

    save_json_report(summary, JSON_PATH)
    save_csv(
        preview_rows,
        PREVIEW_CSV,
        fieldnames=["source", "page", "extraction_method", "ocr_char_count", "text_preview"],
    )

    print(f"OCR ready: {deps.get('ready')} — {deps.get('message')}")
    for t in ocr_test_results:
        print(f"  {t['source']}: ocr_usable={t['ocr_usable']}, chars={t['total_ocr_chars_sample']}")
    print(f"Saved: {JSON_PATH}")
    print(f"Saved: {PREVIEW_CSV}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR diagnostics")
    parser.add_argument("--full", action="store_true", help="OCR all pages (slow)")
    parser.add_argument("--source", type=str, default=None, help="Single PDF filename")
    args = parser.parse_args()
    run_inspection(full=args.full, source_filter=args.source)


if __name__ == "__main__":
    main()
