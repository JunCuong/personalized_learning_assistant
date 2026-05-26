"""OCR utilities for scanned PDF pages (Tesseract + pdf2image)."""

from __future__ import annotations

import logging
from pathlib import Path

from src.config import OCR_DPI, OCR_LANG, POPPLER_PATH, TESSERACT_CMD

logger = logging.getLogger(__name__)

POPPLER_ERROR_MSG = (
    "pdf2image requires Poppler on Windows. Please install Poppler and add its "
    "bin folder to PATH, or set POPPLER_PATH in .env / config to the Poppler bin directory. "
    "Download: https://github.com/oschwartz10612/poppler-windows/releases"
)


def configure_tesseract() -> dict:
    """Configure pytesseract and verify Tesseract executable exists."""
    import pytesseract

    cmd_path = Path(TESSERACT_CMD)
    pytesseract.pytesseract.tesseract_cmd = str(cmd_path)
    if not cmd_path.exists():
        return {
            "status": "error",
            "tesseract_cmd": str(cmd_path),
            "message": f"Tesseract not found at {cmd_path}",
        }
    return {
        "status": "ok",
        "tesseract_cmd": str(cmd_path),
        "message": "Tesseract configured",
    }


def _poppler_available() -> tuple[bool, str | None]:
    """Return (available, poppler_path_used)."""
    import shutil

    if shutil.which("pdftoppm"):
        return True, None
    if POPPLER_PATH:
        poppler_bin = Path(POPPLER_PATH)
        if (poppler_bin / "pdftoppm.exe").exists() or (poppler_bin / "pdftoppm").exists():
            return True, str(poppler_bin)
    return False, None


def check_ocr_dependencies() -> dict:
    """Return diagnostic dict for OCR stack."""
    result: dict = {
        "tesseract": {"configured": False},
        "pytesseract": {"import_ok": False},
        "pdf2image": {"import_ok": False},
        "poppler": {"available": False, "path": POPPLER_PATH},
        "ready": False,
    }

    try:
        import pytesseract  # noqa: F401

        result["pytesseract"]["import_ok"] = True
    except ImportError as exc:
        result["pytesseract"]["error"] = str(exc)
        result["message"] = "Install pytesseract: pip install pytesseract"
        return result

    try:
        import pdf2image  # noqa: F401

        result["pdf2image"]["import_ok"] = True
    except ImportError as exc:
        result["pdf2image"]["error"] = str(exc)
        result["message"] = "Install pdf2image: pip install pdf2image"
        return result

    tess = configure_tesseract()
    result["tesseract"] = tess

    poppler_ok, poppler_used = _poppler_available()
    result["poppler"]["available"] = poppler_ok
    result["poppler"]["path_used"] = poppler_used or POPPLER_PATH

    result["ready"] = (
        tess.get("status") == "ok"
        and result["pytesseract"]["import_ok"]
        and result["pdf2image"]["import_ok"]
        and poppler_ok
    )
    if not poppler_ok:
        result["message"] = POPPLER_ERROR_MSG
    elif tess.get("status") != "ok":
        result["message"] = tess.get("message")
    else:
        result["message"] = "OCR dependencies ready"
    return result


def _convert_pdf_page(pdf_path: Path, page_number: int, dpi: int = OCR_DPI):
    """Convert one PDF page to PIL image. page_number is 1-based."""
    from pdf2image import convert_from_path

    poppler_ok, poppler_path = _poppler_available()
    if not poppler_ok:
        raise RuntimeError(POPPLER_ERROR_MSG)

    kwargs: dict = {
        "pdf_path": str(pdf_path),
        "dpi": dpi,
        "first_page": page_number,
        "last_page": page_number,
    }
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    images = convert_from_path(**kwargs)
    if not images:
        raise RuntimeError(f"No image rendered for page {page_number} of {pdf_path.name}")
    return images[0]


def ocr_pdf_page(
    pdf_path: Path | str, page_number: int, dpi: int = OCR_DPI
) -> str:
    """
    OCR a single PDF page. page_number is 1-based (first page = 1).
    """
    import pytesseract

    configure_tesseract()
    pdf_path = Path(pdf_path)
    image = _convert_pdf_page(pdf_path, page_number, dpi=dpi)
    text = pytesseract.image_to_string(image, lang=OCR_LANG)
    return (text or "").strip()


def ocr_pdf_document(pdf_path: Path | str, dpi: int = OCR_DPI) -> list[dict]:
    """
    OCR all pages of a PDF. Returns page-level records with ocr_used=True.
    page is 1-based.
    """
    from pypdf import PdfReader

    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    records: list[dict] = []

    for page_num in range(1, len(reader.pages) + 1):
        try:
            text = ocr_pdf_page(pdf_path, page_num, dpi=dpi)
        except Exception as exc:
            logger.warning("OCR failed %s page %s: %s", pdf_path.name, page_num, exc)
            text = ""
        records.append(
            {
                "page": page_num,
                "text": text,
                "ocr_used": True,
                "ocr_char_count": len(text),
            }
        )
    return records
