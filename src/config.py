"""Project configuration with Windows-compatible pathlib paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root: parent of src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"
DIAGNOSTICS_DIR = PROJECT_ROOT / "data" / "diagnostics"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store" / "faiss_index"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
DOCUMENTS_PATH = PROCESSED_DIR / "documents.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

GEMINI_MODEL_NAME = "gemini-2.0-flash"
LOCAL_FALLBACK_MODEL_NAME = "google/flan-t5-base"

MIN_PAGE_CHARS_WARNING = 50

# OCR settings
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() in ("1", "true", "yes")
OCR_DPI = int(os.getenv("OCR_DPI", "200"))
OCR_LANG = os.getenv("OCR_LANG", "eng")
OCR_MIN_TEXT_CHARS = int(os.getenv("OCR_MIN_TEXT_CHARS", "30"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "100"))
MERGE_SHORT_CHUNKS = os.getenv("MERGE_SHORT_CHUNKS", "true").lower() in (
    "1",
    "true",
    "yes",
)
# Set POPPLER_PATH in .env to Poppler bin folder on Windows if not on PATH
_poppler = os.getenv("POPPLER_PATH", "").strip()
POPPLER_PATH = _poppler if _poppler else None
