"""Shared utilities for JSON I/O and dataset inspection."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import DATASET_DIR


def slugify_filename(filename: str) -> str:
    """Convert PDF filename to a stable doc_id (no extension)."""
    stem = Path(filename).stem
    slug = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug.lower() or "document"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def inspect_dataset(dataset_dir: Path | None = None) -> list[dict]:
    """Inspect all PDFs; returns detailed records with backward-compatible fields."""
    from src.dataset_inspection import inspect_all_pdfs

    return inspect_all_pdfs(dataset_dir)


def print_dataset_report(reports: list[dict]) -> None:
    """Print a formatted dataset inspection table."""
    if not reports:
        print("No PDF files found in dataset folder.")
        return

    print(f"{'Filename':<50} {'Pages':>6} {'Chars':>10} {'Status':<10}")
    print("-" * 80)
    for r in reports:
        print(
            f"{r['filename']:<50} {r['pages']:>6} {r['char_count']:>10} "
            f"{r['status']:<10}"
        )
    ok = sum(1 for r in reports if r["status"] == "OK")
    warn = sum(1 for r in reports if r["status"] == "WARNING")
    err = sum(1 for r in reports if r["status"] == "ERROR")
    print("-" * 80)
    print(f"Total: {len(reports)} | OK: {ok} | WARNING: {warn} | ERROR: {err}")


if __name__ == "__main__":
    report = inspect_dataset()
    print_dataset_report(report)
