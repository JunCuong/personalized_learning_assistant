"""Inspect all PDFs in dataset/ and export diagnostic reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASET_DIR, DIAGNOSTICS_DIR
from src.dataset_inspection import build_dataset_summary, inspect_all_pdfs
from src.diagnostics_io import save_csv, save_json_report

CSV_PATH = DIAGNOSTICS_DIR / "dataset_inspection.csv"
JSON_PATH = DIAGNOSTICS_DIR / "dataset_inspection_summary.json"


def print_console_summary(reports: list[dict], summary: dict) -> None:
    if not reports:
        print("No PDF files found in dataset folder.")
        return

    print(f"{'Filename':<50} {'Pages':>6} {'Chars':>10} {'Status':<10}")
    print("-" * 80)
    for r in reports:
        print(
            f"{r['filename']:<50} {r['page_count']:>6} "
            f"{r['total_extracted_char_count']:>10} {r['status']:<10}"
        )
        if r["warning_reason"]:
            print(f"  -> {r['warning_reason']}")
    print("-" * 80)
    print(
        f"Total: {summary['total_pdfs']} | OK: {summary['ok_pdfs']} | "
        f"WARNING: {summary['warning_pdfs']} | ERROR: {summary['error_pdfs']}"
    )
    print(f"Conclusion: {summary['conclusion']}")


def run_inspection(dataset_dir: Path | None = None) -> tuple[list[dict], dict]:
    reports = inspect_all_pdfs(dataset_dir)
    summary = build_dataset_summary(reports)

    save_csv(reports, CSV_PATH)
    save_json_report(summary, JSON_PATH)
    print_console_summary(reports, summary)
    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {JSON_PATH}")
    return reports, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset PDFs")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()
    run_inspection(args.dataset_dir)


if __name__ == "__main__":
    main()
