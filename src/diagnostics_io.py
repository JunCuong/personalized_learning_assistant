"""Shared helpers for diagnostic report export."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DIAGNOSTICS_DIR


def ensure_diagnostics_dir() -> Path:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    return DIAGNOSTICS_DIR


def save_csv(rows: list[dict], path: Path, fieldnames: list[str] | None = None) -> Path:
    ensure_diagnostics_dir()
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames or [])
        return path
    df = pd.DataFrame(rows)
    if fieldnames:
        df = df[[c for c in fieldnames if c in df.columns]]
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def save_json_report(data: Any, path: Path) -> Path:
    ensure_diagnostics_dir()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def mask_secret(value: str | None, visible: int = 4) -> str:
    """Mask API keys: first 4 + last 4 chars only."""
    if not value or not str(value).strip():
        return "(not set)"
    value = str(value).strip()
    if len(value) <= visible * 2:
        return "(set; too short to display)"
    return f"{value[:visible]}...{value[-visible:]}"
