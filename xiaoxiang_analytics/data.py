"""Workbook loading and conservative numeric normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_project_sheet(path: str | Path, sheet_name: str = "Sheet2") -> pd.DataFrame:
    """Load one worksheet without embedding a machine-specific file path."""

    workbook = Path(path).expanduser()
    if not workbook.is_file():
        raise FileNotFoundError(f"Workbook not found: {workbook}")
    return pd.read_excel(workbook, sheet_name=sheet_name)


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a useful error when a workbook does not match the expected schema."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def parse_number(value: object, *, percentage: bool = False) -> float:
    """Parse numbers containing commas, spaces, or an explicit percent sign."""

    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace(" ", "")
        has_percent_sign = cleaned.endswith("%")
        if has_percent_sign:
            cleaned = cleaned[:-1]
        number = float(cleaned)
        return number / 100.0 if has_percent_sign else number
    return float(value)


def normalize_known_numeric_columns(
    frame: pd.DataFrame,
    numeric_columns: Iterable[str],
    percentage_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Return a copy with known numeric and percentage fields parsed."""

    result = frame.copy()
    percentage_set = set(percentage_columns)
    for column in numeric_columns:
        if column in result.columns:
            result[column] = result[column].map(
                lambda value: parse_number(value, percentage=column in percentage_set)
            )
    return result


def coerce_numeric_like_columns(frame: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Convert mostly numeric object columns while preserving real categories."""

    result = frame.copy()
    for column in result.select_dtypes(exclude=[np.number]).columns:
        series = result[column]
        cleaned = series.astype("string").str.replace(",", "", regex=False).str.strip()
        converted = pd.to_numeric(cleaned, errors="coerce")
        non_missing = int(series.notna().sum())
        if non_missing and int(converted.notna().sum()) / non_missing >= threshold:
            result[column] = converted
    return result
