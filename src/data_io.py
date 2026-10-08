"""Bounded UTF-8 CSV ingestion; avoid ambiguous schemas and silent truncation."""
from __future__ import annotations

import csv
import io

import pandas as pd

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BATCH_ROWS = 5_000
MAX_COLUMNS = 30


class UploadError(ValueError):
    """Unsupported/oversized/corrupted CSV upload."""


def read_batch_csv(content: bytes, filename: str) -> pd.DataFrame:
    if not filename.lower().endswith('.csv'):
        raise UploadError('Only .csv files are accepted for batch inference.')
    if not content or len(content) > MAX_FILE_BYTES:
        raise UploadError('CSV must be nonempty and at most 10 MiB.')
    try:
        decoded = content.decode('utf-8-sig')
        headings = next(csv.reader(io.StringIO(decoded), strict=True))
        if len(headings) > MAX_COLUMNS or not headings or any(not h.strip() for h in headings):
            raise UploadError('CSV must have between 1 and 30 named columns.')
        if len(set(headings)) != len(headings):
            raise UploadError('CSV contains duplicate header names.')
        # Explicit field-count checks prevent silent dropping of surplus CSV values.
        checked_rows = 0
        reader = csv.reader(io.StringIO(decoded), strict=True)
        next(reader)
        for row in reader:
            if len(row) != len(headings):
                raise UploadError(f'CSV row {checked_rows + 2} has {len(row)} fields; expected {len(headings)}.')
            checked_rows += 1
            if checked_rows > MAX_BATCH_ROWS:
                raise UploadError(f'CSV is limited to {MAX_BATCH_ROWS:,} rows per run.')
        # dtype=str delays coercion until our strict numeric checks in pipeline.
        data = pd.read_csv(io.StringIO(decoded), dtype=str, nrows=MAX_BATCH_ROWS+1,
                           on_bad_lines='error', index_col=False)
    except (UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError,
            StopIteration, ValueError) as exc:
        if isinstance(exc, UploadError):
            raise
        raise UploadError('Cannot parse CSV. Supply an uncompressed UTF-8 CSV with a header.') from exc
    if data.empty:
        raise UploadError('CSV must contain at least one data row.')
    if len(data) > MAX_BATCH_ROWS:
        raise UploadError(f'CSV is limited to {MAX_BATCH_ROWS:,} rows per run.')
    return data


def to_safe_csv(frame: pd.DataFrame) -> bytes:
    """Prevent CSV formula injection if a future export includes free-text fields."""
    safe = frame.copy()
    for col in safe.select_dtypes(include=['object', 'string']).columns:
        safe[col] = safe[col].map(
            lambda v: "'"+v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@', '\t', '\r'))
            else v
        )
    return safe.to_csv(index=False).encode('utf-8-sig')
