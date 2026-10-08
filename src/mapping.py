"""Strict scalar validation and model-independent category encoding.

Never replace or relabel a trained model's output feature columns using a
user-supplied mapping: user mappings only assign new numeric *codes* to fixed
training categories. Embedding vectors are illustrative and require retraining.
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import OCEAN_LABEL_TO_FEATURE, STRATEGY_FEATURES, STRATEGY_NAMES

MAX_MAPPING_BYTES = 32 * 1024


class MappingError(ValueError):
    """An invalid scalar, code mapping, or strategy was supplied."""


@dataclass(frozen=True)
class MappingConfig:
    code_to_label: tuple[tuple[int, str], ...]
    embeddings: tuple[tuple[str, tuple[float, float, float]], ...]

    @property
    def codes(self) -> dict[int, str]:
        return dict(self.code_to_label)

    @property
    def vectors(self) -> dict[str, tuple[float, float, float]]:
        return dict(self.embeddings)

    @property
    def minimum(self) -> int:
        return min(self.codes)

    @property
    def maximum(self) -> int:
        return max(self.codes)


def validate_code_mapping(mapping: Any) -> tuple[tuple[int, str], ...]:
    if not isinstance(mapping, dict):
        raise MappingError('Mapping must be a JSON object or a CSV with code,label columns.')
    decoded: dict[int, str] = {}
    for raw_code, label in mapping.items():
        if isinstance(raw_code, bool) or not str(raw_code).isdigit():
            raise MappingError(f'Category code must be a non-negative integer: {raw_code!r}')
        code = int(raw_code)
        if str(raw_code) != str(code):
            raise MappingError(f'Category code must have canonical digits, e.g. "1": {raw_code!r}')
        if code in decoded:
            raise MappingError(f'Duplicate numeric code {code}.')
        if not isinstance(label, str) or label not in OCEAN_LABEL_TO_FEATURE:
            raise MappingError(f'Unknown trained category {label!r}.')
        decoded[code] = label
    if set(decoded) != set(range(5)):
        raise MappingError('Mapping must contain exactly the five codes 0, 1, 2, 3, 4.')
    if set(decoded.values()) != set(OCEAN_LABEL_TO_FEATURE):
        raise MappingError('Mapping must cover every trained category exactly once (no duplicates).')
    return tuple(sorted(decoded.items()))


def _validate_embeddings(raw: Any) -> tuple[tuple[str, tuple[float, float, float]], ...]:
    if not isinstance(raw, dict) or set(raw) != set(OCEAN_LABEL_TO_FEATURE):
        raise MappingError('Embeddings must cover the exact five trained categories.')
    validated = []
    for label in OCEAN_LABEL_TO_FEATURE:
        vec = raw[label]
        if not isinstance(vec, list) or len(vec) != 3:
            raise MappingError('Each illustrative embedding must have 3 dimensions.')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vec):
            raise MappingError(f'Embedding for {label} contains invalid numeric values.')
        validated.append((label, tuple(float(v) for v in vec)))
    return tuple(validated)


def load_default_config(path: str | Path) -> MappingConfig:
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    return MappingConfig(
        code_to_label=validate_code_mapping(raw['code_to_label']),
        embeddings=_validate_embeddings(raw['embeddings']),
    )


def override_codes(config: MappingConfig, payload: bytes, filename: str) -> MappingConfig:
    """Only JSON/UTF-8 CSV; no pickle, Python or eval from users."""
    if not payload or len(payload) > MAX_MAPPING_BYTES:
        raise MappingError('Mapping must be non-empty and at most 32 KiB.')
    try:
        decoded = payload.decode('utf-8-sig')
        if filename.lower().endswith('.json'):
            raw = json.loads(decoded)
            if isinstance(raw, dict) and 'code_to_label' in raw:
                raw = raw['code_to_label']
        elif filename.lower().endswith('.csv'):
            rows = list(csv.DictReader(io.StringIO(decoded), strict=True))
            if len(rows) != 5 or any(set(row) != {'code', 'label'} for row in rows):
                raise MappingError('CSV must contain only code,label columns and exactly five rows.')
            raw = {}
            for row in rows:
                if row['code'] in raw:
                    raise MappingError(f'Duplicate CSV code {row["code"]}.')
                raw[row['code']] = row['label']
        else:
            raise MappingError('Only .json and .csv mapping files are supported.')
        return MappingConfig(validate_code_mapping(raw), config.embeddings)
    except (UnicodeError, json.JSONDecodeError, csv.Error, KeyError) as exc:
        raise MappingError('Could not parse the mapping file as UTF-8 JSON/CSV.') from exc


def validate_sense(value: Any, config: MappingConfig, strategy: str) -> tuple[float, str]:
    """Returns finite value + human readable label (or 'continuous' for fractions)."""
    if strategy not in STRATEGY_NAMES:
        raise MappingError(f'Unknown mapping strategy: {strategy!r}')
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise MappingError('Human-sense value must be a finite number, not a boolean or blank.')
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise MappingError('Human-sense value cannot be blank.')
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MappingError(f'Human-sense value must be numeric: {value!r}') from exc
    if not math.isfinite(x):
        raise MappingError('Human-sense value must be finite.')
    if not config.minimum <= x <= config.maximum:
        raise MappingError(f'Human-sense value must be in [{config.minimum}, {config.maximum}].')
    if x.is_integer():
        return x, config.codes[int(x)]
    if strategy != 'normalized_scalar':
        raise MappingError('Categorical one-hot/embedding strategies need a whole-number code (e.g. 3 or 3.0).')
    return x, 'continuous (no discrete category)'


@lru_cache(maxsize=256)
def _encode_one(code: float, label: str, strategy: str,
                minimum: int, maximum: int,
                vector: tuple[float, float, float] | None) -> tuple[float, ...]:
    """Cache only small deterministic mapping transforms; never full private records."""
    if strategy == 'one_hot':
        return tuple(float(label == lab) for lab in OCEAN_LABEL_TO_FEATURE)
    if strategy == 'normalized_scalar':
        return ((code - minimum) / (maximum - minimum),)
    if strategy == 'embedding' and vector is not None:
        return vector
    raise MappingError('Embedding not found for category.')


def encode_human_sense(values: pd.Series, config: MappingConfig,
                       strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds encoding DataFrame plus per-row explanatory audit table."""
    if strategy not in STRATEGY_NAMES:
        raise MappingError(f'Unsupported mapping strategy: {strategy}')
    encoded_rows, explanations = [], []
    for idx, raw in values.items():
        try:
            code, label = validate_sense(raw, config, strategy)
        except MappingError as exc:
            raise MappingError(f'Row {idx}: {exc}') from exc
        vec = config.vectors.get(label) if strategy == 'embedding' else None
        encoded = _encode_one(code, label, strategy, config.minimum, config.maximum, vec)
        encoded_rows.append(encoded)
        explanations.append({
            'row': idx,
            'human_sense': code,
            'category': label,
            'strategy': strategy,
            'encoded_as': ', '.join(f'{k}={v:g}' for k, v in zip(STRATEGY_FEATURES[strategy], encoded)),
        })
    result = pd.DataFrame(encoded_rows, columns=STRATEGY_FEATURES[strategy], index=values.index)
    explanation = pd.DataFrame(explanations, index=values.index)
    return result, explanation
