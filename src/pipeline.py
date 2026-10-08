"""Pandas input validation -> mapping -> exact model feature matrix -> inference."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .mapping import MappingConfig, MappingError, encode_human_sense
from .schema import (DIRECT_INPUT_COLUMNS, ENGINEERED_COLUMNS,
                     HIDDEN_MODEL_DEFAULTS, INPUT_NUMERIC_COLUMNS,
                     NUMERIC_COLUMNS, REQUIRED_INPUT)

LOGGER = logging.getLogger(__name__)


class InputValidationError(ValueError):
    """Tabular data or model-schema mismatch."""


def prepare_features(raw: pd.DataFrame, config: MappingConfig, strategy: str,
                     required_model_features: Sequence[str] | None = None,
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise InputValidationError('At least one data row is required.')
    if not raw.columns.is_unique:
        raise InputValidationError('Input has duplicate column names.')
    missing = sorted(set(REQUIRED_INPUT).difference(raw.columns))
    if missing:
        raise InputValidationError(f'Missing required column(s): {", ".join(missing)}')
    user_numeric: dict[str, pd.Series] = {}
    for feature in INPUT_NUMERIC_COLUMNS:
        values = raw[feature]
        try:
            parsed = pd.to_numeric(values, errors='raise')
            if parsed.isna().any() or not np.isfinite(parsed.to_numpy(dtype=float)).all():
                raise ValueError('null/NaN/infinity')
            user_numeric[feature] = parsed.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise InputValidationError(f'Column {feature!r} must contain only finite numeric values.') from exc

    parsed_input = pd.DataFrame(user_numeric, index=raw.index)
    if (parsed_input['households'] <= 0).any():
        raise InputValidationError("Column 'households' must contain values greater than zero.")
    if (parsed_input['total_rooms'] <= 0).any():
        raise InputValidationError("Column 'total_rooms' must contain values greater than zero.")
    for feature in ('total_bedrooms', 'population'):
        if (parsed_input[feature] < 0).any():
            raise InputValidationError(f'Column {feature!r} cannot contain negative values.')

    numeric: dict[str, pd.Series | float] = {
        feature: parsed_input[feature] for feature in DIRECT_INPUT_COLUMNS
    }
    numeric['spatial_cluster'] = HIDDEN_MODEL_DEFAULTS['spatial_cluster']
    numeric['rooms_per_household'] = (
        parsed_input['total_rooms'] / parsed_input['households']
    )
    numeric['bedrooms_per_room'] = (
        parsed_input['total_bedrooms'] / parsed_input['total_rooms']
    )
    numeric['population_per_household'] = (
        parsed_input['population'] / parsed_input['households']
    )
    numeric['min_dist_to_hub'] = HIDDEN_MODEL_DEFAULTS['min_dist_to_hub']
    try:
        encoded, explanation = encode_human_sense(raw['human_sense'], config, strategy)
    except MappingError as exc:
        raise InputValidationError(str(exc)) from exc
    frame = pd.DataFrame(numeric, index=raw.index).loc[:, NUMERIC_COLUMNS]
    frame = pd.concat([frame, encoded], axis=1)
    if required_model_features is not None:
        expected = list(required_model_features)
        if len(expected) != len(set(expected)):
            raise InputValidationError('Model feature contract contains duplicate column names.')
        if set(frame.columns) != set(expected):
            missing_feat = sorted(set(expected) - set(frame.columns))
            extra_feat = sorted(set(frame.columns) - set(expected))
            raise InputValidationError(
                f'Encoding/model schema mismatch. Model missing inputs={missing_feat}; '
                f'produced unknown features={extra_feat}. The chosen strategy needs a separately trained model.'
            )
        frame = frame.loc[:, expected]
    LOGGER.info('Prepared %d rows with %s mapping', len(frame), strategy)
    return frame, explanation


def predict_batch(raw: pd.DataFrame, model: Any, model_features: Sequence[str],
                  config: MappingConfig, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, explanation = prepare_features(raw, config, strategy, model_features)
    predictions = np.asarray(model.predict(features), dtype=float)
    if predictions.ndim != 1 or len(predictions) != len(raw) or not np.isfinite(predictions).all():
        raise InputValidationError('Model returned invalid or unexpected prediction values.')
    # Export only approved fields, never extra untrusted raw CSV columns.
    # Export numeric fields as real numbers, not untrusted CSV strings.
    # This also prevents safe-CSV escaping from altering negative coordinates.
    result = raw.loc[:, list(INPUT_NUMERIC_COLUMNS)].copy()
    for feature in INPUT_NUMERIC_COLUMNS:
        result[feature] = pd.to_numeric(result[feature], errors='raise').astype(float)
    for feature in ENGINEERED_COLUMNS:
        result[feature] = features[feature].to_numpy(dtype=float)
    result['human_sense'] = explanation['human_sense'].to_numpy(dtype=float)
    result['resolved_category'] = explanation['category']
    result['predicted_median_house_value'] = predictions
    LOGGER.info('Predicted %d rows', len(result))
    return result, explanation
