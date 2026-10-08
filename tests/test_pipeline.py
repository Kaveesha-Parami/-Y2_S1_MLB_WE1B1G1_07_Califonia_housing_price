from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.model_store import load_model_artifacts
from src.pipeline import InputValidationError, prepare_features, predict_batch
from src.schema import SINGLE_DEFAULTS

ROOT = Path(__file__).resolve().parents[1]


def row():
    return pd.DataFrame([{**SINGLE_DEFAULTS, 'human_sense': 3.0}])


def test_schema_and_feature_order(cfg):
    actual = load_model_artifacts(ROOT)
    frame, audit = prepare_features(row(), cfg, 'one_hot', actual.feature_names)
    assert list(frame.columns) == list(actual.feature_names)
    assert list(frame.shape) == [1, 14]
    assert audit.loc[0, 'category'] == 'NEAR BAY'


def test_ratio_features_are_derived_from_raw_housing_counts(cfg):
    frame, _ = prepare_features(row(), cfg, 'one_hot')
    assert frame.loc[0, 'rooms_per_household'] == pytest.approx(880 / 126)
    assert frame.loc[0, 'bedrooms_per_room'] == pytest.approx(129 / 880)
    assert frame.loc[0, 'population_per_household'] == pytest.approx(322 / 126)


@pytest.mark.parametrize('feature', ['households', 'total_rooms'])
def test_ratio_denominators_must_be_positive(feature, cfg):
    bad = row()
    bad.loc[0, feature] = 0
    with pytest.raises(InputValidationError, match=feature):
        prepare_features(bad, cfg, 'one_hot')


def test_real_uploaded_xgboost_prediction(cfg):
    actual = load_model_artifacts(ROOT)
    result, explanation = predict_batch(row(), actual.model, actual.feature_names, cfg, 'one_hot')
    assert np.isfinite(result['predicted_median_house_value'].iloc[0])
    assert result['predicted_median_house_value'].iloc[0] > 0
    assert explanation.loc[0, 'category'] == 'NEAR BAY'


def test_missing_inputs_are_rejected(cfg):
    with pytest.raises(InputValidationError, match='Missing required'):
        prepare_features(row().drop(columns='longitude'), cfg, 'one_hot')


def test_nonfinite_numeric_rejected(cfg):
    bad = row()
    bad.loc[0, 'median_income'] = np.inf
    with pytest.raises(InputValidationError, match='median_income'):
        prepare_features(bad, cfg, 'one_hot')


def test_strategy_mismatch_fails_closed(cfg):
    actual = load_model_artifacts(ROOT)
    with pytest.raises(InputValidationError, match='needs a separately trained model'):
        prepare_features(row(), cfg, 'normalized_scalar', actual.feature_names)


def test_extra_columns_never_reach_model(cfg):
    actual = load_model_artifacts(ROOT)
    raw = row()
    raw['median_house_value'] = 123456789
    features, _ = prepare_features(raw, cfg, 'one_hot', actual.feature_names)
    assert 'median_house_value' not in features.columns


def test_export_preserves_negative_numeric_values(cfg):
    from src.data_io import read_batch_csv, to_safe_csv
    actual = load_model_artifacts(ROOT)
    raw = pd.DataFrame([{**SINGLE_DEFAULTS, 'human_sense': 3}]).astype(str)
    result, _ = predict_batch(raw, actual.model, actual.feature_names, cfg, 'one_hot')
    exported = to_safe_csv(result)
    assert b"-122.23" in exported and b"'-122.23" not in exported


def test_prediction_output_uses_raw_and_derived_fields_not_hidden_model_fields(cfg):
    actual = load_model_artifacts(ROOT)
    result, _ = predict_batch(row(), actual.model, actual.feature_names, cfg, 'one_hot')
    assert 'households' in result.columns
    assert 'rooms_per_household' in result.columns
    assert 'spatial_cluster' not in result.columns
    assert 'min_dist_to_hub' not in result.columns
