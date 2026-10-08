import json
import pandas as pd
import pytest
from src.mapping import MappingError, encode_human_sense, override_codes, validate_code_mapping, validate_sense
from src.schema import OCEAN_COLUMNS


def test_onehot_maps_code_three_to_bay(cfg):
    frame, audit = encode_human_sense(pd.Series([3, '3.0', 0, 4]), cfg, 'one_hot')
    assert frame.columns.tolist() == list(OCEAN_COLUMNS)
    assert frame.iloc[0].sum() == 1
    assert frame.iloc[0]['ocean_proximity_NEAR BAY'] == 1
    assert frame.iloc[2]['ocean_proximity__1H OCEAN'] == 1
    assert audit.iloc[0]['category'] == 'NEAR BAY'


@pytest.mark.parametrize('value', ['-1', '5', 'nan', 'inf', '', 'hello', True, None, 2.5])
def test_categorical_values_rejected(value, cfg):
    with pytest.raises(MappingError):
        validate_sense(value, cfg, 'one_hot')


def test_normalized_float_and_limits(cfg):
    frame, audit = encode_human_sense(pd.Series([0, 1.5, 4]), cfg, 'normalized_scalar')
    assert frame['ocean_proximity_normalized'].tolist() == [0., 0.375, 1.]
    assert audit.iloc[1]['category'] == 'continuous (no discrete category)'


def test_embedding_vector_is_demo_only(cfg):
    matrix, _ = encode_human_sense(pd.Series([3]), cfg, 'embedding')
    assert matrix.iloc[0].tolist() == [0.7, 0.3, 0.6]


def test_permutation_mapping_preserves_feature_columns(cfg):
    new_mapping = {'0': 'INLAND', '1': '<1H OCEAN', '2': 'NEAR OCEAN', '3': 'NEAR BAY', '4': 'ISLAND'}
    changed = override_codes(cfg, json.dumps(new_mapping).encode(), 'mapping.json')
    encoded, _ = encode_human_sense(pd.Series([0]), changed, 'one_hot')
    assert encoded.iloc[0]['ocean_proximity_INLAND'] == 1.0
    assert encoded.columns.tolist() == list(OCEAN_COLUMNS)


def test_duplicate_category_rejected():
    with pytest.raises(MappingError):
        validate_code_mapping({'0': 'INLAND', '1': 'INLAND', '2': 'ISLAND',
                               '3': 'NEAR BAY', '4': 'NEAR OCEAN'})


def test_csv_duplicate_codes_rejected(cfg):
    payload = b'code,label\n0,INLAND\n0,<1H OCEAN\n2,ISLAND\n3,NEAR BAY\n4,NEAR OCEAN\n'
    with pytest.raises(MappingError):
        override_codes(cfg, payload, 'mapping.csv')


def test_bad_embedding_fraction(cfg):
    with pytest.raises(MappingError):
        encode_human_sense(pd.Series([1.25]), cfg, 'embedding')
