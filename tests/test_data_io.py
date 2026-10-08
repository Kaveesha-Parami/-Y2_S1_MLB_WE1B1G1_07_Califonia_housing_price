import pandas as pd
import pytest
from src.data_io import MAX_BATCH_ROWS, UploadError, read_batch_csv, to_safe_csv


def test_csv_reads_utf8():
    frame = read_batch_csv(b'human_sense,longitude\n3,-122.23\n', 'input.csv')
    assert frame.iloc[0]['human_sense'] == '3'


def test_csv_duplicate_columns_rejected():
    with pytest.raises(UploadError, match='duplicate'):
        read_batch_csv(b'a,a\n1,2\n', 'input.csv')


def test_csv_row_limit():
    content = ('code\n' + '1\n'*(MAX_BATCH_ROWS+1)).encode()
    with pytest.raises(UploadError, match='limited'):
        read_batch_csv(content, 'too_many.csv')


def test_non_utf8_rejected():
    with pytest.raises(UploadError):
        read_batch_csv(b'\xff\x80\xff', 'bad.csv')


def test_safe_csv_formula_injection():
    data = pd.DataFrame({'name': ['=1+1', 'near bay'], 'value': [1, 2]})
    csvbytes = to_safe_csv(data)
    assert b"'=1+1" in csvbytes


def test_extra_csv_values_rejected():
    with pytest.raises(UploadError, match='has 3 fields'):
        read_batch_csv(b'a,b\n1,2,3\n', 'malformed.csv')
