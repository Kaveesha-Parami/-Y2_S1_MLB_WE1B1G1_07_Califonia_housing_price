from pathlib import Path
import pytest
from src.mapping import load_default_config

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def cfg():
    return load_default_config(ROOT / 'config/default_mapping.json')
