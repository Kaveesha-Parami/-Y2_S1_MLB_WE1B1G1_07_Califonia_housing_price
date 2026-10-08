"""Local trusted model loading and strict feature-contract validation."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .schema import STRATEGY_NAMES

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArtifacts:
    model: Any
    feature_names: tuple[str, ...]
    input_strategy: str
    target_name: str


def _local_path(root: Path, rel: str) -> Path:
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Model artifact path must be inside the application directory.')
    return path


def load_model_artifacts(root: Path) -> ModelArtifacts:
    """ONLY load joblib from source-controlled trusted local files, NEVER from uploads."""
    root = root.resolve()
    manifest = json.loads((root / 'config/model_manifest.json').read_text(encoding='utf-8'))
    strategy = manifest['input_strategy']
    if strategy not in STRATEGY_NAMES:
        raise ValueError('Manifest contains unknown input strategy.')
    model_path = _local_path(root, manifest['model_path'])
    feature_path = _local_path(root, manifest['feature_names_path'])
    LOGGER.info('Loading trusted local XGBoost model: %s', model_path.name)
    model = joblib.load(model_path)
    names = joblib.load(feature_path)
    if not isinstance(names, (list, tuple)) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError('Feature-name artifact must be a non-empty list of strings.')
    names = tuple(names)
    if len(names) != len(set(names)):
        raise ValueError('Feature-name artifact has duplicates.')
    if not hasattr(model, 'predict'):
        raise ValueError('Model has no predict method.')
    if hasattr(model, 'feature_names_in_') and tuple(model.feature_names_in_) != names:
        raise ValueError('Saved model feature names do not match the names artifact.')
    if hasattr(model, 'n_features_in_') and model.n_features_in_ != len(names):
        raise ValueError('Saved model input width does not match the names artifact.')
    return ModelArtifacts(model, names, strategy, manifest['target_name'])
