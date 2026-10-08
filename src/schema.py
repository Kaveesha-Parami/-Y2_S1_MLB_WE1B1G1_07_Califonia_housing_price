"""Stable feature schema from the supplied training notebook / model artifacts."""
from __future__ import annotations

from typing import Final

OCEAN_LABEL_TO_FEATURE: Final[dict[str, str]] = {
    "<1H OCEAN": "ocean_proximity__1H OCEAN",  # notebook replaced '<' with '_'
    "INLAND": "ocean_proximity_INLAND",
    "ISLAND": "ocean_proximity_ISLAND",
    "NEAR BAY": "ocean_proximity_NEAR BAY",
    "NEAR OCEAN": "ocean_proximity_NEAR OCEAN",
}
OCEAN_COLUMNS: Final[tuple[str, ...]] = tuple(OCEAN_LABEL_TO_FEATURE.values())
NUMERIC_COLUMNS: Final[tuple[str, ...]] = (
    "longitude",
    "latitude",
    "housing_median_age",
    "median_income",
    "spatial_cluster",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "min_dist_to_hub",
)

# Values entered by users. The three ratio features in NUMERIC_COLUMNS are
# deliberately not user inputs; prepare_features derives them from these raw
# housing counts before assembling the model matrix.
DIRECT_INPUT_COLUMNS: Final[tuple[str, ...]] = (
    "longitude",
    "latitude",
    "housing_median_age",
    "median_income",
)
RAW_HOUSING_COLUMNS: Final[tuple[str, ...]] = (
    "households",
    "total_rooms",
    "total_bedrooms",
    "population",
)
INPUT_NUMERIC_COLUMNS: Final[tuple[str, ...]] = DIRECT_INPUT_COLUMNS + RAW_HOUSING_COLUMNS
ENGINEERED_COLUMNS: Final[tuple[str, ...]] = (
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
)
REQUIRED_INPUT: Final[tuple[str, ...]] = INPUT_NUMERIC_COLUMNS + ("human_sense",)

# These trained features cannot be reconstructed from the available artifacts.
# Keep their sample-row values internal so they are not exposed as user inputs.
HIDDEN_MODEL_DEFAULTS: Final[dict[str, float]] = {
    "spatial_cluster": 0.0,
    "min_dist_to_hub": -0.818014,
}
STRATEGY_FEATURES: Final[dict[str, tuple[str, ...]]] = {
    "one_hot": OCEAN_COLUMNS,
    "normalized_scalar": ("ocean_proximity_normalized",),
    "embedding": ("ocean_proximity_emb_0", "ocean_proximity_emb_1", "ocean_proximity_emb_2"),
}
STRATEGY_NAMES: Final[tuple[str, ...]] = tuple(STRATEGY_FEATURES)

# The four direct model inputs retain the sample-row values shown in the training
# notebook. Raw housing counts are from the corresponding California housing row.
SINGLE_DEFAULTS: Final[dict[str, float]] = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 0.784314,
    "median_income": 2.635927,
    "households": 126.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
}
