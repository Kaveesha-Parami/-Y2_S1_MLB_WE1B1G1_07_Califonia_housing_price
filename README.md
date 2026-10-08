# Human-Sense Housing Predictor — Streamlit + XGBoost

A working, modular Streamlit application for a supplied XGBoost **regression** model predicting `median_house_value` from **14 model-ready features**, including five one-hot ocean-proximity columns. Instead of asking the user for one-hot vectors, the app accepts a single intuitive **human-sense number** (`human_sense`).

> **Important modeling boundary:** The app now derives `rooms_per_household`, `bedrooms_per_room`, and `population_per_household` from four raw counts. However, the supplied `model_training_XGB.ipynb` loads an **already preprocessed** `df_clean5.csv`, and its fitted upstream scaler/transformer was not supplied. The four direct numerical fields must still use the training representation, while two unavailable internal model features use bundled sample defaults. Predictions should therefore be treated as a UI/integration demonstration until the full fitted preprocessing pipeline is recovered or retrained.

## Quickstart

Requirements: Python 3.10–3.13 and a local shell. Windows PowerShell commands:

```powershell
cd human_sense_housing_app
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
cd human_sense_housing_app
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL reported by Streamlit (normally `http://localhost:8501`). **Load `.joblib` only from trusted sources**: deserialization may execute code. This project packages only the two files supplied for this assignment; the app deliberately does not accept arbitrary uploaded models.

The page starts in dark mode. Use the **Dark mode** switch beside the title to move between dark and light palettes; the choice persists for the current Streamlit session.

## Directory map

```text
human_sense_housing_app/
├── app.py                            # Widgets, preview, single/batch inference, background state
├── requirements.txt
├── README.md
├── .streamlit/config.toml            # 10 MiB upload cap, XSRF/CORS and UI error settings
├── config/
│   ├── default_mapping.json          # Category code/label choices + illustrative demo embeddings
│   └── model_manifest.json           # Trusted local model path, feature list, encoding strategy
├── models/
│   ├── xgboost_top_features_model.joblib    # Supplied trained XGBRegressor
│   └── xgboost_top_feature_names.joblib     # Supplied 14 feature names
├── notebooks/model_training_XGB.ipynb  # Original supplied training reference (pre-cleaned CSV absent)
├── src/
│   ├── schema.py                     # Canonical trained 14-feature contract
│   ├── mapping.py                    # Strict ocean-code validation and trained one-hot encoding
│   ├── data_io.py                    # Bounded UTF-8 CSV parser, safe CSV export
│   ├── pipeline.py                   # Validation -> feature preparation -> model.predict
│   ├── model_store.py                # Local joblib load + feature-contract verification
│   └── workers.py                    # Bounded background thread pool
├── examples/
│   ├── housing_batch.csv             # Five example rows with raw housing counts
│   └── custom_mapping.json           # Permutation of numeric codes
└── tests/
    ├── conftest.py
    ├── test_mapping.py
    ├── test_pipeline.py
    ├── test_data_io.py
    └── test_workers.py
```

## What counts as a human-sense number?

The original one-hot model has five trained categorical values, **not** a single numerical ocean-proximity measurement. The default application code-to-category mapping (arbitrary identifiers, **not meaningful distances**) is:

| Code typed or selected | Trained category | Exact feature set to 1 |
|---:|---|---|
| `0` | `<1H OCEAN` | `ocean_proximity__1H OCEAN` |
| `1` | `INLAND` | `ocean_proximity_INLAND` |
| `2` | `ISLAND` | `ocean_proximity_ISLAND` |
| `3` | `NEAR BAY` | `ocean_proximity_NEAR BAY` |
| `4` | `NEAR OCEAN` | `ocean_proximity_NEAR OCEAN` |

All other one-hot columns are assigned `0`. For example, `human_sense = 3` yields:

```text
ocean_proximity__1H OCEAN  = 0
ocean_proximity_INLAND     = 0
ocean_proximity_ISLAND     = 0
ocean_proximity_NEAR BAY   = 1
ocean_proximity_NEAR OCEAN = 0
```

The odd `__1H` spelling is intentional: the training notebook replaced `<` with `_` in column names before fitting XGBoost. The *exact* saved name, feature order, and width are validated before inference.

### Single prediction

1. Choose an ocean-proximity category from the detailed dropdown, or adjust its synchronized integer code with the −/+ stepper; `3` resolves to `NEAR BAY`.
2. Enter longitude, latitude, housing median age, and median income in the model's training representation.
3. Enter raw `households`, `total_rooms`, `total_bedrooms`, and `population` counts.
4. Inspect the three automatically calculated ratios, then choose **Predict median house value**.

### Batch inference

Upload `examples/housing_batch.csv` in the **Batch CSV** tab and choose **Run batch predictions**. The app previews the uploaded records, checks every requested input row, shows the encoding audit, and returns downloadable predictions. Files above 10 MiB, batches larger than 5,000 rows, duplicate/malformed headers, invalid UTF-8, malformed CSV records, infinities, missing numeric fields, and invalid category codes are rejected with readable errors; the app **does not silently truncate** excess rows. Unexpected columns (including a supplied target) are ignored and not passed to the model.

Required headers (any order):

```csv
longitude,latitude,housing_median_age,median_income,households,total_rooms,total_bedrooms,population,human_sense
-122.23,37.88,0.784314,2.635927,126,880,129,322,3
```

The app calculates:

```text
rooms_per_household      = total_rooms / households
bedrooms_per_room        = total_bedrooms / total_rooms
population_per_household = population / households
```

`households` and `total_rooms` must be greater than zero. Counts must be non-negative. The other four numerical columns are still model-ready values because the original fitted preprocessing artifact is unavailable.

For batches **500–5,000** rows, predictions are submitted to a bounded worker pool (two worker threads, at most four in-flight jobs including the queue). Click **Check status** to collect the finished result without blocking the interactive request. Batches smaller than 500 are handled synchronously. This is *in-process*, session-dependent task handling, not a durable enterprise job queue; restarts/disconnects may lose results. For larger multi-user workloads deploy Celery/RQ + durable storage and an authenticated job API instead.

## Fixed model encoding

The UI uses the bundled model's `one_hot` encoding directly. Mapping-strategy controls and custom mapping uploads are intentionally not displayed. The lower-level mapping module still validates the fixed code-to-category contract and rejects fractional or out-of-range codes.

Example manifest for the supplied artifact:

```json
{
  "model_path": "models/xgboost_top_features_model.joblib",
  "feature_names_path": "models/xgboost_top_feature_names.joblib",
  "input_strategy": "one_hot",
  "target_name": "median_house_value",
  "description": "California housing XGBRegressor; expects already engineered numeric features"
}
```

Do not let users upload joblib models dynamically. Restrict manifest paths to trusted, reviewed files inside the project. For long-term model portability, consider a versioned XGBoost JSON/UBJSON export plus a pinned preprocessing manifest instead of pickle-based persistence.

## Public API examples

```python
import pandas as pd
from pathlib import Path
from src.mapping import load_default_config
from src.pipeline import prepare_features, predict_batch
from src.model_store import load_model_artifacts

config = load_default_config('config/default_mapping.json')
model = load_model_artifacts(Path('.'))
raw = pd.read_csv('examples/housing_batch.csv')

matrix, explanation = prepare_features(
    raw, config, strategy='one_hot', required_model_features=model.feature_names
)
print(explanation[['human_sense', 'category', 'encoded_as']].head())
print(matrix.shape)  # (5, 14)

results, audit = predict_batch(raw, model.model, model.feature_names, config, 'one_hot')
print(results[['human_sense', 'predicted_median_house_value']])
```

Pipeline boundary:

```text
Ocean code 0–4 ──► validated one-hot columns ──────────────────────────┐
                                                                      │
Four raw housing counts ──► three calculated ratio features ──────────┤
                                                                      │
Four direct model values ──► finite-value validation ─────────────────┤
                                                                      ▼
                        exact feature order + schema check ──► prediction
```

## Test and verification

```bash
python -m pytest -q
python -m compileall -q app.py src tests
```

Tests cover code-to-trained-label mapping, mapping permutation, numeric validation (NaN, infinities, blanks, booleans, fractional inputs), normalized scalars, illustrative embeddings, invalid mapping uploads, CSV parsing/size rules, formula-safe CSV export, missing features, contract mismatches, bounded job queue, and **an actual prediction with the two supplied joblib artifacts**.

**Known artifact portability warning:** Loading this older XGBoost pickle with more recent XGBoost can emit a compatibility warning even if prediction works. For a production release, re-export with the original compatible environment, record Python/XGBoost/scikit-learn package versions and a model SHA-256, then run a golden prediction test after upgrading. The included `requirements.txt` specifies supported version ranges but does not reconstruct the *original* training environment.

## Security, operations, and remaining deployment work

- **Uploads:** Only UTF-8 CSV is exposed in the UI; byte-size and row limits, explicit field-count validation, no arbitrary user file paths, and no loading untrusted pickles. Streamlit's `file_uploader(type=...)` is not itself a security boundary: content is explicitly parsed and validated. CSV exports neutralize potential spreadsheet formulas in string cells. Browser UI error details are suppressed.
- **Concurrency:** Background prediction uses 2 workers and a maximum of 4 submitted jobs; tasks run without Streamlit calls. `st.cache_resource` retains the trusted model and worker pool; `st.cache_data` caches only a bundled public CSV example; `lru_cache` caches small code-to-vector transforms, not private uploaded records.
- **Logging:** Logs events, counts and failure stack traces to standard output; does not intentionally log uploaded data or raw numerical values. Configure central monitoring and log retention in your deployment.
- **Guardrails:** User values and calculated ratios must be finite, ratio denominators must be positive, category codes are not rounded, and the model receives the exact saved feature order. The two unavailable internal model inputs use explicit bundled sample defaults until a complete fitted preprocessing pipeline is available.
- **Deployment:** Put behind HTTPS with authentication/authorization if prediction data is sensitive. Add request/session quotas and CSRF-aware reverse proxy configuration, secrets management, robust model artifact supply-chain checks, tests at runtime pin upgrades, observability, drift monitoring, and a durable async queue for high concurrency. Do not expose this project to untrusted public traffic without those controls.
- **ML caveats:** Training notebook displayed approximately `R² = 0.8369` and `RMSE = 46,826.69` on its test evaluation. These are notebook outputs, **not a prospective accuracy guarantee**. Verify partition methodology, training leakage risks, out-of-distribution inputs, and model bias before business/financial use.

### Possible next improvement

Rebuild an end-to-end sklearn `Pipeline` / `ColumnTransformer` (or equivalent) from the raw California housing dataset *including fitted* imputation, scaling, cluster inference, hub-distance derivation, and target/model steps. Save all preprocessing parameters together with the model so end users can enter genuinely human-readable raw measures while still providing a scalar for ocean proximity.
