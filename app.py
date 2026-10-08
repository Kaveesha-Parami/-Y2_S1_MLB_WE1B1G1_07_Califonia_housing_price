"""Interactive Streamlit front end for model-ready housing predictions.

Run from the project root: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_io import UploadError, read_batch_csv, to_safe_csv
from src.mapping import MappingError, load_default_config
from src.model_store import load_model_artifacts
from src.pipeline import InputValidationError, predict_batch, prepare_features
from src.schema import INPUT_NUMERIC_COLUMNS, RAW_HOUSING_COLUMNS, REQUIRED_INPUT, SINGLE_DEFAULTS
from src.workers import BoundedWorkers, QueueFullError

ROOT = Path(__file__).resolve().parent
BATCH_THREAD_THRESHOLD = 500
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
LOGGER = logging.getLogger('human_sense_app')

st.set_page_config(page_title='Human-Sense Housing Predictor', layout='wide')

OCEAN_CATEGORY_DETAILS = {
    '<1H OCEAN': 'Homes in the dataset described as less than one hour from the ocean.',
    'INLAND': 'Homes located farther inland, away from the immediate coast.',
    'ISLAND': 'Homes located on an island.',
    'NEAR BAY': 'Homes located near a bay.',
    'NEAR OCEAN': 'Homes located near the ocean or coastline.',
}


def apply_color_theme(dark_mode: bool) -> None:
    """Apply a session-local light/dark palette without changing model behavior."""
    if dark_mode:
        background = '#0e1117'
        surface = '#262730'
        border = '#3b3d49'
        text = '#fafafa'
        muted = '#b8bcc8'
        accent = '#ff4b4b'
    else:
        background = '#ffffff'
        surface = '#f3f5f8'
        border = '#d8dde6'
        text = '#17202a'
        muted = '#56606b'
        accent = '#d93434'
    color_scheme = 'dark' if dark_mode else 'light'
    st.markdown(
        f'''
        <style>
        :root {{ color-scheme: {color_scheme}; }}
        .stApp, [data-testid="stAppViewContainer"] {{
            background-color: {background};
            color: {text};
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background-color: {background};
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp p, .stApp label, .stApp li,
        [data-testid="stWidgetLabel"] p,
        [data-testid="stExpander"] summary {{
            color: {text} !important;
        }}
        .stApp [data-testid="stCaptionContainer"] p {{
            color: {muted} !important;
        }}
        .stApp [data-testid="stExpander"],
        .stApp div[data-baseweb="select"] > div,
        .stApp div[data-baseweb="base-input"] > div,
        .stApp input,
        [data-baseweb="popover"], [role="listbox"] {{
            background-color: {surface} !important;
            color: {text} !important;
            border-color: {border} !important;
        }}
        [role="option"] {{ color: {text} !important; }}
        .stTabs [data-baseweb="tab"] {{ color: {text}; }}
        .stTabs [aria-selected="true"] {{ color: {accent} !important; }}
        hr {{ border-color: {border} !important; }}
        </style>
        ''',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner='Loading model...')
def get_model(project_root: str):
    return load_model_artifacts(Path(project_root))


@st.cache_resource(show_spinner=False)
def get_worker_pool() -> BoundedWorkers:
    """Bounded background executor, shared across sessions; no st.* calls inside tasks."""
    return BoundedWorkers(max_workers=2, max_inflight=4)


@st.cache_data(show_spinner=False)
def get_example_csv() -> bytes:
    """Cache a bundled public example, never a private user dataset."""
    return (ROOT / 'examples/housing_batch.csv').read_bytes()


def get_batch_key(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sync_ocean_from_dropdown() -> None:
    st.session_state.ocean_code_stepper = st.session_state.ocean_code_dropdown


def sync_ocean_from_stepper() -> None:
    st.session_state.ocean_code_dropdown = st.session_state.ocean_code_stepper


def show_single(artifacts, config, strategy: str) -> None:
    st.subheader('Single prediction')
    st.write('Enter the housing values below. The app calculates the three ratio features automatically.')
    st.markdown('### Ocean proximity')
    st.write(
        'Choose the category that best describes the property. The numbers are category codes used '
        'to activate the model’s matching one-hot feature; they are not distances, scores, or rankings.'
    )
    st.session_state.setdefault('ocean_code_dropdown', 3)
    st.session_state.setdefault('ocean_code_stepper', 3)
    dropdown_col, stepper_col = st.columns([2, 1])
    with dropdown_col:
        st.selectbox(
            'Select a category from the dropdown',
            options=tuple(config.codes),
            format_func=lambda code: f'{code} — {config.codes[code]}',
            key='ocean_code_dropdown',
            on_change=sync_ocean_from_dropdown,
        )
    with stepper_col:
        st.number_input(
            'Or adjust the code with − / +',
            min_value=config.minimum,
            max_value=config.maximum,
            step=1,
            key='ocean_code_stepper',
            on_change=sync_ocean_from_stepper,
        )
    human_sense = st.session_state.ocean_code_stepper
    selected_label = config.codes[human_sense]
    with st.expander('Ocean proximity code guide', expanded=True):
        for code, label in config.code_to_label:
            marker = ' **(selected)**' if code == human_sense else ''
            st.markdown(f'**{code} — {label}**{marker}: {OCEAN_CATEGORY_DETAILS[label]}')
        st.caption(
            f'Current selection: code {human_sense} activates the {selected_label} category; '
            'the other four ocean-category inputs are set to 0.'
        )
    with st.expander('Housing inputs', expanded=True):
        st.caption(
            '`rooms_per_household = total_rooms / households` · '
            '`bedrooms_per_room = total_bedrooms / total_rooms` · '
            '`population_per_household = population / households`'
        )
        numeric_values: dict[str, float] = {}
        cols = st.columns(2)
        for i, name in enumerate(INPUT_NUMERIC_COLUMNS):
            kwargs = {
                'label': name.replace('_', ' ').title(),
                'value': float(SINGLE_DEFAULTS[name]),
                'key': f'numeric_{name}',
            }
            if name in RAW_HOUSING_COLUMNS:
                kwargs['format'] = '%.0f'
                kwargs['min_value'] = 1.0 if name in ('households', 'total_rooms') else 0.0
                kwargs['help'] = f'Raw {name.replace("_", " ")} count.'
            else:
                kwargs['format'] = '%.6f'
                kwargs['help'] = f'Model-ready value for {name}; do not apply scaling again.'
            with cols[i % 2]:
                numeric_values[name] = st.number_input(**kwargs)
    raw = pd.DataFrame([{**numeric_values, 'human_sense': human_sense}])
    try:
        matrix, audit = prepare_features(raw, config, strategy)
        st.caption(
            f"Calculated ratios — rooms/household: {matrix['rooms_per_household'].iloc[0]:.6f}, "
            f"bedrooms/room: {matrix['bedrooms_per_room'].iloc[0]:.6f}, "
            f"population/household: {matrix['population_per_household'].iloc[0]:.6f}. "
            f"Ocean category: {audit['category'].iloc[0]}."
        )
    except (InputValidationError, MappingError) as exc:
        st.warning(f'Fix input: {exc}')
        return

    if st.button('Predict median house value', type='primary'):
        try:
            output, _ = predict_batch(raw, artifacts.model, artifacts.feature_names, config, strategy)
            score = float(output['predicted_median_house_value'].iloc[0])
            st.metric('Predicted median house value (training target units)', f'{score:,.2f}')
            st.caption('Prediction, not an appraisal. Model performance depends on preprocessing '
                       'consistency, input distribution, and historical training data.')
        except Exception:
            LOGGER.exception('Single inference failed')
            st.error('Prediction failed. Confirm model compatibility and inspect server logs.')


def show_batch(artifacts, config, strategy: str) -> None:
    st.subheader('Batch inference from CSV')
    st.write('Upload a CSV containing **human_sense**, four direct model values, and the four raw housing counts. '
             'The ratio features are calculated automatically for every row.')
    st.download_button('Download example CSV', data=get_example_csv(),
                       file_name='housing_example.csv', mime='text/csv')
    uploaded = st.file_uploader('Choose input CSV (10 MiB max; 5,000 rows)', type=['csv'],
                                help='UTF-8 CSV only. Use households, total_rooms, total_bedrooms, and population.')
    if uploaded is None:
        return
    try:
        payload = uploaded.getvalue()
        df = read_batch_csv(payload, uploaded.name)
        st.caption(f'{len(df):,} rows × {len(df.columns)} columns')
        st.dataframe(df.head(12), use_container_width=True, hide_index=True)
        if extra := sorted(set(df.columns) - set(REQUIRED_INPUT)):
            st.info(f'Additional uploaded columns will be ignored: {", ".join(extra)}')
        _, audit = prepare_features(df, config, strategy)
        with st.expander('Preview first 10 resolved ocean categories'):
            st.dataframe(audit.head(10), hide_index=True, use_container_width=True)
    except (UploadError, InputValidationError, MappingError) as exc:
        st.error(str(exc))
        return
    key = get_batch_key(payload)
    task = st.session_state.get('batch_task')
    running = bool(task and not task['future'].done())
    if running and task['key'] != key:
        st.warning('A previous inference is still running; complete it before submitting a changed file.')
    if st.button('Run batch predictions', type='primary', disabled=running):
        st.session_state.pop('batch_result', None)
        try:
            if len(df) >= BATCH_THREAD_THRESHOLD:
                # Only pure Python/Pandas/XGBoost work in worker; no Streamlit API in worker.
                future = get_worker_pool().submit(
                    predict_batch, df.copy(deep=True), artifacts.model,
                    artifacts.feature_names, config, strategy,
                )
                st.session_state.batch_task = {'key': key, 'future': future}
                st.info('Batch submitted to bounded worker pool. Use Check status to collect results.')
            else:
                with st.spinner('Predicting...'):
                    result, explanation = predict_batch(
                        df, artifacts.model, artifacts.feature_names, config, strategy
                    )
                st.session_state.batch_result = (key, result, explanation)
                st.session_state.pop('batch_task', None)
        except QueueFullError as exc:
            st.warning(str(exc))
        except Exception:
            LOGGER.exception('Batch inference failed')
            st.error('Batch prediction failed. Confirm model compatibility and inspect server logs.')

    task = st.session_state.get('batch_task')
    if task and task['key'] == key:
        if task['future'].done():
            try:
                result, explanation = task['future'].result()
                st.session_state.batch_result = (key, result, explanation)
                st.success('Background inference completed.')
            except Exception:
                LOGGER.exception('Background inference failed')
                st.error('Background prediction failed. Check server logs; confirm feature compatibility.')
            finally:
                st.session_state.pop('batch_task', None)
        else:
            st.info('Prediction is running in a background thread (bounded concurrency).')
            st.button('Check status')  # reruns the app, which checks Future.done()

    saved = st.session_state.get('batch_result')
    if saved and saved[0] == key:
        _, result, explanation = saved
        st.success(f'Predictions ready for {len(result):,} rows.')
        st.dataframe(result.head(100), hide_index=True, use_container_width=True)
        with st.expander('Show mapping audit for predicted rows'):
            st.dataframe(explanation.head(100), hide_index=True, use_container_width=True)
        st.download_button('Download all predictions as CSV', data=to_safe_csv(result),
                           file_name='housing_predictions.csv', mime='text/csv',
                           key=f'download_{key[:12]}')


def main() -> None:
    st.session_state.setdefault('dark_mode', True)
    apply_color_theme(bool(st.session_state.dark_mode))
    title_col, theme_col = st.columns([5, 1], vertical_alignment='center')
    with title_col:
        st.title('Human-Sense Housing Predictor')
    with theme_col:
        st.toggle('Dark mode', key='dark_mode', help='Switch between dark and light page colors.')
    st.caption('XGBoost · automatic housing-ratio feature engineering · single & batch inference')
    try:
        artifacts = get_model(str(ROOT))
        config = load_default_config(ROOT / 'config/default_mapping.json')
    except (OSError, ValueError, KeyError) as exc:
        LOGGER.exception('Initialization failed')
        st.error(f'Cannot initialize the trusted model/configuration: {exc}')
        st.stop()

    strategy = artifacts.input_strategy
    tab1, tab2, tab3 = st.tabs(['Single prediction', 'Batch CSV', 'About / model contract'])
    with tab1:
        show_single(artifacts, config, strategy)
    with tab2:
        show_batch(artifacts, config, strategy)
    with tab3:
        st.markdown('''
**Problem:** predict `median_house_value` from 14 already-engineered inputs. Users provide
an intuitive `human_sense` number to choose one of five trained `ocean_proximity` categories.

**Automatic feature engineering:** users enter `households`, `total_rooms`, `total_bedrooms`, and
`population`. The application calculates `rooms_per_household`, `bedrooms_per_room`, and
`population_per_household` immediately before prediction.

**Preprocessing boundary:** training loaded an already-processed `df_clean5.csv`. The fitted
upstream transformer was not supplied, so longitude, latitude, housing median age, and median
income must still be entered in the same representation used during training.
        ''')
        st.caption('Numeric feature values are strictly parsed as finite floats; target columns and other '
                   'unexpected upload columns are not passed to prediction.')


if __name__ == '__main__':
    main()
