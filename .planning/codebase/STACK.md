# Technology Stack

**Analysis Date:** 2026-05-18

## Languages

**Primary:**
- Python 3.14.2 - All ML, data pipeline, backend, and web UI code

**Secondary:**
- None detected (no JavaScript/TypeScript frontend; `frontend/` directory is empty)

## Runtime

**Environment:**
- CPython 3.14.2 (via local venv at `venv/lib/python3.14/`)

**Package Manager:**
- pip (standard)
- Lockfile: Not present — only `requirements.txt` with minimum version pins (`>=`)

## Frameworks

**Core ML:**
- scikit-learn >=1.3.0 — preprocessing (StandardScaler, OneHotEncoder, ColumnTransformer), classifiers (RandomForest, LogisticRegression, KNN, MLP), regressors (RandomForest, GradientBoosting), MultiOutputRegressor, GridSearchCV, train_test_split, metrics
- xgboost >=2.0.0 — XGBClassifier used as Stage 1 candidate in `src/train_models.py`

**Web UI:**
- Streamlit >=1.30.0 — single-page app in `app.py`; session state manages multi-stage chat + portfolio state
- Plotly >=5.18.0 — Pie chart (`plotly.graph_objects.Pie`) for allocation display in `app.py`

**LLM:**
- openai >=1.30.0 — `gpt-4o-mini` for portfolio explanation and tool-calling chat in `src/llm.py`

**Data Fetching:**
- yfinance >=0.2.0 — Historical prices, fund info, geographic holdings, sector weightings in `src/etf_fetcher.py`
- finnhub-python >=2.4.0 — Real-time quote data (current price, day high/low) in `src/etf_fetcher.py`

**Data / Numeric:**
- pandas >=2.0.0 — DataFrames throughout; CSV I/O for all pipeline artifacts
- numpy >=1.24.0 — Numeric computation; stochastic sampling (`np.random.choice`) in `src/allocations.py`

**Serialization:**
- joblib >=1.3.0 — `.pkl` serialization for trained models and preprocessor (`models/*.pkl`, `data/training/preprocessor.pkl`)

**Visualization (non-UI):**
- matplotlib >=3.7.0 — Pipeline evaluation plots (confusion matrix, feature importance) saved to `evaluation_reports/`
- seaborn >=0.12.0 — Heatmap for confusion matrix in `src/train_models.py`

**Testing:**
- pytest >=7.0 — Test runner for `tests/test_data_validation.py` (24 unit tests)

**Build/Dev:**
- jupyter >=1.0.0 / notebook >=7.0.0 — Exploratory notebooks in `notebooks/`
- python-dotenv >=1.0.0 — Loads `.env` secrets in `src/etf_fetcher.py` and `app.py`

## Key Dependencies

**Critical (runtime path):**
- `scikit-learn` — preprocessor, all 3 ML model stages; removing it breaks the entire inference pipeline
- `openai` — LLM advisory layer; `src/llm.py` instantiates client at module load (`client = OpenAI(...)`)
- `joblib` — model artifact load at inference time (`_load_artifacts()` in `src/inference.py`)
- `yfinance` — ETF data source; required to run Step 1 of the pipeline
- `finnhub-python` — ETF real-time quote enrichment; requires `FINNHUB_API_KEY` env var

**Infrastructure:**
- `streamlit` — entire web UI is Streamlit; no other HTTP server present
- `xgboost` — imported at module level in `src/train_models.py`; MLP note: MLPClassifier is sklearn built-in

## Configuration

**Environment:**
- Loaded via `python-dotenv` from project-root `.env` file (gitignored; never committed)
- `app.py` calls `load_dotenv()` at startup
- `src/etf_fetcher.py` loads `.env` explicitly: `env_path = Path(__file__).parent.parent / '.env'`
- Required keys: `FINNHUB_API_KEY`, `OPENAI_API_KEY`

**Build:**
- No build system (no `pyproject.toml`, `setup.py`, or `Makefile`)
- Entry point for training: `python pipeline.py`
- Entry point for web UI: `venv/bin/streamlit run app.py`
- Entry point for inference CLI: `venv/bin/python src/inference.py`

## Platform Requirements

**Development:**
- Python 3.14.2
- Virtual environment at `venv/` (gitignored)
- `.env` file with `FINNHUB_API_KEY` and `OPENAI_API_KEY`
- Internet access for yfinance and Finnhub API calls during ETF data fetch step

**Production:**
- No deployment configuration detected (no Dockerfile, Procfile, or cloud config)
- All data and model artifacts (`data/`, `models/`, `evaluation_reports/`) are gitignored and must be regenerated
- Pipeline regeneration order: `python pipeline.py` → `src/prepare_data.py` → `src/train_models.py`

---

*Stack analysis: 2026-05-18*
