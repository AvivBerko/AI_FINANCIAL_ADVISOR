# Codebase Structure

**Analysis Date:** 2026-05-18

## Directory Layout

```
AI_FINANCIAL_ADVISOR/
├── app.py                      # Streamlit web application (entry point)
├── pipeline.py                 # Training pipeline orchestrator (entry point)
├── requirements.txt            # Python dependencies
├── .env                        # API keys (gitignored, never commit)
│
├── src/                        # All Python source modules
│   ├── allocations.py          # Rule-based ETF selection + training label generator
│   ├── data_generation.py      # Synthetic investor profile generator (5,000 rows)
│   ├── data_validation.py      # ETF + investor DataFrame validation
│   ├── etf_fetcher.py          # yfinance + Finnhub ETF data acquisition
│   ├── inference.py            # 4-stage inference pipeline (public API: recommend())
│   ├── llm.py                  # OpenAI gpt-4o-mini advisory layer
│   ├── morning_star_rating.py  # Sharpe-based custom star rating calculator
│   ├── prepare_data.py         # Feature engineering + train/test split
│   ├── train_models.py         # Model training, comparison, and persistence
│   └── validation_rules.py     # Shared validation contracts (field names, ranges)
│
├── models/                     # Persisted ML artifacts (gitignored, regenerate-able)
│   ├── risk_profile_classifier.pkl   # Stage 1 winner (best macro F1)
│   ├── risk_profile_rf.pkl           # Stage 1 RF (always saved for feature importance)
│   ├── allocation_regressor.pkl      # Stage 2 winner (best avg MAE)
│   ├── etf_count_regressor.pkl       # Stage 3 Random Forest
│   └── label_map.json                # Winner algorithm name + int-to-label map
│
├── data/                       # All data artifacts (gitignored, regenerate-able)
│   ├── raw/
│   │   ├── combined_etf_data.csv                  # Raw ETF data from APIs
│   │   └── combined_etf_with_morning_star.csv      # ETF data + star ratings
│   ├── processed/
│   │   └── synthetic_investor_data_<timestamp>.csv # Generated investor profiles
│   ├── training/
│   │   ├── X_train.csv, X_test.csv                # 19-feature processed inputs
│   │   ├── y_train_profile.csv, y_test_profile.csv # Stage 1 labels
│   │   ├── y_train_allocation.csv, y_test_allocation.csv # Stage 2 labels
│   │   ├── y_train_etfs.csv, y_test_etfs.csv      # Stage 3 labels
│   │   └── preprocessor.pkl                       # Fitted ColumnTransformer
│   └── validation_reports/     # Markdown validation reports (gitignored)
│
├── evaluation_reports/         # Training metrics + plots (gitignored)
│   ├── model_metrics.json
│   ├── confusion_matrix_risk_profile.png
│   ├── feature_importance_risk_profile.png
│   ├── allocation_predictions.png
│   ├── etf_count_distribution.png
│   └── stage1_algorithm_comparison.png
│
├── notebooks/                  # Jupyter notebooks (experiments + reporting)
│   ├── phase2_data_preparation.ipynb
│   ├── phase3_model_training.ipynb
│   ├── phase3A_classification_experiments.ipynb
│   ├── phase3A_classification_experiments_no_finetuning.ipynb
│   ├── phase3B_allocation_regression_experiments.ipynb
│   ├── phase3B_allocation_regression_experiments wide_params.ipynb
│   ├── phase3C_etf_count_regression_experiments.ipynb
│   ├── phase3C_etf_count_regression_experiments wide_params.ipynb
│   ├── phase4_etf_prediction.ipynb
│   ├── models/                 # Notebook-local model copies (separate from root models/)
│   └── evaluation_reports/     # Notebook-local report copies
│
├── tests/
│   ├── __init__.py
│   └── test_data_validation.py # 24 unit tests for src/data_validation.py
│
├── docs/
│   └── superpowers/
│       ├── plans/
│       └── specs/
│
├── .planning/
│   └── codebase/               # GSD architecture documents (this directory)
│
├── frontend/                   # Empty — frontend not yet implemented
│
└── subbmited_files/            # Submitted project deliverables (gitignored)
```

## Directory Purposes

**`src/`:**
- Purpose: All application Python modules
- Contains: 10 modules covering data acquisition, validation, training, inference, and LLM layer
- Key files: `src/inference.py` (main API), `src/allocations.py` (shared rule logic), `src/validation_rules.py` (shared contracts)

**`models/`:**
- Purpose: Persisted joblib model artifacts produced by `src/train_models.py`
- Contains: Three `.pkl` model files, one RF-specific `.pkl`, one `label_map.json`
- Generated: Yes — by `python src/train_models.py` or `python pipeline.py`
- Committed: No — gitignored (`models/*.pkl`); only directory placeholder `.gitkeep` is committed

**`data/`:**
- Purpose: All data artifacts organised by pipeline stage
- Contains: `raw/` (API data), `processed/` (generated investor CSVs), `training/` (feature matrices + labels + preprocessor), `validation_reports/` (per-run markdown)
- Generated: Yes — by `pipeline.py` steps 1-3
- Committed: No — gitignored (`*.csv`, `data/`); only `.gitkeep` is committed

**`data/training/`:**
- Purpose: Ready-to-train feature matrices and label vectors + fitted preprocessor
- Key files: `X_train.csv` (19 features, ~4,000 rows), `preprocessor.pkl` (must match inference transform)
- Note: The preprocessor in `data/training/preprocessor.pkl` MUST be used at inference time — `src/inference.py` loads it from this path

**`evaluation_reports/`:**
- Purpose: Model performance metrics and visualisation plots
- Contains: `model_metrics.json` (machine-readable metrics for all stages), five PNG plots
- Generated: Yes — by `src/train_models.py`
- Committed: No — gitignored

**`notebooks/`:**
- Purpose: Interactive experiments and academic reporting
- Contains: Eight Jupyter notebooks mirroring the training pipeline stages
- Note: `notebooks/models/` and `notebooks/evaluation_reports/` are local copies independent from the project-root `models/` and `evaluation_reports/` directories used by the production code

**`tests/`:**
- Purpose: Unit tests — currently covers data validation only
- Key files: `tests/test_data_validation.py` (24 tests for `src/data_validation.py`)

**`frontend/`:**
- Purpose: Reserved for a separate frontend (currently empty)
- Note: The current web UI is in `app.py` (Streamlit); this directory is a placeholder

## Key File Locations

**Entry Points:**
- `pipeline.py`: Training pipeline orchestrator — run to regenerate all artifacts from scratch
- `app.py`: Streamlit web application — run with `venv/bin/streamlit run app.py`
- `src/inference.py`: CLI demo and importable inference API — `from src.inference import recommend`

**Configuration:**
- `.env`: `OPENAI_API_KEY`, `FINNHUB_API_KEY` — required at runtime for LLM and ETF fetching
- `requirements.txt`: All Python dependencies
- `src/validation_rules.py`: Canonical field names, valid categorical values, numeric ranges — edit here to change data contracts

**Core Logic:**
- `src/inference.py`: The `recommend(user: dict) -> dict` function — the single public API for the 4-stage pipeline
- `src/allocations.py`: `CONFLICT_GROUPS`, `assign_portfolio_numeric_aum()`, `run_allocations_pipeline()` — Stage 4 logic shared between training and inference
- `src/train_models.py`: Full training comparison logic for all three stages

**Testing:**
- `tests/test_data_validation.py`: 24 unit tests — run with `python -m pytest tests/`

**Documentation:**
- `model_architecture.md`: Detailed description of ML model choices
- `project_flowchart.md`: Visual flowchart of the full system
- `llm.md`: Notes on the LLM integration design
- `PROJECT_WALKTHROUGH.md`: End-to-end project walkthrough

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `train_models.py`, `data_generation.py`)
- Model artifacts: descriptive with stage prefix (e.g., `risk_profile_classifier.pkl`, `allocation_regressor.pkl`)
- Training CSV files: `X_train.csv`, `y_train_<target>.csv`, `y_test_<target>.csv`
- Data CSVs: `combined_etf_data.csv`, `synthetic_investor_data_<YYYYMMDDHHMMSS>.csv`

**Directories:**
- All lowercase with underscores: `data/`, `evaluation_reports/`, `validation_reports/`
- Data subdirs reflect pipeline stage: `raw/` → `processed/` → `training/`

**Python code:**
- Functions: `snake_case` (e.g., `recommend`, `validate_input`, `assign_portfolio_numeric_aum`)
- Classes: `PascalCase` (e.g., `ValidationReport`, `FinnhubETFDataHandler`, `YFinanceETFDataHandler`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `CONFLICT_GROUPS`, `FEATURE_COLUMNS`, `VALID_VALUES`)
- Module-level singletons: prefixed with `_` (e.g., `_ARTIFACTS`, `_INFERENCE_TO_CSV`)

**Investor profile keys:**
- CSV / data_generation canonical form: `CapitalCase` (`Age`, `Income`, `InvestmentHorizon`, `RiskTolerance`, `ExperienceYears`)
- Inference dict form: mixed (`age`, `income`, `horizon`, `risk_tolerance`, `experience` for numerics; `CapitalCase` for categoricals)
- Mapping table: `src/inference.py:58` `_INFERENCE_TO_CSV`

## Where to Add New Code

**New ML model for an existing stage:**
- Add training logic in `src/train_models.py` within the appropriate stage section
- Save as `models/<stage>_<algo>.pkl` using joblib
- Update `label_map.json` writing if needed for Stage 1 label encoding
- Update `src/inference.py` model loading in `_load_artifacts()` if changing the winning model

**New investor feature:**
- Add generation logic to `src/data_generation.py`
- Add field to `INVESTOR_FEATURE_COLUMNS` in `src/validation_rules.py`
- Add range/valid-values entry to `VALID_VALUES` or `NUMERIC_RANGES` in `src/validation_rules.py`
- Add to `feature_columns` list in `src/prepare_data.py:99`
- Add to `FEATURE_COLUMNS` list in `src/inference.py:53`
- Add to `categorical_features` or `numerical_features` in `src/prepare_data.py:106`
- Add UI input to `app.py` form and profile dict construction at `app.py:97`

**New ETF asset class or conflict group:**
- Add category keywords to `get_category_mapping()` in `src/allocations.py:126`
- Add new conflict group to `CONFLICT_GROUPS` in `src/allocations.py:11`
- Mirror keywords in `KNOWN_CATEGORY_KEYWORDS` in `src/validation_rules.py`

**New LLM tool:**
- Add tool schema to `TOOLS` list in `src/llm.py:23`
- Add dispatch branch in `app.py` tool dispatch section around `app.py:264`

**New test:**
- Add test file `tests/test_<module>.py`
- Follow existing pattern in `tests/test_data_validation.py` using pytest

**New notebook experiment:**
- Place in `notebooks/` with prefix `phase<N><letter>_<description>.ipynb`

## Special Directories

**`models/` (project root):**
- Purpose: Production model artifacts used by `src/inference.py`
- Generated: Yes — by `src/train_models.py`
- Committed: No

**`notebooks/models/`:**
- Purpose: Notebook-local model artifacts — NOT used by `src/inference.py`
- Generated: Yes — when notebooks are run
- Committed: No
- Warning: Do not confuse with project-root `models/`; the two are independent

**`data/training/`:**
- Purpose: Bridge between data engineering and model training; `preprocessor.pkl` here must stay in sync with `src/inference.py`'s transform
- Generated: Yes — by `src/prepare_data.py`
- Committed: No

**`.planning/codebase/`:**
- Purpose: GSD architecture documents consumed by planning and execution agents
- Generated: Yes — by `/gsd:map-codebase`
- Committed: Yes — should be committed alongside code

---

*Structure analysis: 2026-05-18*
