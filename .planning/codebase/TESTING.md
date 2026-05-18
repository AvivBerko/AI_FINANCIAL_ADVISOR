# Testing Patterns

**Analysis Date:** 2026-05-18

## Test Framework

**Runner:**
- pytest 9.0.3 (Python 3.14.2)
- Config: none — no `pytest.ini`, `pyproject.toml`, or `setup.cfg` present; pytest discovers tests with default settings

**Assertion Library:**
- pytest built-in `assert` statements
- `pytest.raises` for exception assertions

**Run Commands:**
```bash
python -m pytest tests/                # Run all tests
python -m pytest tests/ -v             # Verbose output
python -m pytest tests/ -k "investor"  # Filter by name pattern
```

No coverage command configured. No watch mode configured.

## Test File Organization

**Location:** All tests live in `tests/` at the project root, separate from `src/`.

**Naming:**
- Test file: `test_<module_name>.py` — currently only `tests/test_data_validation.py`
- `tests/__init__.py` present (empty), enabling `from src.X import Y` inside tests

**Structure:**
```
tests/
├── __init__.py
└── test_data_validation.py   # 24 tests for src/data_validation.py
```

## Test Structure

**Suite Organization:**

Tests are grouped by subject with `# ---` separator comments:

```python
# ---------------------------------------------------------------------------
# Fixtures for investor tests
# ---------------------------------------------------------------------------
def _make_good_investor_row(**overrides):
    ...

def _make_good_investor_df(n=10):
    ...

# ---------------------------------------------------------------------------
# validate_investor_data
# ---------------------------------------------------------------------------
def test_investor_happy_path_keeps_all_rows():
    ...
```

No `class`-based test grouping — all tests are top-level functions.

**Import style inside test functions:**

Imports are deferred inside the test body rather than at module top:

```python
def test_investor_happy_path_keeps_all_rows():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(10)
    clean, rpt = validate_investor_data(df)
    assert rpt.n_input == 10
```

The only module-level imports are `pytest`, `pandas`, `numpy`, and `ValidationReport` (the dataclass, needed for dataclass construction tests).

**Test naming convention:**

`test_<subject>_<scenario_description>` — descriptive enough to read as a sentence:
- `test_investor_happy_path_keeps_all_rows`
- `test_investor_drops_row_with_nan_in_feature`
- `test_etf_imputes_nan_aum_with_class_median`
- `test_etf_floor_violation_raises`
- `test_map_to_asset_class_matches_allocations_get_category_mapping`

## Fixtures and Test Data Factories

**Pattern:** Plain helper functions prefixed with `_make_` instead of pytest `@fixture` decorators.

```python
def _make_good_investor_row(**overrides):
    row = {
        'Age': 30, 'Gender': 'Male', 'Education': 'Bachelor',
        'MaritalStatus': 'Single', 'HouseholdSize': 1,
        'Income': 80000, 'InvestmentGoal': 'Retirement',
        'InvestmentHorizon': 30, 'InvestmentCapital': 50000,
        'RiskTolerance': 'Medium', 'FinancialInvolvement': 'Medium',
        'ExperienceYears': 5,
    }
    row.update(overrides)
    return row

def _make_good_investor_df(n=10):
    return pd.DataFrame([_make_good_investor_row() for _ in range(n)])
```

```python
def _make_good_etf_row(**overrides):
    row = {
        'Fund Symbol': 'VOO',
        'Category': 'Large Blend',
        'Assets Under Management (AUM)': 1_000_000_000.0,
        'custom_star_rating': 4.0,
        'Volatility (Annual STD)': 0.18,
    }
    row.update(overrides)
    return row

def _make_good_etf_df():
    """Returns 12 ETFs: 6 equity, 5 bond, 1 alternative — passes all floors."""
    rows = (
        [_make_good_etf_row(**{'Fund Symbol': f'E{i}', 'Category': 'Large Blend'})    for i in range(6)] +
        [_make_good_etf_row(**{'Fund Symbol': f'B{i}', 'Category': 'Corporate Bond'}) for i in range(5)] +
        [_make_good_etf_row(**{'Fund Symbol': 'GLD',   'Category': 'Gold'})]
    )
    return pd.DataFrame(rows)
```

The `**overrides` keyword argument pattern lets tests mutate a single field from a valid baseline:

```python
df.loc[2, 'Age'] = np.nan  # or via override in row construction
```

**No pytest fixtures** (`@pytest.fixture`) are used anywhere.

## Mocking

**No mocking framework used.** Tests for `src/data_validation.py` are pure unit tests that operate entirely on in-memory DataFrames. No external calls, no I/O, no model loading — so mocking is not needed for the currently tested code.

Tests for `src/inference.py` validation (`test_inference_validate_input_still_works`) call the real `validate_input()` function, which only does dict key checks — again requiring no mocking.

When writing tests for code that loads models from disk (`_load_artifacts` in `src/inference.py`) or calls external APIs (`src/etf_fetcher.py`, `src/llm.py`), mocking will be required. The recommended approach would be `unittest.mock.patch` or `pytest-mock`.

## What Is Tested vs. Not Tested

**Tested (24 tests in `tests/test_data_validation.py`):**
- `ValidationReport` dataclass construction and `to_markdown()` output
- `validate_investor_data`: happy path, missing-column schema error, NaN drop, invalid categorical drop, out-of-range numeric drop, consistency violation (age vs. experience), multiple-violation row dropped once, duplicate warning only, class-balance warning
- `validate_etf_data`: happy path, missing-column schema error, missing Fund Symbol drop, missing Category drop, NaN AUM imputation (class median), negative volatility treated as bad, global default fallback when class has < 3 samples, asset-class counts reported, floor violation (single class), floor violation (multiple classes)
- `validate_input` in `src/inference.py`: good input passes, bad Gender raises, bad age raises
- Cross-module sync: `_map_to_asset_class` in `src/data_validation.py` must match `get_category_mapping` in `src/allocations.py`
- Refactor guard: `src/inference.py` must import from `src.validation_rules` not define its own constants

**NOT tested:**
- `src/allocations.py` — `get_profile_probabilities`, `get_allocation_mix`, `get_num_etfs`, `assign_portfolio_numeric_aum`, `run_allocations_pipeline`
- `src/train_models.py` — ML model training, evaluation helpers, model persistence
- `src/prepare_data.py` — data preparation pipeline, feature engineering, train/test split
- `src/data_generation.py` — synthetic investor generation, risk score calculation
- `src/etf_fetcher.py` — API fetch logic, data processing
- `src/morning_star_rating.py` — star rating calculation
- `src/inference.py::recommend` — full 4-stage inference pipeline
- `src/llm.py` — LLM chat, tool call handling, allocation math helper
- `app.py` — Streamlit UI

## Data Validation Pattern

The validation architecture is two-layer:

**Layer 1 — `src/validation_rules.py` (pure constants):**
Defines all contracts as Python data structures. No functions. Constants are imported by both the validation layer and inference:
- `INVESTOR_FEATURE_COLUMNS` — required columns list
- `VALID_VALUES` — allowed values per categorical column
- `NUMERIC_RANGES` — `(lo, hi)` per numeric column
- `ETF_REQUIRED_COLUMNS`, `ETF_DROP_IF_MISSING`, `ETF_IMPUTE_COLUMNS`
- `ASSET_CLASS_FLOORS`, `KNOWN_CATEGORY_KEYWORDS`

**Layer 2 — `src/data_validation.py` (logic):**
Implements validation using rules from Layer 1. Returns `(clean_df, ValidationReport)` tuples. Raises `ValueError` for unrecoverable schema errors; logs per-row issues to the report. Saves human-readable Markdown reports to `data/validation_reports/`.

**Integration point:**
`src/prepare_data.py` calls both validators before running the allocation pipeline:
```python
etf_df,       etf_report = validate_etf_data(etf_df)
investors_df, inv_report = validate_investor_data(investors_df)
```

**Inference uses Layer 1 constants directly** (remapping lowercase inference keys to canonical CSV keys):
```python
from src.validation_rules import VALID_VALUES as _CSV_VALID_VALUES, NUMERIC_RANGES as _CSV_NUMERIC_RANGES
VALID_VALUES   = {k: _CSV_VALID_VALUES[_to_csv(k)]   for k in FEATURE_COLUMNS if _to_csv(k) in _CSV_VALID_VALUES}
NUMERIC_RANGES = {k: _CSV_NUMERIC_RANGES[_to_csv(k)] for k in FEATURE_COLUMNS if _to_csv(k) in _CSV_NUMERIC_RANGES}
```

## Notebook Experimental Conventions

ML experiments are conducted in Jupyter notebooks before being consolidated into `src/` scripts.

**Notebook naming convention:**
- `phase<N>_<description>.ipynb` — phase number prefix ties notebooks to pipeline stages
- Variant suffixes: `_wide_params` (broader hyperparameter grids), `_no_finetuning` (baseline without tuning)

**Notebooks present:**
- `notebooks/phase2_data_preparation.ipynb` — exploratory feature engineering
- `notebooks/phase3_model_training.ipynb` — consolidated training (moved to `src/train_models.py`)
- `notebooks/phase3A_classification_experiments.ipynb` — classifier comparison experiments
- `notebooks/phase3A_classification_experiments_no_finetuning.ipynb` — baseline without GridSearch
- `notebooks/phase3B_allocation_regression_experiments.ipynb` — allocation regressor experiments
- `notebooks/phase3B_allocation_regression_experiments wide_params.ipynb` — wider param grid
- `notebooks/phase3C_etf_count_regression_experiments.ipynb` — ETF count model experiments
- `notebooks/phase3C_etf_count_regression_experiments wide_params.ipynb` — wider param grid
- `notebooks/phase4_etf_prediction.ipynb` — inference demo and multi-label approaches

**Convention:** Once a notebook experiment is finalized, the winning approach is extracted into a corresponding `src/` script (`src/train_models.py`, `src/prepare_data.py`). Notebooks are kept as academic record.

## Coverage

**Requirements:** None enforced. No `pytest-cov` in `requirements.txt`.

**Current state:** 24 tests cover `src/data_validation.py` (the validation layer) comprehensively. All other `src/` modules have zero test coverage.

**View coverage (if pytest-cov were installed):**
```bash
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

*Testing analysis: 2026-05-18*
