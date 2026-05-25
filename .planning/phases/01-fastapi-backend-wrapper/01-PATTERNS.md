# Phase 1: FastAPI Backend Wrapper — Pattern Map

**Mapped:** 2026-05-19
**Files analyzed:** 14 new + 2 modified
**Analogs found:** 14 / 14 (every new file has at least a role-match or data-flow-match analog)
**Phase directory:** `.planning/phases/01-fastapi-backend-wrapper/`

> The wrapper is **additive**: no existing source file is rewritten. The only existing-file edit is an additive helper on `src/allocations.py` to surface per-ticker selection metadata for `/api/explain`. Most "analogs" below are pulled from the Python service layer the wrapper will import (`src/inference.py`, `src/allocations.py`, etc.); the FastAPI patterns themselves come from RESEARCH.md because **no prior FastAPI code exists in this repo**.

---

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality | Notes |
|---|---|---|---|---|---|
| `api/main.py` (NEW) | app-entry | startup/lifespan | `pipeline.py` (orchestrator, fail-fast on missing artifacts) + `src/inference.py::_get_artifacts` (cached loader) | role-match (orchestrator pattern) | No FastAPI app exists; lifespan pattern is from RESEARCH.md Pattern 1 |
| `api/schemas.py` (NEW) | model/validation | request-response | `src/validation_rules.py` (VALID_VALUES, NUMERIC_RANGES, INVESTOR_FEATURE_COLUMNS) + `src/inference.py::validate_input` | exact role-match (validation contract) | Pydantic Literals must mirror `VALID_VALUES` constants verbatim |
| `api/deps.py` (NEW) | DI helper | request-response | `src/inference.py::_get_artifacts()` (singleton-cache idiom) | role-match | FastAPI Depends pulls from `app.state` set in lifespan |
| `api/routers/recommend.py` (NEW) | controller | request-response (CRUD-like) | `app.py` lines 96-117 (Streamlit form-submit → `recommend(profile)`) | data-flow-match | Streamlit is the only place that calls `recommend()` today |
| `api/routers/prices.py` (NEW) | controller | batch / partial-failure | `src/etf_fetcher.py::FinnhubETFDataHandler.get_finnhub_daily_prices` lines 23-43 (per-ticker try/except) | data-flow-match | Per-ticker error envelope pattern already exists |
| `api/routers/explain.py` (NEW) | controller | request-response | `src/inference.py::recommend` + new `assign_portfolio_with_metadata` adapter | role-match | Composes existing `recommend()` + a small new metadata-capturing variant |
| `api/routers/etfs.py` (NEW) | controller | lookup (single-record) | `src/inference.py::_load_artifacts` lines 84-102 (CSV → DataFrame, filter by symbol) | data-flow-match | Reads the same `combined_etf_with_morning_star.csv` |
| `api/services/recommend_service.py` (NEW) | service | transform | `src/inference.py::recommend` lines 139-235 (4-stage orchestration) + `app.py` lines 97-117 (call site) | exact role-match | Thin wrapper that converts PascalCase → lowercase keys and renormalizes |
| `api/services/prices_service.py` (NEW) | service | batch / streaming-like | `src/etf_fetcher.py::YFinanceETFDataHandler.fetch_etf_data` lines 76-112 (single-ticker yfinance call) | data-flow-match | Pattern: `yf.Ticker(symbol)` then `.info` / `.fast_info` with try/except |
| `api/services/explain_service.py` (NEW) | service | transform | `src/allocations.py::assign_portfolio_numeric_aum` lines 173-243 (selection logic where metadata lives) | role-match | Captures Stage-1 `feature_importances_`, Stage-2 raw output, Stage-4 per-ticker metadata |
| `api/services/etf_service.py` (NEW) | service | lookup | `src/inference.py::_load_artifacts` lines 84-104 (CSV load + column rename + numeric coerce) + `src/morning_star_rating.py::calculate_star_ratings` (already merged into CSV at training time) | exact role-match | The CSV is the supported-universe oracle |
| `tests/conftest.py` (NEW) | test fixture | n/a | `tests/test_data_validation.py` lines 32-46 (`_make_good_investor_row`, `_make_good_investor_df`) | exact role-match | Reuse the same fixture row shape, add a `TestClient` fixture |
| `tests/test_api_recommend.py` (NEW) | test | request-response | `tests/test_data_validation.py` lines 52-119 (pytest functions + assert on dict keys) | role-match (test style) | Same `pytest` + module-level functions, no class-based grouping |
| `tests/test_api_prices.py` (NEW) | test | batch | `tests/test_data_validation.py` lines 187-258 (per-row drop/impute assertions) | role-match (test style) | Same pytest idiom, mock `yf.Ticker` |
| `tests/test_api_explain.py` (NEW) | test | request-response | `tests/test_data_validation.py` lines 52-141 | role-match (test style) | — |
| `tests/test_api_etfs.py` (NEW) | test | lookup | `tests/test_data_validation.py` lines 188-282 | role-match (test style) | 404 assertion for unknown ticker |
| `src/allocations.py` (MODIFIED — additive only) | service | transform | itself, lines 173-243 (`assign_portfolio_numeric_aum`) | self | Add `assign_portfolio_with_metadata()` returning `{ticker: {asset_class, aum_rank, candidate_pool_size, star_rating}}` alongside the existing function |
| `requirements.txt` (MODIFIED — additive only) | config | n/a | existing `requirements.txt` (already has `streamlit`, `pytest`, `joblib`, `yfinance`, `openai`, etc.) | self | Add `fastapi>=0.115`, `uvicorn[standard]>=0.30` |
| `BACKEND.md` (NEW) | docs | n/a | `README.md` lines 1-50 (Quick Start sections, code-fenced bash) | role-match | Mirror header style + emoji conventions used in existing README |

---

## Pattern Assignments

### `api/schemas.py` (validation, request-response)

**Primary analog:** `src/validation_rules.py` (entire file is the source-of-truth contract).
**Secondary analog:** `src/inference.py::validate_input` lines 118-133 (the current Python-level validator the API replaces with Pydantic).

**Source-of-truth constants to mirror exactly** (`src/validation_rules.py` lines 18-34):

```python
VALID_VALUES = {
    'Gender':               ['Male', 'Female'],
    'Education':            ['High school', 'Bachelor', 'Master'],
    'MaritalStatus':        ['Married', 'Single', 'Divorced', 'Widowed'],
    'InvestmentGoal':       ['Retirement', 'Home purchase', 'Child education', 'Other'],
    'RiskTolerance':        ['Low', 'Medium', 'High'],
    'FinancialInvolvement': ['Low', 'Medium', 'High'],
}

NUMERIC_RANGES = {
    'Age':               (18, 100),
    'HouseholdSize':     (1, 10),
    'Income':            (0, 10_000_000),
    'InvestmentHorizon': (1, 50),
    'InvestmentCapital': (0, 100_000_000),
    'ExperienceYears':   (0, 80),
}
```

**Frozen contract mismatch to surface** (RESEARCH Pattern 2 callout, NOT to be fixed in Phase 1): `MaritalStatus` in `validation_rules.py` is 4-value `['Married','Single','Divorced','Widowed']`, but `DEC-investor-profile-schema-frozen` locks the API surface to 2 values `['Single','Married']`. **The locked schema wins** — the Pydantic Literal MUST be `Literal["Single", "Married"]` and the executor must NOT "fix" the upstream constant.

**Existing field-key mapping to mirror** (`src/inference.py` lines 53-65):

```python
FEATURE_COLUMNS = [
    'age', 'Gender', 'Education', 'MaritalStatus', 'HouseholdSize',
    'income', 'InvestmentGoal', 'horizon', 'InvestmentCapital',
    'risk_tolerance', 'FinancialInvolvement', 'experience'
]
_INFERENCE_TO_CSV = {
    'age': 'Age',
    'income': 'Income',
    'horizon': 'InvestmentHorizon',
    'risk_tolerance': 'RiskTolerance',
    'experience': 'ExperienceYears',
}
```

This is the **landmine**: `recommend()` expects `age` / `income` / `horizon` / `risk_tolerance` / `experience` (lowercase) but the locked HTTP schema is PascalCase. The Pydantic model needs a `to_inference_dict()` method that does this PascalCase → lowercase mapping (see RESEARCH Pattern 2, lines 424-439).

**Pattern for Pydantic v2 Literal + Field bounds** (from RESEARCH Pattern 2 — no prior analog in repo):

```python
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class InvestorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Age: int = Field(..., ge=18, le=100)
    Gender: Literal["Male", "Female"]
    # ... (see RESEARCH.md lines 395-440 for full example)
```

---

### `api/services/recommend_service.py` (service, transform)

**Primary analog:** `src/inference.py::recommend` lines 139-235 (the function being wrapped).
**Call-site analog:** `app.py` lines 96-117 (current consumer pattern).

**The function being wrapped** (`src/inference.py` lines 139-235) — return shape the wrapper must consume:

```python
def recommend(user: dict) -> dict:
    """
    Returns dict with keys:
        risk_profile   : str   — Aggressive / Moderate / Conservative
        allocation     : dict  — {equity_pct, bond_pct, alt_pct}  (sum = 1.0)
        total_etfs     : int   — total ETFs in portfolio
        etf_counts     : dict  — {equity, bond, alternative} counts
        portfolio      : dict  — {equity: [...], bond: [...], alternative: [...]}
    """
```

**Current consumer pattern to copy** (`app.py` lines 96-117):

```python
profile = {
    "age":                  age,
    "Gender":               gender,
    "Education":            education,
    "MaritalStatus":        marital_status,
    "HouseholdSize":        household_size,
    "income":               income,
    "InvestmentGoal":       investment_goal,
    "horizon":              horizon,
    "InvestmentCapital":    investment_cap,
    "risk_tolerance":       risk_tolerance,
    "FinancialInvolvement": financial_involvement,
    "experience":           experience,
}
try:
    result = recommend(profile)
except Exception as e:
    st.error(f"Error running ML pipeline: {e}")
```

The service-layer wrapper does the same call, plus (a) flattens `result['portfolio']` (per-class lists) + `result['allocation']` (per-class pct) → flat `{ticker: weight}`, and (b) renormalizes to sum=1.0. See RESEARCH Code Examples lines 632-674 for the flattening pattern.

**Renormalization pattern** (per `CON-allocated-assets-jsonb-shape`):

```python
total = sum(flat.values())
if total > 0:
    flat = {t: w / total for t, w in flat.items()}
```

**Seed pattern for reproducibility** (per RESEARCH Pitfall 4 — `np.random.choice` in Stage 4 is non-deterministic):

```python
import numpy as np
if seed is not None:
    np.random.seed(seed)
```

---

### `api/services/prices_service.py` (service, batch / partial-failure)

**Primary analog:** `src/etf_fetcher.py::FinnhubETFDataHandler.get_finnhub_daily_prices` lines 23-43 (per-ticker try/except returning a per-ticker dict).
**Secondary analog:** `src/etf_fetcher.py::YFinanceETFDataHandler.fetch_etf_data` lines 76-112 (yfinance `Ticker(...).info` access pattern).

**Existing per-ticker try/except pattern to copy** (`src/etf_fetcher.py` lines 23-43):

```python
def get_finnhub_daily_prices(self, symbol):
    try:
        quote = self.finnhub_client.quote(symbol)
        data = {
            "Symbol": symbol,
            "Current Price": quote.get("c"),
            ...
        }
        return data
    except Exception as e:
        print(f"Error retrieving data for {symbol}: {e}")
        return None
```

**yfinance access pattern to copy** (`src/etf_fetcher.py` lines 76-78):

```python
etf = yf.Ticker(ticker_symbol)
info = etf.info  # for prices use fast_info instead (see RESEARCH Pitfall 5)
```

**Critical adaptation per locked contract** (vs. analog): the existing fetcher returns `None` on failure; the HTTP contract requires a 200 response with per-ticker `error` fields. Adapt the analog by replacing `return None` with `results[ticker] = {"price": None, "error": str(e)[:200]}`. Full pattern in RESEARCH Pitfall 5 lines 578-592.

---

### `api/services/explain_service.py` (service, transform)

**Primary analog:** `src/allocations.py::assign_portfolio_numeric_aum` lines 173-243 (the selection logic where per-ticker metadata is currently *computed but discarded*).
**Secondary analog:** `src/inference.py::recommend` lines 159-235 (the full pipeline to re-run while capturing intermediates).

**The metadata that already exists internally but is not returned** (`src/allocations.py` lines 198-220):

```python
# Inside assign_portfolio_numeric_aum, per pick:
max_aum    = candidate_df['AUM'].max()
aum_power  = np.power(candidate_df['AUM'] / max_aum, 1.5)
if asset_class == 'equity':
    vol = candidate_df['volatility'] + 0.01
    quality = candidate_df['star_rating'] / vol
else:
    quality = candidate_df['star_rating']
weights = aum_power * quality
# These per-ticker numbers (AUM rank, star_rating, volatility, weight) are
# the rationale we want to surface in /api/explain — currently thrown away.
picked_symbol = np.random.choice(candidate_df['symbol'].values, p=weights)
```

**Minimal extension pattern** (per CONTEXT.md Specific Idea #3, Option (a) — "small extension to the function signature"):

```python
# NEW additive helper in src/allocations.py (do NOT modify the existing function).
def assign_portfolio_with_metadata(investor_row, etf_by_class):
    """Same as assign_portfolio_numeric_aum but also returns a metadata dict
    {ticker: {asset_class, aum, aum_rank_in_class, star_rating, volatility,
              candidate_pool_size_at_pick, selection_weight}}.
    """
    # Mirror the body of assign_portfolio_numeric_aum (lines 173-243),
    # appending to a metadata dict at each np.random.choice call.
    ...
    return portfolio, metadata
```

**Stage 1 feature_importances pattern** (sklearn standard; no prior analog in repo — `feature_importances_` is a `RandomForestClassifier` attribute already accessible on `app.state.classifier`):

```python
# Top-3 driver pattern (RESEARCH Don't Hand-Roll line 490):
importances = classifier.feature_importances_   # shape: (19,)
feature_names = preprocessor.get_feature_names_out()
# Multiply by absolute value of the scaled input row → top contributors
contributions = np.abs(X_processed[0]) * importances
top_3_idx = np.argsort(contributions)[::-1][:3]
```

---

### `api/services/etf_service.py` (service, lookup)

**Primary analog:** `src/inference.py::_load_artifacts` lines 84-104 (the CSV load + column-rename + numeric coerce pattern).

**CSV load + clean pattern to copy** (`src/inference.py` lines 84-102):

```python
etf_df = pd.read_csv(ETF_FILE)
etf_features = etf_df.rename(columns={
    'Fund Symbol':                    'symbol',
    'Category':                       'category',
    'Assets Under Management (AUM)':  'AUM',
    'custom_star_rating':             'star_rating',
    'Volatility (Annual STD)':        'volatility',
})
etf_features['AUM']         = pd.to_numeric(etf_features['AUM'],         errors='coerce').fillna(1e9)
etf_features['star_rating'] = pd.to_numeric(etf_features['star_rating'],  errors='coerce').fillna(3)
etf_features['volatility']  = pd.to_numeric(etf_features['volatility'],   errors='coerce').fillna(0.15)
etf_features['asset_class'] = etf_features['category'].apply(get_category_mapping)
```

**404 contract translation** (per `DEC-fastapi-endpoint-surface` — unknown ticker → 404 with `{"detail": "Ticker {ticker} not in supported universe"}`):

```python
row = etf_features[etf_features['symbol'] == ticker]
if row.empty:
    raise HTTPException(404, f"Ticker {ticker} not in supported universe")
```

**Fields the response should include** (per `DEC-fastapi-endpoint-surface`: name, asset class, expense ratio, top holdings, AUM, sector breakdown, Morningstar rating). All present in the CSV — `src/etf_fetcher.py::YFinanceETFDataHandler.fetch_etf_data` lines 89-112 lists the column names that exist in the raw upstream data; `src/morning_star_rating.py` adds `custom_star_rating` lines 23-40.

---

### `api/main.py` (app-entry, startup/lifespan)

**Primary analog:** `pipeline.py` lines 30-101 (orchestrator with fail-fast file existence checks + actionable error messages).
**Secondary analog:** `src/inference.py::_get_artifacts` lines 106-112 (cached-singleton idiom).

**Fail-fast-on-missing-file pattern to copy** (`pipeline.py` lines 76-81 — already used elsewhere in this codebase):

```python
raw_file = 'data/raw/combined_etf_data.csv'
if not os.path.exists(raw_file):
    print(f"❌ Raw ETF data not found: {raw_file}")
    print("   Please run Step 1 first.")
    raise FileNotFoundError(f"Missing {raw_file}")
```

**Cached-singleton idiom to copy** (`src/inference.py` lines 106-112):

```python
_ARTIFACTS = None

def _get_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = _load_artifacts()
    return _ARTIFACTS
```

**Recommended lifespan composition** (RESEARCH Pattern 1 lines 374-385 — *reuse existing cache, no new loaders*):

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.inference import _get_artifacts
    try:
        _get_artifacts()    # triggers module-level cache; fails fast on missing files
    except (FileNotFoundError, OSError) as e:
        raise RuntimeError(
            f"Cannot start API: {e}. Run `python pipeline.py` to regenerate models."
        ) from e
    yield

app = FastAPI(title="AI Financial Advisor API", version="0.1.0", lifespan=lifespan)
```

**Actual artifact paths in this repo** (verified via `ls`):

```
models/
  risk_profile_classifier.pkl
  allocation_regressor.pkl
  etf_count_regressor.pkl
  label_map.json
data/training/
  preprocessor.pkl
data/raw/
  combined_etf_with_morning_star.csv
```

Use these exact paths in lifespan validation.

---

### `api/routers/recommend.py` (controller, request-response)

**Primary analog:** `app.py` lines 96-143 (the only existing call-site of `recommend()`).

**Existing call pattern to copy** (`app.py` lines 110-117):

```python
with st.spinner("Analyzing your profile..."):
    try:
        result = recommend(profile)
    except Exception as e:
        st.error(f"Error running ML pipeline: {e}")
        st.stop()
```

**HTTP adapter pattern** (from RESEARCH Pattern 3 lines 451-466 — no prior FastAPI router in repo):

```python
from fastapi import APIRouter, HTTPException, Request
from api.schemas import InvestorProfile, RecommendResponse
from api.services.recommend_service import run_recommendation

router = APIRouter(prefix="/api", tags=["recommend"])

@router.post("/recommend", response_model=RecommendResponse)
def recommend_endpoint(profile: InvestorProfile, request: Request) -> RecommendResponse:
    try:
        alloc_map = run_recommendation(profile.to_inference_dict())
    except FileNotFoundError as e:
        raise HTTPException(503, f"Backend not ready: {e}. Run `python pipeline.py`.")
    return RecommendResponse(allocation=alloc_map)
```

---

### `tests/test_api_*.py` and `tests/conftest.py`

**Primary analog:** `tests/test_data_validation.py` (entire file — same idiomatic style applies).

**Test style — pytest module-level functions, no class grouping** (`tests/test_data_validation.py` lines 52-67):

```python
def test_investor_happy_path_keeps_all_rows():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(10)
    clean, rpt = validate_investor_data(df)
    assert rpt.n_input == 10
    assert rpt.n_output == 10
```

**Fixture builder pattern to reuse verbatim for API tests** (`tests/test_data_validation.py` lines 32-46) — this is the **canonical valid investor profile** for the entire repo, mirror it in `conftest.py`:

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
```

**Error-path assertion style to copy** (`tests/test_data_validation.py` lines 63-67):

```python
def test_investor_schema_missing_column_raises():
    from src.data_validation import validate_investor_data
    df = _make_good_investor_df(5).drop(columns=['Age'])
    with pytest.raises(ValueError, match="Age"):
        validate_investor_data(df)
```

**For API tests, swap `pytest.raises` for response-code assertions** (RESEARCH Standard Stack — `fastapi.testclient.TestClient`):

```python
def test_recommend_rejects_missing_field(client):
    payload = _make_good_investor_row()
    del payload['Age']
    r = client.post("/api/recommend", json=payload)
    assert r.status_code == 422
    assert "Age" in r.text
```

**`conftest.py` fixtures to add** (no prior conftest exists):

```python
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def good_profile():
    return _make_good_investor_row()   # same dict shape as test_data_validation
```

---

### `src/allocations.py` (MODIFIED — additive helper only)

**Constraint:** Do NOT modify `assign_portfolio_numeric_aum` (lines 173-243) — it's called by `src/inference.py::recommend` and `run_allocations_pipeline`. The existing function and its callers MUST continue to work post-Phase-1 (per `CON-financial-brain-stays-python` reuse mandate).

**Pattern: add a sibling function** that mirrors the body but appends to a metadata dict before each `np.random.choice` call. Re-use `CONFLICT_GROUPS` (lines 11-20) and `get_category_mapping` (lines 126-140) verbatim.

**Existing function body to mirror** (`src/allocations.py` lines 173-243):

```python
def assign_portfolio_numeric_aum(investor_row, etf_by_class):
    portfolio = {}
    asset_mapping = {'equity': 'num_equity_etfs', 'bond': 'num_bond_etfs', 'alternative': 'num_alt_etfs'}
    for asset_class, n_etfs_key in asset_mapping.items():
        full_df = etf_by_class.get(asset_class)
        if full_df is None or len(full_df) == 0: continue
        candidate_df = full_df.copy()
        n_slots = int(investor_row.get(n_etfs_key, 0))
        selected_tickers = []
        for _ in range(n_slots):
            if len(candidate_df) == 0: break
            try:
                max_aum = candidate_df['AUM'].max()
                aum_power = np.power(candidate_df['AUM'] / max_aum, 1.5)
                if asset_class == 'equity':
                    vol = candidate_df['volatility'] + 0.01
                    quality = candidate_df['star_rating'] / vol
                else:
                    quality = candidate_df['star_rating']
                weights = aum_power * quality
                weights = None if weights.sum() == 0 else weights / weights.sum()
                picked_symbol = np.random.choice(candidate_df['symbol'].values, p=weights)
                selected_tickers.append(picked_symbol)
                candidate_df = candidate_df[candidate_df['symbol'] != picked_symbol]
                for group in CONFLICT_GROUPS:
                    if picked_symbol in group:
                        siblings_to_ban = [s for s in group if s != picked_symbol]
                        candidate_df = candidate_df[~candidate_df['symbol'].isin(siblings_to_ban)]
            except Exception:
                ...
        portfolio[asset_class] = selected_tickers
    return portfolio
```

Per-pick metadata to capture in the new helper: `asset_class`, `aum_at_pick`, `star_rating_at_pick`, `volatility_at_pick` (equity only), `selection_weight` (normalized), `candidate_pool_size_at_pick`, `aum_rank_in_pool`, `conflict_group_banned` (list of siblings removed).

---

## Shared Patterns

### Logging
**Source:** stdlib `logging` (no prior `logger = logging.getLogger(__name__)` convention exists in repo — current code uses `print()`).
**Apply to:** All `api/services/*.py` files.
**Recommendation:** Use stdlib `logging` (RESEARCH Standard Stack lock-in). The existing `print()` style in `pipeline.py` / `src/etf_fetcher.py` is fine for CLI scripts but not for an HTTP service.

```python
import logging
logger = logging.getLogger(__name__)
```

### Path constants
**Source:** `src/inference.py` lines 42-44:
```python
MODEL_DIR    = 'models'
TRAINING_DIR = 'data/training'
ETF_FILE     = 'data/raw/combined_etf_with_morning_star.csv'
```
**Apply to:** Any file that needs to reference model/data paths.
**Recommendation:** Import these from `src.inference` rather than redefining — single source of truth.

### Singleton/cache idiom
**Source:** `src/inference.py` lines 106-112 (`_ARTIFACTS = None` + `_get_artifacts()` guard).
**Apply to:** `api/main.py` lifespan (reuses `_get_artifacts()` directly), `api/services/etf_service.py` if it needs a process-wide cached DataFrame.

### Error handling — fail-fast with actionable message
**Source:** `pipeline.py` lines 76-81, `src/inference.py` lines 119-121.
**Apply to:** Lifespan, all service-layer functions.
**Pattern:** Raise a clear exception that names the missing artifact AND the command to regenerate (`python pipeline.py`).

### PascalCase ↔ lowercase key translation
**Source:** `src/inference.py` lines 58-65 (`_INFERENCE_TO_CSV`).
**Apply to:** `api/schemas.py::InvestorProfile.to_inference_dict()`.
**Critical:** HTTP schema is PascalCase (`Age`, `Income`, `InvestmentHorizon`, `RiskTolerance`, `ExperienceYears`), but `recommend()` expects lowercase (`age`, `income`, `horizon`, `risk_tolerance`, `experience`). The other 7 fields (`Gender`, `Education`, `MaritalStatus`, `HouseholdSize`, `InvestmentGoal`, `InvestmentCapital`, `FinancialInvolvement`) stay PascalCase in both surfaces.

### Try/except per-ticker for partial-failure batches
**Source:** `src/etf_fetcher.py::FinnhubETFDataHandler.get_finnhub_daily_prices` lines 23-43.
**Apply to:** `api/services/prices_service.py`.
**Adaptation:** Replace `return None` on failure with `{"price": None, "error": str(e)[:200]}` so the locked-contract per-ticker envelope works.

### Test fixture style
**Source:** `tests/test_data_validation.py` lines 32-46 (`_make_good_investor_row`).
**Apply to:** All `tests/test_api_*.py` + new `tests/conftest.py`.
**Pattern:** Module-private builders (`_make_*`) with `**overrides`, no class-based grouping.

---

## No Analog Found

No file lacks an analog — every new file has either a role-match (existing Python file with the same job) or a data-flow-match (existing Python file with the same I/O pattern). For the **FastAPI-specific scaffolding** (`lifespan`, `APIRouter`, `Depends`, `TestClient`), the repo has **no prior FastAPI code**, so those patterns come from RESEARCH.md (Pattern 1, 2, 3) rather than the codebase. This is expected and called out in the task brief.

---

## Metadata

**Analog search scope:** `src/`, `app.py`, `pipeline.py`, `tests/`, `README.md`, `requirements.txt`
**Files scanned (full read):** 7 (`src/inference.py`, `src/allocations.py`, `src/etf_fetcher.py`, `src/morning_star_rating.py`, `src/validation_rules.py`, `app.py`, `pipeline.py`, `tests/test_data_validation.py`)
**Files inspected (head/ls only):** `README.md`, `requirements.txt`, `models/`, `data/training/`, `data/raw/`
**Codebase has no prior FastAPI code:** confirmed (no `api/`, no `app/`, no `main.py` referencing `FastAPI`).
**Pattern extraction date:** 2026-05-19

---

## PATTERN MAPPING COMPLETE
