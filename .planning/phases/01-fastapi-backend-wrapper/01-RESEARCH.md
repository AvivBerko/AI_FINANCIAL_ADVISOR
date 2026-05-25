# Phase 1: FastAPI Backend Wrapper - Research

**Researched:** 2026-05-19
**Domain:** Python HTTP service wrapping an existing sklearn-based ML inference pipeline + yfinance live-price layer + ETF metadata + Morningstar-style ratings
**Confidence:** HIGH (codebase fully inspected; FastAPI/Pydantic/yfinance behavior verified)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Endpoint surface (`DEC-fastapi-endpoint-surface`):** Exactly 4 endpoints in v1, no more, no fewer:
- `POST /api/recommend` — 12 investor fields → `{ticker: weight}` map summing to 1.0
- `POST /api/current-prices` — list of tickers → live yfinance prices, per-ticker error surfacing
- `POST /api/explain` — 12 fields + allocation map → structured rationale (3 sections)
- `GET /api/etfs/{ticker}` — ETF metadata, 404 on unknown ticker

**Investor profile schema (`DEC-investor-profile-schema-frozen`):** EXACT 12 fields, EXACT names (PascalCase as listed), EXACT enum literals (e.g., `"High school"`, `"Child education"`):
- `Age` (int), `Gender` (Male|Female), `Education` (High school|Bachelor|Master), `MaritalStatus` (Single|Married), `HouseholdSize` (int), `Income` (number), `InvestmentGoal` (Child education|Retirement|Home purchase|Other), `InvestmentHorizon` (int years), `InvestmentCapital` (number), `RiskTolerance` (Low|Medium|High), `FinancialInvolvement` (Low|Medium|High), `ExperienceYears` (int)

**Allocation output (`CON-allocated-assets-jsonb-shape`):** `{ticker: weight}` JSON object, weights 0.0-1.0 floats summing to **exactly 1.0** within float tolerance. **Server MUST renormalize before returning** if pipeline produces drift.

**Architectural boundary (`CON-financial-brain-stays-python`):** ALL ML inference, yfinance calls, ETF metadata happen in Python. Wrapper is a leaf node; never calls out to other services.

**Reuse vs rewrite:** Wrapper is ADDITIVE. `python pipeline.py` and `src/inference.py` MUST continue to work post-implementation. `src/inference.py` already exists (verified — see Architecture Patterns).

**Model loading:** Load `models/*.pkl` at FastAPI startup (lifespan event). If missing, fail fast pointing at `python pipeline.py`. Service-layer cache, never per-request.

**Explain endpoint:** Structured object (not free-text), 3 sections: (1) risk-class with feature contributions, (2) equity/bond split reasoning, (3) per-ticker selection reasoning sourced from `src/allocations.py` selection logic.

**Repo layout:** Wrapper lives at repo root alongside `pipeline.py` and `src/`. `frontend/` is RESERVED for Next.js (Phase 3+) — DO NOT touch.

**Error contracts:**
- 422 (FastAPI default) for invalid 12-field profile
- 404 with `{"detail": "Ticker {ticker} not in supported universe"}` for unknown ETF
- 200 with per-ticker error fields on `/api/current-prices` partial failures (NOT 5xx)
- 503 with actionable message on model unavailable / missing pickle

**Local-run:** `uvicorn api.main:app --reload --port 8000` from repo root. Default port **8000**. README or new `BACKEND.md` documents the command. **No CORS** (no frontend caller yet; CORS added in Phase 3).

### Claude's Discretion

- FastAPI directory layout (`api/` vs `app/` vs root-level `app.py`)
- Pydantic split between request/response models and shared types
- Whether `/api/explain` extends the `/api/recommend` input model or has a dedicated model
- Service-layer abstraction shape (functions vs classes)
- Logging library (stdlib `logging` recommended)
- Test framework (pytest + FastAPI TestClient conventional)
- Whether to add `/health` or `/api/health` (recommended but not required)
- Exact structured schema inside the 3-section explain rationale (JSON-serializable, 3 sections)
- Optional `seed` parameter on `/api/recommend` for reproducibility (recommendation: include it, default None → stochastic)

### Deferred Ideas (OUT OF SCOPE)

- Authentication / API keys (Phase 1 is local-only, Next.js is the only caller)
- Rate limiting on yfinance
- Dockerization / production deploy
- Caching `/api/etfs/{ticker}` responses (yfinance is cached client-side)
- WebSocket/SSE for `/api/recommend` progress
- CORS configuration (Phase 3)
- A `/api/portfolio-value` endpoint (frontend composes it from `/api/current-prices` + `InvestmentCapital`)
- Improving Stage 1 classifier accuracy (tracked as `BACKEND-V2-02`)
- Containerization, observability beyond stdlib logging, changing model pickle format
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| BACKEND-01 | FastAPI server runnable from origin repo wraps the 4-stage ML pipeline as HTTP endpoints | Architecture Patterns (project structure, lifespan loading); Standard Stack |
| BACKEND-02 | `POST /api/recommend` accepts 12 frozen fields, returns ticker→weight summing to 1.0 | Pydantic models for 12 fields (Standard Stack), feature-engineering parity via persisted `preprocessor.pkl` (Common Pitfalls #3), renormalization step (Code Examples) |
| BACKEND-03 | `POST /api/current-prices` accepts ticker list, returns live yfinance prices | yfinance integration pattern (Code Examples), partial-failure envelope (Common Pitfalls #6) |
| BACKEND-04 | yfinance stays in Python; Next.js never imports it | Scope Fence — verified by Success Criterion #7; implementation discipline |
| BACKEND-05 | `POST /api/explain` accepts 12 fields + allocation, returns 3-section structured rationale | Per-ticker metadata capture (Code Examples / Don't Hand-Roll); risk-class feature contributions (Standard Stack — sklearn `feature_importances_`) |
| BACKEND-06 | `GET /api/etfs/{ticker}` returns metadata, 404 on unknown | ETF universe lookup (Code Examples), wraps `src/etf_fetcher.py` + `src/morning_star_rating.py` (Architecture Patterns) |
</phase_requirements>

---

## Summary

The project already has a clean Python inference surface: `src/inference.py::recommend(user: dict) -> dict` exists, runs the full 4-stage pipeline, and is currently consumed by the Streamlit `app.py`. The persisted `data/training/preprocessor.pkl` (a `sklearn.compose.ColumnTransformer`) handles the 12→19 feature engineering — **the API does not need to hand-replicate one-hot encoding**. This is the single most important codebase fact for Phase 1: it collapses what would have been the trickiest task (BACKEND-02 feature parity) into "call the existing function."

The wrapper is therefore a **thin adapter**, not a translation layer. The four endpoints map almost 1:1 onto existing functions: `recommend()` from `src/inference.py`, `yf.Ticker(symbol).fast_info` for live prices, the cached `data/raw/combined_etf_with_morning_star.csv` + `calculate_star_ratings()` for ETF metadata, and a small new helper that re-runs the pipeline while capturing selection metadata for `/api/explain`.

The real risks are: (1) **sklearn version drift** — models were pickled with sklearn 1.3.x but venv now has 1.8.0, which raises `InconsistentVersionWarning` and can produce silent misbehavior; (2) **missing artifacts** — `models/*.pkl`, `data/training/preprocessor.pkl`, and `data/raw/combined_etf_with_morning_star.csv` are all gitignored and must exist at startup; (3) **stochasticity in Stage 4 ETF selection** — `np.random.choice` makes `/api/recommend` non-deterministic by default, which complicates testing.

**Primary recommendation:** Create a top-level `api/` package with `api/main.py` (FastAPI app + lifespan), `api/schemas.py` (Pydantic v2 models), `api/services/` (one module per endpoint), and `api/deps.py` (DI helpers for cached models). Reuse `src/inference.py::recommend()` verbatim for `/api/recommend`. For `/api/explain`, write a small `recommend_with_explanation()` adapter that re-runs the pipeline once and captures per-ticker selection metadata in the process. Pin `numpy.random.seed()` per-request when a `seed` is supplied. Pin sklearn at the version models were trained on (1.3.x or whatever was actually used) in `requirements.txt` to silence `InconsistentVersionWarning`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| HTTP request/response shape | API / Backend (FastAPI) | — | Single owner of contract validation; downstream tiers consume |
| 12-field validation (types + enums) | API / Backend (Pydantic) | — | Validation MUST happen at HTTP edge; never trust frontend |
| Feature engineering (12 → 19) | API / Backend (sklearn `ColumnTransformer` loaded from `data/training/preprocessor.pkl`) | — | Owns by definition; transforming raw input is server's job |
| 4-stage ML inference | API / Backend (`src/inference.py::recommend`) | — | Per `CON-financial-brain-stays-python` — all ML in Python |
| Live ETF price retrieval (yfinance) | API / Backend (yfinance Python lib) | — | Per `CON-financial-brain-stays-python` — yfinance never crosses the boundary |
| ETF metadata lookup | API / Backend (cached CSV + `src/etf_fetcher.py`) | — | Cached file is the supported-universe oracle |
| Morningstar-style star rating | API / Backend (`src/morning_star_rating.py`) | — | Existing implementation, reused as-is |
| Allocation renormalization to exactly 1.0 | API / Backend | — | Per `CON-allocated-assets-jsonb-shape` — server-side guarantee |
| Per-ticker selection rationale capture | API / Backend (new adapter around `src/allocations.py`) | — | Existing selection logic doesn't expose metadata; wrapper extends it |
| Partial-failure envelope per ticker | API / Backend (`/api/current-prices`) | — | Per locked error contract — server decides per-ticker fate |
| Composing `get_portfolio_value` | Frontend (Next.js, Phase 5) | API / Backend (`/api/current-prices`) | Explicit deferral in CONTEXT.md — frontend multiplies prices × `InvestmentCapital`, no 5th endpoint |
| Model pickle loading | API / Backend (FastAPI lifespan) | — | One-time cost amortized across all requests |

---

## Standard Stack

### Core

| Library | Version (verified) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `fastapi` | 0.136.1 (latest) | HTTP framework + Pydantic-driven request validation + OpenAPI docs | Standard FastAPI choice; project memory confirms intent |
| `uvicorn[standard]` | 0.47.0 (latest) | ASGI server; `[standard]` adds `httptools`, `uvloop`, `watchfiles` for `--reload` | Tiangolo's recommended runner; CONTEXT specifies `uvicorn` command |
| `pydantic` | 2.12.5 (already installed) | Request/response model validation | **Already in venv** — FastAPI 0.100+ defaults to v2 |
| `joblib` | 1.5.3 (already installed) | Pickled model loading (used by `src/inference.py::_load_artifacts`) | Already used by inference.py |
| `scikit-learn` | 1.8.0 (installed) → **PIN to training version** | RandomForestClassifier, ColumnTransformer | See Common Pitfalls #1 — version drift landmine |
| `pandas` | 2.3.3 (already installed) | DataFrame construction for ML input, ETF CSV loading | Already used everywhere |
| `numpy` | (installed) | Stage 4 stochastic selection | Already used |
| `yfinance` | 1.2.0 (already installed) | Live ETF prices | Already used by `src/etf_fetcher.py` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `httpx` | 0.28.1 (already installed) | Test client for endpoint tests (`fastapi.testclient.TestClient` wraps it) | Test code only |
| `pytest` | 9.0.3 (already installed) | Test runner | Already configured (`tests/` dir + 24 tests for `data_validation.py`) |
| `python-dotenv` | (already in requirements.txt) | Load `.env` (Finnhub key not needed at runtime but `src/etf_fetcher.py` loads it on import) | Loaded transitively via existing modules |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| FastAPI | Flask + Flask-Pydantic | Flask would also work but lacks built-in async, OpenAPI generation, and dependency injection. CONTEXT.md already locks FastAPI. |
| Pydantic v2 | Pydantic v1 | v1 syntax differs (`@validator` vs `@field_validator`, `.dict()` vs `.model_dump()`). v2 is already installed; no reason to downgrade. |
| `uvicorn` | `hypercorn`, `granian` | Tiangolo-recommended pairing is uvicorn; CONTEXT.md specifies it. |
| SHAP for feature attribution | sklearn `feature_importances_` × scaled input | SHAP adds heavy dependency (~50MB binary + `numba`) for an academic project where the bar is "structured rationale, not statistically rigorous attribution". See research focus #7 + Code Examples. |
| Re-implement preprocessing | Load persisted `preprocessor.pkl` | `data/training/preprocessor.pkl` IS the source of truth — re-implementing is bug-prone. **Use the persisted artifact.** |

**Installation (additive — only fastapi + uvicorn are new):**

```bash
venv/bin/pip install "fastapi>=0.115" "uvicorn[standard]>=0.30"
# Then freeze:
venv/bin/pip freeze > requirements.txt
# Or, recommended: edit requirements.txt to add the two pinned lines explicitly.
```

**Version verification (PyPI, 2026-05-19):**
- `fastapi==0.136.1` — verified via `pip index versions fastapi`
- `uvicorn==0.47.0` — verified via `pip index versions uvicorn`
- `httpx==0.28.1` — already installed; verified via `pip index versions httpx`
- `pydantic==2.12.5` — already installed
- `pytest==9.0.3` — already installed

---

## Package Legitimacy Audit

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| fastapi | PyPI | 7+ yrs (since 2018) | ~70M/mo | github.com/fastapi/fastapi (>70k stars) | `[ASSUMED OK]` (slopcheck CLI not on PATH in shell) | Approved — well-known mainstream |
| uvicorn | PyPI | 7+ yrs (since 2018) | ~50M/mo | github.com/encode/uvicorn (>8k stars) | `[ASSUMED OK]` | Approved — encode org maintains httpx/starlette too |
| pydantic | PyPI | 8+ yrs | ~200M/mo | github.com/pydantic/pydantic (>20k stars) | `[ASSUMED OK]` | Already installed |
| httpx | PyPI | 6+ yrs | ~100M/mo | github.com/encode/httpx | `[ASSUMED OK]` | Already installed |

**Packages removed due to slopcheck `[SLOP]` verdict:** none
**Packages flagged as suspicious `[SUS]`:** none

`slopcheck` is not currently on this machine's PATH and pip install is blocked (sandbox), so packages are tagged `[ASSUMED OK]` per the protocol. All four are blue-chip Python libraries with millions of monthly downloads and >5-year history — the verification gap is process, not risk. Planner does NOT need to insert a `checkpoint:human-verify` task for these specific packages.

---

## Architecture Patterns

### System Architecture Diagram

```
                      ┌──────────────────────────────────────┐
                      │  Next.js orchestrator (Phase 3+)     │
                      │  - Onboarding form                   │
                      │  - Dashboard                         │
                      │  - Vercel AI SDK chat (7 tools)      │
                      └──────────────┬───────────────────────┘
                                     │
                                     │ HTTP (localhost:8000, no CORS yet)
                                     │
                     ┌───────────────▼──────────────────┐
                     │  FastAPI app  (api/main.py)      │
                     │                                  │
                     │  lifespan:  load 4 pickles,      │
                     │             load ETF CSV         │
                     │             stash in app.state   │
                     │                                  │
                     │   ┌─────────────────────────┐    │
                     │   │   Pydantic validation   │    │
                     │   │   (12 fields, enums,    │    │
                     │   │    ranges) — 422 on bad │    │
                     │   └────────────┬────────────┘    │
                     │                │                 │
                     │   ┌────────────┴─────────────┐   │
                     │   │   Routers (per endpoint) │   │
                     │   │   /api/recommend         │──┐│
                     │   │   /api/current-prices    │  ││
                     │   │   /api/explain           │  ││
                     │   │   /api/etfs/{ticker}     │  ││
                     │   └──────────────────────────┘  ││
                     └─────────────────────────────────│┘
                                                       │
                                                       ▼
                     ┌─────────────────────────────────────┐
                     │  Service layer (api/services/)      │
                     │  - recommend_service (wraps         │
                     │    src/inference.py::recommend)     │
                     │  - prices_service (yfinance.Ticker  │
                     │    .fast_info, per-ticker try/except)│
                     │  - explain_service (re-runs pipeline│
                     │    while capturing metadata)        │
                     │  - etf_service (CSV lookup +        │
                     │    src/morning_star_rating)         │
                     └────────┬───────────────────────┬────┘
                              │                       │
              ┌───────────────▼──┐         ┌──────────▼──────────┐
              │  src/inference   │         │  yfinance.Ticker     │
              │  src/allocations │         │  .fast_info          │
              │  src/etf_fetcher │         │  (only for live      │
              │  src/morning_star│         │   prices; ETF metadata
              │                  │         │   read from CSV)     │
              └────────┬─────────┘         └─────────────────────┘
                       │
        ┌──────────────┼───────────────────────┐
        ▼              ▼                       ▼
  models/*.pkl   data/training/      data/raw/
  - risk_clf     preprocessor.pkl    combined_etf_
  - alloc_reg    (ColumnTransformer) with_morning_
  - etf_count_reg                    star.csv
  + label_map.json                   (supported universe)
```

### Component Responsibilities

| Path | Responsibility |
|------|----------------|
| `api/main.py` | FastAPI app instantiation, lifespan event (load all artifacts), router includes, OpenAPI metadata (title, version), `/docs` enabled |
| `api/schemas.py` (or `api/schemas/`) | All Pydantic v2 models: `InvestorProfile`, `RecommendResponse`, `CurrentPricesRequest`/`Response`, `ExplainRequest`/`Response`, `EtfDetail`; shared `Literal` enums for the 6 categorical fields |
| `api/routers/recommend.py` | `POST /api/recommend` route; calls service, applies renormalization to sum=1.0 |
| `api/routers/prices.py` | `POST /api/current-prices` route; per-ticker try/except partial-failure envelope |
| `api/routers/explain.py` | `POST /api/explain` route; calls `explain_service` |
| `api/routers/etfs.py` | `GET /api/etfs/{ticker}` route; raises `HTTPException(404)` on unsupported ticker |
| `api/services/recommend_service.py` | Thin wrapper: maps PascalCase request fields → `src/inference.py` lowercase keys → calls `recommend()` → flattens to `{ticker: weight}` |
| `api/services/prices_service.py` | `yf.Ticker(t).fast_info` per ticker with timeout + exception handling; returns `{ticker: {price, currency, error?}}` |
| `api/services/explain_service.py` | Re-runs ML pipeline (or accepts pre-computed result); captures Stage 1 `feature_importances_`, Stage 2 raw output, Stage 4 per-ticker selection metadata (AUM rank, asset class, candidate pool size) by calling a small **new** adapter `assign_portfolio_with_metadata()` in `src/allocations.py` |
| `api/services/etf_service.py` | Loads `data/raw/combined_etf_with_morning_star.csv` at startup; returns metadata for a single ticker; raises `KeyError`/`LookupError` translated to 404 |
| `api/deps.py` | FastAPI dependency-injection helpers: `get_models(request: Request) -> dict`, `get_etf_universe(request: Request) -> pd.DataFrame` — pull from `app.state` |

### Recommended Project Structure

```
AI_FINANCIAL_ADVISOR/
├── api/                              # NEW — wrapper code lives here
│   ├── __init__.py
│   ├── main.py                       # FastAPI app + lifespan
│   ├── schemas.py                    # Pydantic v2 models (all in one file is fine; ~150 lines)
│   ├── deps.py                       # Dependency injection helpers
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── recommend.py
│   │   ├── prices.py
│   │   ├── explain.py
│   │   └── etfs.py
│   └── services/
│       ├── __init__.py
│       ├── recommend_service.py
│       ├── prices_service.py
│       ├── explain_service.py
│       └── etf_service.py
├── src/                              # UNCHANGED (with one MINIMAL extension to allocations.py)
│   ├── allocations.py                # + new assign_portfolio_with_metadata() helper (additive)
│   ├── inference.py                  # UNCHANGED — reused as-is
│   ├── etf_fetcher.py                # UNCHANGED
│   ├── morning_star_rating.py        # UNCHANGED
│   ├── data_generation.py            # UNCHANGED
│   ├── prepare_data.py               # UNCHANGED
│   ├── train_models.py               # UNCHANGED
│   ├── validation_rules.py           # UNCHANGED — Pydantic enums can reference these directly
│   └── data_validation.py            # UNCHANGED
├── tests/
│   ├── test_data_validation.py       # EXISTING — untouched
│   ├── test_api_recommend.py         # NEW
│   ├── test_api_prices.py            # NEW
│   ├── test_api_explain.py           # NEW
│   ├── test_api_etfs.py              # NEW
│   └── conftest.py                   # NEW — TestClient fixture, model-mock fixtures
├── pipeline.py                       # UNCHANGED
├── app.py                            # UNCHANGED (Streamlit still works)
├── requirements.txt                  # + fastapi, uvicorn[standard]
├── BACKEND.md                        # NEW — run instructions, endpoint summary, regeneration command
├── README.md                         # MINOR EDIT — point at BACKEND.md
└── frontend/                         # RESERVED — DO NOT TOUCH (Phase 3+)
```

**Rationale for `api/` (vs `app/` or root-level `app.py`):**
1. `app.py` is **already taken** by Streamlit. Adding another `app.py` for FastAPI would collide.
2. `app/` package name conflicts with FastAPI tutorials' default but is also a common convention; using `api/` makes the *intent* obvious (it's the HTTP surface) and avoids confusion.
3. The `api/main.py` → `uvicorn api.main:app` command pattern is unambiguous and matches the CONTEXT.md example verbatim.

### Pattern 1: FastAPI Lifespan for One-Time Artifact Loading

**What:** Use `@asynccontextmanager` to load pickles, the ETF CSV, and the preprocessor exactly once at server start; expose via `app.state`.

**When to use:** Whenever inference cost is dominated by `joblib.load()` (~1-2s per pickle here) — request-time loading would crush latency.

**Example (verified pattern from FastAPI docs):**
```python
# api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib, json
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("models")
TRAINING_DIR = Path("data/training")
ETF_CSV = Path("data/raw/combined_etf_with_morning_star.csv")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: validate everything exists, then load ---
    required = [
        MODEL_DIR / "risk_profile_classifier.pkl",
        MODEL_DIR / "allocation_regressor.pkl",
        MODEL_DIR / "etf_count_regressor.pkl",
        MODEL_DIR / "label_map.json",
        TRAINING_DIR / "preprocessor.pkl",
        ETF_CSV,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        # Fail-fast with actionable error — surfaces as 503 on every request
        # via the dependency, but also crashes uvicorn on boot which is the
        # clearer signal for a dev environment.
        raise RuntimeError(
            f"Missing required artifacts: {missing}. "
            f"Run `python pipeline.py` from the repo root to regenerate."
        )

    app.state.classifier = joblib.load(MODEL_DIR / "risk_profile_classifier.pkl")
    app.state.alloc_regressor = joblib.load(MODEL_DIR / "allocation_regressor.pkl")
    app.state.count_regressor = joblib.load(MODEL_DIR / "etf_count_regressor.pkl")
    app.state.preprocessor = joblib.load(TRAINING_DIR / "preprocessor.pkl")
    with open(MODEL_DIR / "label_map.json") as f:
        app.state.label_info = json.load(f)
    app.state.etf_universe = pd.read_csv(ETF_CSV)
    yield
    # --- Shutdown: nothing to release for sklearn pickles ---

app = FastAPI(
    title="AI Financial Advisor API",
    version="0.1.0",
    lifespan=lifespan,
)
```

**Alternative (simpler — reuse existing `src/inference.py` caching):**
`src/inference.py` already memoizes via `_get_artifacts()` at module level. The lifespan can simply call `_get_artifacts()` once to warm the cache and *also* surface the explicit "missing files → 503" error:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.inference import _get_artifacts
    try:
        _get_artifacts()  # Triggers the module-level cache; fail-fast on missing
    except (FileNotFoundError, OSError) as e:
        raise RuntimeError(
            f"Cannot start API: {e}. Run `python pipeline.py` to regenerate models."
        ) from e
    yield
```

**Recommendation:** Use the second pattern (reuses existing cache). It also automatically handles the ETF DataFrame because `_load_artifacts` already constructs `etf_by_class` from the CSV. Only add a separate `app.state.etf_universe` if `/api/etfs/{ticker}` needs the **full** CSV (with columns not present in the `etf_by_class` dict) — see Code Examples.

### Pattern 2: Pydantic v2 with Literal Enums + Field Validation

**What:** Use `typing.Literal` for the 6 enum fields (faster than `Enum`, surfaces exact strings in OpenAPI schema) + `Field(..., ge=, le=)` for numeric ranges matching `src/validation_rules.NUMERIC_RANGES`.

**When to use:** Always, for the 12-field profile. The frozen enum strings have spaces ("High school", "Child education") — a `StrEnum` subclass works but `Literal` is leaner.

**Example:**
```python
# api/schemas.py
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

GenderLit = Literal["Male", "Female"]
EducationLit = Literal["High school", "Bachelor", "Master"]
MaritalStatusLit = Literal["Single", "Married"]
InvestmentGoalLit = Literal["Child education", "Retirement", "Home purchase", "Other"]
RiskToleranceLit = Literal["Low", "Medium", "High"]
FinancialInvolvementLit = Literal["Low", "Medium", "High"]

class InvestorProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")  # 422 on unknown keys

    Age: int = Field(..., ge=18, le=100)
    Gender: GenderLit
    Education: EducationLit
    MaritalStatus: MaritalStatusLit
    HouseholdSize: int = Field(..., ge=1, le=10)
    Income: float = Field(..., ge=0, le=10_000_000)
    InvestmentGoal: InvestmentGoalLit
    InvestmentHorizon: int = Field(..., ge=1, le=50)
    InvestmentCapital: float = Field(..., ge=0, le=100_000_000)
    RiskTolerance: RiskToleranceLit
    FinancialInvolvement: FinancialInvolvementLit
    ExperienceYears: int = Field(..., ge=0, le=80)

    def to_inference_dict(self) -> dict:
        """Map PascalCase HTTP fields → lowercase keys expected by src/inference.py::recommend."""
        return {
            "age": self.Age,
            "Gender": self.Gender,
            "Education": self.Education,
            "MaritalStatus": self.MaritalStatus,
            "HouseholdSize": self.HouseholdSize,
            "income": self.Income,
            "InvestmentGoal": self.InvestmentGoal,
            "horizon": self.InvestmentHorizon,
            "InvestmentCapital": self.InvestmentCapital,
            "risk_tolerance": self.RiskTolerance,
            "FinancialInvolvement": self.FinancialInvolvement,
            "experience": self.ExperienceYears,
        }
```

**Note (BACKEND-02 enum-exactness landmine):** `src/validation_rules.VALID_VALUES['MaritalStatus']` is `['Married', 'Single', 'Divorced', 'Widowed']` (4 values) but `DEC-investor-profile-schema-frozen` and `app.py` Streamlit select are `Single | Married` (2 values). **The locked schema wins** — the Pydantic Literal must be `Literal["Single", "Married"]`. Existing data may contain Divorced/Widowed but the API surface does not accept them. The planner should not "fix" this mismatch in either direction in Phase 1 — surface it as a discovered inconsistency in the implementation notes.

### Pattern 3: Per-Endpoint Routers + Service-Layer Indirection

**What:** Each endpoint is a thin router calling into a service module. Routers do HTTP/validation; services do work.

**When to use:** Standard FastAPI hygiene; makes unit-testing services without TestClient trivial.

**Example:**
```python
# api/routers/recommend.py
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

### Anti-Patterns to Avoid

- **Reimplementing the 12→19 feature engineering by hand in the API layer.** The persisted `data/training/preprocessor.pkl` (a `ColumnTransformer`) IS the transform. Load it and call `.transform(df)`. Hand-rolling `pd.get_dummies` will produce mismatched dummy columns (see Common Pitfalls #3).
- **Per-request `joblib.load`.** Each pickle is 2-10MB; loading per request would add ~2s latency. Use lifespan + `app.state`.
- **`async def` route handlers that call yfinance.** yfinance is synchronous and uses `requests` under the hood. Either keep handlers `def` (FastAPI runs them in a threadpool) or `async def` and wrap calls in `asyncio.to_thread`. Synchronous handlers are simpler and totally adequate here.
- **Returning the full sklearn `RandomForestClassifier` representation in `/api/explain`.** It's huge and not JSON-serializable. Extract only `feature_importances_` × scaled input → top-N drivers.
- **Trusting yfinance success.** Even valid tickers occasionally return NaN/None. Always validate the response and surface `error` per ticker.
- **Calling `pd.read_csv` per-request for the ETF universe.** Load once at lifespan.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 12→19 feature engineering | Hand-coded `pd.get_dummies` + `StandardScaler` re-fit | `joblib.load("data/training/preprocessor.pkl").transform(df)` | `ColumnTransformer` is persisted at training time; re-fitting on a single row of input produces wrong scaling and missing dummy columns |
| 12-field validation | Custom `if/else` on field values | Pydantic v2 `Literal` types + `Field(ge=, le=)` | FastAPI auto-generates 422 with field-level errors; matches `DEC-investor-profile-schema-frozen` exactly |
| Live ETF prices | `requests.get("https://finance.yahoo.com/...")` HTML scraping | `yf.Ticker(symbol).fast_info["last_price"]` | yfinance handles Yahoo cookie crumb auth, retries, and caching; rolling your own breaks within months |
| Star ratings | New rating logic | `src.morning_star_rating.calculate_star_ratings()` | Already exists, already tested in practice (5-star bucketing logic at lines 28-43) |
| ETF selection logic | New selection algorithm | `src.allocations.assign_portfolio_numeric_aum()` | Already implements AUM-weighted sampling + conflict resolution (lines 173-243). Extend it minimally to also return metadata (see Code Examples). |
| Risk-profile classification | Re-implement decision tree | `src.inference.recommend()` | Calls all 3 sklearn models + label map → already returns `risk_profile` |
| Allocation renormalization | Custom `for ticker in alloc:` logic | `total = sum(values); {k: v/total for k, v in alloc.items()}` (single line) | Sufficient and easy to verify in tests |
| Local SHAP-style explanations | `shap` library | sklearn `feature_importances_` × abs(scaled_input_value) → top 3-5 | Adds heavy dependency (~50MB binary) for academic project where bar is "structured rationale, not statistically rigorous". See Standard Stack alternatives. |

**Key insight:** This is a **wrapper, not a rewrite**. The temptation to "improve" things while wrapping must be resisted. Every existing function (`recommend`, `assign_portfolio_numeric_aum`, `calculate_star_ratings`) gets called, not rebuilt. The one minimal extension allowed by CONTEXT.md is capturing per-ticker selection metadata for `/api/explain` (CONTEXT.md "Specific Ideas" bullet 3 endorses Option (a): "capturing it via a small extension to the function signature").

---

## Runtime State Inventory

Phase 1 is **purely additive** — no rename, refactor, migration, or data mutation. Section omitted by design.

**Verified:**
- No string is being renamed.
- No data is being migrated.
- No OS-level state is being modified.
- No env var names are changing (`.env` has `FINNHUB_API_KEY` which is unused at API runtime — `etf_fetcher.py`'s `FinnhubETFDataHandler` is for bulk data fetching, not live API).
- No package installs change existing artifact names.

---

## Common Pitfalls

### Pitfall 1: sklearn Version Drift on Pickle Load

**What goes wrong:** `joblib.load("models/risk_profile_classifier.pkl")` raises `InconsistentVersionWarning` (or worse, silently produces incorrect predictions) because the venv currently has scikit-learn 1.8.0 but `requirements.txt` says `scikit-learn>=1.3.0` and the pickles were likely created with 1.3.x or 1.4.x (file timestamps: May 12 2026; check `evaluation_reports/` for the actual version if recorded).

**Why it happens:** sklearn explicitly states: *"There are no supported ways to load a model trained with a different version of scikit-learn. Models saved using one version might load in other versions, however this is entirely unsupported and inadvisable."* — sklearn 1.8 docs.

**How to avoid:**
1. **Pin sklearn to the version that produced the pickles.** Edit `requirements.txt` to `scikit-learn==1.3.X` (or whatever version is in the pickle's metadata).
2. **OR regenerate the pickles** by running `python pipeline.py` before starting the API for the first time. Documented in `BACKEND.md`.
3. **Catch `InconsistentVersionWarning` at startup** and log a clear "models may misbehave; regenerate via python pipeline.py" message:
   ```python
   import warnings
   from sklearn.exceptions import InconsistentVersionWarning
   warnings.filterwarnings("error", category=InconsistentVersionWarning)
   try:
       app.state.classifier = joblib.load(...)
   except InconsistentVersionWarning as e:
       logger.warning(f"Model version mismatch: {e}. Predictions may be unreliable.")
       app.state.classifier = joblib.load(...)  # Load anyway
   ```

**Warning signs:** `InconsistentVersionWarning` in startup logs; `risk_profile` predictions that differ between successive calls with identical input (when seed is fixed).

### Pitfall 2: Missing Artifacts at Startup

**What goes wrong:** Per `.gitignore`, ALL of these are not in git: `models/*.pkl`, `data/`, `evaluation_reports/`. A fresh clone has nothing. FastAPI lifespan crashes with cryptic `FileNotFoundError`.

**Why it happens:** The training pipeline produces gitignored artifacts; the API consumes them. There's no automatic regeneration trigger.

**How to avoid:**
- Lifespan validates ALL required files before loading; raises `RuntimeError` with a list of missing files AND the exact command to regenerate (`python pipeline.py`).
- Document the regeneration command prominently in `BACKEND.md`.
- Add a smoke test that asserts artifacts exist before calling the API in CI/local-test runs.

**Warning signs:** `FileNotFoundError: data/training/preprocessor.pkl` on `uvicorn` startup; first deployment to a clean machine.

### Pitfall 3: Feature Engineering Mismatch (the Big One)

**What goes wrong:** Hand-rolled `pd.get_dummies(profile_df)` on a single-row input produces columns like `["Gender_Male"]` instead of the full set `["Gender_Female"]` (the `drop="first"` column happens to be the input value). When passed to the classifier expecting 19 features, sklearn raises a shape error — or worse, silently aligns columns wrong via positional matching.

**Why it happens:** `OneHotEncoder(drop="first")` learns which category to drop at fit-time. The persisted `preprocessor.pkl` knows; a freshly-instantiated encoder does not.

**How to avoid:** **Load `data/training/preprocessor.pkl` and call `.transform()`. Never refit.** This is the **single most important implementation detail in Phase 1.** `src/inference.py::_load_artifacts` already does this correctly — reuse it.

**Warning signs:** `ValueError: X has N features, but RandomForestClassifier is expecting 19 features`; predictions that look "random" because dummy columns are misaligned.

### Pitfall 4: Stochastic Stage 4 Selection (Non-Deterministic Output)

**What goes wrong:** `src/allocations.py::assign_portfolio_numeric_aum` uses `np.random.choice(..., p=weights)` for ETF selection. Calling `/api/recommend` twice with identical input returns different `{ticker: weight}` maps.

**Why it happens:** By design — the original pipeline samples from AUM-weighted distributions to produce variety. This is desired in production (slight randomness adds diversification) but breaks test determinism.

**How to avoid:**
- Add optional `seed: int | None = None` field to `InvestorProfile` (recommended by CONTEXT.md Specific Idea #5).
- In `recommend_service`, if `seed is not None`, call `np.random.seed(seed)` before invoking `recommend()`.
- Tests pass a fixed seed (e.g., `seed=42`); production calls omit it.
- Document in `BACKEND.md` that default behavior is stochastic.

**Warning signs:** Flaky tests that occasionally fail with "expected SPY in allocation, got VOO"; user reports identical profiles getting different recommendations.

### Pitfall 5: yfinance Returns NaN / None / Empty

**What goes wrong:** `yf.Ticker("INVALID_TICKER").fast_info["last_price"]` may return `NaN`, `None`, raise `KeyError`, or hang for many seconds depending on network conditions. A single bad ticker in a batch of 10 currently has no isolation — without per-ticker try/except, one bad value kills the whole batch.

**Why it happens:** yfinance scrapes/queries Yahoo Finance; Yahoo's response is unreliable for delisted, illiquid, or just-launched tickers. Rate limits and network blips also surface here.

**How to avoid:**
- Per-ticker `try/except` in `prices_service`:
  ```python
  results = {}
  for ticker in tickers:
      try:
          info = yf.Ticker(ticker).fast_info
          price = info.get("last_price") or info.get("regularMarketPrice")
          if price is None or pd.isna(price):
              results[ticker] = {"price": None, "error": "no_price_available"}
          else:
              results[ticker] = {"price": float(price), "currency": info.get("currency", "USD")}
      except Exception as e:
          results[ticker] = {"price": None, "error": str(e)[:200]}
  ```
- Wrap in `concurrent.futures.ThreadPoolExecutor` if latency matters (typical 10-ticker batch ≈ 5-10s sequential, ~1-2s parallel).
- 200 response with per-ticker error fields — never 5xx the whole batch.

**Warning signs:** `/api/current-prices` returning 500 for batches containing one delisted ticker; intermittent test failures for tickers that don't exist anymore.

### Pitfall 6: ETF "Universe" Defined by CSV, Not by Pickle

**What goes wrong:** `/api/etfs/{ticker}` returns 404 for tickers NOT in `data/raw/combined_etf_with_morning_star.csv` — but that CSV is also gitignored. A fresh clone has no universe; every request 404s.

**Why it happens:** Same as Pitfall #2 — gitignored artifacts.

**How to avoid:** Same fix as Pitfall #2. Additionally, the etf_service should treat "CSV not loaded" as a 503, not a 404, to distinguish "system not ready" from "ticker not in supported universe".

### Pitfall 7: enum-string Casing / Spacing Drift

**What goes wrong:** Frontend sends `"high_school"` or `"high school"` (lowercase) → Pydantic 422 because Literal expects `"High school"`. Or worse, sends `"HighSchool"` (snake_case → CamelCase fix) → 422.

**Why it happens:** Easy mistake when frontend devs autoformat or apply convention conversions.

**How to avoid:**
- Document EXACT strings in `BACKEND.md` and in Pydantic model field docstrings (which appear in `/docs`).
- In Phase 3, Zod schemas in Next.js must mirror these literals exactly.
- 422 error message from Pydantic includes the expected values — leave the default message visible.

### Pitfall 8: `model_config` Forbidden Fields Breaking Existing Calls

**What goes wrong:** Setting `model_config = ConfigDict(extra="forbid")` on `InvestorProfile` causes a 422 if the frontend ever sends an extra `name` or `id` field for its own bookkeeping.

**Why it happens:** Defensive validation is great until it isn't.

**How to avoid:** Decision: keep `extra="forbid"` for explicit contract enforcement in Phase 1 (the spec is locked, frontend will mirror exactly). Document this so Phase 3 frontend devs don't add extra fields.

---

## Code Examples

Verified patterns from official sources + this codebase.

### `/api/recommend` Service (full implementation pattern)

```python
# api/services/recommend_service.py
from typing import Optional
import numpy as np
from src.inference import recommend as run_4stage_pipeline

def run_recommendation(profile_dict: dict, seed: Optional[int] = None) -> dict[str, float]:
    """
    Wraps src/inference.py::recommend, flattens the per-asset-class portfolio
    into a flat {ticker: weight} map, and renormalizes to sum=1.0.
    """
    if seed is not None:
        np.random.seed(seed)

    result = run_4stage_pipeline(profile_dict)
    # result["portfolio"]   == {"equity": [...], "bond": [...], "alternative": [...]}
    # result["allocation"]  == {"equity_pct": 0.55, "bond_pct": 0.35, "alt_pct": 0.10}

    alloc = result["allocation"]
    portfolio = result["portfolio"]

    # Equal-weight WITHIN each asset class (the existing recommend() doesn't
    # produce per-ticker weights — only counts. Standard practice when wrapping
    # a count-based pipeline is intra-class equal-weighting unless otherwise
    # specified. The locked contract says ticker→weight summing to 1.0; the
    # planner should confirm this is the intended distribution OR ask whether
    # AUM-weighted intra-class is wanted.)
    flat: dict[str, float] = {}
    for asset_class, tickers in portfolio.items():
        if not tickers:
            continue
        class_pct = alloc.get("alt_pct" if asset_class == "alternative" else f"{asset_class}_pct", 0.0)
        per_ticker = class_pct / len(tickers)
        for t in tickers:
            flat[t] = per_ticker

    # Renormalize to handle float drift — locked contract: sum == 1.0
    total = sum(flat.values())
    if total > 0:
        flat = {t: w / total for t, w in flat.items()}

    return flat
```

**Decision needed (flag for planner):** The original `recommend()` returns asset-class percentages (e.g., 55% equity) and per-class tickers but NOT per-ticker weights. The contract is `{ticker: weight}` summing to 1.0. Two options:
1. **Equal-weight within class** (above) — simple, defensible, locked contract is silent on intra-class distribution.
2. **AUM-weighted within class** — uses Stage 4's AUM data; more "principled" but adds complexity.

**Recommendation:** Equal-weight within class. Document the choice in BACKEND.md. The planner SHOULD include a task to call out this design decision explicitly (and may want to flag for the user during plan execution if they have a strong preference).

### `/api/current-prices` Service

```python
# api/services/prices_service.py
import yfinance as yf
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def _fetch_one(ticker: str) -> tuple[str, dict]:
    try:
        info = yf.Ticker(ticker).fast_info
        # fast_info attributes vary across yfinance versions; try common ones
        price = None
        for attr in ("last_price", "lastPrice", "regularMarketPrice"):
            try:
                price = info[attr] if hasattr(info, "__getitem__") else getattr(info, attr, None)
                if price is not None and not pd.isna(price):
                    break
                price = None
            except (KeyError, AttributeError):
                continue

        if price is None:
            return ticker, {"price": None, "error": "no_price_available"}

        currency = "USD"
        try:
            currency = info["currency"] if hasattr(info, "__getitem__") else getattr(info, "currency", "USD")
        except (KeyError, AttributeError):
            pass

        return ticker, {"price": float(price), "currency": currency}
    except Exception as e:
        logger.warning(f"yfinance failed for {ticker}: {e}")
        return ticker, {"price": None, "error": str(e)[:200]}

def fetch_prices(tickers: list[str]) -> dict[str, dict]:
    """Returns {ticker: {price: float|None, currency: str?, error: str?}}."""
    results: dict[str, dict] = {}
    # Parallel — yfinance is synchronous so threadpool is the right tool
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            ticker, payload = fut.result()
            results[ticker] = payload
    return results
```

### `/api/explain` — Capturing Selection Metadata

The minimal extension to `src/allocations.py` (per CONTEXT.md "Specific Ideas" Option (a)):

```python
# src/allocations.py — ADD this function alongside the existing one. DO NOT modify the existing one.

def assign_portfolio_with_metadata(investor_row, etf_by_class):
    """
    Same as assign_portfolio_numeric_aum but returns BOTH the portfolio AND a
    per-ticker metadata dict explaining why each ticker was chosen.

    Returns
    -------
    portfolio : dict[str, list[str]]   # same shape as assign_portfolio_numeric_aum
    metadata  : dict[str, dict]
        {
            ticker: {
                "asset_class": "equity"|"bond"|"alternative",
                "aum": float,
                "aum_rank_in_class": int,        # 1 = largest in candidate pool at selection time
                "star_rating": float,
                "volatility": float,
                "selection_pool_size": int,      # how many candidates were eligible at the time
            },
            ...
        }
    """
    portfolio = {}
    metadata = {}
    asset_mapping = {
        "equity": "num_equity_etfs",
        "bond": "num_bond_etfs",
        "alternative": "num_alt_etfs",
    }

    for asset_class, n_etfs_key in asset_mapping.items():
        full_df = etf_by_class.get(asset_class)
        if full_df is None or len(full_df) == 0:
            continue
        candidate_df = full_df.copy()
        n_slots = int(investor_row.get(n_etfs_key, 0))
        selected = []

        for _ in range(n_slots):
            if len(candidate_df) == 0: break
            # ... [reuse the AUM-weighted selection logic from assign_portfolio_numeric_aum] ...
            pool_size_at_pick = len(candidate_df)
            picked_symbol = np.random.choice(candidate_df["symbol"].values, p=weights)
            picked_row = candidate_df[candidate_df["symbol"] == picked_symbol].iloc[0]

            # Capture metadata at selection time
            aum_rank = (full_df["AUM"] >= picked_row["AUM"]).sum()  # rank in ORIGINAL pool
            metadata[picked_symbol] = {
                "asset_class": asset_class,
                "aum": float(picked_row["AUM"]),
                "aum_rank_in_class": int(aum_rank),
                "star_rating": float(picked_row["star_rating"]),
                "volatility": float(picked_row["volatility"]),
                "selection_pool_size": pool_size_at_pick,
            }
            selected.append(picked_symbol)
            candidate_df = candidate_df[candidate_df["symbol"] != picked_symbol]
            for group in CONFLICT_GROUPS:
                if picked_symbol in group:
                    candidate_df = candidate_df[~candidate_df["symbol"].isin([s for s in group if s != picked_symbol])]
        portfolio[asset_class] = selected
    return portfolio, metadata
```

The explain service then:
1. Calls a small wrapper that runs the full pipeline like `recommend()` but uses `assign_portfolio_with_metadata` instead of `assign_portfolio_numeric_aum`.
2. Extracts Stage 1 feature importances:
   ```python
   classifier = app.state.classifier  # sklearn RandomForestClassifier
   importances = classifier.feature_importances_  # shape (19,)
   feature_names = preprocessor.get_feature_names_out()  # if available; else hand-construct
   x_scaled = preprocessor.transform(X_raw)[0]  # shape (19,)
   # Local contribution proxy: importance × abs(scaled_value)
   contributions = sorted(
       zip(feature_names, importances * np.abs(x_scaled)),
       key=lambda kv: kv[1], reverse=True
   )[:5]
   ```
3. Builds the response envelope:
   ```python
   {
       "risk_class": {
           "predicted": "Moderate",
           "top_feature_contributions": [
               {"feature": "age", "contribution_score": 0.42},
               {"feature": "horizon", "contribution_score": 0.31},
               ...
           ]
       },
       "allocation_split": {
           "equity_pct": 0.55, "bond_pct": 0.35, "alt_pct": 0.10,
           "reasoning": "Moderate risk profile with 10-year horizon → balanced 55/35/10 split."
       },
       "per_ticker": {
           "SPY": {"asset_class": "equity", "aum_rank_in_class": 1, "star_rating": 4.0, "reasoning": "Largest equity ETF by AUM in candidate pool"},
           ...
       }
   }
   ```

### `/api/etfs/{ticker}` — Universe Lookup

```python
# api/services/etf_service.py
import pandas as pd
from fastapi import HTTPException

def get_etf_detail(ticker: str, etf_df: pd.DataFrame) -> dict:
    row = etf_df[etf_df["Fund Symbol"] == ticker]
    if row.empty:
        raise HTTPException(404, f"Ticker {ticker} not in supported universe")
    r = row.iloc[0]

    # Map asset class using existing logic
    from src.allocations import get_category_mapping
    asset_class = get_category_mapping(r.get("Category", ""))

    return {
        "ticker": ticker,
        "name": r.get("Fund Name"),
        "asset_class": asset_class,
        "category": r.get("Category"),
        "expense_ratio": _none_if_nan(r.get("expense_ratio")),
        "aum": _none_if_nan(r.get("Assets Under Management (AUM)")),
        "morningstar_rating": _none_if_nan(r.get("custom_star_rating")),
        "top_holdings": [],  # NOT in current CSV — flag for the planner: deferred or fetch live via yfinance?
        "sector_breakdown": _extract_sector_cols(r),  # rows have "sector_weightings_*" prefixed cols
        "dividend_yield": _none_if_nan(r.get("Dividend Yield")),
        "one_year_return": _none_if_nan(r.get("1 Year Return")),
        "volatility_annual": _none_if_nan(r.get("Volatility (Annual STD)")),
    }

def _none_if_nan(v):
    if pd.isna(v): return None
    return v if not hasattr(v, "item") else v.item()

def _extract_sector_cols(row):
    return {
        k.replace("sector_weightings_", ""): _none_if_nan(v)
        for k, v in row.items()
        if str(k).startswith("sector_weightings_") and not pd.isna(v)
    }
```

**Open question for planner:** The CSV does NOT contain `top_holdings` (that data lives in yfinance `ticker.funds_data.top_holdings`). Options:
1. Fetch live from yfinance on each `/api/etfs/{ticker}` call (slow, ~1-3s extra).
2. Return `top_holdings: []` and document as deferred.
3. Pre-cache during one-time `python src/etf_fetcher.py` run.

**Recommendation:** Option 2 for Phase 1 (return `[]`), document the limitation. Phase 3 frontend can simply hide the section if empty. Add a future ticket to extend `src/etf_fetcher.py` to capture top_holdings into the cached CSV.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `@app.on_event("startup")` decorator | `lifespan` context manager (`@asynccontextmanager`) | FastAPI 0.93 (2023) | Old API is deprecated; lifespan is the modern way to load models. |
| Pydantic v1 `@validator` | Pydantic v2 `@field_validator` + `model_config = ConfigDict(...)` | Pydantic v2 (2023) | Project already on v2. Use v2 syntax throughout. |
| `yf.Ticker(t).info["regularMarketPrice"]` | `yf.Ticker(t).fast_info["last_price"]` | yfinance 0.2.x | `info` is slow (~1s/ticker); `fast_info` is <100ms. |
| `OneHotEncoder(sparse=False)` | `OneHotEncoder(sparse_output=False)` | sklearn 1.2 (2023) | Already correctly applied in `src/prepare_data.py` line 113. |

**Deprecated/outdated:**
- `BaseModel.dict()` → `BaseModel.model_dump()` (Pydantic v2)
- `BaseModel.parse_obj()` → `BaseModel.model_validate()` (Pydantic v2)
- `OneHotEncoder(sparse=False)` → `OneHotEncoder(sparse_output=False)` (sklearn 1.2+)

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | sklearn pickles were created with version 1.3.x or 1.4.x (~mid-2024) | Common Pitfalls #1 | If wrong, the recommended version pin in requirements.txt is incorrect; mitigation is `python pipeline.py` regenerate, which always works. |
| A2 | `/api/recommend` ticker weights should be equal-weighted within asset class | Code Examples / `recommend_service` | If user wants AUM-weighted intra-class, implementation needs the AUM lookup; ~30 min of extra work, planner can flip. |
| A3 | `top_holdings` is acceptable to defer (return `[]`) in Phase 1 | Code Examples / `etf_service` | If user wants top_holdings now, `src/etf_fetcher.py` extension is required — adds a sub-task. |
| A4 | `feature_importances_` × scaled-input is a defensible proxy for local feature contribution | Standard Stack / Code Examples | If user demands SHAP, ~1 day extra to integrate `shap` lib + ensure pickles are compatible. |
| A5 | `data/raw/combined_etf_with_morning_star.csv` is the canonical "supported universe" | Architecture Patterns / Code Examples | If user defines universe differently (e.g., a hand-curated subset), the etf_service universe check needs updating. |
| A6 | yfinance `fast_info["last_price"]` returns USD by default for US ETFs | Code Examples / `prices_service` | Currency mismatch is benign (always USD for the 95 ETFs in this universe) but the code returns `currency` field anyway. |
| A7 | All requirements (BACKEND-01..06) are satisfied by the 4 endpoints + service layer as designed; no extra plumbing needed | Phase Requirements table | Low risk — requirements are mechanically traceable to the 4 endpoint contracts. |

---

## Open Questions

1. **Intra-asset-class weighting in `/api/recommend`.**
   - What we know: Locked contract says `{ticker: weight}` summing to 1.0; per-class percentages come from Stage 2 regression; per-class ticker count comes from Stage 3; specific tickers come from Stage 4.
   - What's unclear: How should the per-class percentage be distributed across the N tickers in that class? Equal-weight is simplest; AUM-weighted is more "principled".
   - Recommendation: Equal-weight in Phase 1. Document explicitly. Planner can ask user to confirm OR generate a small explainer in BACKEND.md.

2. **`top_holdings` in `/api/etfs/{ticker}`.**
   - What we know: Spec lists `top_holdings` as a required field; current CSV does not contain it.
   - What's unclear: Whether to fetch live from yfinance (slow) or defer.
   - Recommendation: Defer (return `[]`); document; create v2 ticket.

3. **`MaritalStatus` enum drift.**
   - What we know: Locked schema is `Single | Married` (2 values); `src/validation_rules.py` has 4 values (`Married, Single, Divorced, Widowed`); the synthetic data generator may have produced rows with Divorced/Widowed.
   - What's unclear: Whether existing trained models have seen Divorced/Widowed and whether the API should accept them for backward-compat.
   - Recommendation: API enforces the locked 2-value enum (`Single | Married` only). Existing models will route through one-hot encoding via `preprocessor.transform` which handles unknown categories per its `handle_unknown` setting (defaults to "ignore" in `OneHotEncoder` after sklearn 1.1; verify in `preprocessor.pkl`).

4. **Optional `seed` field on `/api/recommend`.**
   - What we know: Stage 4 is stochastic; CONTEXT.md endorses optional seed.
   - What's unclear: Whether to expose in the public OpenAPI surface or only via test-only header.
   - Recommendation: Public field, documented as "for reproducible testing". The locked contract doesn't forbid it. No data harm.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.14 | Runtime | ✓ | 3.14.2 (system) and 3.14 (venv) | — |
| venv at `./venv/` | Runtime | ✓ | Python 3.14 | — |
| `pandas` | `recommend_service`, `etf_service` | ✓ | 2.3.3 | — |
| `numpy` | `recommend_service`, `explain_service` | ✓ | bundled with pandas | — |
| `scikit-learn` | model loading | ✓ but **version-drifted** | 1.8.0 installed; pickles likely 1.3.x | regenerate via `python pipeline.py` |
| `joblib` | model loading | ✓ | 1.5.3 | — |
| `yfinance` | `prices_service` | ✓ | 1.2.0 | — |
| `pydantic` | schemas | ✓ | 2.12.5 | — |
| `httpx` | tests (`TestClient`) | ✓ | 0.28.1 | — |
| `pytest` | tests | ✓ | 9.0.3 | — |
| **`fastapi`** | core | **✗** | — | **must install** |
| **`uvicorn`** | runner | **✗** | — | **must install** |
| `models/risk_profile_classifier.pkl` | startup | ✓ | dated 2026-05-12 | regenerate via `python pipeline.py` |
| `models/allocation_regressor.pkl` | startup | ✓ | dated 2026-05-12 | regenerate via `python pipeline.py` |
| `models/etf_count_regressor.pkl` | startup | ✓ | dated 2026-05-12 | regenerate via `python pipeline.py` |
| `models/label_map.json` | startup | ✓ | dated 2026-05-18 | regenerate via `python pipeline.py` |
| `data/training/preprocessor.pkl` | startup, feature engineering | ✓ | dated 2026-05-11 | regenerate via `python pipeline.py` |
| `data/raw/combined_etf_with_morning_star.csv` | startup, etf_service | ✓ | dated 2026-05-11 | regenerate via `python pipeline.py` (steps 1 + 1.5) |
| `.env` with `FINNHUB_API_KEY` | only used by `src/etf_fetcher.py` at training time | ✓ | — | not needed at API runtime |
| Internet access (Yahoo Finance) | `/api/current-prices` runtime | assumed ✓ on dev box | — | partial-failure envelope handles outage |
| Port 8000 free | local-run | assumed ✓ | — | uvicorn `--port` flag |

**Missing dependencies with no fallback:** `fastapi`, `uvicorn` — installation is a planned task in Phase 1.

**Missing dependencies with fallback:** sklearn version mismatch — fallback is `python pipeline.py` regenerate (already documented).

---

## Validation Architecture

> Project has no `.planning/config.json`, so Nyquist validation is treated as enabled.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest` 9.0.3 (already installed) + `fastapi.testclient.TestClient` (uses installed `httpx` 0.28.1) |
| Config file | None currently; existing `pytest.ini` could be added but `pytest` auto-discovers `tests/test_*.py` |
| Quick run command | `venv/bin/pytest tests/test_api_recommend.py -x -q` (single endpoint) |
| Full suite command | `venv/bin/pytest -x -q` (all tests including pre-existing `test_data_validation.py`) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BACKEND-01 | Server boots; all 4 endpoints registered on the router | smoke (TestClient init + `app.routes` introspection) | `pytest tests/test_api_smoke.py -x` | ❌ Wave 0 |
| BACKEND-02 | `POST /api/recommend` with valid 12-field payload returns 200 + `{ticker: weight}` summing to 1.0 (within 1e-6) | endpoint contract | `pytest tests/test_api_recommend.py::test_recommend_returns_normalized_allocation -x` | ❌ Wave 0 |
| BACKEND-02 | Invalid 12-field payload (missing field, wrong enum) returns 422 | endpoint contract | `pytest tests/test_api_recommend.py::test_recommend_rejects_invalid_payload -x` | ❌ Wave 0 |
| BACKEND-02 | `seed=42` makes output deterministic across two calls | endpoint contract | `pytest tests/test_api_recommend.py::test_recommend_with_seed_is_deterministic -x` | ❌ Wave 0 |
| BACKEND-03 | `POST /api/current-prices` with known tickers returns prices | endpoint contract (integration — hits live yfinance unless mocked) | `pytest tests/test_api_prices.py::test_prices_for_known_tickers -x` | ❌ Wave 0 |
| BACKEND-03 | Unknown ticker returns `{"ticker": {"price": null, "error": "..."}}` not 5xx | endpoint contract | `pytest tests/test_api_prices.py::test_prices_partial_failure -x` | ❌ Wave 0 |
| BACKEND-04 | (verification by absence) `frontend/` does not import yfinance | structural (Phase 3+ concern; in Phase 1, verify `api/` files do not import via yfinance only in `services/prices_service.py`) | `grep -r "import yfinance" api/ \| wc -l` should equal 1 | ❌ Wave 0 (structural check) |
| BACKEND-05 | `POST /api/explain` returns response with the 3 required sections | endpoint contract | `pytest tests/test_api_explain.py::test_explain_has_three_sections -x` | ❌ Wave 0 |
| BACKEND-05 | Risk-class section includes top feature contributions | endpoint contract | `pytest tests/test_api_explain.py::test_explain_includes_feature_contributions -x` | ❌ Wave 0 |
| BACKEND-05 | Per-ticker section includes aum_rank + asset_class | endpoint contract | `pytest tests/test_api_explain.py::test_explain_per_ticker_metadata -x` | ❌ Wave 0 |
| BACKEND-06 | `GET /api/etfs/SPY` returns 200 + metadata | endpoint contract | `pytest tests/test_api_etfs.py::test_etfs_known_ticker -x` | ❌ Wave 0 |
| BACKEND-06 | `GET /api/etfs/ZZZZ` returns 404 with locked error message | endpoint contract | `pytest tests/test_api_etfs.py::test_etfs_unknown_ticker_returns_404 -x` | ❌ Wave 0 |
| BACKEND-01..06 | (additivity) `python pipeline.py` still runs to completion | manual smoke | `python pipeline.py --skip-fetch` (must not error) | covered by existing pipeline |
| BACKEND-01..06 | (additivity) `python src/inference.py` still runs to completion | manual smoke | `venv/bin/python src/inference.py` (must print 3 recommendations) | covered by existing inference |
| BACKEND-01..06 | (additivity) `venv/bin/streamlit run app.py` still launches | manual-only (UI smoke; not in CI) | manual visual check | manual |

### Sampling Rate

- **Per task commit:** `venv/bin/pytest tests/test_api_*.py -x -q` (run only the API tests touched, plus any test that names the changed router)
- **Per wave merge:** `venv/bin/pytest -x -q` (entire suite, including pre-existing `test_data_validation.py`)
- **Phase gate (`/gsd:verify-work`):** Full suite green AND manual smoke of `python pipeline.py --skip-fetch --skip-training` AND `venv/bin/python src/inference.py`

### Wave 0 Gaps

- [ ] `tests/test_api_smoke.py` — covers BACKEND-01 (server boots, 4 routes registered)
- [ ] `tests/test_api_recommend.py` — covers BACKEND-02 (3 tests above)
- [ ] `tests/test_api_prices.py` — covers BACKEND-03 (2 tests above; consider `respx` mock for httpx or `unittest.mock.patch('yfinance.Ticker')` for offline runs)
- [ ] `tests/test_api_explain.py` — covers BACKEND-05 (3 tests above)
- [ ] `tests/test_api_etfs.py` — covers BACKEND-06 (2 tests above)
- [ ] `tests/conftest.py` — shared fixtures:
  - `client` (FastAPI TestClient with lifespan startup); requires fixtures so models load
  - `mock_models` (skips actual pickle loading; useful when CI doesn't have artifacts)
  - `sample_profile` (a valid 12-field dict for reuse)
- [ ] Framework install: `venv/bin/pip install "fastapi>=0.115" "uvicorn[standard]>=0.30"` — add to requirements.txt then `pip install -r requirements.txt`

### Test fixture pattern for "models may be missing in CI"

```python
# tests/conftest.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

ARTIFACTS_PRESENT = all([
    Path("models/risk_profile_classifier.pkl").exists(),
    Path("data/training/preprocessor.pkl").exists(),
    Path("data/raw/combined_etf_with_morning_star.csv").exists(),
])

@pytest.fixture(scope="session")
def client():
    """Real TestClient — requires artifacts on disk. Skip if missing."""
    if not ARTIFACTS_PRESENT:
        pytest.skip("Required ML artifacts missing — run `python pipeline.py` to generate")
    from fastapi.testclient import TestClient
    from api.main import app
    with TestClient(app) as c:
        yield c

@pytest.fixture
def sample_profile():
    return {
        "Age": 30, "Gender": "Male", "Education": "Bachelor",
        "MaritalStatus": "Single", "HouseholdSize": 1,
        "Income": 80000, "InvestmentGoal": "Retirement",
        "InvestmentHorizon": 30, "InvestmentCapital": 50000,
        "RiskTolerance": "Medium", "FinancialInvolvement": "Medium",
        "ExperienceYears": 3,
    }
```

This pattern means CI without artifacts gracefully skips integration tests instead of crashing; the developer regenerates artifacts and reruns to get full coverage.

---

## Security Domain

> `security_enforcement` not specified in config → treat as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | **no** (Phase 1 is local-only by locked decision) | — (deferred per CONTEXT.md scope-fence) |
| V3 Session Management | no | — |
| V4 Access Control | no (no users yet) | — |
| V5 Input Validation | **yes** | Pydantic v2 `BaseModel` with `Literal` enums, `Field(ge=, le=)`, `extra="forbid"` |
| V6 Cryptography | no (no secrets at API runtime; `.env` is for training only) | — |
| V7 Error Handling & Logging | **yes** | FastAPI default 422 + custom `HTTPException(503)` for missing artifacts; `logging.getLogger(__name__)` per module |
| V8 Data Protection | partial | No PII storage (profiles are POSTed not stored in Phase 1) — Phase 2 (DB) takes ownership |
| V12 API & Web Service | **yes** | OpenAPI schema auto-generated at `/docs`; rate limiting deferred |
| V14 Configuration | partial | No secrets needed at runtime; pin sklearn version in requirements.txt |

### Known Threat Patterns for FastAPI + Python ML Service

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Pickle deserialization of arbitrary path | T (Tampering) | `joblib.load` is called only on `models/*.pkl` paths hardcoded at startup; no user-controlled path |
| Adversarial profile inputs (extreme values causing model crash) | D (Denial-of-Service) | `Field(ge=, le=)` ranges from `validation_rules.py` clamp inputs to reasonable bounds before reaching model |
| yfinance return values used as numeric without validation | T (Tampering, indirect) | All yfinance returns coerced via `float(...)` with try/except; `NaN` checks before serializing |
| Resource exhaustion via large `tickers` array on `/api/current-prices` | D (DoS) | Pydantic `max_items` on the list (suggest 100 cap) + `ThreadPoolExecutor(max_workers=8)` cap |
| Slow yfinance call hanging request | D (DoS) | Per-call timeout (use `requests.get(timeout=5)` if dropping to raw HTTP; for yfinance, run inside `ThreadPoolExecutor` with overall budget) |
| ETF universe defined client-side, bypass attempt | T (Tampering) | `/api/etfs/{ticker}` filters strictly against the on-disk CSV; no SQL injection vector |
| Verbose error leaking internal paths | I (Information Disclosure) | Catch broad exceptions in services, return 500 with `"internal error"` message (no traceback in response); log full traceback server-side |

**Phase 1 security posture summary:** Local-only service with one trusted caller (Next.js on same host). Input validation is the primary control; everything else is best-effort defense-in-depth.

---

## Sources

### Primary (HIGH confidence)
- **Codebase (verified by direct read):**
  - `src/inference.py` (lines 1-301) — confirms `recommend(user: dict)` exists, model loading pattern, preprocessor usage
  - `src/allocations.py` (lines 1-316) — confirms `assign_portfolio_numeric_aum`, `CONFLICT_GROUPS`, stochastic `np.random.choice`
  - `src/prepare_data.py` (lines 95-175) — confirms 12→19 feature engineering via `ColumnTransformer`, `preprocessor.pkl` persisted
  - `src/validation_rules.py` (lines 12-67) — confirms enum values, numeric ranges (note: MaritalStatus has 4 values here vs 2 in locked schema)
  - `src/etf_fetcher.py` (lines 1-308) — confirms yfinance usage patterns
  - `src/morning_star_rating.py` (lines 1-110) — confirms star rating logic and re-use surface
  - `app.py` (lines 96-117) — confirms `recommend()` is the current consumption surface (used by Streamlit)
  - `requirements.txt` — current deps; FastAPI/uvicorn NOT present
  - `.gitignore` (lines 16-26) — confirms `data/`, `*.pkl`, `models/*.pkl` are gitignored
  - `data/training/preprocessor.pkl` (file exists, 4561 bytes, dated 2026-05-11) — verified by `ls -la`
  - `models/*.pkl` (3 pickles + label_map.json) — verified by `ls -la`
- **`.planning/phases/01-fastapi-backend-wrapper/01-CONTEXT.md`** — locked user decisions for Phase 1
- **`.planning/intel/decisions.md`** + **`constraints.md`** — locked architectural decisions/contracts
- **`docs/MIGRATION-SPEC.md`** — PRD source of truth
- **FastAPI official docs** — https://fastapi.tiangolo.com/advanced/events/ (lifespan pattern)
- **scikit-learn official docs** — https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.InconsistentVersionWarning.html and https://scikit-learn.org/stable/model_persistence.html (version-mismatch warning)
- **PyPI direct query (2026-05-19):** fastapi 0.136.1, uvicorn 0.47.0, httpx 0.28.1 — verified via `pip index versions`

### Secondary (MEDIUM confidence)
- **yfinance ranaroussi docs** — https://ranaroussi.github.io/yfinance/reference/yfinance.ticker_tickers.html (Tickers class; `fast_info` reference is sparse in current docs, hence A6 is flagged)
- **GitHub issues for yfinance** — `fast_info` field-name variation across versions documented in issue threads

### Tertiary (LOW confidence)
- None — every code-actionable claim is grounded in a verified source above.

---

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** — every package verified on PyPI; versions installed match recommendations
- Architecture: **HIGH** — based on direct read of all relevant files in the codebase
- Pitfalls: **HIGH** — version drift verified by inspecting actual installed sklearn 1.8 vs `requirements.txt` `>=1.3.0`; preprocessor.pkl confirmed to exist
- Code examples: **HIGH** — patterns are either copied from FastAPI docs or directly derived from `src/inference.py`'s existing logic
- Test strategy: **MEDIUM** — pytest is conventional but actual `TestClient` lifespan behavior with the model-loading path needs a smoke run to fully de-risk

**Research date:** 2026-05-19
**Valid until:** 2026-06-19 (30 days — FastAPI / sklearn / yfinance are all stable; only meaningful invalidator is a pickle regeneration changing the artifact set)

---

## RESEARCH COMPLETE

**Phase:** 1 - FastAPI Backend Wrapper
**Confidence:** HIGH

### Key Findings

1. **`src/inference.py::recommend()` already exists and is the right consumption surface.** Project memory's note that "`src/inference.py` may be missing" is outdated — it's a 301-line, well-structured module that loads all artifacts and runs all 4 stages. The API is a thin adapter around it.

2. **`data/training/preprocessor.pkl` (a `ColumnTransformer`) is persisted and handles the 12→19 feature engineering.** This collapses the hardest-anticipated task (BACKEND-02 feature parity) into "call the existing function." The API does NOT need to hand-replicate one-hot encoding or scaling.

3. **sklearn version drift is the #1 landmine.** Installed sklearn is 1.8.0; pickles were created with 1.3.x or 1.4.x. Pin sklearn in `requirements.txt` to the training version, OR run `python pipeline.py` to regenerate at the current version. The lifespan startup should detect this and log clearly.

4. **The four endpoints map nearly 1:1 onto existing functions.** Only one minimal extension to `src/allocations.py` is needed: a new `assign_portfolio_with_metadata()` adapter that captures per-ticker selection rationale (AUM rank, pool size, etc.) for `/api/explain`. The existing function is untouched.

5. **Two open design questions deserve user attention before plan execution:** (a) intra-asset-class weighting in `/api/recommend` (equal-weight vs AUM-weighted within class); (b) `top_holdings` in `/api/etfs/{ticker}` (defer with empty array vs fetch live from yfinance). Recommendations are made (equal-weight + defer), but these are flag-worthy.

### File Created
`/Users/avivberkovich/Documents/AI_FINANCIAL_ADVISOR/.planning/phases/01-fastapi-backend-wrapper/01-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | All packages PyPI-verified; pydantic/httpx/pytest already installed |
| Architecture | HIGH | Direct codebase read of every relevant file; thin wrapper is mechanically sound |
| Pitfalls | HIGH | sklearn version drift verified by actual installed-vs-required diff |
| Feature engineering | HIGH | `data/training/preprocessor.pkl` confirmed to exist and load via existing `_get_artifacts()` |
| Test strategy | MEDIUM | Pattern is conventional; needs first run to confirm TestClient lifespan loads pickles cleanly |
| `/api/explain` SHAP-vs-importance | MEDIUM | Recommendation is defensible for academic project but a domain reviewer might push for richer attribution |

### Open Questions for Planner

1. Intra-class weight distribution in `/api/recommend` (equal vs AUM-weighted) — flagged in Open Questions #1 and Assumptions A2.
2. `top_holdings` field handling — flagged in Open Questions #2 and Assumptions A3.
3. Whether `MaritalStatus` discrepancy between locked schema (2 values) and `validation_rules.py` (4 values) needs addressing in Phase 1 (recommendation: no — accept locked 2 values).
4. Whether the optional `seed` field should be public on `/api/recommend` (recommendation: yes; A4 mitigation).

### Ready for Planning

Research complete. The planner can now create PLAN.md files. Recommended task ordering:
1. Install fastapi + uvicorn, update `requirements.txt`
2. Pin sklearn version (or document regeneration step)
3. Create `api/` package skeleton (`main.py`, `schemas.py`, empty routers + services)
4. Implement Pydantic models (12-field profile + 4 response shapes)
5. Implement `lifespan` + `/api/health` (Claude's discretion endpoint, recommended)
6. Implement `/api/recommend` end-to-end + test
7. Implement `/api/etfs/{ticker}` (simplest endpoint, validates ETF universe loading) + test
8. Implement `/api/current-prices` (yfinance integration, partial-failure envelope) + test
9. Implement `/api/explain` (requires the minimal `src/allocations.py` extension) + test
10. Write `BACKEND.md` + update `README.md`
11. Verify additivity: `python pipeline.py --skip-fetch`, `python src/inference.py`, `streamlit run app.py` all still work
