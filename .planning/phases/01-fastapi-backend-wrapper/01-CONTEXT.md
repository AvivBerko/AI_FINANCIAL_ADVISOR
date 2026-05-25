# Phase 1: FastAPI Backend Wrapper — Context

**Gathered:** 2026-05-19
**Status:** Ready for planning
**Source:** Synthesized from `.planning/intel/` (PRD ingest of `docs/MIGRATION-SPEC.md`) + `.planning/ROADMAP.md` Phase 1 success criteria

<domain>
## Phase Boundary

Phase 1 wraps the existing Python 4-stage ML pipeline, the yfinance live-price layer, and the ETF metadata/explanation surfaces as HTTP endpoints, so a future Next.js orchestrator can call them. The **financial brain is untouched** — no model retraining, no logic rewrites, no Streamlit changes. The wrapper is **additive** and lives inside the current Python origin repo.

**In scope:**
- A FastAPI server runnable from the repo root.
- Four HTTP endpoints (POST /api/recommend, POST /api/current-prices, POST /api/explain, GET /api/etfs/{ticker}).
- Pydantic request/response schemas for each endpoint matching the locked contracts.
- A thin service layer that adapts the existing `src/` modules and `pipeline.py` artifacts to the HTTP surface.
- Loading the persisted RandomForest models (`models/*.pkl`) at app startup.
- Error handling: 404 for unknown tickers, 422 for invalid 12-field profiles, 5xx surfaced clearly when yfinance fails.
- Local-run instructions (uvicorn command, port, README delta).

**Out of scope (for this phase):**
- Authentication/authorization on endpoints (no auth in Phase 1; deferred to a future hardening pass — Next.js orchestrator is the only caller and runs on the same host or trusted network during dev).
- Production deployment, containerization, reverse-proxy config.
- Hyperparameter retuning, model improvements, or addressing the circular-labeling issue (tracked separately as BACKEND-V2-02).
- Any frontend code (Phase 3+).
- Database persistence (Phase 2).
- LLM integration (Phase 5).

</domain>

<decisions>
## Implementation Decisions

### Endpoint surface (locked — `DEC-fastapi-endpoint-surface`)

Exactly these 4 endpoints, no more, no fewer in v1:
- `POST /api/recommend` — accepts 12 investor profile fields, returns `{ticker: weight}` map summing to 1.0. Re-runs the full 4-stage pipeline. Backs onboarding + `update_investor_profile` LLM tool.
- `POST /api/current-prices` — accepts list of tickers, returns live yfinance prices per ticker. Errors per-ticker surfaced clearly when invalid.
- `POST /api/explain` — accepts 12 fields + allocation map, returns structured rationale: predicted risk class with feature contributions, equity/bond split reasoning, per-ticker selection reasoning. Backs `explain_allocation` LLM tool.
- `GET /api/etfs/{ticker}` — returns ETF metadata (name, asset class, expense ratio, top holdings, AUM, sector breakdown, Morningstar rating). 404 on unknown tickers. Backs `get_etf_details` LLM tool.

### Investor profile schema (locked — `DEC-investor-profile-schema-frozen`)

The 12 fields with EXACT types/enums:
- `Age` (int)
- `Gender` (enum: Male, Female)
- `Education` (enum: High school, Bachelor, Master)
- `MaritalStatus` (enum: Single, Married)
- `HouseholdSize` (int)
- `Income` (number)
- `InvestmentGoal` (enum: Child education, Retirement, Home purchase, Other)
- `InvestmentHorizon` (int, years)
- `InvestmentCapital` (number)
- `RiskTolerance` (enum: Low, Medium, High)
- `FinancialInvolvement` (enum: Low, Medium, High)
- `ExperienceYears` (int)

The Pydantic request model for `/api/recommend` and `/api/explain` MUST validate these exactly. Field names MUST match (PascalCase as listed) so contract is stable across all 3 tiers.

### Allocation output shape (locked — `CON-api-recommend-contract`, `CON-allocated-assets-jsonb-shape`)

`/api/recommend` response is a `{ticker: weight}` JSON object where weights are 0.0-1.0 floats summing to **exactly 1.0** (within float tolerance — server MUST renormalize before returning if the underlying pipeline produces drift).

### Architectural boundary (locked — `CON-financial-brain-stays-python`)

- ALL ML inference, yfinance calls, and ETF metadata lookups MUST happen in Python.
- The wrapper MUST NOT call out to other services; it is a leaf node in the dependency graph.
- The Next.js frontend (Phase 3+) is the ONLY consumer in production; never call yfinance from Next.js.

### Reuse vs. rewrite

- The wrapper is **additive**. The existing `python pipeline.py` (training entry point) and `src/inference.py` (CLI) MUST continue to work after Phase 1 ships — verified by running them post-implementation.
- Phase 1 SHOULD reuse existing modules: `src/allocations.py`, `src/etf_fetcher.py`, `src/morning_star_rating.py`, plus the model artifacts in `models/`.
- If `src/inference.py` does not yet exist (per project memory it is missing), Phase 1 MAY create it as part of building the recommend endpoint, since the HTTP layer needs a clean function-call surface anyway. The CLI form is secondary; the HTTP service is primary.

### Model loading strategy

- Load `models/*.pkl` at FastAPI startup (lifespan event), not per request. Cache the loaded estimators in app state.
- If model files are missing at startup, fail fast with a clear error pointing at `python pipeline.py` to regenerate.

### Explain endpoint composition

The `/api/explain` rationale is a **structured object** (not a free-text string). Three concerns are surfaced:
1. **Risk-class prediction with feature contributions** — Stage 1 classifier output + the most influential features (e.g., feature importance × actual value, or local SHAP-style attribution if cheap).
2. **Equity/bond split reasoning** — Stage 2 regression output (equity_pct, bond_pct) plus a brief narrative-template tying it to risk class + horizon.
3. **Per-ticker selection reasoning** — for each ticker in the input allocation, a small object describing why it was chosen (asset-class fit, AUM rank within class, Morningstar rating, etc.). The data already exists in `src/allocations.py` selection logic; surface it rather than recompute.

The exact structured schema for the rationale is **Claude's Discretion** within these constraints (3 sections, JSON-serializable).

### Repo layout

- Wrapper lives at the **repo root** alongside `pipeline.py` and `src/`. Suggested layout (Claude's Discretion):
  - `api/` directory (or `app.py` + `api/routers/`) for FastAPI code.
  - Pydantic schemas in `api/schemas/` (or single `api/schemas.py`).
  - Service layer adapters in `api/services/`.
  - The exact directory split is Claude's Discretion as long as imports of `src/` modules remain stable.
- The `frontend/` directory (Phase 3+) is reserved for Next.js — DO NOT touch it.

### Error handling contracts

- Invalid 12-field profile → FastAPI's default 422 (Pydantic validation error response). No custom error envelope.
- Unknown ETF ticker on `GET /api/etfs/{ticker}` → 404 with `{"detail": "Ticker {ticker} not in supported universe"}`.
- yfinance failure on `/api/current-prices` → return 200 with per-ticker objects where failed tickers carry an explicit error field, rather than failing the whole batch. (Rationale: dashboard caller wants partial results.)
- Model unavailable / missing pickle → 503 with actionable message.

### Local-run posture

- `uvicorn api.main:app --reload --port 8000` (or equivalent) MUST work from repo root.
- Default port: **8000**.
- README (or a new `BACKEND.md`) MUST document the run command.
- No CORS in Phase 1 (no frontend caller yet); add CORS in Phase 3 when Next.js is wired up.

### Claude's Discretion

The following are deliberately left to the planner/executor to choose:
- FastAPI directory layout (`api/`, `app/`, root-level `app.py`, etc.).
- Pydantic split between request/response models and shared types.
- Whether `/api/explain` reuses the same input model as `/api/recommend` + extends it, or has a dedicated model.
- Internal service-layer abstraction shape (functions vs. classes).
- Logging library choice (stdlib `logging` is fine — Claude's Discretion).
- Test framework (pytest is conventional; `httpx.AsyncClient`/`TestClient` for endpoint tests).
- Whether to add a `/health` or `/api/health` endpoint (recommended but not required by any locked contract).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### PRD (source of truth)
- `docs/MIGRATION-SPEC.md` — locked migration architecture, feature definitions, execution plan.

### Project planning intel (already synthesized from PRD)
- `.planning/intel/decisions.md` — locked decisions, esp. `DEC-fastapi-endpoint-surface`, `DEC-investor-profile-schema-frozen`, `DEC-3tier-architecture`, `DEC-llm-tools-closed-set-v1`.
- `.planning/intel/constraints.md` — API contracts: `CON-api-recommend-contract`, `CON-api-current-prices-contract`, `CON-api-explain-contract`, `CON-api-etfs-detail-contract`, `CON-financial-brain-stays-python`.
- `.planning/intel/context.md` — migration rationale, sequencing, repo layout intent.
- `.planning/intel/requirements.md` — `REQ-feat-fastapi-backend-wrapper`, `REQ-phase-1-fastapi-backend`.
- `.planning/REQUIREMENTS.md` — BACKEND-01 through BACKEND-06 (acceptance-checkable line items).
- `.planning/ROADMAP.md` — Phase 1 goal + 7 success criteria, dependency graph.

### Code (existing implementation to wrap)
- `pipeline.py` — training pipeline orchestrator (must continue to work).
- `src/inference.py` — CLI inference entry (may be missing; HTTP layer is primary).
- `src/allocations.py` — rule-based allocation + per-ticker selection logic (drives explain rationale).
- `src/etf_fetcher.py` — yfinance metadata fetcher.
- `src/morning_star_rating.py` — Morningstar-style star rating computation.
- `src/data_generation.py` — synthetic investor data generator (NOT called at request time; training only).
- `models/` — persisted RF model pickles (`stage1_risk_classifier.pkl`, `stage2_allocation_regressor.pkl`, `stage3_etf_count.pkl` or equivalents — verify actual filenames).
- `notebooks/phase2_data_preparation.ipynb` — feature engineering (the 19 engineered features the model expects — the API must apply identical transforms to incoming 12-field payload).
- `notebooks/phase3_model_training.ipynb` — model training (reference for what was persisted).

</canonical_refs>

<specifics>
## Specific Ideas

- The Pydantic enum literals MUST match the exact strings used during training (e.g., "High school" not "high_school"), otherwise the feature-encoding step will fail silently or skew predictions. The encoder used in `phase2_data_preparation.ipynb` is the source of truth — the API needs to apply the same encoding.
- `/api/current-prices` should accept `{"tickers": ["SPY","QQQ"]}` (object with list) rather than a bare array — easier to extend with optional params (e.g., `as_of_date`) without a breaking change.
- The `/api/explain` per-ticker rationale data lives in `src/allocations.py` — when the API computes the allocation for the explain endpoint, it should capture the *selection metadata* (asset class, AUM rank, candidate pool size) emitted by the selection logic. If that information isn't currently returned by `src/allocations.py`, the wrapper SHOULD adapt by either (a) capturing it via a small extension to the function signature or (b) re-deriving from `etf_fetcher.py` lookups. Option (a) is cleaner.
- For local development, exposing an interactive `/docs` (Swagger UI) is FastAPI's default and SHOULD be left enabled — it's the easiest way for the Phase 3 frontend developer to sanity-check the contract.
- Stochastic seed: the existing `run_allocations_pipeline` uses `np.random.choice`. Decide at planning time whether the API should accept an optional `seed` field for reproducibility or remain stochastic. Recommendation: **seed parameter optional, default None (stochastic)** — supports both deterministic testing and production variance.

</specifics>

<deferred>
## Deferred Ideas

- Authentication / API keys (Phase 1 is local-only).
- Rate limiting on yfinance calls (yfinance has its own client-side caching; add only if it becomes a problem).
- Dockerization / production deploy (separate workstream).
- Caching `/api/etfs/{ticker}` responses (premature optimization — yfinance is cached client-side).
- WebSocket or SSE for streaming `/api/recommend` progress (not needed; pipeline is fast enough).
- CORS configuration (added in Phase 3 when Next.js comes online).
- An `/api/portfolio-value` endpoint that does the current-prices × allocation math server-side. Per `DEC-llm-tools-closed-set-v1`, `get_portfolio_value` is a frontend-composed tool that calls `/api/current-prices` + multiplies by `InvestmentCapital` — keep that composition on the frontend side; no extra endpoint needed.

</deferred>

<scope_fence>
## Scope Fence

**This phase does NOT:**
- Touch any frontend code.
- Touch the database (Supabase doesn't exist yet in Phase 1; that's Phase 2).
- Add auth.
- Improve ML model accuracy.
- Replace Streamlit (the Streamlit prototype can be ignored or left in place; the migration spec doesn't mandate deletion in this phase).
- Add observability beyond stdlib `logging` (no Sentry, no OpenTelemetry, no Prometheus — premature).
- Change the persistence format of the model pickles.

**If a task in this phase would touch any of the above, STOP — it belongs in a different phase.**

</scope_fence>

---

*Phase: 01-fastapi-backend-wrapper*
*Context gathered: 2026-05-19 from .planning/intel/* (PRD ingest)*
