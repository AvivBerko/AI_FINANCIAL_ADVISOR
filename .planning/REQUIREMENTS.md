# Requirements: AI-Driven Portfolio Advisor

**Defined:** 2026-05-18
**Core Value:** A user fills in 12 financial profile fields, sees an ML-recommended ETF portfolio they can trust, and can conversationally adjust it — without the LLM ever silently overriding the model.

## v1 Requirements

Requirements for the migration release (Streamlit prototype → Next.js + Supabase + FastAPI). Each maps to exactly one roadmap phase. IDs in parentheses link back to synthesized intel in `.planning/intel/`.

### Backend (Python FastAPI Wrapper)

- [ ] **BACKEND-01**: A FastAPI server runnable from the Python origin repo wraps the existing 4-stage ML pipeline as HTTP endpoints (`REQ-feat-fastapi-backend-wrapper`, `REQ-phase-1-fastapi-backend`)
- [ ] **BACKEND-02**: `POST /api/recommend` accepts a JSON body with the 12 frozen investor profile fields (with type/enum constraints from `DEC-investor-profile-schema-frozen`) and returns a ticker→weight allocation map summing to 1.0 — re-running the full 4-stage pipeline (`CON-api-recommend-contract`)
- [ ] **BACKEND-03**: `POST /api/current-prices` accepts a list of tickers and returns live yfinance market prices per ticker, suitable for dashboard valuation (`CON-api-current-prices-contract`)
- [ ] **BACKEND-04**: Real-time ETF price fetching (yfinance) remains inside Python — Next.js does not call yfinance directly; it goes through `/api/current-prices` (`CON-financial-brain-stays-python`)
- [ ] **BACKEND-05**: `POST /api/explain` accepts the 12 profile fields plus an allocation map and returns a structured rationale (risk-class prediction with feature contributions, equity/bond split reasoning, per-ticker selection reasoning). Backs the `explain_allocation` LLM tool. (`CON-api-explain-contract`, `DEC-fastapi-endpoint-surface`)
- [ ] **BACKEND-06**: `GET /api/etfs/{ticker}` returns ETF metadata (name, asset class, expense ratio, top holdings, AUM, sector breakdown, Morningstar rating) for any ticker in the supported universe; returns 404 otherwise. Wraps `src/etf_fetcher.py` + `src/morning_star_rating.py`. Backs the `get_etf_details` LLM tool. (`CON-api-etfs-detail-contract`, `DEC-fastapi-endpoint-surface`)

### Database (Supabase Postgres)

- [ ] **DB-01**: A single SQL migration file is produced that can be pasted into the Supabase SQL Editor and creates all 4 tables, types, and constraints in one execution (`REQ-phase-2-database-init`, `CON-single-sql-migration-file`)
- [ ] **DB-02**: The migration creates exactly 4 tables and no others: `profiles`, `portfolios`, `portfolio_history`, `chat_messages` (`DEC-db-tables-fixed`)
- [ ] **DB-03**: `profiles` is linked to Supabase `auth.users` by foreign key on user id, and stores user basic metadata only (not the 12 financial fields) (`CON-profiles-table-auth-link`)
- [ ] **DB-04**: `portfolios` stores both the 12 explicit input features (each as its own typed column matching the frozen enums) AND an `allocated_assets` JSONB column shaped as a ticker→weight map summing to 1.0 (`CON-portfolios-table-stores-12-features`, `CON-allocated-assets-jsonb-shape`)
- [ ] **DB-05**: Row-Level Security policies are enabled on all 4 tables with per-user ownership via `auth.uid()` (resolves open question 6 from `.planning/INGEST-CONFLICTS.md`)

### Onboarding (Next.js + Shadcn)

- [ ] **ONBOARD-01**: A Next.js (App Router) project is bootstrapped in `frontend/` with Tailwind CSS + Shadcn/ui installed and styling tokens defined (`REQ-phase-3-frontend-scaffold-and-onboarding`)
- [ ] **ONBOARD-02**: A multi-step investor onboarding form (not a single long page) captures all 12 frozen profile fields using Shadcn/ui components, with type/enum constraints enforced (`REQ-feat-onboarding-form`)
- [ ] **ONBOARD-03**: All client-side form validation uses Zod schemas matching the frozen 12-field types/enums (`DEC-frontend-stack`, `REQ-feat-onboarding-form`)
- [ ] **ONBOARD-04**: On submit, the profile is persisted to Supabase (`profiles` + `portfolios`), the form calls `POST /api/recommend`, and the returned baseline allocation is stored as `ml_result` in `portfolios.allocated_assets` (`REQ-feat-onboarding-form`)
- [ ] **ONBOARD-05**: Supabase authentication is wired in (sign-up / sign-in) so each onboarding submission is tied to an authenticated `auth.users` id (resolves open question 6)

### Dashboard (Recharts/Tremor)

- [ ] **DASH-01**: An authenticated dashboard route renders the active allocation as a donut/pie chart (Recharts or Tremor) for any user who has completed onboarding (`REQ-feat-realtime-dashboard`, `REQ-phase-4-dashboard`)
- [ ] **DASH-02**: The dashboard calls `POST /api/current-prices` with the active allocation's tickers and computes an aggregate current asset value displayed alongside the chart (`REQ-feat-realtime-dashboard`)
- [ ] **DASH-03**: A Recharts-driven performance timeline is rendered from `portfolio_history` rows, alongside ROI highlights (`REQ-feat-realtime-dashboard`)
- [ ] **DASH-04**: The dashboard visibly distinguishes the frozen baseline (`ml_result`) from the live `current_allocation` whenever they differ — both states are present in UI state and the user can see the split (`CON-state-isolation-invariants`)
- [ ] **DASH-05**: When `current_allocation ≠ ml_result`, the comparative banner is rendered verbatim (or near-verbatim) per spec: "Your original ML-recommended allocation was X, but it has been adjusted to Y based on your conversational request." (`CON-state-isolation-invariants`)

### Chat (Vercel AI SDK + 7-Tool Closed Set)

- [ ] **CHAT-01**: A slide-out drawer (or split-screen) chat surface is present on the dashboard, built on the Vercel AI SDK (`REQ-feat-conversational-finetuning-drawer`, `DEC-vercel-ai-sdk-chat-layer`)
- [ ] **CHAT-02**: The chat layer registers exactly the 7-tool closed set and no others: `explain_allocation`, `get_etf_details`, `get_portfolio_value`, `adjust_allocation`, `swap_etf`, `reset_to_baseline`, `update_investor_profile` (`DEC-llm-tools-closed-set-v1`, `CON-llm-tool-surface-is-closed-set`)
- [ ] **CHAT-03**: `adjust_allocation` accepts signed percentage-point deltas (e.g., `equity_delta: +0.08`, `bond_delta: -0.08`), applies them to `current_allocation`, and re-normalizes the total back to exactly 1.0; tickers remain stable, only weights shift; does NOT re-run the ML pipeline (`CON-adjust-allocation-tool-contract`). Resolves open question 3 by enumerating the asset-class delta-key taxonomy and the clamp/renormalize order before implementation.
- [ ] **CHAT-04**: `update_investor_profile` accepts a partial dictionary of the 12 profile fields, merges onto the stored profile, wipes any qualitative overrides accumulated via `adjust_allocation`/`swap_etf`, and re-calls `POST /api/recommend` with the merged 12-field payload to establish a brand-new `ml_result`; `current_allocation` is then reset to equal the new `ml_result` (resolves open question 4) (`CON-update-investor-profile-tool-contract`)
- [ ] **CHAT-05**: Tool execution mutates the active chart state in the current view instantly (no full reload), and chat sequence is persisted to Supabase `chat_messages`; allocation changes are also persisted to `portfolios` and (where applicable) `portfolio_history` (`REQ-feat-conversational-finetuning-drawer`)
- [ ] **CHAT-06**: `explain_allocation` accepts an optional `target` (`"ml_result"` | `"current_allocation"`, default current) and calls `POST /api/explain`, returning structured rationale to the LLM (read-only; no state mutation) (`CON-explain-allocation-tool-contract`)
- [ ] **CHAT-07**: `get_etf_details` accepts a single `ticker` and calls `GET /api/etfs/{ticker}`, returning ETF metadata to the LLM (read-only; no state mutation) (`CON-get-etf-details-tool-contract`)
- [ ] **CHAT-08**: `get_portfolio_value` accepts an optional `target` and combines `POST /api/current-prices` results with the in-memory allocation and `InvestmentCapital` to return total USD value + per-ticker valuation to the LLM (read-only; no state mutation) (`CON-get-portfolio-value-tool-contract`)
- [ ] **CHAT-09**: `swap_etf` accepts `old_ticker` and `new_ticker`, validates that `new_ticker` is in the supported ETF universe AND shares the same asset class as `old_ticker`, then transfers the weight 1:1 in `current_allocation`; rejects with an error message to the LLM when validation fails (`CON-swap-etf-tool-contract`)
- [ ] **CHAT-10**: `reset_to_baseline` takes no inputs, copies `ml_result` into `current_allocation`, makes no backend call, and causes the comparative banner to disappear (`CON-reset-to-baseline-tool-contract`)

## v2 Requirements

Deferred to future releases. Tracked but not in current roadmap.

### Backend extensions

- **BACKEND-V2-01**: Additional FastAPI endpoints beyond the initial 2 (e.g., portfolio analytics, backtesting, explanations)
- **BACKEND-V2-02**: Improving Stage 1 risk profile classifier accuracy above the current ~46% baseline by addressing circular labeling (separate workstream from migration)

### LLM extensions

- **CHAT-V2-01**: Additional LLM tools beyond the v1 closed set of 7 (requires re-opening `DEC-llm-tools-closed-set-v1`). Candidates: `get_performance_history`, `save_portfolio_snapshot`, `simulate_scenario`, `lock_holding`, `get_rebalancing_recommendation`.
- **CHAT-V2-02**: Streaming explanations rendered in the chat surface (the `explain_allocation` tool exists in v1; v2 turns its output into a streamed UI experience)

### Platform

- **PLAT-V2-01**: Native mobile client
- **PLAT-V2-02**: Real-money brokerage integration / order execution

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Rewriting the ML pipeline in TypeScript | `CON-financial-brain-stays-python` — Python brain is wrapped, not rewritten |
| Letting Next.js call yfinance directly | `CON-financial-brain-stays-python` — must route through FastAPI |
| LLM tools outside the v1 closed set of 7 | `DEC-llm-tools-closed-set-v1`, `CON-llm-tool-surface-is-closed-set` |
| Letting LLM ingest raw user profiles or arbitrarily override ML | Anti-decision in `DEC-llm-tools-closed-set-v1` |
| More than 4 database tables | `DEC-db-tables-fixed` |
| Adding or removing investor profile fields | `DEC-investor-profile-schema-frozen` |
| Real-money brokerage / order execution | Advisory only; out of scope for this migration |
| Native mobile app | Web-first; responsive Next.js is the only client surface |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BACKEND-01 | Phase 1 | Pending |
| BACKEND-02 | Phase 1 | Pending |
| BACKEND-03 | Phase 1 | Pending |
| BACKEND-04 | Phase 1 | Pending |
| BACKEND-05 | Phase 1 | Pending |
| BACKEND-06 | Phase 1 | Pending |
| DB-01 | Phase 2 | Pending |
| DB-02 | Phase 2 | Pending |
| DB-03 | Phase 2 | Pending |
| DB-04 | Phase 2 | Pending |
| DB-05 | Phase 2 | Pending |
| ONBOARD-01 | Phase 3 | Pending |
| ONBOARD-02 | Phase 3 | Pending |
| ONBOARD-03 | Phase 3 | Pending |
| ONBOARD-04 | Phase 3 | Pending |
| ONBOARD-05 | Phase 3 | Pending |
| DASH-01 | Phase 4 | Pending |
| DASH-02 | Phase 4 | Pending |
| DASH-03 | Phase 4 | Pending |
| DASH-04 | Phase 4 | Pending |
| DASH-05 | Phase 4 | Pending |
| CHAT-01 | Phase 5 | Pending |
| CHAT-02 | Phase 5 | Pending |
| CHAT-03 | Phase 5 | Pending |
| CHAT-04 | Phase 5 | Pending |
| CHAT-05 | Phase 5 | Pending |
| CHAT-06 | Phase 5 | Pending |
| CHAT-07 | Phase 5 | Pending |
| CHAT-08 | Phase 5 | Pending |
| CHAT-09 | Phase 5 | Pending |
| CHAT-10 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-18*
*Last updated: 2026-05-19 after LLM tool-surface expansion (2 → 7 tools; FastAPI surface 2 → 4 endpoints)*
