# Roadmap: AI-Driven Portfolio Advisor

## Overview

Migrate the existing Streamlit AI Financial Advisor prototype into a 3-tier production architecture (Next.js + Supabase + Python FastAPI) in 5 sequential phases. Phase 1 exposes the existing Python ML pipeline over HTTP — this unblocks every downstream phase that needs a recommendation or live prices. Phase 2 creates the Supabase schema — this unblocks anything that needs to persist a user. Phases 3, 4, and 5 build out the frontend in user-flow order: first you onboard, then you see your portfolio, then you talk to it.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4, 5): Planned milestone work
- Decimal phases (2.1, 2.2, etc.): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: FastAPI Backend Wrapper** - Expose the Python 4-stage ML pipeline and live ETF prices as HTTP endpoints
- [ ] **Phase 2: Supabase Database Initialization** - Single SQL migration creating the 4 fixed tables with RLS
- [ ] **Phase 3: Next.js Scaffold + 12-Field Onboarding Form** - Bootstrap `frontend/` and ship the multi-step Shadcn onboarding flow that produces a baseline allocation
- [ ] **Phase 4: Real-Time Dashboard with Baseline-vs-Current Split** - Render allocation, live valuation, performance timeline, and the `ml_result` vs `current_allocation` divergence
- [ ] **Phase 5: Vercel AI SDK Chat with 7-Tool Closed Set** - Conversational fine-tuning drawer wired to the 7-tool closed set (3 read + 4 mutating), persisting chat to Supabase

## Phase Details

### Phase 1: FastAPI Backend Wrapper
**Goal**: The Python 4-stage ML pipeline, the yfinance live-price layer, and the ETF metadata + explanation surfaces are callable as HTTP endpoints from the Next.js orchestrator, with the financial brain untouched and the wrapper living inside the Python origin repo.
**Depends on**: Nothing (first phase; unblocks Phases 3, 4, 5)
**Requirements**: BACKEND-01, BACKEND-02, BACKEND-03, BACKEND-04, BACKEND-05, BACKEND-06
**Success Criteria** (what must be TRUE):
  1. A FastAPI server starts from the repo root and exposes all 4 endpoints on a documented local port: `POST /api/recommend`, `POST /api/current-prices`, `POST /api/explain`, `GET /api/etfs/{ticker}`.
  2. Sending a valid 12-field JSON profile to `/api/recommend` returns a ticker→weight allocation map whose weights sum to 1.0 (verified by running the existing 4-stage pipeline end-to-end).
  3. Sending a list of tickers to `/api/current-prices` returns live yfinance prices per ticker, with errors surfaced clearly when a ticker is invalid.
  4. Sending a 12-field profile + an allocation map to `/api/explain` returns a structured rationale (risk-class with feature contributions, equity/bond split with reasoning, per-ticker selection reasoning).
  5. Calling `/api/etfs/{ticker}` for a supported ticker returns full ETF metadata (name, asset class, expense ratio, top holdings, AUM, Morningstar rating); unknown tickers return 404.
  6. The existing `python pipeline.py` training entry point and `src/inference.py` CLI still work — the wrapper is additive, not a rewrite.
  7. No yfinance import exists anywhere in the (yet-to-be-built) `frontend/` Next.js code: all live-price calls route through `/api/current-prices`.
**Plans**: TBD

### Phase 2: Supabase Database Initialization
**Goal**: A Supabase Postgres database has the 4 fixed tables, the 12-field investor schema, and per-user RLS policies in place, ready to back the frontend.
**Depends on**: Nothing structurally (can run parallel to Phase 1, but blocks Phases 3, 4, 5)
**Requirements**: DB-01, DB-02, DB-03, DB-04, DB-05
**Success Criteria** (what must be TRUE):
  1. A single `.sql` migration file in the repo can be pasted into the Supabase SQL Editor and run cleanly in one execution, creating all 4 tables, types, and constraints.
  2. After running, exactly these tables exist in Supabase: `profiles`, `portfolios`, `portfolio_history`, `chat_messages` — no more, no fewer.
  3. `profiles` carries a foreign key to `auth.users(id)`; `portfolios` has one typed column per frozen profile field PLUS an `allocated_assets JSONB` column.
  4. Row-Level Security is enabled on all 4 tables and per-user ownership is enforced via `auth.uid()` — a user cannot read or write another user's rows when authenticated.
  5. Inserting a sample portfolio row with a JSONB allocation map (e.g., `{"SPY": 0.40, "QQQ": 0.35, "AGG": 0.25}`) succeeds and round-trips intact on read.
**Plans**: TBD

### Phase 3: Next.js Scaffold + 12-Field Onboarding Form
**Goal**: A new user can authenticate, fill in the 12 frozen investor profile fields through a multi-step Shadcn form, and have their profile persisted alongside a baseline ML-generated portfolio.
**Depends on**: Phase 1, Phase 2
**Requirements**: ONBOARD-01, ONBOARD-02, ONBOARD-03, ONBOARD-04, ONBOARD-05
**Success Criteria** (what must be TRUE):
  1. A Next.js (App Router) project lives in `frontend/` with Tailwind + Shadcn/ui installed, theme tokens defined, and `pnpm dev` (or equivalent) boots the app cleanly.
  2. An unauthenticated visitor is prompted to sign up / sign in; once authenticated they land on the onboarding flow.
  3. The onboarding form is multi-step (not a single long page), uses Shadcn components, and collects all 12 frozen fields with the correct input controls per type (integer / numeric / enum select).
  4. Zod schemas reject invalid input client-side: e.g., enum values outside the allowed set, non-integer ages, missing required fields.
  5. On submit, a row is written to `profiles` and `portfolios` for the authenticated user, `POST /api/recommend` is called with the 12 fields, and the returned allocation is stored as `ml_result` (initial `allocated_assets`) in `portfolios`.
**Plans**: TBD
**UI hint**: yes

### Phase 4: Real-Time Dashboard with Baseline-vs-Current Split
**Goal**: An onboarded user lands on a dashboard that shows their active allocation, live valuation, performance history, and a clear visual separation between the frozen ML baseline and the live (potentially user-tweaked) allocation.
**Depends on**: Phase 1, Phase 2, Phase 3
**Requirements**: DASH-01, DASH-02, DASH-03, DASH-04, DASH-05
**Success Criteria** (what must be TRUE):
  1. An authenticated, onboarded user navigating to the dashboard sees their active allocation rendered as a donut/pie chart (Recharts or Tremor) within a single viewport.
  2. The dashboard fetches live prices via `POST /api/current-prices` for the active tickers and renders an aggregate current asset value alongside the chart, refreshed on view.
  3. A Recharts performance timeline driven by `portfolio_history` rows is visible, along with ROI highlights computed from baseline cost vs current value.
  4. The dashboard internally holds two distinct states — `ml_result` (frozen baseline) and `current_allocation` (mutable) — and visually renders both when they differ (e.g., side-by-side donut comparison or a clearly labelled overlay).
  5. When `current_allocation` diverges from `ml_result`, the comparative banner is shown verbatim per spec: *"Your original ML-recommended allocation was X, but it has been adjusted to Y based on your conversational request."* (At this phase, the divergence is unreachable from the UI — Phase 5 introduces the mutation path — but the banner must render correctly when the two states are seeded differently in a dev/test fixture.)
**Plans**: TBD
**UI hint**: yes

### Phase 5: Vercel AI SDK Chat with 7-Tool Closed Set
**Goal**: A user can open a chat drawer on the dashboard, ask read-only questions about their allocation (rationale, ETF details, current value), and mutate the active portfolio via 4 well-defined write paths — and have the dashboard update instantly with the conversation persisted.
**Depends on**: Phase 1, Phase 2, Phase 3, Phase 4
**Requirements**: CHAT-01, CHAT-02, CHAT-03, CHAT-04, CHAT-05, CHAT-06, CHAT-07, CHAT-08, CHAT-09, CHAT-10
**Success Criteria** (what must be TRUE):
  1. A slide-out (or split-screen) chat drawer is accessible from the dashboard, built on the Vercel AI SDK, with the 7-tool closed set registered and no others: `explain_allocation`, `get_etf_details`, `get_portfolio_value`, `adjust_allocation`, `swap_etf`, `reset_to_baseline`, `update_investor_profile`.
  2. The 3 read-only tools (`explain_allocation`, `get_etf_details`, `get_portfolio_value`) execute without mutating any state; their outputs are surfaced in the chat. `explain_allocation` calls `POST /api/explain`; `get_etf_details` calls `GET /api/etfs/{ticker}`; `get_portfolio_value` calls `POST /api/current-prices` and combines with `InvestmentCapital`.
  3. Calling `adjust_allocation` with signed deltas (per the resolved delta-key taxonomy from open question 3) mutates `current_allocation` only, re-normalizes weights to exactly 1.0, keeps tickers stable, and updates the dashboard chart instantly — without re-calling `POST /api/recommend`.
  4. Calling `swap_etf` with valid `old_ticker` + `new_ticker` (same-asset-class) transfers the weight 1:1 in `current_allocation`; invalid swaps (wrong asset class, unknown ticker) are rejected with a clear error message back to the LLM and no mutation happens.
  5. Calling `reset_to_baseline` copies `ml_result` into `current_allocation` with no backend call; the divergence banner disappears and the chart re-renders to the frozen baseline.
  6. Calling `update_investor_profile` with a partial 12-field dictionary merges it onto the stored profile, wipes qualitative overrides, calls `POST /api/recommend` with the merged payload, and resets BOTH `ml_result` and `current_allocation` to the new recommendation (resolves open question 4).
  7. Every chat message (user + assistant + tool calls) is written to `chat_messages`; allocation changes are persisted to `portfolios` (and, where applicable, `portfolio_history`); on page reload the conversation and the active allocation both reappear intact.
  8. After any of `adjust_allocation` or `swap_etf` causes `current_allocation ≠ ml_result`, the comparative banner from Phase 4 actually appears in-app (now reachable via real user mutation), and disappears the moment `reset_to_baseline` or `update_investor_profile` re-aligns the two states.
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5

Note: Phase 2 has no structural dependency on Phase 1 and could run in parallel by a sufficiently parallel operator; the canonical execution order is sequential as listed.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. FastAPI Backend Wrapper | 0/TBD | Not started | - |
| 2. Supabase Database Initialization | 0/TBD | Not started | - |
| 3. Next.js Scaffold + 12-Field Onboarding Form | 0/TBD | Not started | - |
| 4. Real-Time Dashboard with Baseline-vs-Current Split | 0/TBD | Not started | - |
| 5. Vercel AI SDK Chat with 7-Tool Closed Set | 0/TBD | Not started | - |
