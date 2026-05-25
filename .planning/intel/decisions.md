# Decisions Intel

Synthesized from classified ADRs and PRD-asserted fixed architecture decisions.
Each entry preserves provenance. LOCKED entries cannot be silently overridden downstream.

---

## DEC-3tier-architecture

- **Source:** docs/MIGRATION-SPEC.md (PRD, asserted as fixed)
- **Status:** locked-ish (PRD asserts as the migration target — treat as architectural decision)
- **Scope:** overall system topology
- **Decision:** The system MUST be a 3-tier distributed architecture composed of:
  1. **Next.js frontend orchestrator** (TypeScript, Tailwind, Shadcn/ui, Recharts/Tremor) — handles UI, onboarding, and orchestrates calls between Supabase and the Python financial API.
  2. **Supabase PostgreSQL database** — stores user profiles, portfolios, historical valuations, and chat history.
  3. **Python FastAPI backend** — retains the 4-stage ML pipeline and real-time ETF price fetching (yfinance); the financial brain stays in Python.
- **Rationale (from spec):** Migration from single-file Streamlit prototype to a production-grade distributed architecture; the financial logic stays in Python while UI/state/chat move to the modern web stack.

---

## DEC-investor-profile-schema-frozen

- **Source:** docs/MIGRATION-SPEC.md § "Data Model — 12 Investor Profile Fields"
- **Status:** locked-ish (PRD declares "exactly 12 specific investor profile fields")
- **Scope:** investor profile data schema across all 3 tiers (frontend form, DB, ML pipeline)
- **Decision:** Investor profile schema is FROZEN to exactly 12 fields with the following types/constraints:
  - `Age` (integer)
  - `Gender` (enum: Male, Female)
  - `Education` (enum: High school, Bachelor, Master)
  - `MaritalStatus` (enum: Single, Married)
  - `HouseholdSize` (integer)
  - `Income` (numeric)
  - `InvestmentGoal` (enum: Child education, Retirement, Home purchase, Other)
  - `InvestmentHorizon` (integer, years)
  - `InvestmentCapital` (numeric)
  - `RiskTolerance` (enum: Low, Medium, High)
  - `FinancialInvolvement` (enum: Low, Medium, High)
  - `ExperienceYears` (integer)
- **Rationale (from spec):** Mapped from the original data schema; the onboarding form, DB tables, and ML pipeline must all conform.

---

## DEC-db-tables-fixed

- **Source:** docs/MIGRATION-SPEC.md § "Database Requirements"
- **Status:** locked-ish
- **Scope:** Supabase schema
- **Decision:** Exactly 4 database tables, no more, no fewer:
  1. **`profiles`** — linked to `auth.users`, user basic metadata
  2. **`portfolios`** — stores the 12 explicit input features ("dry data") AND the output `allocated_assets` as a JSONB allocation map (e.g., `{"SPY": 0.40, "QQQ": 0.35, "AGG": 0.25}`)
  3. **`portfolio_history`** — time-series valuation records for dashboard equity/performance curves
  4. **`chat_messages`** — persists the Vercel AI SDK chat sequence
- **Note:** A single SQL migration file MUST be generated for the Supabase SQL Editor that creates all four.

---

## DEC-llm-tools-closed-set-v1

- **Source:** docs/MIGRATION-SPEC.md § "LLM Layer Architecture / Tool surface"
- **Status:** locked-ish
- **Scope:** LLM tool-calling contract
- **Decision:** The LLM layer exposes a **closed set of EXACTLY 7 first-class tools**. Adding an 8th tool requires re-opening this decision.

  **Read-only tools (3):**
  1. **`explain_allocation`** — returns structured ML rationale for an allocation (risk-profile feature contributions, equity/bond split reasoning, per-ETF selection reasoning). Backed by `POST /api/explain`.
  2. **`get_etf_details`** — returns metadata for a single ETF (name, asset class, expense ratio, top holdings, AUM, Morningstar rating). Backed by `GET /api/etfs/{ticker}`.
  3. **`get_portfolio_value`** — returns live aggregate USD value + per-ticker valuation for `ml_result` or `current_allocation`, using `POST /api/current-prices` + `InvestmentCapital`.

  **Mutating tools (4):**
  4. **`adjust_allocation`** — qualitative/nuanced risk tweaks via signed percentage-point deltas (e.g., `equity_delta: +0.08`, `bond_delta: -0.08`). Applies onto `current_allocation`, re-normalizes to 1.0, tickers stable.
  5. **`swap_etf`** — replace one ETF with another at the same weight. Validates same-asset-class; rejects otherwise. Mutates `current_allocation` only.
  6. **`reset_to_baseline`** — copy `ml_result` → `current_allocation`. No backend call. Eliminates divergence.
  7. **`update_investor_profile`** — used ONLY when the user explicitly corrects a factual profile feature. Accepts a partial dictionary of the 12 profile fields, merges, wipes any qualitative overrides, re-runs the full Python ML pipeline via `POST /api/recommend` to establish a brand-new `ml_result`, and resets `current_allocation := new ml_result`.

- **Anti-decision (explicit):** The LLM never arbitrarily overrides the ML model. All `ml_result` mutations flow exclusively through `update_investor_profile`. The LLM does not ingest raw user profiles manually.
- **Closed-set invariant:** No tool outside this set is registered with the Vercel AI SDK. Any future tool addition is a deliberate amendment, not an emergent capability.
- **Supersedes:** previously named `DEC-llm-tools-exactly-two` (2026-05-18); expanded 2026-05-19 after user requested a richer tool surface while preserving the state-isolation contract.

---

## DEC-state-isolation-baseline-vs-current

- **Source:** docs/MIGRATION-SPEC.md § "State isolation rules"
- **Status:** locked-ish
- **Scope:** application state model (frontend + DB)
- **Decision:** Two distinct allocation states must coexist:
  - **`ml_result` (frozen baseline)** — immutable unless a profile is factually updated via `update_investor_profile`.
  - **`current_allocation` (mutable)** — what the Next.js UI actively renders and passes to charts; mutated by `adjust_allocation`.
- **UI requirement:** When the two diverge, display a comparative banner:
  > "Your original ML-recommended allocation was X, but it has been adjusted to Y based on your conversational request."

---

## DEC-vercel-ai-sdk-chat-layer

- **Source:** docs/MIGRATION-SPEC.md § "LLM Layer Architecture / Overview"
- **Status:** locked-ish
- **Scope:** chat/LLM SDK choice
- **Decision:** The Vercel AI SDK is the LLM chat layer. It sits on top of the financial engine and hosts the conversational fine-tuning environment using the two tool calls defined above.

---

## DEC-fastapi-endpoint-surface

- **Source:** docs/MIGRATION-SPEC.md § "Feature 1 — Python FastAPI Backend Wrapper"
- **Status:** locked-ish
- **Scope:** Python backend HTTP surface
- **Decision:** The FastAPI wrapper exposes 4 endpoints (v1 set):
  - `POST /api/recommend` — takes the 12 inputs, returns asset weights.
  - `POST /api/current-prices` — takes tickers, uses yfinance to return live market prices for real-time dashboard calculations.
  - `POST /api/explain` — returns structured ML rationale for an allocation (risk-profile feature contributions, equity/bond split reasoning, per-ETF selection reasoning). Backs `explain_allocation`.
  - `GET /api/etfs/{ticker}` — returns metadata for a single ETF (name, asset class, expense ratio, top holdings, AUM, Morningstar rating). Wraps `src/etf_fetcher.py` + `src/morning_star_rating.py`. Backs `get_etf_details`.
- **Note:** Expanded 2026-05-19 from 2 endpoints to 4 alongside the LLM tool-surface expansion (`DEC-llm-tools-closed-set-v1`).

---

## DEC-frontend-stack

- **Source:** docs/MIGRATION-SPEC.md § "Frontend orchestrator"
- **Status:** locked-ish
- **Scope:** frontend technology choices
- **Decision:** Frontend stack is Next.js (App Router) + TypeScript + Tailwind CSS + Shadcn/ui + Recharts/Tremor. Client-side form validation uses Zod (per Feature 2).
