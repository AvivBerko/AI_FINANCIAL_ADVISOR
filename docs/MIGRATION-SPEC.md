# AI-Driven Portfolio Advisor — Migration Specification

**Status:** Draft PRD — Streamlit prototype → Next.js + Supabase + FastAPI
**Last updated:** 2026-05-18

## Context

We are migrating an investment portfolio advisor application from a single-file Python Streamlit prototype into a modern, production-grade distributed architecture. The state, UI, and LLM chat layer are moving to a Next.js (App Router) and Supabase stack. However, the financial brain — including the 4-stage ML pipeline and real-time ETF price fetching — remains in the Python repository, exposed via a lightweight FastAPI backend.

## System Architecture

### Frontend orchestrator
**Next.js** (TypeScript, Tailwind CSS, Shadcn/ui, Recharts/Tremor). Handles user interactions, onboarding, and orchestrates calls between Supabase and the Python financial API.

### Database layer
**Supabase** (PostgreSQL). Stores user profiles, historical portfolio weights, frozen baseline records, snapshot valuations, and chat histories.

### Financial backend API
**Python origin repo** (FastAPI). Retains all core financial formulas and machine learning inference. Exposes high-performance endpoints for portfolio generation and real-time ETF metrics.

## Data Model — 12 Investor Profile Fields

The onboarding form and database must accommodate exactly 12 specific investor profile fields ("dry data") mapped from the original data schema:

| Field | Type | Constraints |
|---|---|---|
| `Age` | integer | — |
| `Gender` | enum | `Male`, `Female` |
| `Education` | enum | `High school`, `Bachelor`, `Master` |
| `MaritalStatus` | enum | `Single`, `Married` |
| `HouseholdSize` | integer | — |
| `Income` | numeric | — |
| `InvestmentGoal` | enum | `Child education`, `Retirement`, `Home purchase`, `Other` |
| `InvestmentHorizon` | integer | years |
| `InvestmentCapital` | numeric | — |
| `RiskTolerance` | enum | `Low`, `Medium`, `High` |
| `FinancialInvolvement` | enum | `Low`, `Medium`, `High` |
| `ExperienceYears` | integer | — |

## Database Requirements

Generate a SQL migration file for the Supabase SQL Editor creating:

1. **`profiles`** — linked to `auth.users`, user basic metadata.
2. **`portfolios`** — stores the 12 explicit input features ("dry data") and the output `allocated_assets` JSONB allocation maps (e.g., `{"SPY": 0.40, "QQQ": 0.35, "AGG": 0.25}`).
3. **`portfolio_history`** — time-series valuation records for the dashboard equity/performance curves.
4. **`chat_messages`** — persists the Vercel AI SDK chat sequence.

## LLM Layer Architecture

### Overview
The LLM layer sits on top of the financial engine via the **Vercel AI SDK**. It does not ingest raw user profiles manually and never arbitrarily overrides the ML model. Instead, it provides plain-language analysis and hosts an interactive chat environment for conversational fine-tuning using a **closed set of 7 first-class tool calls** — 3 read-only and 4 mutating. Every tool either reads state or mutates it through one of four well-defined, state-isolated paths.

### State isolation rules
- Maintain a **frozen `ml_result` baseline**. Immutable unless a profile is factually updated.
- Maintain a separate **mutable `current_allocation`** state. This is what the Next.js UI actively renders and passes to charts.
- If allocations diverge, display a comparative UI banner:
  > "Your original ML-recommended allocation was X, but it has been adjusted to Y based on your conversational request."

### Tool surface (closed set of 7)

The LLM has access to **exactly these 7 tools and no others**. Tools are categorized by their effect on state:

| Tool | Effect | Mutates |
|---|---|---|
| `explain_allocation` | read | — |
| `get_etf_details` | read | — |
| `get_portfolio_value` | read | — |
| `adjust_allocation` | write | `current_allocation` only |
| `swap_etf` | write | `current_allocation` only |
| `reset_to_baseline` | write | `current_allocation` only (resets to `ml_result`) |
| `update_investor_profile` | write | `ml_result` (re-runs pipeline) + resets `current_allocation` |

> Explicitly NOT a tool: `compare_to_baseline` (the baseline-vs-current diff) is a deterministic UI computation derived from the two states; the system prompt can reference it without invoking a tool call.

#### Read-only tools

##### `explain_allocation`
- **Description:** Returns the structured rationale behind an allocation — why the ML pipeline chose this risk profile, this equity/bond split, and these specific ETFs.
- **Inputs:** optional `target: "ml_result" | "current_allocation"` (default: `"current_allocation"`).
- **Output:** an object with the predicted risk class, the top profile-feature drivers, the equity/bond split, and per-ticker selection reasoning (AUM rank, asset-class fit).
- **Backend:** calls `POST /api/explain` on the FastAPI wrapper.

##### `get_etf_details`
- **Description:** Looks up metadata for a single ETF.
- **Inputs:** `ticker: string` (e.g., `"SPY"`).
- **Output:** name, asset class, expense ratio, top holdings, AUM, sector breakdown, Morningstar-style rating.
- **Backend:** calls `GET /api/etfs/{ticker}` (wraps existing `src/etf_fetcher.py` + `src/morning_star_rating.py`).

##### `get_portfolio_value`
- **Description:** Returns the live aggregate value of an allocation given the user's `InvestmentCapital` and current ETF prices.
- **Inputs:** optional `target: "ml_result" | "current_allocation"` (default: `"current_allocation"`).
- **Output:** total USD value + per-ticker valuation + percent change from purchase basis (if `portfolio_history` is non-empty).
- **Backend:** calls `POST /api/current-prices` and combines the response with the in-memory allocation and `InvestmentCapital`. No new endpoint required.

#### Mutating tools

##### `adjust_allocation`
- **Description:** Use for qualitative or nuanced risk tweaks (e.g., "a bit more equity", "reduce bonds slightly").
- **Behavior:** Takes signed percentage-point deltas (e.g., `equity_delta: +0.08`, `bond_delta: -0.08`) and applies them directly onto `current_allocation`. Re-normalizes the total back to 1.0. Tickers/ETFs remain stable; only weights shift.
- **State effect:** mutates `current_allocation`; may trigger the divergence banner.

##### `swap_etf`
- **Description:** Replace one ETF in the active allocation with another at the same weight, preserving the asset-class composition.
- **Inputs:** `old_ticker: string`, `new_ticker: string`.
- **Behavior:** Locate `old_ticker` in `current_allocation`, validate that `new_ticker` belongs to the supported ETF universe AND the same asset class as `old_ticker` (rejected otherwise), then transfer the weight 1:1. Equity/bond split is preserved; total still sums to 1.0.
- **State effect:** mutates `current_allocation`; may trigger the divergence banner.

##### `reset_to_baseline`
- **Description:** Discard all conversational tweaks and restore the frozen ML baseline.
- **Inputs:** none.
- **Behavior:** Set `current_allocation := ml_result`. No backend call.
- **State effect:** eliminates any divergence; the comparative banner disappears.

##### `update_investor_profile`
- **Description:** Use ONLY when the user explicitly corrects a factual profile feature (e.g., "my horizon is 30 years, not 10", "change my income to $120k").
- **Behavior:** Accepts a partial dictionary of the 12 profile fields, merges onto the stored profile, wipes any qualitative overrides, and re-runs the full Python ML pipeline from scratch via `POST /api/recommend` to establish a brand-new `ml_result`. `current_allocation` is reset to equal the new `ml_result`.
- **State effect:** mutates both `ml_result` and `current_allocation`; the banner disappears (they're equal again).

### Anti-decisions (locked)

- The LLM never arbitrarily overrides the ML model. Every mutation goes through one of the 3 write tools above.
- The LLM never ingests raw user profiles manually — only via `update_investor_profile`.
- The tool surface is a **closed set**. Adding an 8th tool requires re-opening this section as an architectural decision.

## Core Features

### Feature 1 — Python FastAPI Backend Wrapper
Wrap the legacy ML pipeline in FastAPI exposing:
- `POST /api/recommend` — takes the 12 inputs, returns asset weights.
- `POST /api/current-prices` — takes tickers, uses yfinance to return live market prices for real-time dashboard calculations.
- `POST /api/explain` — returns structured ML rationale for a given allocation (feature contributions for risk-profile classification, equity/bond split reasoning, per-ETF selection reasoning).
- `GET /api/etfs/{ticker}` — returns metadata for a single ETF (name, asset class, expense ratio, top holdings, AUM, Morningstar rating), wrapping `src/etf_fetcher.py` + `src/morning_star_rating.py`.

These 4 endpoints back the FastAPI surface. The chat layer's read-only tools (`explain_allocation`, `get_etf_details`, `get_portfolio_value`) route through them; `get_portfolio_value` reuses `/api/current-prices` rather than introducing a 5th endpoint.

### Feature 2 — Shadcn Multi-Step Investor Onboarding Form
A premium multi-step frontend form capturing the 12 inputs with clean client-side Zod validation.

### Feature 3 — Real-Time Financial Dashboard
Displays active asset allocations via beautiful donut/pie charts. Queries the Python API for current prices to compute aggregate current asset values, displaying Recharts performance timelines and ROI highlights.

### Feature 4 — Conversational Fine-Tuning Drawer
A slide-out or split-screen chat UI built with Vercel AI SDK that registers the **7-tool closed set** defined in the LLM Layer Architecture section above — 3 read-only (`explain_allocation`, `get_etf_details`, `get_portfolio_value`) and 4 mutating (`adjust_allocation`, `swap_etf`, `reset_to_baseline`, `update_investor_profile`). All write tools alter state instantly in the active view per the state isolation rules.

## Execution Plan

| Phase | Title | Scope |
|---|---|---|
| 1 | **FastAPI backend** | Build Python FastAPI server endpoints inside the origin repo to handle ML logic and current ETF price endpoints. |
| 2 | **Database init** | Write and supply the PostgreSQL initialization script for Supabase tables, types, and constraints. |
| 3 | **Frontend scaffold + onboarding** | Initialize the Next.js frontend repository (in `frontend/`), styling definitions, and the 12-field Shadcn onboarding form. |
| 4 | **Dashboard** | Construct the dashboard layout tracking the split between baseline ML responses and live overridden weights. |
| 5 | **LLM chat integration** | Wire up the Vercel AI SDK chat agent to execute tool handlers that mutate frontend chart states and save records to Supabase. |
