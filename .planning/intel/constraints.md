# Constraints Intel

Synthesized from SPECs (and SPEC-like clauses inside the PRD). Constraints are non-negotiable technical contracts.

---

## CON-api-recommend-contract

- **Source:** docs/MIGRATION-SPEC.md § "Feature 1"
- **Type:** api-contract
- **Endpoint:** `POST /api/recommend`
- **Contract:**
  - Input: JSON object containing the 12 investor profile fields with the types/enums frozen in `DEC-investor-profile-schema-frozen`.
  - Output: an asset-weights allocation. Shape implied by spec example: a map of ticker → weight (e.g., `{"SPY": 0.40, "QQQ": 0.35, "AGG": 0.25}`), with weights summing to 1.0.
- **Notes:** This is the canonical entry point that re-runs the full 4-stage ML pipeline. Called both during onboarding and on every `update_investor_profile` invocation.

---

## CON-api-current-prices-contract

- **Source:** docs/MIGRATION-SPEC.md § "Feature 1"
- **Type:** api-contract
- **Endpoint:** `POST /api/current-prices`
- **Contract:**
  - Input: a list of tickers.
  - Output: live market prices per ticker, sourced from yfinance.
- **Notes:** Used by the dashboard to compute aggregate current asset value in real time.

---

## CON-allocated-assets-jsonb-shape

- **Source:** docs/MIGRATION-SPEC.md § "Database Requirements"
- **Type:** schema
- **Constraint:** The `portfolios.allocated_assets` column is JSONB. Its shape is an object mapping ticker symbol → numeric weight (0.0–1.0), with weights summing to 1.0. Example: `{"SPY": 0.40, "QQQ": 0.35, "AGG": 0.25}`.

---

## CON-portfolios-table-stores-12-features

- **Source:** docs/MIGRATION-SPEC.md § "Database Requirements"
- **Type:** schema
- **Constraint:** The `portfolios` table MUST store both:
  - The 12 explicit input features ("dry data"), each as its own column with the type/enum from `DEC-investor-profile-schema-frozen`.
  - The output `allocated_assets` JSONB allocation map.

---

## CON-profiles-table-auth-link

- **Source:** docs/MIGRATION-SPEC.md § "Database Requirements"
- **Type:** schema
- **Constraint:** The `profiles` table MUST be linked to Supabase's `auth.users` (foreign-key relationship by user id). It holds basic user metadata only — the 12 financial input features live in `portfolios`, not `profiles`.

---

## CON-single-sql-migration-file

- **Source:** docs/MIGRATION-SPEC.md § "Database Requirements" + Phase 2
- **Type:** schema
- **Constraint:** A single SQL migration file MUST be produced, suitable for paste-execution in the Supabase SQL Editor, creating all 4 tables, types, and constraints in one shot.

---

## CON-adjust-allocation-tool-contract

- **Source:** docs/MIGRATION-SPEC.md § "Tool definitions / adjust_allocation"
- **Type:** protocol (LLM tool contract)
- **Constraint:**
  - Inputs: signed percentage-point deltas, e.g., `equity_delta: +0.08`, `bond_delta: -0.08`.
  - Behavior: applies deltas to the current active allocation, then re-normalizes the total back to exactly 1.0.
  - Invariants: tickers/ETFs in the allocation remain stable — only their weights shift. Does NOT re-run the ML pipeline.

---

## CON-update-investor-profile-tool-contract

- **Source:** docs/MIGRATION-SPEC.md § "Tool definitions / update_investor_profile"
- **Type:** protocol (LLM tool contract)
- **Constraint:**
  - Inputs: a partial dictionary of the 12 profile fields (only changed fields supplied).
  - Behavior: merges the partial onto the stored profile, WIPES any qualitative overrides accumulated via `adjust_allocation`, and re-runs the full Python ML pipeline from scratch to establish a brand new baseline portfolio.
  - Trigger discipline: only invoked when the user explicitly corrects a factual profile feature (e.g., "my horizon is 30 years, not 10").

---

## CON-state-isolation-invariants

- **Source:** docs/MIGRATION-SPEC.md § "State isolation rules"
- **Type:** protocol (state machine)
- **Constraint:**
  - `ml_result` is immutable except via `update_investor_profile`.
  - `current_allocation` is the only state mutated by `adjust_allocation`.
  - When `current_allocation ≠ ml_result`, the comparative banner must be displayed in the UI verbatim (or near-verbatim) to the spec copy.

---

## CON-llm-tool-surface-is-closed-set

- **Source:** docs/MIGRATION-SPEC.md § "LLM Layer Architecture / Tool surface (closed set of 7)"
- **Type:** protocol
- **Constraint:** The LLM has access to EXACTLY 7 tools — a closed set — and no others. The closed set is:
  - Read-only: `explain_allocation`, `get_etf_details`, `get_portfolio_value`
  - Mutating: `adjust_allocation`, `swap_etf`, `reset_to_baseline`, `update_investor_profile`
- **Invariants:** The LLM must not ingest raw user profiles manually (only via `update_investor_profile`), must not arbitrarily override the ML model (only `update_investor_profile` writes `ml_result`), and must not register any tool outside this closed set with the Vercel AI SDK.
- **Amendment history:** v1 (2026-05-18) declared a 2-tool surface; expanded to 7 on 2026-05-19 while preserving the closed-set + state-isolation invariants.

---

## CON-swap-etf-tool-contract

- **Source:** docs/MIGRATION-SPEC.md § "Tool surface / swap_etf"
- **Type:** protocol (LLM tool contract)
- **Constraint:**
  - Inputs: `old_ticker: string`, `new_ticker: string`.
  - Pre-conditions: `old_ticker` must exist in `current_allocation`; `new_ticker` must belong to the supported ETF universe AND share the same asset class as `old_ticker`.
  - Behavior: transfer the weight from `old_ticker` to `new_ticker` 1:1; remove the old key, add the new key at the same weight. Total still sums to 1.0. Asset-class composition (equity/bond split) is preserved.
  - Failure mode: when pre-conditions fail, the tool returns an error to the LLM; no mutation occurs.
- **State effect:** mutates `current_allocation` only; may trigger the divergence banner.

---

## CON-reset-to-baseline-tool-contract

- **Source:** docs/MIGRATION-SPEC.md § "Tool surface / reset_to_baseline"
- **Type:** protocol (LLM tool contract)
- **Constraint:**
  - Inputs: none.
  - Behavior: set `current_allocation := ml_result` (structural copy). No backend call.
  - Post-condition: divergence is zero; comparative banner disappears.
- **State effect:** mutates `current_allocation` only; eliminates any divergence from `ml_result`.

---

## CON-explain-allocation-tool-contract

- **Source:** docs/MIGRATION-SPEC.md § "Tool surface / explain_allocation"
- **Type:** protocol (LLM tool contract)
- **Constraint:**
  - Inputs: optional `target: "ml_result" | "current_allocation"` (default: `"current_allocation"`).
  - Output: structured object — predicted risk class, top profile-feature drivers, equity/bond split, per-ticker selection reasoning.
  - Behavior: read-only. Calls `POST /api/explain` on the FastAPI wrapper.
- **State effect:** none.

---

## CON-get-etf-details-tool-contract

- **Source:** docs/MIGRATION-SPEC.md § "Tool surface / get_etf_details"
- **Type:** protocol (LLM tool contract)
- **Constraint:**
  - Inputs: `ticker: string`.
  - Output: name, asset class, expense ratio, top holdings, AUM, sector breakdown, Morningstar rating.
  - Behavior: read-only. Calls `GET /api/etfs/{ticker}` (wraps `src/etf_fetcher.py` + `src/morning_star_rating.py`).
- **State effect:** none.

---

## CON-get-portfolio-value-tool-contract

- **Source:** docs/MIGRATION-SPEC.md § "Tool surface / get_portfolio_value"
- **Type:** protocol (LLM tool contract)
- **Constraint:**
  - Inputs: optional `target: "ml_result" | "current_allocation"` (default: `"current_allocation"`).
  - Output: total USD value + per-ticker valuation + percent change from purchase basis (if `portfolio_history` is non-empty).
  - Behavior: read-only. Calls `POST /api/current-prices` and combines with the in-memory allocation and `InvestmentCapital`. No new endpoint required.
- **State effect:** none.

---

## CON-api-explain-contract

- **Source:** docs/MIGRATION-SPEC.md § "Feature 1" (expanded 2026-05-19)
- **Type:** api-contract
- **Endpoint:** `POST /api/explain`
- **Contract:**
  - Input: JSON object containing (a) the 12 investor profile fields and (b) the allocation map to explain (ticker → weight).
  - Output: structured rationale — predicted risk class with feature contributions, equity/bond split with reasoning, per-ticker selection reasoning (AUM rank, asset-class fit).
- **Notes:** Backs the `explain_allocation` LLM tool. Reuses the existing 4-stage models without retraining.

---

## CON-api-etfs-detail-contract

- **Source:** docs/MIGRATION-SPEC.md § "Feature 1" (expanded 2026-05-19)
- **Type:** api-contract
- **Endpoint:** `GET /api/etfs/{ticker}`
- **Contract:**
  - Input: ticker symbol in the URL path.
  - Output: ETF metadata object — name, asset class, expense ratio, top holdings, AUM, sector breakdown, Morningstar rating. Returns 404 when ticker is not in the supported ETF universe.
- **Notes:** Backs the `get_etf_details` LLM tool. Wraps `src/etf_fetcher.py` + `src/morning_star_rating.py`.

---

## CON-financial-brain-stays-python

- **Source:** docs/MIGRATION-SPEC.md § "System Architecture"
- **Type:** nfr (architectural boundary)
- **Constraint:** All core financial formulas, the 4-stage ML pipeline, and real-time ETF price fetching MUST remain in the Python origin repo. Next.js does not implement ML inference or call yfinance directly; it goes through FastAPI.
