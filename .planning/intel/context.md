# Context Intel

Running notes synthesized from DOC-level material and PRD context sections. Verbatim or near-verbatim, with source attribution.

---

## Topic: Migration origin and goal

- **Source:** docs/MIGRATION-SPEC.md § "Context"
- Migrating an investment portfolio advisor from a single-file Python Streamlit prototype into a modern, production-grade distributed architecture.
- The state, UI, and LLM chat layer move to a Next.js (App Router) + Supabase stack.
- The financial brain — the 4-stage ML pipeline and real-time ETF price fetching — remains in the Python repository, exposed via a lightweight FastAPI backend.
- Migration spec status (per spec front matter): "Draft PRD — Streamlit prototype → Next.js + Supabase + FastAPI", last updated 2026-05-18.

---

## Topic: Why this migration architecture (rationale)

- **Source:** docs/MIGRATION-SPEC.md § "System Architecture"
- Frontend orchestrator chosen as Next.js to handle user interactions, onboarding, and inter-service orchestration between Supabase and the Python financial API.
- Supabase (PostgreSQL) chosen for the database layer to store user profiles, historical portfolio weights, frozen baseline records, snapshot valuations, and chat histories.
- Python FastAPI chosen as the financial backend because it retains all the legacy financial formulas and ML inference and exposes them as high-performance endpoints. The legacy ML logic is not being rewritten in TypeScript.

---

## Topic: LLM design philosophy

- **Source:** docs/MIGRATION-SPEC.md § "LLM Layer Architecture / Overview"
- The LLM sits ON TOP of the financial engine, not inside it.
- It provides plain-language analysis and hosts an interactive chat environment for conversational fine-tuning.
- Design constraint stated explicitly: the LLM does not ingest raw user profiles manually and never arbitrarily overrides the ML model. All mutations to allocation flow through the two defined tools.

---

## Topic: Baseline-vs-current allocation UX intent

- **Source:** docs/MIGRATION-SPEC.md § "State isolation rules"
- Two distinct allocation states by design: frozen `ml_result` baseline + mutable `current_allocation`.
- When they diverge, the UI shows a comparative banner so the user can see the delta between the original ML recommendation and the conversationally-adjusted state. Verbatim copy from spec:
  > "Your original ML-recommended allocation was X, but it has been adjusted to Y based on your conversational request."

---

## Topic: Execution sequencing (5-phase plan)

- **Source:** docs/MIGRATION-SPEC.md § "Execution Plan"
- Phase 1 — FastAPI backend (Python repo)
- Phase 2 — Database init (Supabase SQL migration)
- Phase 3 — Frontend scaffold + onboarding form (Next.js `frontend/`)
- Phase 4 — Dashboard (Next.js)
- Phase 5 — LLM chat integration (Vercel AI SDK + Supabase persistence)
- Sequencing implies Phase 1 unblocks Phases 3–5 (frontend depends on `/api/recommend` and `/api/current-prices`), and Phase 2 unblocks Phases 3–5 (need DB tables to persist anything).

---

## Topic: Repo layout implication

- **Source:** docs/MIGRATION-SPEC.md § "Execution Plan / Phase 3"
- The Next.js frontend lives in a `frontend/` directory (implied: monorepo or sibling-folder layout inside the current Python repo, since the FastAPI work happens "inside the origin repo" per Phase 1).
- This is not stated as definitively monorepo vs separate-repo; only the `frontend/` path is named.
