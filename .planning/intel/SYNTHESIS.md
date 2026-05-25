# Synthesis Summary

Entry point for downstream consumers (e.g., `gsd-roadmapper`). Reflects the synthesized intel state after ingesting the single classified PRD.

## Ingest snapshot

- Mode: `new`
- Existing `.planning/` decisions/requirements: none (only `codebase/` reference docs present)
- Precedence applied: `ADR > SPEC > PRD > DOC` (default)
- Cycle detection: N/A (single doc, no cross-refs)

## Doc counts by type

- ADR: 0
- SPEC: 0
- PRD: 1 — `docs/MIGRATION-SPEC.md` (classified high confidence, manifest_override=true, locked=false, precedence=2)
- DOC: 0
- UNKNOWN: 0
- **Total synthesized:** 1

## Decisions captured (8)

All recorded in `.planning/intel/decisions.md`. None are formally LOCKED (no ADRs exist), but all are asserted as fixed by the PRD and tagged "locked-ish":

- `DEC-3tier-architecture` — Next.js + Supabase Postgres + Python FastAPI
- `DEC-investor-profile-schema-frozen` — exactly 12 fields with stated types/enums
- `DEC-db-tables-fixed` — exactly 4 tables: `profiles`, `portfolios`, `portfolio_history`, `chat_messages`
- `DEC-llm-tools-exactly-two` — `adjust_allocation`, `update_investor_profile` (closed set)
- `DEC-state-isolation-baseline-vs-current` — frozen `ml_result` vs mutable `current_allocation`
- `DEC-vercel-ai-sdk-chat-layer` — Vercel AI SDK as chat layer
- `DEC-fastapi-endpoint-surface` — `POST /api/recommend`, `POST /api/current-prices`
- `DEC-frontend-stack` — Next.js + TS + Tailwind + Shadcn/ui + Recharts/Tremor + Zod

## Requirements extracted (9)

All recorded in `.planning/intel/requirements.md`:

- Feature requirements (4):
  - `REQ-feat-fastapi-backend-wrapper`
  - `REQ-feat-onboarding-form`
  - `REQ-feat-realtime-dashboard`
  - `REQ-feat-conversational-finetuning-drawer`
- Phase requirements (5):
  - `REQ-phase-1-fastapi-backend`
  - `REQ-phase-2-database-init`
  - `REQ-phase-3-frontend-scaffold-and-onboarding`
  - `REQ-phase-4-dashboard`
  - `REQ-phase-5-llm-chat-integration`

Phase requirements implicitly compose the feature requirements; the roadmapper can collapse or sequence them as needed.

## Constraints captured (11)

All recorded in `.planning/intel/constraints.md`. Breakdown by type:

- api-contract: 2 (`CON-api-recommend-contract`, `CON-api-current-prices-contract`)
- schema: 4 (`CON-allocated-assets-jsonb-shape`, `CON-portfolios-table-stores-12-features`, `CON-profiles-table-auth-link`, `CON-single-sql-migration-file`)
- protocol: 4 (`CON-adjust-allocation-tool-contract`, `CON-update-investor-profile-tool-contract`, `CON-state-isolation-invariants`, `CON-llm-tool-surface-is-closed-set`)
- nfr: 1 (`CON-financial-brain-stays-python`)

## Context topics (6)

All recorded in `.planning/intel/context.md`:

- Migration origin and goal
- Why this migration architecture (rationale)
- LLM design philosophy
- Baseline-vs-current allocation UX intent
- Execution sequencing (5-phase plan)
- Repo layout implication

## Conflicts

- BLOCKERs: 0
- WARNINGs (competing variants): 0
- INFO (auto-resolved / open questions): 6

Full report: `.planning/INGEST-CONFLICTS.md`

The 6 INFO notes flag open questions that the roadmapper or user should address downstream (delta taxonomy in `adjust_allocation`, current_allocation reset semantics in `update_investor_profile`, monorepo vs split-repo, RLS/auth policy, etc.) — none are blocking.

## Pointers

- Decisions: `.planning/intel/decisions.md`
- Requirements: `.planning/intel/requirements.md`
- Constraints: `.planning/intel/constraints.md`
- Context: `.planning/intel/context.md`
- Conflicts report: `.planning/INGEST-CONFLICTS.md`
- Source doc: `docs/MIGRATION-SPEC.md`
- Source classification: `.planning/intel/classifications/MIGRATION-SPEC-7a3b9c2e.json`
