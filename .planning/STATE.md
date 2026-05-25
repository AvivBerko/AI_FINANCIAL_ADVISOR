# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-18)

**Core value:** A user fills in 12 financial profile fields, sees an ML-recommended ETF portfolio they can trust, and can conversationally adjust it — without the LLM ever silently overriding the model.
**Current focus:** Phase 1 — FastAPI Backend Wrapper

## Current Position

Phase: 1 of 5 (FastAPI Backend Wrapper)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-05-18 — Roadmap created from ingested PRD (docs/MIGRATION-SPEC.md)

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: —

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent decisions affecting current work:

- Migration: 3-tier architecture (Next.js + Supabase + Python FastAPI) locked-ish per PRD
- Migration: Python ML pipeline is wrapped, not rewritten — Phase 1 scope
- Migration: FastAPI surface initially exactly `POST /api/recommend` + `POST /api/current-prices`
- Migration: Monorepo confirmed — Python at root, Next.js in `frontend/`

### Pending Todos

None yet.

### Blockers/Concerns

Open questions (non-blocking, see PROJECT.md "Open questions / first-week decisions"):
- Phase 5 prerequisite: define `adjust_allocation` delta-key taxonomy + clamp/renormalize order (open Q3)
- Phase 5 prerequisite: confirm `current_allocation` reset semantics on `update_investor_profile` (open Q4 — currently codified in CHAT-04 as "reset to new ml_result")
- Phase 2/3 input: decide RLS policies + sign-up/sign-in UX (open Q6 — currently codified in DB-05 and ONBOARD-05)

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-05-18
Stopped at: ROADMAP.md / STATE.md / REQUIREMENTS.md / PROJECT.md written from intel ingest
Resume file: None
