-- =============================================================================
-- Migration 002 — add `holdings` JSONB column to `portfolios`
-- =============================================================================
-- Paste into the Supabase SQL Editor and run.
--
-- `holdings` snapshots per-ticker shares + buying price at portfolio creation
-- so the dashboard can compute P&L (current value vs cost basis) without
-- re-querying historical prices.
--
-- Shape:
--   {
--     "VTI":  {"shares": 12.345, "buying_price": 245.10, "bought_at": "2026-05-25T14:30:00Z"},
--     "AGG":  {"shares":  9.876, "buying_price": 102.30, "bought_at": "2026-05-25T14:30:00Z"},
--     ...
--   }
--
-- Nullable so existing rows (created before this migration) stay valid.
-- =============================================================================

alter table public.portfolios
    add column if not exists holdings jsonb;
