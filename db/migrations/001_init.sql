-- =============================================================================
-- AI Financial Advisor — Option A schema (Streamlit + Supabase)
-- =============================================================================
-- Paste this entire file into the Supabase SQL Editor and run once.
-- It is idempotent (uses IF NOT EXISTS / DROP POLICY IF EXISTS) so re-running
-- on an already-initialised project is safe.
--
-- Tables:
--   profiles         — one row per user, holds the 12 frozen investor fields
--   portfolios       — one row per user, holds ml_result + current_allocation
--   chat_messages    — one row per chat turn, in display order
--
-- Row-Level Security: every table is locked to auth.uid() = user_id, so a
-- logged-in user can only ever see or mutate their own rows.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- profiles — 12 frozen investor fields
-- -----------------------------------------------------------------------------
create table if not exists public.profiles (
    user_id               uuid primary key references auth.users(id) on delete cascade,
    age                   integer  not null check (age between 18 and 100),
    gender                text     not null check (gender in ('Male','Female')),
    education             text     not null check (education in ('High school','Bachelor','Master')),
    marital_status        text     not null check (marital_status in ('Single','Married','Divorced','Widowed')),
    household_size        integer  not null check (household_size between 1 and 10),
    income                numeric  not null check (income >= 0),
    investment_goal       text     not null check (investment_goal in ('Retirement','Home purchase','Child education','Other')),
    investment_horizon    integer  not null check (investment_horizon between 1 and 50),
    investment_capital    numeric  not null check (investment_capital >= 0),
    risk_tolerance        text     not null check (risk_tolerance in ('Low','Medium','High')),
    financial_involvement text     not null check (financial_involvement in ('Low','Medium','High')),
    experience_years      integer  not null check (experience_years between 0 and 80),
    updated_at            timestamptz not null default now()
);

alter table public.profiles enable row level security;

drop policy if exists "profiles_select_own"  on public.profiles;
drop policy if exists "profiles_insert_own"  on public.profiles;
drop policy if exists "profiles_update_own"  on public.profiles;
drop policy if exists "profiles_delete_own"  on public.profiles;

create policy "profiles_select_own"  on public.profiles for select using (auth.uid() = user_id);
create policy "profiles_insert_own"  on public.profiles for insert with check (auth.uid() = user_id);
create policy "profiles_update_own"  on public.profiles for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "profiles_delete_own"  on public.profiles for delete using (auth.uid() = user_id);

-- -----------------------------------------------------------------------------
-- portfolios — the ML result + the live (possibly user-tweaked) allocation
-- -----------------------------------------------------------------------------
create table if not exists public.portfolios (
    user_id               uuid primary key references auth.users(id) on delete cascade,
    risk_profile          text     not null,
    ml_result             jsonb    not null,          -- raw output of recommend()
    current_allocation    jsonb    not null,          -- {equity_pct, bond_pct, alt_pct}
    current_portfolio     jsonb    not null,          -- {equity:[...], bond:[...], alternative:[...]}
    allocation_overridden boolean  not null default false,
    updated_at            timestamptz not null default now()
);

alter table public.portfolios enable row level security;

drop policy if exists "portfolios_select_own" on public.portfolios;
drop policy if exists "portfolios_insert_own" on public.portfolios;
drop policy if exists "portfolios_update_own" on public.portfolios;
drop policy if exists "portfolios_delete_own" on public.portfolios;

create policy "portfolios_select_own" on public.portfolios for select using (auth.uid() = user_id);
create policy "portfolios_insert_own" on public.portfolios for insert with check (auth.uid() = user_id);
create policy "portfolios_update_own" on public.portfolios for update using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "portfolios_delete_own" on public.portfolios for delete using (auth.uid() = user_id);

-- -----------------------------------------------------------------------------
-- chat_messages — append-only conversation log (display turns only)
-- -----------------------------------------------------------------------------
create table if not exists public.chat_messages (
    id          bigserial primary key,
    user_id     uuid not null references auth.users(id) on delete cascade,
    role        text not null check (role in ('user','assistant')),
    content     text not null,
    created_at  timestamptz not null default now()
);

create index if not exists chat_messages_user_time_idx
    on public.chat_messages (user_id, created_at);

alter table public.chat_messages enable row level security;

drop policy if exists "chat_select_own" on public.chat_messages;
drop policy if exists "chat_insert_own" on public.chat_messages;
drop policy if exists "chat_delete_own" on public.chat_messages;

create policy "chat_select_own" on public.chat_messages for select using (auth.uid() = user_id);
create policy "chat_insert_own" on public.chat_messages for insert with check (auth.uid() = user_id);
create policy "chat_delete_own" on public.chat_messages for delete using (auth.uid() = user_id);
