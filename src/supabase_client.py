"""Supabase client factory.

Two client modes:
  * `get_anon_client()` — public anon key. Use for auth (sign_in / sign_up).
  * Per-user authenticated client — the same anon client after the user's
    access token has been attached via `postgrest_auth()`. All RLS-gated
    reads/writes go through this so `auth.uid()` resolves correctly inside
    Postgres policies.
"""
from __future__ import annotations

import os
from functools import lru_cache

from supabase import Client, create_client


def _env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Add SUPABASE_URL and SUPABASE_ANON_KEY to your .env file."
        )
    return val


@lru_cache(maxsize=1)
def get_anon_client() -> Client:
    return create_client(_env("SUPABASE_URL"), _env("SUPABASE_ANON_KEY"))


def with_user_session(access_token: str, refresh_token: str) -> Client:
    """Return a client whose subsequent calls run as the authenticated user.

    Sets the JWT on the underlying postgrest + auth instances so Row-Level
    Security policies that key off `auth.uid()` see the right identity.
    """
    client = get_anon_client()
    client.auth.set_session(access_token, refresh_token)
    client.postgrest.auth(access_token)
    return client
