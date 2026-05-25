"""CRUD helpers for the Streamlit + Supabase Option A surface.

Each function takes an authenticated `client` (see `supabase_client.with_user_session`)
and a `user_id`. RLS guarantees a user can only ever touch their own rows.

The Streamlit app builds an in-memory `profile` dict using lowercase/PascalCase
keys mirroring `src/inference.py`'s `FEATURE_COLUMNS`. The DB uses snake_case
columns — `_profile_to_row()` and `_row_to_profile()` translate between them.
"""
from __future__ import annotations

from typing import Any, Optional

from supabase import Client


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to JSON-serialisable types.

    The ML pipeline returns numpy int64 / float64 inside dicts and lists.
    Supabase's jsonb columns serialise via stdlib json, which chokes on
    those types. Normalising at the DB boundary keeps callers simple.
    """
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # numpy / pandas scalars expose .item() returning a native Python scalar
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


# ---------------------------------------------------------------------------
# Profile <-> DB row translation
# ---------------------------------------------------------------------------
# Streamlit-side profile keys (mirror src/inference.py FEATURE_COLUMNS) →
# Postgres column names. Lowercase keys stay lowercase; PascalCase keys are
# converted to snake_case.
_PROFILE_TO_COLUMN = {
    "age":                  "age",
    "Gender":               "gender",
    "Education":             "education",
    "MaritalStatus":         "marital_status",
    "HouseholdSize":         "household_size",
    "income":                "income",
    "InvestmentGoal":        "investment_goal",
    "horizon":               "investment_horizon",
    "InvestmentCapital":     "investment_capital",
    "risk_tolerance":        "risk_tolerance",
    "FinancialInvolvement":  "financial_involvement",
    "experience":            "experience_years",
}
_COLUMN_TO_PROFILE = {v: k for k, v in _PROFILE_TO_COLUMN.items()}


def _profile_to_row(profile: dict, user_id: str) -> dict:
    row = {"user_id": user_id}
    for k, col in _PROFILE_TO_COLUMN.items():
        row[col] = profile[k]
    return row


def _row_to_profile(row: dict) -> dict:
    return {pk: row[col] for col, pk in _COLUMN_TO_PROFILE.items()}


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------
def load_profile(client: Client, user_id: str) -> Optional[dict]:
    resp = (
        client.table("profiles")
        .select("*")
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    if not resp or not getattr(resp, "data", None):
        return None
    return _row_to_profile(resp.data)


def save_profile(client: Client, user_id: str, profile: dict) -> None:
    row = _profile_to_row(profile, user_id)
    client.table("profiles").upsert(row, on_conflict="user_id").execute()


# ---------------------------------------------------------------------------
# portfolios
# ---------------------------------------------------------------------------
def load_portfolio(client: Client, user_id: str) -> Optional[dict]:
    resp = (
        client.table("portfolios")
        .select("*")
        .eq("user_id", user_id)
        .maybe_single()
        .execute()
    )
    if not resp or not getattr(resp, "data", None):
        return None
    d = resp.data
    return {
        "risk_profile":          d["risk_profile"],
        "ml_result":             d["ml_result"],
        "current_allocation":    d["current_allocation"],
        "current_portfolio":     d["current_portfolio"],
        "allocation_overridden": d["allocation_overridden"],
        "holdings":              d.get("holdings"),
    }


def save_portfolio(
    client: Client,
    user_id: str,
    *,
    risk_profile: str,
    ml_result: dict,
    current_allocation: dict,
    current_portfolio: dict,
    allocation_overridden: bool,
    holdings: Optional[dict] = None,
) -> None:
    row = {
        "user_id":               user_id,
        "risk_profile":          risk_profile,
        "ml_result":             _jsonable(ml_result),
        "current_allocation":    _jsonable(current_allocation),
        "current_portfolio":     _jsonable(current_portfolio),
        "allocation_overridden": allocation_overridden,
        "holdings":              _jsonable(holdings) if holdings is not None else None,
    }
    client.table("portfolios").upsert(row, on_conflict="user_id").execute()


# ---------------------------------------------------------------------------
# chat_messages
# ---------------------------------------------------------------------------
def load_messages(client: Client, user_id: str) -> list[dict]:
    resp = (
        client.table("chat_messages")
        .select("role,content,created_at")
        .eq("user_id", user_id)
        .order("created_at")
        .execute()
    )
    rows = getattr(resp, "data", None) or []
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def append_message(client: Client, user_id: str, role: str, content: str) -> None:
    client.table("chat_messages").insert(
        {"user_id": user_id, "role": role, "content": content}
    ).execute()


def clear_messages(client: Client, user_id: str) -> None:
    client.table("chat_messages").delete().eq("user_id", user_id).execute()
