"""Streamlit auth widgets backed by Supabase.

Renders a sign-in / sign-up screen if `st.session_state["auth"]` is empty.
On success, stores `{user_id, email, access_token, refresh_token}` in
session_state and triggers a rerun so the main app proceeds.

Session lives in `st.session_state` only — Streamlit forgets it on a hard
refresh and the user has to log back in. Their profile / portfolio /
chat history all persist in Supabase regardless.
"""
from __future__ import annotations

import streamlit as st

from src.supabase_client import get_anon_client


_AUTH_KEY = "auth"


def is_authenticated() -> bool:
    return bool(st.session_state.get(_AUTH_KEY))


def current_user() -> dict | None:
    return st.session_state.get(_AUTH_KEY)


def sign_out() -> None:
    auth = st.session_state.pop(_AUTH_KEY, None)
    if auth:
        try:
            get_anon_client().auth.sign_out()
        except Exception:
            pass
    # Wipe app state so the next user starts fresh.
    for k in (
        "user_inputs", "ml_result", "current_allocation", "current_portfolio",
        "risk_profile", "chat_history", "display_messages",
        "portfolio_ready", "allocation_overridden", "loaded_from_db",
    ):
        st.session_state.pop(k, None)


def _store_session(resp) -> None:
    st.session_state[_AUTH_KEY] = {
        "user_id":       resp.user.id,
        "email":         resp.user.email,
        "access_token":  resp.session.access_token,
        "refresh_token": resp.session.refresh_token,
    }


def render_auth_gate() -> None:
    """Render the sign-in / sign-up panel. Call `st.stop()` after this if
    the user is still unauthenticated."""
    st.title("📈 AI Financial Advisor")
    st.caption("Sign in to access your saved portfolio and chat history.")

    tab_login, tab_signup = st.tabs(["Sign in", "Create account"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email    = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            try:
                resp = get_anon_client().auth.sign_in_with_password(
                    {"email": email, "password": password}
                )
                _store_session(resp)
                st.rerun()
            except Exception as e:
                st.error(f"Sign-in failed: {e}")

    with tab_signup:
        with st.form("signup_form", clear_on_submit=False):
            email_s     = st.text_input("Email", key="signup_email")
            password_s  = st.text_input("Password (min 6 characters)", type="password", key="signup_password")
            password_s2 = st.text_input("Confirm password",            type="password", key="signup_password2")
            submitted_s = st.form_submit_button("Create account", use_container_width=True)
        if submitted_s:
            if password_s != password_s2:
                st.error("Passwords do not match.")
            elif len(password_s) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                try:
                    resp = get_anon_client().auth.sign_up(
                        {"email": email_s, "password": password_s}
                    )
                    if resp.session is None:
                        st.info(
                            "Account created. Check your email to confirm, then sign in. "
                            "(If your Supabase project has email confirmation disabled, "
                            "just sign in now.)"
                        )
                    else:
                        _store_session(resp)
                        st.rerun()
                except Exception as e:
                    st.error(f"Sign-up failed: {e}")
