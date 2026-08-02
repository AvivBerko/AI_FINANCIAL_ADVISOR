"""
AI Financial Advisor — Streamlit Web App
=========================================
Run with:
    venv/bin/streamlit run app.py
"""

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import plotly.graph_objects as go

from src.inference import recommend
from src.llm import (
    apply_adjust_allocation,
    chat,
    get_initial_explanation,
    get_tool_followup,
)
from src.auth_ui import current_user, is_authenticated, render_auth_gate, sign_out
from src import db
from src.pricing import compute_pnl, fetch_current_prices, snapshot_holdings
from src.supabase_client import with_user_session

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Auth gate — anything below only renders for signed-in users
# ---------------------------------------------------------------------------
if not is_authenticated():
    render_auth_gate()
    st.stop()

_auth = current_user()
_client = with_user_session(_auth["access_token"], _auth["refresh_token"])
_user_id = _auth["user_id"]

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        # Core ML data
        "user_inputs":           {},
        "ml_result":             {},
        # Live view (may diverge from ml_result if LLM adjusts allocation)
        "current_allocation":    {},
        "current_portfolio":     {},
        "risk_profile":          "",
        # LLM chat
        "chat_history":          [],   # full OpenAI messages (includes tool artifacts)
        "display_messages":      [],   # only what renders in the chat UI
        # Flags
        "portfolio_ready":       False,
        "allocation_overridden": False,
        # Holdings snapshot for P&L tracking
        "holdings":              {},
        "investment_capital":    0.0,
        # One-shot DB restore guard
        "loaded_from_db":        False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# First-load: hydrate session_state from Supabase if the user already has
# a saved profile + portfolio.
# ---------------------------------------------------------------------------
if not st.session_state["loaded_from_db"]:
    try:
        saved_profile   = db.load_profile(_client,   _user_id)
        saved_portfolio = db.load_portfolio(_client, _user_id)
        saved_messages  = db.load_messages(_client,  _user_id)
    except Exception as e:
        st.warning(f"Could not load your saved data: {e}")
        saved_profile = saved_portfolio = None
        saved_messages = []

    if saved_profile and saved_portfolio:
        st.session_state["user_inputs"]           = saved_profile
        st.session_state["ml_result"]             = saved_portfolio["ml_result"]
        st.session_state["current_allocation"]    = saved_portfolio["current_allocation"]
        st.session_state["current_portfolio"]     = saved_portfolio["current_portfolio"]
        st.session_state["risk_profile"]          = saved_portfolio["risk_profile"]
        st.session_state["allocation_overridden"] = saved_portfolio["allocation_overridden"]
        st.session_state["holdings"]              = saved_portfolio.get("holdings") or {}
        st.session_state["investment_capital"]    = float(saved_profile.get("InvestmentCapital") or 0.0)
        st.session_state["display_messages"]      = saved_messages
        # Rebuild a minimal chat_history so the LLM has context (display
        # turns only — tool-call artifacts from past sessions are not replayed).
        st.session_state["chat_history"] = [
            {"role": m["role"], "content": m["content"]} for m in saved_messages
        ]
        st.session_state["portfolio_ready"] = True

    st.session_state["loaded_from_db"] = True

# ---------------------------------------------------------------------------
# Sidebar — account
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"**Signed in as**\n\n{_auth['email']}")
    if st.button("Sign out", use_container_width=True):
        sign_out()
        st.rerun()
    st.divider()
    if st.session_state["portfolio_ready"]:
        if st.button("Start over (clear profile)", use_container_width=True):
            try:
                db.clear_messages(_client, _user_id)
            except Exception:
                pass
            for k in (
                "user_inputs", "ml_result", "current_allocation",
                "current_portfolio", "risk_profile", "chat_history",
                "display_messages", "portfolio_ready", "allocation_overridden",
                "holdings", "investment_capital",
            ):
                st.session_state.pop(k, None)
            st.session_state["loaded_from_db"] = True  # don't re-hydrate
            st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 AI Financial Advisor")
st.caption("Enter your investor profile to receive a personalized ETF portfolio recommendation.")

if _err := st.session_state.get("persist_error"):
    st.error(_err)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("investor_profile"):
    st.subheader("Your Investor Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal**")
        age             = st.number_input("Age",              min_value=18, max_value=100, value=35)
        gender          = st.selectbox("Gender",              ["Male", "Female"])
        education       = st.selectbox("Education",           ["High school", "Bachelor", "Master"])
        marital_status  = st.selectbox("Marital Status",      ["Single", "Married", "Divorced", "Widowed"])
        household_size  = st.number_input("Household Size",   min_value=1, max_value=10, value=1)

    with col2:
        st.markdown("**Financial**")
        income           = st.number_input("Annual Income ($)",          min_value=0,   max_value=10_000_000,  value=80_000,  step=1_000)
        investment_cap   = st.number_input("Investment Capital ($)",     min_value=0,   max_value=100_000_000, value=50_000,  step=1_000)
        investment_goal  = st.selectbox("Investment Goal",               ["Retirement", "Home purchase", "Child education", "Other"])
        horizon          = st.number_input("Investment Horizon (years)", min_value=1,   max_value=50,          value=20)

    with col3:
        st.markdown("**Risk & Experience**")
        risk_tolerance        = st.selectbox("Risk Tolerance",         ["Low", "Medium", "High"])
        financial_involvement = st.selectbox("Financial Involvement",  ["Low", "Medium", "High"])
        experience            = st.number_input("Years of Experience", min_value=0, max_value=80, value=5)

    submitted = st.form_submit_button("Get My Portfolio Recommendation", use_container_width=True)

# ---------------------------------------------------------------------------
# Handle form submission
# ---------------------------------------------------------------------------
if submitted:
    profile = {
        "age":                  age,
        "Gender":               gender,
        "Education":            education,
        "MaritalStatus":        marital_status,
        "HouseholdSize":        household_size,
        "income":               income,
        "InvestmentGoal":       investment_goal,
        "horizon":              horizon,
        "InvestmentCapital":    investment_cap,
        "risk_tolerance":       risk_tolerance,
        "FinancialInvolvement": financial_involvement,
        "experience":           experience,
    }

    with st.spinner("Analyzing your profile..."):
        try:
            result = recommend(profile)
        except Exception as e:
            st.error(f"Error running ML pipeline: {e}")
            st.stop()

    with st.spinner("Generating your personalized explanation..."):
        try:
            explanation = get_initial_explanation(profile, result)
        except Exception as e:
            explanation = "Your portfolio has been generated. Ask me anything about it below."

    # Snapshot live prices so we can track P&L from this purchase point.
    with st.spinner("Snapshotting purchase prices..."):
        all_tickers = [t for tickers in result["portfolio"].values() for t in tickers]
        purchase_prices = fetch_current_prices(all_tickers)
        holdings = snapshot_holdings(
            portfolio=result["portfolio"],
            allocation=result["allocation"],
            investment_capital=float(profile["InvestmentCapital"]),
            current_prices=purchase_prices,
        )

    # Store everything in session state
    st.session_state["user_inputs"]           = profile
    st.session_state["ml_result"]             = result
    st.session_state["current_allocation"]    = result["allocation"].copy()
    st.session_state["current_portfolio"]     = result["portfolio"].copy()
    st.session_state["risk_profile"]          = result["risk_profile"]
    st.session_state["allocation_overridden"] = False
    st.session_state["holdings"]              = holdings
    st.session_state["investment_capital"]    = float(profile["InvestmentCapital"])

    # Seed chat_history with the trigger + explanation (LLM needs this context going forward)
    st.session_state["chat_history"] = [
        {"role": "user",      "content": "Please explain my portfolio recommendation."},
        {"role": "assistant", "content": explanation},
    ]
    # display_messages only shows what the user sees
    st.session_state["display_messages"] = [
        {"role": "assistant", "content": explanation},
    ]
    st.session_state["portfolio_ready"] = True

    # Persist to Supabase. Reset chat_messages so the saved log mirrors
    # the in-memory display_messages for this fresh portfolio.
    try:
        db.save_profile(_client, _user_id, profile)
        db.save_portfolio(
            _client, _user_id,
            risk_profile=result["risk_profile"],
            ml_result=result,
            current_allocation=result["allocation"],
            current_portfolio=result["portfolio"],
            allocation_overridden=False,
            holdings=holdings,
        )
        db.clear_messages(_client, _user_id)
        db.append_message(_client, _user_id, "assistant", explanation)
        st.session_state.pop("persist_error", None)
    except Exception as e:
        # Stash the error so it survives the st.rerun() below and the
        # user actually sees what went wrong.
        st.session_state["persist_error"] = f"Recommendation generated, but failed to save: {e}"

    st.rerun()

# ---------------------------------------------------------------------------
# Portfolio display + chat (only rendered after a portfolio has been generated)
# ---------------------------------------------------------------------------
if not st.session_state["portfolio_ready"]:
    st.stop()

alloc     = st.session_state["current_allocation"]
portfolio = st.session_state["current_portfolio"]
rp        = st.session_state["risk_profile"]

st.divider()
st.subheader("Your Portfolio Recommendation")

# Risk profile badge
badge_color = {"Aggressive": "#e74c3c", "Moderate": "#f39c12", "Conservative": "#27ae60"}.get(rp, "#888")
st.markdown(
    f"<div style='display:inline-block; background:{badge_color}; color:white; "
    f"padding:6px 20px; border-radius:20px; font-size:1.1rem; font-weight:600;'>"
    f"Risk Profile: {rp}</div>",
    unsafe_allow_html=True,
)

if st.session_state["allocation_overridden"]:
    st.info("⚙️ Allocation adjusted via chat. The ETF selection reflects the original ML recommendation.")

st.markdown("")

# Allocation pie chart + metrics
left, right = st.columns([1, 1])

with left:
    labels = ["Equity", "Bonds", "Alternatives"]
    values = [alloc["equity_pct"], alloc["bond_pct"], alloc["alt_pct"]]
    colors = ["#2980b9", "#27ae60", "#8e44ad"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont_size=14,
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    ml_alloc = st.session_state["ml_result"]["allocation"]
    st.metric("Equity",       f"{alloc['equity_pct']*100:.1f}%",
              delta=f"{(alloc['equity_pct'] - ml_alloc['equity_pct'])*100:+.1f}pp"
              if st.session_state["allocation_overridden"] else None)
    st.metric("Bonds",        f"{alloc['bond_pct']*100:.1f}%",
              delta=f"{(alloc['bond_pct'] - ml_alloc['bond_pct'])*100:+.1f}pp"
              if st.session_state["allocation_overridden"] else None)
    st.metric("Alternatives", f"{alloc['alt_pct']*100:.1f}%",
              delta=f"{(alloc['alt_pct'] - ml_alloc['alt_pct'])*100:+.1f}pp"
              if st.session_state["allocation_overridden"] else None)
    st.metric("Total ETFs",   sum(len(tickers) for tickers in portfolio.values()))

# ETF breakdown
st.markdown("#### Portfolio ETFs")
col_eq, col_bd, col_alt = st.columns(3)

def _etf_card(col, title, tickers, pct, color):
    with col:
        st.markdown(
            f"<div style='border-left: 4px solid {color}; padding-left: 10px;'>"
            f"<b>{title}</b> &nbsp;<span style='color:{color}'>{pct*100:.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        for ticker in tickers:
            st.markdown(f"- **{ticker}**")

_etf_card(col_eq,  "Equity",       portfolio.get("equity",      []), alloc["equity_pct"], "#2980b9")
_etf_card(col_bd,  "Bonds",        portfolio.get("bond",        []), alloc["bond_pct"],   "#27ae60")
_etf_card(col_alt, "Alternatives", portfolio.get("alternative", []), alloc["alt_pct"],    "#8e44ad")

# ---------------------------------------------------------------------------
# Performance — live P&L vs purchase-time snapshot
# ---------------------------------------------------------------------------
_holdings = st.session_state.get("holdings") or {}
if _holdings:
    st.divider()
    st.subheader("📊 Performance — live vs purchase price")

    with st.spinner("Fetching live prices..."):
        _live_prices = fetch_current_prices(list(_holdings.keys()))
    _pnl = compute_pnl(_holdings, _live_prices)

    _bought_at_iso = None
    for _h in _holdings.values():
        if _h.get("bought_at"):
            _bought_at_iso = _h["bought_at"]
            break
    if _bought_at_iso:
        st.caption(f"Purchase snapshot taken at **{_bought_at_iso}** UTC")

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Invested",      f"${_pnl['total_cost']:,.2f}")
    _m2.metric("Current value", f"${_pnl['total_value']:,.2f}")
    _pnl_delta = f"{_pnl['total_pnl']:+,.2f}"
    _m3.metric("Gain / Loss",   f"${_pnl['total_pnl']:+,.2f}", delta=_pnl_delta)
    _pct = _pnl.get("total_pnl_pct")
    _m4.metric("Return",        f"{(_pct or 0)*100:+.2f}%",
               delta=f"{(_pct or 0)*100:+.2f}%" if _pct is not None else None)

    # Per-ticker breakdown table
    import pandas as _pd
    _rows = []
    for r in _pnl["per_ticker"]:
        _rows.append({
            "Ticker":    r["ticker"],
            "Shares":    f"{r['shares']:.4f}" if r["shares"] else "—",
            "Buy $":     f"{r['buying_price']:.2f}"  if r["buying_price"]  is not None else "—",
            "Now $":     f"{r['current_price']:.2f}" if r["current_price"] is not None else "—",
            "Cost":      f"${r['cost_basis']:,.2f}",
            "Value":     f"${r['current_value']:,.2f}" if r["current_value"] is not None else "—",
            "P&L":       f"${r['pnl']:+,.2f}"          if r["pnl"]          is not None else "—",
            "Return":    f"{r['pnl_pct']*100:+.2f}%"   if r["pnl_pct"]      is not None else "—",
        })
    st.dataframe(_pd.DataFrame(_rows), hide_index=True, use_container_width=True)

    if any(r["current_value"] is None for r in _pnl["per_ticker"]):
        st.caption("⚠️  Tickers shown as '—' lack a current price from yfinance — "
                   "their cost basis is included but their gain/loss is excluded.")

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
st.divider()
st.subheader("💬 Chat with your Advisor")

with st.expander("What can I ask?", expanded=False):
    st.markdown(
        """
**Just chat — ask anything about your portfolio.** A few examples of what the advisor can do for you:

**🧠 Understand**
- *"Why did I get an aggressive risk profile?"*
- *"Why is SPY in my equity bucket?"*
- *"What does the alternatives bucket protect against?"*
- *"Walk me through the trade-offs in this allocation."*

**🎚️ Adjust the allocation** *(percentages only — same ETFs, different weights)*
- *"Make me a bit more conservative."*
- *"Shift 10% from equity to bonds."*
- *"Add 5% to alternatives, take it from equity."*

**🪪 Change your profile** *(re-runs the full recommendation — new ETFs possible)*
- *"My income just went up to $120,000."*
- *"I'm now married with one child."*
- *"My horizon is 15 years, not 30."*
- *"Bump my experience to 10 years."*

When the advisor changes anything — allocation or profile — the donut chart and ETF list update on the next message, and everything is saved automatically.
        """
    )

for msg in st.session_state["display_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def _persist_message(role: str, content: str) -> None:
    try:
        db.append_message(_client, _user_id, role, content)
    except Exception as e:
        st.warning(f"Could not save message: {e}")


def _persist_portfolio_state() -> None:
    try:
        db.save_portfolio(
            _client, _user_id,
            risk_profile=st.session_state["risk_profile"],
            ml_result=st.session_state["ml_result"],
            current_allocation=st.session_state["current_allocation"],
            current_portfolio=st.session_state["current_portfolio"],
            allocation_overridden=st.session_state["allocation_overridden"],
            holdings=st.session_state.get("holdings") or {},
        )
    except Exception as e:
        st.warning(f"Could not save portfolio update: {e}")


if prompt := st.chat_input("Ask me anything about your portfolio..."):
    # Show user message immediately
    st.session_state["display_messages"].append({"role": "user", "content": prompt})
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    _persist_message("user", prompt)

    with st.spinner("Thinking..."):
        text, tool_call = chat(
            st.session_state["chat_history"],
            st.session_state["user_inputs"],
            st.session_state["current_allocation"],
            st.session_state["current_portfolio"],
            st.session_state["risk_profile"],
        )

    if tool_call is None:
        # Plain response — just display it
        st.session_state["display_messages"].append({"role": "assistant", "content": text})
        st.session_state["chat_history"].append({"role": "assistant", "content": text})
        _persist_message("assistant", text)
        st.rerun()

    else:
        # ---- Tool dispatch ----
        # 1. Store the assistant's tool-call message in chat_history (required by OpenAI)
        st.session_state["chat_history"].append(tool_call["assistant_msg"])

        args = tool_call["arguments"]

        if tool_call["name"] == "adjust_allocation":
            new_alloc = apply_adjust_allocation(
                st.session_state["current_allocation"],
                equity_delta=args["equity_delta"],
                bond_delta=args["bond_delta"],
                alt_delta=args["alt_delta"],
            )
            st.session_state["current_allocation"]    = new_alloc
            st.session_state["allocation_overridden"] = True

            tool_result = (
                f"Allocation updated. "
                f"Equity: {new_alloc['equity_pct']*100:.1f}%, "
                f"Bonds: {new_alloc['bond_pct']*100:.1f}%, "
                f"Alternatives: {new_alloc['alt_pct']*100:.1f}%. "
                f"Reason: {args['reason']}"
            )

        elif tool_call["name"] == "update_investor_profile":
            new_inputs = {**st.session_state["user_inputs"], **args["updates"]}

            with st.spinner("Re-running ML pipeline with updated profile..."):
                try:
                    new_result = recommend(new_inputs)
                except Exception as e:
                    st.error(f"Error re-running pipeline: {e}")
                    st.stop()

            # Re-snapshot holdings: profile change wipes the old purchase
            # baseline because the recommendation just changed.
            with st.spinner("Re-snapshotting purchase prices..."):
                _new_tickers = [t for ts in new_result["portfolio"].values() for t in ts]
                _new_prices  = fetch_current_prices(_new_tickers)
                _new_capital = float(new_inputs.get("InvestmentCapital",
                                                    st.session_state.get("investment_capital") or 0.0))
                new_holdings = snapshot_holdings(
                    portfolio=new_result["portfolio"],
                    allocation=new_result["allocation"],
                    investment_capital=_new_capital,
                    current_prices=_new_prices,
                )

            st.session_state["user_inputs"]           = new_inputs
            st.session_state["ml_result"]             = new_result
            st.session_state["current_allocation"]    = new_result["allocation"].copy()
            st.session_state["current_portfolio"]     = new_result["portfolio"].copy()
            st.session_state["risk_profile"]          = new_result["risk_profile"]
            st.session_state["allocation_overridden"] = False
            st.session_state["holdings"]              = new_holdings
            st.session_state["investment_capital"]    = _new_capital

            new_alloc = new_result["allocation"]
            tool_result = (
                f"Profile updated and ML pipeline re-run. "
                f"New risk profile: {new_result['risk_profile']}. "
                f"New allocation — Equity: {new_alloc['equity_pct']*100:.1f}%, "
                f"Bonds: {new_alloc['bond_pct']*100:.1f}%, "
                f"Alternatives: {new_alloc['alt_pct']*100:.1f}%. "
                f"Reason: {args['reason']}"
            )

        else:
            tool_result = "Tool executed."

        # 2. Append tool result to chat_history
        st.session_state["chat_history"].append({
            "role":         "tool",
            "tool_call_id": tool_call["id"],
            "content":      tool_result,
        })

        # 3. Get LLM follow-up explanation using the updated context
        with st.spinner("Preparing explanation..."):
            followup = get_tool_followup(
                st.session_state["chat_history"],
                st.session_state["user_inputs"],
                st.session_state["current_allocation"],
                st.session_state["current_portfolio"],
                st.session_state["risk_profile"],
            )

        st.session_state["chat_history"].append({"role": "assistant", "content": followup})
        st.session_state["display_messages"].append({"role": "assistant", "content": followup})

        # The chat tool just mutated either the allocation or the full
        # portfolio + profile. Persist everything that changed.
        if tool_call["name"] == "update_investor_profile":
            try:
                db.save_profile(_client, _user_id, st.session_state["user_inputs"])
            except Exception as e:
                st.warning(f"Could not save updated profile: {e}")
        _persist_portfolio_state()
        _persist_message("assistant", followup)

        st.rerun()
