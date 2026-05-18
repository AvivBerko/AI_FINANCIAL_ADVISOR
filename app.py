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

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="📈",
    layout="wide",
)

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
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 AI Financial Advisor")
st.caption("Enter your investor profile to receive a personalized ETF portfolio recommendation.")

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

    # Store everything in session state
    st.session_state["user_inputs"]           = profile
    st.session_state["ml_result"]             = result
    st.session_state["current_allocation"]    = result["allocation"].copy()
    st.session_state["current_portfolio"]     = result["portfolio"].copy()
    st.session_state["risk_profile"]          = result["risk_profile"]
    st.session_state["allocation_overridden"] = False

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
    st.metric("Total ETFs",   st.session_state["ml_result"]["total_etfs"])

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
# Chat interface
# ---------------------------------------------------------------------------
st.divider()
st.subheader("💬 Chat with your Advisor")

for msg in st.session_state["display_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything about your portfolio..."):
    # Show user message immediately
    st.session_state["display_messages"].append({"role": "user", "content": prompt})
    st.session_state["chat_history"].append({"role": "user", "content": prompt})

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

            st.session_state["user_inputs"]           = new_inputs
            st.session_state["ml_result"]             = new_result
            st.session_state["current_allocation"]    = new_result["allocation"].copy()
            st.session_state["current_portfolio"]     = new_result["portfolio"].copy()
            st.session_state["risk_profile"]          = new_result["risk_profile"]
            st.session_state["allocation_overridden"] = False

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

        st.rerun()
