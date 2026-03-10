"""
AI Financial Advisor — Streamlit Web App
=========================================
Run with:
    venv/bin/streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
from src.inference import recommend

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="📈",
    layout="wide",
)

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
        income           = st.number_input("Annual Income ($)",       min_value=0,   max_value=10_000_000, value=80_000,  step=1_000)
        investment_cap   = st.number_input("Investment Capital ($)",  min_value=0,   max_value=100_000_000, value=50_000, step=1_000)
        investment_goal  = st.selectbox("Investment Goal",            ["Retirement", "Home purchase", "Child education", "Other"])
        horizon          = st.number_input("Investment Horizon (years)", min_value=1, max_value=50, value=20)

    with col3:
        st.markdown("**Risk & Experience**")
        risk_tolerance        = st.selectbox("Risk Tolerance",         ["Low", "Medium", "High"])
        financial_involvement = st.selectbox("Financial Involvement",  ["Low", "Medium", "High"])
        experience            = st.number_input("Years of Experience", min_value=0, max_value=80, value=5)

    submitted = st.form_submit_button("Get My Portfolio Recommendation", use_container_width=True)

# ---------------------------------------------------------------------------
# Run inference and display results
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
            st.error(f"Error: {e}")
            st.stop()

    st.divider()
    st.subheader("Your Portfolio Recommendation")

    # --- Risk profile badge ---
    rp = result["risk_profile"]
    badge_color = {"Aggressive": "#e74c3c", "Moderate": "#f39c12", "Conservative": "#27ae60"}.get(rp, "#888")
    st.markdown(
        f"<div style='display:inline-block; background:{badge_color}; color:white; "
        f"padding:6px 20px; border-radius:20px; font-size:1.1rem; font-weight:600;'>"
        f"Risk Profile: {rp}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # --- Allocation pie chart + metrics ---
    alloc = result["allocation"]
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
        st.metric("Equity",       f"{alloc['equity_pct']*100:.1f}%")
        st.metric("Bonds",        f"{alloc['bond_pct']*100:.1f}%")
        st.metric("Alternatives", f"{alloc['alt_pct']*100:.1f}%")
        st.metric("Total ETFs",   result["total_etfs"])

    # --- ETF breakdown ---
    st.markdown("#### Portfolio ETFs")
    portfolio = result["portfolio"]
    etf_counts = result["etf_counts"]

    col_eq, col_bd, col_alt = st.columns(3)

    def etf_card(col, title, tickers, pct, color):
        with col:
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 10px;'>"
                f"<b>{title}</b> &nbsp;<span style='color:{color}'>{pct*100:.1f}%</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            for ticker in tickers:
                st.markdown(f"- **{ticker}**")

    etf_card(col_eq,  "Equity",       portfolio.get("equity",      []), alloc["equity_pct"], "#2980b9")
    etf_card(col_bd,  "Bonds",        portfolio.get("bond",        []), alloc["bond_pct"],   "#27ae60")
    etf_card(col_alt, "Alternatives", portfolio.get("alternative", []), alloc["alt_pct"],    "#8e44ad")
