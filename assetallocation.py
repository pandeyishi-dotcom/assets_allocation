# assetallocation_v2.py
"""
💹 Live Nifty Robo-Advisor — Asset Allocation & Goal Planner (V2)
Personalized version with greeting, quotes & user input sidebar
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# -----------------------------------------
# Page setup and theme
# -----------------------------------------
st.set_page_config(page_title="Live Nifty Robo-Advisor", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #041422 0%, #062b3a 100%); color: #E8F0F2;}
.title { color:#6FF0B0; font-size:28px; font-weight:700; }
.muted { color:#9FB4C8; }
.card { background: rgba(255,255,255,0.03); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); }
.metric-card { text-align:center; background:rgba(255,255,255,0.08); border-radius:10px; padding:10px; margin-bottom:8px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💹 Live Nifty Robo-Advisor — Smart Asset Allocation</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Real-time allocation, goal planning, risk analytics & efficient frontier simulation.</div>', unsafe_allow_html=True)
st.write("")

# -----------------------------------------
# SIDEBAR: Personalized Section
# -----------------------------------------
with st.sidebar:
    st.header("👋 Welcome to your Robo-Advisor")

    # Personalized name input
    user_name = st.text_input("Enter your name:", value="Ishani")
    if not user_name.strip():
        user_name = "Investor"

    # Time-based greeting
    hour = datetime.now().hour
    if 5 <= hour < 12:
        greet = "Good morning"
    elif 12 <= hour < 18:
        greet = "Good afternoon"
    elif 18 <= hour < 22:
        greet = "Good evening"
    else:
        greet = "Working late? Let's make your money work too!"
    st.markdown(f"### 🌞 {greet}, {user_name}!")

    # Motivational quotes
    quotes = [
        "Investing is not about beating others, it's about achieving your goals.",
        "Do not save what is left after spending; spend what is left after saving.",
        "The best investment you can make is in yourself.",
        "Wealth grows quietly, not overnight.",
        "Discipline and patience are the real alpha."
    ]
    st.info(f"💡 {random.choice(quotes)}")

    st.header("Profile & Settings")
    age = st.slider("Age", 18, 75, 34)
    monthly_income = st.number_input("Monthly income (₹)", min_value=0, value=70000, step=5000)
    self_declared_risk = st.selectbox("Risk appetite", ["Low", "Moderate", "High"])
    risk_score = {"Low": 25, "Moderate": 50, "High": 75}[self_declared_risk]

    st.markdown("---")
    st.subheader("Goal Settings")
    if "goals" not in st.session_state:
        st.session_state.goals = [
            {"name": "Retirement", "amount": 8000000, "years": 25},
            {"name": "Home", "amount": 3000000, "years": 8},
        ]
    with st.expander("View / Edit Goals"):
        goals_df = pd.DataFrame(st.session_state.goals)
        edited = st.data_editor(goals_df, num_rows="dynamic")
        st.session_state.goals = edited.to_dict("records")

    st.markdown("---")
    st.subheader("Investment Inputs")
    current_investment = st.number_input("Current investment (₹)", min_value=0, value=500000, step=10000)
    use_sip = st.checkbox("Use SIP (Systematic Investment Plan)", value=True)
    monthly_sip = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500) if use_sip else 0
    horizon = st.slider("Investment Horizon (Years)", 1, 40, 10)

    st.markdown("---")
    st.subheader("Advanced Settings")
    mc_sims = st.slider("Monte Carlo Simulations", 200, 5000, 1000, step=200)
    frontier_samples = st.slider("Efficient Frontier Portfolios", 100, 2000, 400, step=50)
    lookback = st.selectbox("Lookback Period (Years)", [1, 3, 5], index=2)

    st.markdown("---")
    st.subheader("Ticker Overrides (Optional)")
    t_nifty = st.text_input("Nifty Index (Yahoo Finance)", value="^NSEI")
    t_gold = st.text_input("Gold ETF (Yahoo Finance)", value="GOLDBEES.NS")
    t_intl = st.text_input("International ETF", value="VTI")

# -----------------------------------------
# ASSET LOGIC
# -----------------------------------------
st.markdown("### 📊 Asset Allocation Engine")

ASSET_CLASSES = {
    "Equity": 0.5,
    "Debt": 0.2,
    "Gold": 0.15,
    "International": 0.1,
    "Cash": 0.05
}

# Adjust based on age and risk
equity_alloc = max(0.2, min(0.8, (100 - age) / 100 + risk_score / 200))
debt_alloc = 1 - equity_alloc - 0.15 - 0.1 - 0.05
ASSET_CLASSES["Equity"] = equity_alloc
ASSET_CLASSES["Debt"] = max(0.1, debt_alloc)

alloc_df = pd.DataFrame(list(ASSET_CLASSES.items()), columns=["Asset Class", "Allocation"])
alloc_df["Allocation %"] = (alloc_df["Allocation"] * 100).round(2)
st.dataframe(alloc_df, use_container_width=True)

# Allocation pie
fig_alloc = px.pie(alloc_df, names="Asset Class", values="Allocation", color_discrete_sequence=px.colors.sequential.Tealgrn)
st.plotly_chart(fig_alloc, use_container_width=True)

# -----------------------------------------
# LIVE NIFTY DATA
# -----------------------------------------
st.markdown("### 📈 Live Nifty & Gold Tracker")

try:
    nifty = yf.Ticker(t_nifty)
    gold = yf.Ticker(t_gold)
    intl = yf.Ticker(t_intl)
    data = yf.download([t_nifty, t_gold, t_intl], period=f"{lookback}y")["Adj Close"]

    returns = data.pct_change().dropna()
    annual_ret = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    corr = returns.corr()

    st.subheader("Annualized Returns & Risk")
    perf_df = pd.DataFrame({
        "Asset": [t_nifty, t_gold, t_intl],
        "Annual Return": (annual_ret.values * 100).round(2),
        "Volatility": (annual_vol.values * 100).round(2)
    })
    st.dataframe(perf_df, use_container_width=True)

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=annual_vol * 100, y=annual_ret * 100,
        mode="markers+text", text=["Nifty", "Gold", "Intl"],
        textposition="top center", marker=dict(size=10, color=["#6FF0B0", "#FFD700", "#66C2FF"])
    ))
    fig_perf.update_layout(title="Risk vs Return", xaxis_title="Volatility (%)", yaxis_title="Return (%)")
    st.plotly_chart(fig_perf, use_container_width=True)
except Exception as e:
    st.warning("⚠️ Could not fetch live market data. Check ticker symbols or internet connection.")

# -----------------------------------------
# GOAL PROJECTION (SIP + LUMPSUM)
# -----------------------------------------
st.markdown("### 🎯 Goal Projection & SIP Planning")

def future_value(principal, rate, years, monthly_sip=0):
    fv_lumpsum = principal * ((1 + rate) ** years)
    fv_sip = monthly_sip * (((1 + rate / 12) ** (years * 12) - 1) / (rate / 12))
    return fv_lumpsum + fv_sip

proj_results = []
for goal in st.session_state.goals:
    amt = goal["amount"]
    years = goal["years"]
    rate = 0.1 if self_declared_risk == "Moderate" else (0.07 if self_declared_risk == "Low" else 0.13)
    future_val = future_value(current_investment, rate, years, monthly_sip)
    gap = amt - future_val
    proj_results.append({
        "Goal": goal["name"],
        "Target (₹)": amt,
        "Projected Value (₹)": round(future_val, 2),
        "Surplus / Deficit (₹)": round(gap, 2)
    })

proj_df = pd.DataFrame(proj_results)
st.dataframe(proj_df, use_container_width=True)

fig_goal = px.bar(proj_df, x="Goal", y=["Target (₹)", "Projected Value (₹)"], barmode="group",
                  title="Goal vs Projected Corpus", color_discrete_sequence=px.colors.qualitative.Set2)
st.plotly_chart(fig_goal, use_container_width=True)

# -----------------------------------------
# FOOTER
# -----------------------------------------
st.markdown("---")
st.markdown(
    f"<div class='muted'>✨ Built for smart investors like <b>{user_name}</b>. Stay patient, stay invested!</div>",
    unsafe_allow_html=True
)
