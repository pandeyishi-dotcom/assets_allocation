# ✅ AI-Driven Fintech Dashboard
# NIFTY 50 live / LTP view + Asset Allocation + Goal Simulation + Efficient Frontier

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime

# ---------------- CONFIGURATION ---------------- #
st.set_page_config(page_title="AI-Driven Asset Allocation Dashboard", layout="wide", page_icon="📊")

# ---------------- STYLING ---------------- #
st.markdown("""
    <style>
        body {background-color: #0e1117;}
        .main {background-color: #0e1117; color: white;}
        div[data-testid="stSidebar"] {background-color: #111827;}
        h1, h2, h3, h4, h5 {color: #00FFC6; font-family: 'Poppins', sans-serif;}
        .stDataFrame {border-radius: 12px;}
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/NSE_Logo.svg", width=140)
st.sidebar.title("📊 Fintech Dashboard")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "💹 Live Market", "🧩 Asset Allocation", "🎯 Goals & Simulation", "📈 Efficient Frontier"])

# ---------------- LIVE MARKET ---------------- #
if menu == "💹 Live Market":
    st.header("💹 Live Indian Market Tracker (NIFTY 50)")

    try:
        nifty = yf.download("^NSEI", period="5d", interval="15m", progress=False)
        if nifty.empty:
            nifty = yf.download("^NSEI", period="1mo", interval="1d", progress=False)

        if nifty.empty:
            st.warning("⚠️ Could not fetch live data. Please check your internet connection.")
        else:
            latest_price = nifty["Close"].iloc[-1]
            last_date = nifty.index[-1].date()
            today = date.today()

            col1, col2 = st.columns(2)
            if last_date < today:
                col1.subheader("📅 Market Closed")
                col1.metric("Last Traded Price (LTP)", f"₹{latest_price:,.2f}")
            else:
                col1.subheader("✅ Market Live")
                col1.metric("NIFTY 50", f"₹{latest_price:,.2f}")

            if len(nifty) >= 2:
                prev_close = nifty["Close"].iloc[-2]
                pct_change = ((latest_price - prev_close) / prev_close) * 100
                col2.metric("Change (%)", f"{pct_change:+.2f}%")

            st.subheader("📈 Price Movement (Last 5 Days)")
            st.line_chart(nifty["Close"], use_container_width=True, height=400)

            st.subheader("📊 Historical Data (Recent)")
            st.dataframe(nifty.tail(10).round(2), use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Could not fetch live data: {e}")

# ---------------- ASSET ALLOCATION ---------------- #
elif menu == "🧩 Asset Allocation":
    st.header("🧩 Intelligent Asset Allocation")
    st.write("Balance risk and reward with AI-driven diversification across asset classes.")

    options = ["Equity", "Debt", "Gold", "REITs", "Crypto", "Cash"]
    weights = {opt: st.slider(f"{opt} Allocation (%)", 0, 100, 15) for opt in options}
    total = sum(weights.values())

    if total != 100:
        st.warning(f"⚠️ Allocations must total 100%. Current total: {total}%")
    else:
        df_alloc = pd.DataFrame(list(weights.items()), columns=["Asset Class", "Allocation (%)"])
        fig = px.pie(df_alloc, names="Asset Class", values="Allocation (%)", color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Allocation successfully created!")

# ---------------- GOAL SIMULATION ---------------- #
elif menu == "🎯 Goals & Simulation":
    st.header("🎯 Smart Goal Planning")
    st.write("Estimate your future corpus and assess compounding effects over time.")

    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Initial Investment (₹)", 10000, 10000000, 100000)
    with c2:
        years = st.slider("Investment Duration (Years)", 1, 40, 10)
    with c3:
        expected_return = st.slider("Expected Annual Return (%)", 4, 20, 10)

    future_value = amount * ((1 + expected_return / 100) ** years)
    st.metric("Projected Corpus", f"₹{future_value:,.0f}", delta=f"{expected_return}% CAGR")

    data = pd.DataFrame({
        "Year": range(1, years + 1),
        "Value": [amount * ((1 + expected_return / 100) ** i) for i in range(1, years + 1)]
    })
    fig = px.line(data, x="Year", y="Value", title="Investment Growth Over Time", color_discrete_sequence=["#00FFC6"])
    st.plotly_chart(fig, use_container_width=True)

# ---------------- EFFICIENT FRONTIER ---------------- #
elif menu == "📈 Efficient Frontier":
    st.header("📈 Efficient Frontier Simulation (AI Risk-Return Model)")
    st.write("Visualize optimal portfolios balancing expected return and risk.")

    np.random.seed(42)
    n_portfolios = 1000
    returns = np.random.normal(0.1, 0.03, n_portfolios)
    risk = np.random.normal(0.15, 0.05, n_portfolios)

    frontier = pd.DataFrame({"Expected Return": returns, "Risk": risk})
    fig = px.scatter(frontier, x="Risk", y="Expected Return", color="Expected Return",
                     color_continuous_scale="Tealgrn", title="Efficient Frontier Simulation")
    fig.add_trace(go.Scatter(
        x=[0.15], y=[0.1],
        mode="markers+text", text=["Current Portfolio"],
        textposition="top center",
        marker=dict(size=14, color="gold")
    ))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- HOME ---------------- #
else:
    st.title("🤖 AI-Driven Portfolio Planner")
    st.subheader("Transform your investments with automation, analytics, and intelligence.")
    st.markdown("""
        Welcome to the **AI-Driven Asset Allocation Dashboard** — your personal fintech lab that helps you:
        - Track Indian market performance (NIFTY 50 & major stocks)  
        - Simulate personalized asset allocation  
        - Forecast long-term wealth goals  
        - Explore the Efficient Frontier using AI simulation  

        💡 *Designed for modern investors. Built for intelligent decisions.*
    """)
