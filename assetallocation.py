# ✅ ASSET ALLOCATION - AI DRIVEN FINTECH DASHBOARD
# Built for Streamlit Cloud | Indian Market | Professional Dark Theme

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date

# ------------------ CONFIGURATION ------------------ #
st.set_page_config(page_title="AI-Driven Asset Allocation Dashboard", layout="wide")

# ------------------ STYLING ------------------ #
st.markdown("""
    <style>
        body {background-color: #0e1117;}
        .main {
            background-color: #0e1117;
            color: white;
        }
        div[data-testid="stSidebar"] {
            background-color: #111827;
        }
        h1, h2, h3, h4, h5 {
            color: #00FFC6;
            font-family: 'Poppins', sans-serif;
        }
        .stDataFrame {border-radius: 12px;}
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------ #
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/NSE_Logo.svg", width=140)
st.sidebar.title("📊 Fintech Dashboard")
menu = st.sidebar.radio("Navigation", ["🏠 Home", "💹 Live Market", "🧩 Asset Allocation", "🎯 Goals & Simulation", "📈 Efficient Frontier"])

# ------------------ LIVE MARKET ------------------ #
if menu == "💹 Live Market":
    st.header("💹 Live Indian Market Tracker (NIFTY 50)")
    nifty_symbols = ["^NSEI", "^BSESN", "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
    df_list = []

    for symbol in nifty_symbols:
        data = yf.download(symbol, period="5d", interval="1d", progress=False)
        if not data.empty:
            df_list.append({
                "Symbol": symbol,
                "Price": round(data["Close"].iloc[-1], 2),
                "Change (%)": round(((data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2]) * 100, 2)
            })
    df = pd.DataFrame(df_list)
    # Ensure Change (%) column exists before using it if "Change (%)" not in df.columns and "Price" in df.columns and "Prev Close" in df.columns:     df["Change (%)"] = ((df["Price"] - df["Prev Close"]) / df["Prev Close"]) * 100  if "Change (%)" in df.columns:     df["Status"] = np.where(df["Change (%)"] > 0, "▲ Gain", "▼ Loss") else:     df["Status"] = "N/A"

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns),
                    fill_color="#00FFC6",
                    align='center'),
        cells = dict(values=[     df.get("Symbol", ["N/A"] * len(df)),     df.get("Price", ["N/A"] * len(df)),     df.get("Change (%)", ["N/A"] * len(df)),     df.get("Status", ["N/A"] * len(df)), ],
                   fill_color=[["#161a1f"] * len(df)],
                   align='center'))
    ])
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ------------------ ASSET ALLOCATION ------------------ #
elif menu == "🧩 Asset Allocation":
    st.header("🧩 Intelligent Asset Allocation")
    st.write("Balance risk and reward with AI-driven diversification across asset classes.")
    options = ["Equity", "Debt", "Gold", "REITs", "Crypto", "Cash"]
    weights = {}
    for opt in options:
        weights[opt] = st.slider(f"{opt} Allocation (%)", 0, 100, 15)
    total = sum(weights.values())
    if total != 100:
        st.warning(f"⚠️ Allocations must total 100%. Current total: {total}%")
    else:
        df_alloc = pd.DataFrame(list(weights.items()), columns=["Asset Class", "Allocation (%)"])
        fig = px.pie(df_alloc, names="Asset Class", values="Allocation (%)",
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)
        st.success("✅ Allocation successfully created!")

# ------------------ GOAL SIMULATION ------------------ #
elif menu == "🎯 Goals & Simulation":
    st.header("🎯 Smart Goal Planning")
    st.write("Estimate future corpus and assess how your investments compound over time.")
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
        "Year": list(range(1, years + 1)),
        "Value": [amount * ((1 + expected_return / 100) ** i) for i in range(1, years + 1)]
    })
    fig = px.line(data, x="Year", y="Value", title="Investment Growth Over Time", color_discrete_sequence=["#00FFC6"])
    st.plotly_chart(fig, use_container_width=True)

# ------------------ EFFICIENT FRONTIER ------------------ #
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
    fig.add_trace(go.Scatter(x=[0.15], y=[0.1],
                             mode="markers+text",
                             text=["Current Portfolio"],
                             textposition="top center",
                             marker=dict(size=14, color="gold")))
    st.plotly_chart(fig, use_container_width=True)

# ------------------ HOME ------------------ #
else:
    st.title("🤖 AI-Driven Portfolio Planner")
    st.subheader("Transform your investments with automation, analytics, and intelligence.")
    st.markdown("""
        Welcome to the **AI-Driven Asset Allocation Dashboard** — an advanced fintech interface that helps you:
        - Track Indian market performance (NIFTY 50 & major stocks)  
        - Simulate personalized asset allocation  
        - Forecast long-term wealth goals  
        - Explore the Efficient Frontier with AI  
        
        💡 *Designed for modern investors. Built for intelligent decisions.*
    """)
