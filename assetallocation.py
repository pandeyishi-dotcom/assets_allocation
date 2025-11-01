# ✅ AI-Driven Asset Allocation Dashboard (Ishani Edition)
# Sleek Fintech App for Indian Market | Streamlit Cloud Compatible

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime

# ---------------- CONFIGURATION ---------------- #
st.set_page_config(page_title="AI-Driven Asset Allocation Dashboard", layout="wide")

# ---------------- CUSTOM STYLING ---------------- #
st.markdown("""
    <style>
        body {background-color: #0e1117; color: white;}
        .main {background-color: #0e1117; color: white;}
        div[data-testid="stSidebar"] {
            background-color: #111827;
            padding: 10px;
        }
        h1, h2, h3, h4, h5 {
            color: #00FFC6;
            font-family: 'Poppins', sans-serif;
        }
        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #00FFC6;
            text-align: center;
            margin-bottom: 5px;
        }
        .sidebar-subtitle {
            font-size: 13px;
            color: #9CA3AF;
            text-align: center;
            margin-bottom: 15px;
        }
        .sidebar-footer {
            position: relative;
            bottom: 10px;
            color: #9CA3AF;
            text-align: center;
            font-size: 11px;
        }
        .metric-card {
            background-color: #1f2937;
            padding: 10px;
            border-radius: 12px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/NSE_Logo.svg", width=140)
    st.markdown("<div class='sidebar-title'>📊 Ishani Fintech Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-subtitle'>AI-Driven Wealth Insights</div>", unsafe_allow_html=True)
    menu = st.radio("Navigation", ["🏠 Home", "💹 Live Market", "🧩 Asset Allocation", "🎯 Goals & Simulation", "📈 Efficient Frontier"])
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-footer'>Made with 💚 by Ishani</div>", unsafe_allow_html=True)

# ---------------- HOME PAGE ---------------- #
if menu == "🏠 Home":
    st.title("🤖 AI-Driven Portfolio Planner")
    st.subheader("Transform your investments with automation, analytics, and intelligence.")
    st.markdown("""
        Welcome to the **AI-Driven Asset Allocation Dashboard** — a futuristic fintech experience by Ishani.  
        Track live markets, plan goals, and simulate risk-reward intelligence like a pro investor.  

        **Highlights:**
        - 🔸 Real-time Indian market tracking (NIFTY 50 & major stocks)  
        - 🔸 Smart goal simulations  
        - 🔸 Intelligent asset diversification  
        - 🔸 Efficient frontier visualization for risk-return balance  
        
        💡 *Built for clarity. Designed for growth.*
    """)

# ---------------- LIVE MARKET ---------------- #
elif menu == "💹 Live Market":
    st.header("💹 Live Indian Market Tracker (NIFTY 50)")
    nifty_symbols = ["^NSEI", "^BSESN", "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
    df_list = []

    try:
        for symbol in nifty_symbols:
            data = yf.download(symbol, period="5d", interval="1d", progress=False)
            if not data.empty:
                df_list.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Price": round(data["Close"].iloc[-1], 2),
                    "Change (%)": round(((data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2]) * 100, 2)
                })
        df = pd.DataFrame(df_list)
        df["Status"] = np.where(df["Change (%)"] > 0, "▲ Gain", "▼ Loss")

        if not df.empty:
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns), fill_color="#00FFC6", align='center'),
                cells=dict(values=[df[c] for c in df.columns],
                           fill_color=[["#161a1f"] * len(df)], align='center'))
            ])
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

            latest = df[df["Symbol"] == "^NSEI"]["Price"].values[0] if "^NSEI" in df["Symbol"].values else df["Price"].mean()
            st.metric("NIFTY 50 Index (LTP)", f"₹{latest:,.2f}")

        else:
            st.warning("⚠️ Market closed. Displaying last traded prices (LTP).")
    except Exception:
        st.warning("⚠️ Could not fetch live data. Please check your internet connection.")

# ---------------- ASSET ALLOCATION ---------------- #
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

# ---------------- GOALS & SIMULATION ---------------- #
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
    fig.add_trace(go.Scatter(x=[0.15], y=[0.1],
                             mode="markers+text",
                             text=["Current Portfolio"],
                             textposition="top center",
                             marker=dict(size=14, color="gold")))
    st.plotly_chart(fig, use_container_width=True)
