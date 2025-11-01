# ✅ AI-Driven Fintech Dashboard V3
# Personalized with Greeting, Quotes & Full Analytics Suite

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import random

# ---------------- CONFIGURATION ---------------- #
st.set_page_config(page_title="AI-Driven Fintech Dashboard", layout="wide")

# ---------------- STYLING ---------------- #
st.markdown("""
    <style>
        body {background-color: #0e1117;}
        .main {background-color: #0e1117; color: white;}
        div[data-testid="stSidebar"] {background-color: #111827;}
        h1, h2, h3, h4, h5 {color: #00FFC6; font-family: 'Poppins', sans-serif;}
        .stDataFrame {border-radius: 12px;}
        .welcome-box {
            text-align: center;
            padding: 50px;
            border-radius: 20px;
            background: linear-gradient(135deg, #111827, #0e1117);
            color: white;
            font-family: 'Poppins', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- MOTIVATIONAL QUOTES ---------------- #
quotes = [
    "Invest in your dreams. Grind now. Shine later.",
    "Discipline is the bridge between goals and achievement.",
    "Wealth is the ability to fully experience life.",
    "Risk comes from not knowing what you're doing.",
    "Don’t look for the needle in the haystack. Just buy the haystack."
]

# ---------------- USER INPUT / GREETING ---------------- #
if "username" not in st.session_state:
    st.markdown("<div class='welcome-box'>", unsafe_allow_html=True)
    st.title("🤖 Welcome to the AI-Driven Fintech Dashboard")
    st.write("Enter your name to personalize your experience:")
    name = st.text_input("Your Name", placeholder="e.g. Ishani", key="username_input")

    if st.button("Start Dashboard 🚀"):
        if name.strip():
            st.session_state.username = name.strip().title()
            st.session_state.quote = random.choice(quotes)
            st.experimental_rerun()
        else:
            st.warning("Please enter your name to continue.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ---------------- HELPER FUNCTIONS ---------------- #
def greeting_message():
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning"
    elif hour < 17:
        return "Good Afternoon"
    else:
        return "Good Evening"

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/NSE_Logo.svg", width=130)
    st.title("📊 Fintech Dashboard")

    st.markdown(f"### {greeting_message()}, {st.session_state.username} 👋")
    st.markdown(f"💬 *{st.session_state.quote}*")

    menu = st.radio("Navigate", [
        "🏠 Home",
        "💹 Live Market",
        "🧩 Asset Allocation",
        "🎯 Goals & Simulation",
        "📈 Efficient Frontier"
    ])

# ---------------- MODULE: HOME ---------------- #
if menu == "🏠 Home":
    st.title("🤖 AI-Driven Portfolio Planner")
    st.subheader("Transform your investments with automation, analytics, and intelligence.")
    st.markdown("""
        Welcome to the **AI-Driven Fintech Dashboard** — a smart investment platform for the modern investor.

        Features include:
        - 💹 Real-time Indian market tracking (NIFTY 50 + Top Stocks)
        - 🧩 AI-guided asset allocation
        - 🎯 Goal-based wealth simulations
        - 📈 Efficient frontier for risk-return optimization
        
        🌱 *Designed for insight. Powered by intelligence.*
    """)

# ---------------- MODULE: LIVE MARKET ---------------- #
elif menu == "💹 Live Market":
    st.header("💹 Live Indian Market Tracker (NIFTY 50)")
    try:
        nifty = yf.download("^NSEI", period="5d", interval="15m", progress=False)
        if nifty.empty:
            nifty = yf.download("^NSEI", period="1mo", interval="1d", progress=False)

        latest = nifty["Close"].iloc[-1]
        prev = nifty["Close"].iloc[-2]
        change = latest - prev
        pct = (change / prev) * 100

        colA, colB, colC = st.columns(3)
        colA.metric("NIFTY 50 (LTP)", f"{latest:,.2f}")
        colB.metric("Δ (points)", f"{change:+.2f}")
        colC.metric("Change (%)", f"{pct:+.2f}%")

        st.line_chart(nifty["Close"], use_container_width=True)
    except Exception:
        st.warning("⚠️ Could not fetch live data. Displaying placeholder chart.")
        dummy = pd.Series(np.random.randn(20).cumsum(), name="NIFTY (Simulated)")
        st.line_chart(dummy, use_container_width=True)

# ---------------- MODULE: ASSET ALLOCATION ---------------- #
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

# ---------------- MODULE: GOALS ---------------- #
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

# ---------------- MODULE: EFFICIENT FRONTIER ---------------- #
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
