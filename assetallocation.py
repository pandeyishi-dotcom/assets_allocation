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
# ------------------ LIVE MARKET ------------------ #
if menu == "💹 Live Market":
    import datetime
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st

    st.title("📊 Live Indian Market Tracker (NIFTY 50)")

    # Add subtitle with current date & time
    now = datetime.datetime.now().strftime("%d %B %Y, %I:%M %p")
    st.markdown(f"**As of:** {now}")

    # Try to fetch live Nifty 50 data
    try:
        try:     nifty = yf.download("^NSEI", period="5d", interval="15m")     if nifty.empty:         nifty = yf.download("^NSEI", period="1mo", interval="1d") except Exception:     nifty = pd.DataFrame()  if nifty.empty:     nifty = yf.download("^NSEI", period="1mo", interval="1d")
        if not nifty.empty:
            last_price = round(nifty["Close"].iloc[-1], 2)
            prev_close = round(nifty["Close"].iloc[0], 2)
            change = round(last_price - prev_close, 2)
            pct_change = round((change / prev_close) * 100, 2)

            # Market summary
            if change > 0:
                st.success(f"▲ NIFTY 50 is up by {change} points ({pct_change}%) to **{last_price}** 🟢")
            elif change < 0:
                st.error(f"▼ NIFTY 50 is down by {abs(change)} points ({abs(pct_change)}%) to **{last_price}** 🔴")
            else:
                st.info(f"⏸ NIFTY 50 is unchanged at **{last_price}**")

            # Plot intraday chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=nifty.index, y=nifty["Close"], mode="lines", name="NIFTY 50"))
            fig.update_layout(title="📈 Intraday Price Movement", xaxis_title="Time", yaxis_title="Price (INR)")
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ Could not fetch live data. Please check your internet connection.")
    except Exception as e:
        st.error("❌ No live data available. Please check your internet or try again later.")


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
