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
# ------------------ LIVE MARKET (robust + fallback) ------------------ #
if menu == "💹 Live Market":
    import pytz
    from datetime import datetime, date

    st.header("💹 Live Indian Market Tracker (NIFTY 50)")

    # Helper: attempt multiple fetch strategies, return (df, reason)
    def fetch_nifty_safe():
        reason = []
        # 1) Try 15m intraday (5 days)
        try:
            df = yf.download("^NSEI", period="5d", interval="15m", progress=False)
            if not df.empty:
                reason.append("intraday 15m (5d)")
                return df, "live", "; ".join(reason)
        except Exception as e:
            reason.append(f"intraday failed: {e}")

        # 2) Try daily 1 month (fallback to LTP)
        try:
            df = yf.download("^NSEI", period="1mo", interval="1d", progress=False)
            if not df.empty:
                reason.append("daily 1mo (fallback)")
                return df, "ltp", "; ".join(reason)
        except Exception as e:
            reason.append(f"daily failed: {e}")

        # 3) Try Ticker.history (alternative call)
        try:
            t = yf.Ticker("^NSEI")
            df = t.history(period="1mo", interval="1d")
            if not df.empty:
                reason.append("Ticker.history 1mo")
                return df, "ltp", "; ".join(reason)
        except Exception as e:
            reason.append(f"Ticker.history failed: {e}")

        # 4) Final: empty -> return None
        return None, "error", "; ".join(reason)

    # Fetch once
    nifty_df, mode, fetch_reason = fetch_nifty_safe()

    # If fetching failed, use a small sample so UI stays alive
    if nifty_df is None or nifty_df.empty:
        st.warning("⚠️ Could not fetch live data from Yahoo Finance — using local sample data. (Reason logged below)")
        sample = {
            "Datetime": pd.date_range(end=pd.Timestamp.now(), periods=5, freq="D"),
            "Open": [20000, 20100, 19900, 20250, 20300],
            "High": [20150, 20250, 20050, 20400, 20450],
            "Low": [19950, 20050, 19850, 20100, 20200],
            "Close": [20100, 19950, 20200, 20300, 20350],
            "Volume": [1000, 1100, 1050, 1200, 900]
        }
        nifty_df = pd.DataFrame(sample).set_index("Datetime")
        data_mode = "Sample Data (no live)"
        fetch_reason = fetch_reason or "unknown"
    else:
        data_mode = "Live" if mode == "live" else "Market Closed / LTP"

    # Show reason (concise)
    st.caption(f"Data source: {data_mode} — fetch info: {fetch_reason}")

    # Prepare display (latest price, LTP logic)
    latest_price = round(float(nifty_df["Close"].iloc[-1]), 2)
    last_timestamp = nifty_df.index[-1]
    last_date = pd.to_datetime(last_timestamp).date()
    today = date.today()

    col1, col2 = st.columns([2, 1])

    # If last_date < today => market closed (show LTP)
    if last_date < today:
        col1.subheader("📅 Market Closed — showing LTP")
        col1.metric("Last Traded Price (LTP)", f"₹{latest_price:,.2f}")
    else:
        col1.subheader("✅ Market Live")
        # compute previous close robustly
        if len(nifty_df) >= 2:
            prev_close = float(nifty_df["Close"].iloc[-2])
            change = latest_price - prev_close
            pct = (change / prev_close) * 100 if prev_close != 0 else 0.0
            col1.metric("NIFTY 50", f"₹{latest_price:,.2f}", delta=f"{change:+.2f} ({pct:+.2f}%)")
        else:
            col1.metric("NIFTY 50", f"₹{latest_price:,.2f}")

    # Quick chart and table
    st.subheader("📈 Price Movement (recent)")
    # prefer using Close series
    try:
        st.line_chart(nifty_df["Close"].rename("Close"), use_container_width=True, height=350)
    except Exception:
        st.line_chart(nifty_df.iloc[:, 0], use_container_width=True, height=350)

    st.subheader("📊 Recent Data (latest rows)")
    try:
        st.dataframe(nifty_df.tail(10).round(2), use_container_width=True)
    except Exception:
        st.table(nifty_df.tail(5).round(2))

    # extra: top NIFTY samples (attempt fetch; tolerant)
    st.markdown("---")
    st.subheader("🔎 Sample Stocks — TCS | INFY | RELIANCE | HDFCBANK")
    sample_symbols = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
    stocks = []
    for s in sample_symbols:
        try:
            d = yf.download(s, period="5d", interval="1d", progress=False)
            if not d.empty:
                last = round(d["Close"].iloc[-1], 2)
                prev = round(d["Close"].iloc[-2], 2) if len(d) >= 2 else last
                pct = round(((last - prev) / prev) * 100, 2) if prev != 0 else 0.0
                stocks.append({"Symbol": s.replace(".NS", ""), "Price (₹)": last, "Change (%)": pct})
        except Exception:
            stocks.append({"Symbol": s.replace(".NS", ""), "Price (₹)": "N/A", "Change (%)": "N/A"})
    st.table(pd.DataFrame(stocks))

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
