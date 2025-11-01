import streamlit as st
import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from plotly import graph_objs as go

st.set_page_config(page_title="📊 Live Indian Market Tracker (NIFTY 50)", layout="wide")

# --- Title ---
st.title("📊 Live Indian Market Tracker (NIFTY 50)")

# --- Sidebar ---
st.sidebar.header("🔍 Dashboard Filters")
period = st.sidebar.selectbox("Data Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=1)
interval = st.sidebar.selectbox("Interval", ["5m", "15m", "30m", "1h", "1d"], index=2)

# --- Fetch Live Nifty Data ---
try:
    nifty = yf.download("^NSEI", period=period, interval=interval, progress=False)
    if nifty.empty:
        raise ValueError("Empty DataFrame received")

    nifty["Change"] = nifty["Close"].diff()
    nifty["% Change"] = nifty["Close"].pct_change() * 100

    latest_price = nifty["Close"].iloc[-1]
    change = nifty["Change"].iloc[-1]
    pct = nifty["% Change"].iloc[-1]
    last_time = nifty.index[-1].strftime("%d-%b %I:%M %p")

    # Header Metrics
    colA, colB, colC = st.columns(3)
    colA.metric("NIFTY 50 LTP", f"₹{latest_price:,.2f}")
    colB.metric("Recent Δ (pts)", f"{change:+.2f}")
    colC.metric("% Change", f"{pct:+.2f}%")

except Exception:
    # Fallback: If market closed, show LTP only
    st.warning("⚠️ Live data unavailable (Market closed or no connection). Showing last known price.")

    nifty = yf.download("^NSEI", period="1mo", interval="1d", progress=False)
    latest_price = nifty["Close"].iloc[-1]
    change = nifty["Close"].iloc[-1] - nifty["Close"].iloc[-2]
    pct = (change / nifty["Close"].iloc[-2]) * 100
    last_time = nifty.index[-1].strftime("%d-%b")

    colA, colB, colC = st.columns(3)
    colA.metric("NIFTY 50 LTP (Last Close)", f"₹{latest_price:,.2f}")
    colB.metric("Prev Δ (pts)", f"{change:+.2f}")
    colC.metric("% Change", f"{pct:+.2f}%")

# --- Price Chart ---
st.markdown("### 📈 NIFTY 50 Trend")
fig = go.Figure()
fig.add_trace(go.Scatter(x=nifty.index, y=nifty["Close"], mode="lines", name="NIFTY 50"))
fig.update_layout(
    xaxis_title="Time",
    yaxis_title="Price (₹)",
    template="plotly_dark",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# --- Sector ETFs Overview (Mini Heatmap) ---
st.markdown("### 🧩 Sectoral Performance Snapshot")
sectors = {
    "BANK": "^NSEBANK",
    "IT": "^CNXIT",
    "FMCG": "^CNXFMCG",
    "AUTO": "^CNXAUTO",
    "PHARMA": "^CNXPHARMA",
}

sector_data = []
for name, ticker in sectors.items():
    data = yf.download(ticker, period="5d", interval="1d", progress=False)
    if not data.empty:
        chg = ((data["Close"].iloc[-1] / data["Close"].iloc[-2]) - 1) * 100
        sector_data.append([name, f"{chg:+.2f}%"])

if sector_data:
    df_sectors = pd.DataFrame(sector_data, columns=["Sector", "Δ%"])
    st.dataframe(df_sectors, use_container_width=True)

# --- Risk Metrics Section ---
st.markdown("### ⚙️ Risk Metrics (Volatility & CAGR Simulation)")
returns = nifty["Close"].pct_change().dropna()
volatility = returns.std() * np.sqrt(252)
cagr = ((nifty["Close"].iloc[-1] / nifty["Close"].iloc[0]) ** (252 / len(nifty))) - 1

col1, col2 = st.columns(2)
col1.metric("Annualized Volatility", f"{volatility*100:.2f}%")
col2.metric("CAGR (Simulated)", f"{cagr*100:.2f}%")

st.caption(f"Last updated: {last_time}")
st.success("✅ Dashboard loaded successfully with live/fallback data.")
