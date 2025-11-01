# -------------------------------------------
# Live Indian Market Tracker (NIFTY 50)
# -------------------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# ---- Page setup ----
st.set_page_config(page_title="Live Indian Market Tracker", layout="wide", page_icon="📊")

# ---- Sidebar ----
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.info("Track the NIFTY 50 Index live or view the latest LTP if markets are closed.")
refresh = st.sidebar.button("🔄 Refresh Data")

# ---- Title ----
st.title("📊 Live Indian Market Tracker (NIFTY 50)")

# ---- Data fetching ----
try:
    nifty = yf.download("^NSEI", period="5d", interval="15m")

    # Fallback to daily data if intraday is empty
    if nifty.empty:
        nifty = yf.download("^NSEI", period="1mo", interval="1d")

    if nifty.empty:
        st.warning("⚠️ Could not fetch live data. Please check your internet connection or try later.")
    else:
        latest_price = nifty["Close"].iloc[-1]
        last_date = nifty.index[-1].date()
        today = datetime.date.today()

        # ---- Display section ----
        col1, col2 = st.columns(2)
        if last_date < today:
            col1.subheader("📅 Market Closed")
            col1.metric("Last Traded Price (LTP)", f"₹{latest_price:,.2f}")
        else:
            col1.subheader("✅ Market Live")
            col1.metric("NIFTY 50", f"₹{latest_price:,.2f}")

        # ---- Calculate change ----
        if len(nifty) >= 2:
            prev_close = nifty["Close"].iloc[-2]
            pct_change = ((latest_price - prev_close) / prev_close) * 100
            col2.metric("Change (%)", f"{pct_change:+.2f}%")

        # ---- Chart ----
        st.subheader("📈 Price Movement")
        st.line_chart(nifty["Close"], height=400, use_container_width=True)

        # ---- Data Table ----
        st.subheader("📊 Historical Data (Recent)")
        st.dataframe(
            nifty.tail(10).round(2),
            use_container_width=True,
            hide_index=False,
        )

except Exception as e:
    st.warning(f"⚠️ Could not fetch live data: {e}")
