import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Asset Allocation Dashboard",
    layout="wide",
    page_icon="💹"
)

# ----------------------------
# Sidebar: Personalized & Live
# ----------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="
            background: rgba(15, 17, 23, 0.85);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            color: white;
        ">
            <h3 style='margin-bottom: -10px;'>Welcome, <span style='color:#00FFAA;'>Ishani 👋</span></h3>
            <p style='font-size:13px;color:#ccc;'>Your Real-Time Market Cockpit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Live Tickers: NIFTY / BANKNIFTY / USDINR
    def get_ticker_data(symbol, label):
        try:
            df = yf.download(symbol, period="1d", interval="5m", progress=False)
            if df.empty:
                df = yf.download(symbol, period="5d", interval="1d", progress=False)
            ltp = round(df["Close"].iloc[-1], 2)
            prev_close = round(df["Close"].iloc[-2], 2)
            change = round(ltp - prev_close, 2)
            pct = round((change / prev_close) * 100, 2)
            color = "#00FFAA" if change >= 0 else "#FF4C4C"
            st.markdown(
                f"""
                <div style="background-color:#1E1E1E;padding:10px;border-radius:10px;margin-bottom:10px;">
                    <b style="color:white;">{label}</b><br>
                    <span style="color:{color};font-size:18px;">{ltp}</span>
                    <span style="color:{color};font-size:13px;">({change:+.2f}, {pct:+.2f}%)</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            st.markdown(
                f"""
                <div style="background-color:#1E1E1E;padding:10px;border-radius:10px;margin-bottom:10px;">
                    <b style="color:white;">{label}</b><br>
                    <span style="color:#ccc;">⚠️ Data Unavailable</span>
                </div>
                """,
                unsafe_allow_html=True
            )

    get_ticker_data("^NSEI", "NIFTY 50")
    get_ticker_data("^NSEBANK", "BANKNIFTY")
    get_ticker_data("USDINR=X", "USD/INR")

    st.markdown("---")
    st.markdown("#### ⚙️ Quick Links")
    st.write("📊 Portfolio Overview")
    st.write("📈 Market Pulse")
    st.write("💰 SIP Goals")
    st.write("🧮 Allocation Advisor")

# ----------------------------
# Main Layout
# ----------------------------
st.title("💹 Asset Allocation Dashboard")
st.caption("A dynamic market and portfolio analytics platform tailored for Ishani.")

# Example main section content
st.subheader("Market Overview")

try:
    nifty = yf.download("^NSEI", period="5d", interval="1h", progress=False)
    if not nifty.empty:
        nifty["% Change"] = nifty["Close"].pct_change() * 100
        st.line_chart(nifty["Close"], use_container_width=True)
        st.metric("Current NIFTY LTP", f"{nifty['Close'].iloc[-1]:.2f}")
    else:
        st.warning("⚠️ Live NIFTY data unavailable. Showing cached data.")
except Exception:
    st.warning("⚠️ Could not load live market data. Please check your connection.")

st.markdown("---")
st.subheader("📊 Portfolio Allocation (Demo)")

demo_data = {
    "Asset Class": ["Equity", "Debt", "Gold", "REITs", "Cash"],
    "Allocation (%)": [55, 25, 10, 5, 5],
}
df = pd.DataFrame(demo_data)
st.dataframe(df, use_container_width=True)

st.markdown(
    """
    <div style="background-color:#1e1e1e;padding:12px;border-radius:10px;color:#ccc;font-size:13px;">
        Tip: Refine your allocations dynamically based on risk appetite, age, and market sentiment.
    </div>
    """,
    unsafe_allow_html=True
)
