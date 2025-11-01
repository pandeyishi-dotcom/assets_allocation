# Live Indian Market Tracker (NIFTY 50) — V2 + Enhanced Sidebar
# Author: Ishani ❤️
# Built with Streamlit
# This version merges the upgraded sidebar and enhanced UI layout.

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Live Indian Market Tracker (NIFTY 50)",
    page_icon="📈",
    layout="wide"
)

# --- SIDEBAR SECTION ---
with st.sidebar:
    st.title("📊 Ishani's Market Dashboard")
    st.markdown("Welcome, **Ishani 👋**")
    st.divider()

    # Market status logic
    now = datetime.datetime.now()
    market_open = now.weekday() < 5 and now.hour >= 9 and now.hour < 15 and not (
        now.hour == 15 and now.minute > 30
    )

    if market_open:
        st.success("🟢 Market is OPEN")
    else:
        st.error("🔴 Market is CLOSED — Showing last known LTP")

    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    section = st.radio(
        "Choose a section:",
        ["Market Overview", "Asset Allocation", "Analytics"],
        index=0
    )

    st.markdown("---")
    st.caption("Made with ❤️ by Ishani | Powered by Streamlit")

# --- FETCH DATA FUNCTION ---
@st.cache_data(ttl=300)
def fetch_nifty_data():
    try:
        url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        df = pd.DataFrame(data["data"])
        df = df[["symbol", "open", "dayHigh", "dayLow", "lastPrice", "pChange"]]
        df.rename(
            columns={
                "symbol": "Symbol",
                "open": "Open",
                "dayHigh": "High",
                "dayLow": "Low",
                "lastPrice": "LTP",
                "pChange": "% Change",
            },
            inplace=True,
        )
        return df

    except Exception:
        return None

# --- DATA FETCH ---
nifty_df = fetch_nifty_data()

# --- MAIN LAYOUT ---
st.title("🇮🇳 Live Indian Market Tracker (NIFTY 50)")

if section == "Market Overview":
    st.subheader("📈 NIFTY 50 Overview")

    if nifty_df is None or nifty_df.empty:
        st.warning("⚠️ Could not fetch live data. Showing offline fallback.")
        nifty_df = pd.DataFrame({
            "Symbol": ["NIFTY 50"],
            "Open": [0],
            "High": [0],
            "Low": [0],
            "LTP": [0],
            "% Change": [0],
        })
    else:
        st.success("✅ Live market data loaded successfully!")

    # Calculate key metrics
    avg_change = nifty_df["% Change"].mean()
    top_gainer = nifty_df.loc[nifty_df["% Change"].idxmax()]
    top_loser = nifty_df.loc[nifty_df["% Change"].idxmin()]

    col1, col2, col3 = st.columns(3)
    col1.metric("Average % Change", f"{avg_change:.2f}%")
    col2.metric("Top Gainer", f"{top_gainer['Symbol']} ({top_gainer['% Change']:.2f}%)")
    col3.metric("Top Loser", f"{top_loser['Symbol']} ({top_loser['% Change']:.2f}%)")

    st.markdown("---")
    st.dataframe(nifty_df, use_container_width=True)

    # Chart visualization
    st.subheader("📊 Price Movement Visualization")
    st.line_chart(nifty_df.set_index("Symbol")["% Change"], use_container_width=True)

elif section == "Asset Allocation":
    st.subheader("💰 Portfolio Asset Allocation")

    default_df = pd.DataFrame({
        "Asset Class": ["Equity", "Debt", "Gold", "Real Estate", "Cash"],
        "Allocation (%)": [50, 20, 10, 15, 5],
    })

    edited = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)

    st.markdown("#### 📊 Allocation Summary")
    total_alloc = edited["Allocation (%)"].sum()

    if total_alloc != 100:
        st.warning(f"⚠️ Allocation total = {total_alloc}%. Please make sure it sums to 100%.")
    else:
        st.success("✅ Allocation perfectly balanced!")

    st.bar_chart(edited.set_index("Asset Class"))

elif section == "Analytics":
    st.subheader("📉 Portfolio Risk & Performance Analytics")

    try:
        # Simulated data for demo
        np.random.seed(42)
        days = 100
        dates = pd.date_range(end=datetime.datetime.now(), periods=days)
        returns = np.random.normal(0.001, 0.02, days).cumsum()
        df = pd.DataFrame({"Date": dates, "Portfolio Value": 100 * (1 + returns)})

        st.line_chart(df.set_index("Date"), use_container_width=True)
        st.caption("Simulated performance chart for portfolio analytics demonstration.")
    except Exception as e:
        st.error(f"Error generating analytics: {e}")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2025 Ishani | Educational Use Only | Data from NSE India")
