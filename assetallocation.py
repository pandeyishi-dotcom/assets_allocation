import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Asset Allocation Dashboard",
    layout="wide",
    page_icon="💹"
)

# --- Custom Styling ---
st.markdown("""
    <style>
        /* Sidebar Styling */
        .sidebar-container {
            background: rgba(15, 17, 23, 0.9);
            padding: 15px;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        .collapsed {
            width: 70px !important;
            overflow-x: hidden;
            transition: all 0.3s ease;
        }
        .st-emotion-cache-1d391kg {padding-top: 1rem;}
        [data-testid="collapsedControl"] {
            position: absolute;
            right: -25px;
            top: 40px;
        }
        .sidebar-button {
            background: #222;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 6px 10px;
            cursor: pointer;
            margin: 4px;
            transition: 0.3s;
        }
        .sidebar-button:hover {background: #00ffaa33;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Layout ---
with st.sidebar:
    st.markdown('<div class="sidebar-container" id="sidebar">', unsafe_allow_html=True)

    # Collapsible toggle button
    collapse_state = st.session_state.get("collapse_state", False)
    if st.button("🔄 Collapse" if not collapse_state else "⬅️ Expand", key="collapse_btn"):
        st.session_state["collapse_state"] = not collapse_state

    # Sidebar content
    if not st.session_state.get("collapse_state", False):
        st.markdown(
            """
            <h3 style='color:#00FFAA;text-align:center;'>Welcome, Ishani 👋</h3>
            <p style='font-size:13px;color:#ccc;text-align:center;'>Your Real-Time Market Cockpit</p>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        def show_ticker(symbol, label):
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
                    <div style="background:#1E1E1E;padding:10px;border-radius:10px;margin-bottom:10px;">
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
                    <div style="background:#1E1E1E;padding:10px;border-radius:10px;margin-bottom:10px;">
                        <b style="color:white;">{label}</b><br>
                        <span style="color:#ccc;">⚠️ Data Unavailable</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        show_ticker("^NSEI", "NIFTY 50")
        show_ticker("^NSEBANK", "BANKNIFTY")
        show_ticker("USDINR=X", "USD/INR")

        st.divider()
        st.markdown("#### ⚙️ Quick Links")
        st.write("📊 Portfolio Overview")
        st.write("📈 Market Pulse")
        st.write("💰 SIP Goals")
        st.write("🧮 Allocation Advisor")

    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Section ---
st.title("💹 Asset Allocation Dashboard")
st.caption("Dynamic market and portfolio analytics tailored for Ishani.")

st.subheader("📈 Market Overview")

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

st.divider()
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
