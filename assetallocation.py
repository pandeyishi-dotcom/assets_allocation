# assetallocation_final_ishani.py
# AI Portfolio Cockpit — Enhanced Sidebar + Stable Core
# Requirements: streamlit, yfinance, pandas, numpy, plotly, pytz

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from io import StringIO

# -------------------------
# Page config & theme
# -------------------------
st.set_page_config(page_title="AI Portfolio Cockpit — Ishani", layout="wide", page_icon="💠")
ACCENT = "#00FFC6"
BG = "#0e1117"
CARD = "#0f1720"
MUTED = "#9aa5a6"

st.markdown(
    f"""
    <style>
      body {{ background: {BG}; color: #e6eef0; }}
      div[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, #061833, #071a2f);
          color: #f2f6fa;
      }}
      .titlebig{{ font-size:28px; color:{ACCENT}; font-weight:700; margin-bottom:6px; }}
      .muted{{ color:{MUTED}; }}
      .card{{ background:{CARD}; padding:12px; border-radius:10px; }}
      .smallpill{{ font-size:12px; color:#cfeee6; background:#07121a; padding:6px 10px;border-radius:6px; }}
      .footer{{ color:{MUTED}; font-size:12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utility functions
# -------------------------
@st.cache_data(ttl=300)
def fetch_symbol(symbol, period="5d", interval="15m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_safe_nifty():
    try:
        df = fetch_symbol("^NSEI", period="5d", interval="15m")
        if not df.empty:
            return df, "live", "intraday 15m (5d)"
    except:
        pass
    try:
        df = fetch_symbol("^NSEI", period="1mo", interval="1d")
        if not df.empty:
            return df, "ltp", "daily 1mo fallback"
    except:
        pass
    try:
        t = yf.Ticker("^NSEI")
        df = t.history(period="1mo", interval="1d")
        if not df.empty:
            return df, "ltp", "Ticker.history 1mo"
    except:
        pass
    return None, "error", "All fetch methods failed"

def compute_portfolio_value(df_holdings):
    df = df_holdings.copy()
    symbols = df["Symbol"].unique().tolist()
    prices = {}
    for s in symbols:
        try:
            d = fetch_symbol(s, period="5d", interval="1d")
            prices[s] = float(d["Close"].iloc[-1]) if not d.empty else np.nan
        except Exception:
            prices[s] = np.nan
    df["LTP"] = df["Symbol"].map(prices)
    df["Value"] = df["LTP"] * df["Quantity"]
    if "BuyPrice" in df.columns:
        df["Cost"] = df["BuyPrice"] * df["Quantity"]
        df["P&L"] = df["Value"] - df["Cost"]
        df["P&L %"] = np.where(df["Cost"] != 0, df["P&L"] / df["Cost"] * 100, np.nan)
    return df

def suggest_allocation(age, risk_level, horizon_years):
    base_equity = 60
    if age >= 60: base_equity = 30
    elif age >= 45: base_equity = 40
    elif age <= 30: base_equity = 70

    if risk_level == "Low": equity = base_equity - 20
    elif risk_level == "Moderate": equity = base_equity
    else: equity = base_equity + 10

    if horizon_years >= 15: equity += 5
    elif horizon_years <= 5: equity -= 5

    equity = max(10, min(90, int(round(equity))))
    debt = 100 - equity
    gold = int(round(equity * 0.08))
    reits = 5
    cash = 5
    allocation = {
        "Large-cap Equity": int(round(equity * 0.45)),
        "Mid/Small-cap Equity": int(round(equity * 0.25)),
        "International Equity": int(round(equity * 0.10)),
        "Debt — Govt": int(round(debt * 0.6)),
        "Debt — Corporate": int(round(debt * 0.3)),
        "Cash/Liquid": int(round(cash)),
        "Gold": gold,
        "Real Estate / REITs": reits
    }
    total = sum(allocation.values())
    for k in allocation:
        allocation[k] = int(round(allocation[k] * 100 / total))
    diff = 100 - sum(allocation.values())
    allocation["Cash/Liquid"] += diff
    rationale = f"Rule-of-thumb allocation based on age {age}, risk {risk_level}, horizon {horizon_years}y."
    return allocation, rationale

def sip_projection(monthly_sip, years, annual_return, inflation=0.05):
    months = years * 12
    r = annual_return / 12
    fv = monthly_sip * (((1 + r) ** months - 1) / r) if r != 0 else monthly_sip * months
    real = fv / ((1 + inflation) ** years)
    return fv, real

# -------------------------
# Sidebar — Enhanced UI
# -------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:10px;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/NSE_Logo.svg' width='110'>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown(
        f"<h2 style='text-align:center; color:{ACCENT}; font-weight:700;'>AI Portfolio Cockpit</h2>",
        unsafe_allow_html=True
    )
    st.caption("by Ishani — Empowering data-driven investing 💹")
    st.markdown("---")
    nav = st.radio(
        "Navigation",
        [
            "Home", "Live Market", "Market Pulse",
            "Portfolio", "Asset Allocation",
            "Allocation Advisor", "Goals & SIP",
            "Sector Heatmap", "Watchlist"
        ],
        index=1
    )
    st.markdown("<hr style='border:0.5px solid #1b2a3d;'>", unsafe_allow_html=True)
    st.caption("⚙️ Built for Indian markets • Auto-fallbacks enabled")

# -------------------------
# All main sections preserved as-is
# -------------------------

# --- Home ---
if nav == "Home":
    st.markdown(f"<div class='titlebig'>AI Portfolio Cockpit</div>", unsafe_allow_html=True)
    st.write("Your AI-driven financial command center — live market, portfolios, allocations, and projections.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY", "Use Live Market tab")
    c2.metric("Portfolio", "Upload CSV")
    c3.metric("Allocation", "See Asset Allocation tab")
    st.info("Explore your portfolio holistically. Switch tabs on the left to navigate features.")

# --- Market Pulse ---
elif nav == "Market Pulse":
    st.markdown(f"<div class='titlebig'>Market Pulse Snapshot</div>", unsafe_allow_html=True)
    symbols = {
        "NIFTY": "^NSEI", "SENSEX": "^BSESN",
        "BANK NIFTY": "^NSEBANK", "USD/INR": "INR=X",
        "GOLD": "GC=F", "BITCOIN": "BTC-USD"
    }
    cols = st.columns(len(symbols))
    for i, (label, sym) in enumerate(symbols.items()):
        d = fetch_symbol(sym, period="5d", interval="1d")
        if d.empty:
            cols[i].metric(label, "N/A")
        else:
            last = d["Close"].iloc[-1]
            prev = d["Close"].iloc[-2] if len(d) > 1 else last
            pct = (last - prev) / prev * 100 if prev != 0 else 0
            cols[i].metric(label, f"{last:,.2f}", f"{pct:+.2f}%")

# --- Live Market (NIFTY) ---
elif nav == "Live Market":
    st.markdown(f"<div class='titlebig'>Live NIFTY Tracker</div>", unsafe_allow_html=True)
    nifty_df, mode, reason = fetch_safe_nifty()
    if nifty_df is None or nifty_df.empty:
        st.warning("No live data available — showing sample.")
        sample = {
            "Datetime": pd.date_range(end=pd.Timestamp.now(), periods=6, freq="D"),
            "Close": [20100, 20200, 20300, 20450, 20500, 20600]
        }
        nifty_df = pd.DataFrame(sample).set_index("Datetime")
        mode = "sample"
    st.caption(f"Mode: {mode} ({reason})")
    latest = nifty_df["Close"].iloc[-1]
    prev = nifty_df["Close"].iloc[-2] if len(nifty_df) > 1 else latest
    change = latest - prev
    pct = (change / prev) * 100 if prev != 0 else 0
    colA, colB, colC = st.columns(3)
    colA.metric("NIFTY 50", f"₹{latest:,.2f}", f"{pct:+.2f}%")
    colB.metric("Change (pts)", f"{change:+.2f}")
    colC.metric("Change (%)", f"{pct:+.2f}%")
    st.line_chart(nifty_df["Close"], use_container_width=True)

# --- Rest of tabs preserved from your version ---
elif nav == "Portfolio":
    st.markdown(f"<div class='titlebig'>Portfolio Tracker</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV: Symbol,Quantity,BuyPrice", type=["csv"])
    if uploaded:
        raw = pd.read_csv(uploaded, header=None)
        if raw.shape[1] == 2:
            raw.columns = ["Symbol", "Quantity"]
        else:
            raw = raw.iloc[:, :3]
            raw.columns = ["Symbol", "Quantity", "BuyPrice"]
        raw["Symbol"] = raw["Symbol"].astype(str).str.strip().str.upper()
        raw["Quantity"] = raw["Quantity"].astype(float)
        if "BuyPrice" in raw.columns:
            raw["BuyPrice"] = raw["BuyPrice"].astype(float)
        df_hold = compute_portfolio_value(raw)
        st.dataframe(df_hold.round(2), use_container_width=True)
        total_value = df_hold["Value"].sum()
        st.metric("Total Value", f"₹{total_value:,.2f}")
    else:
        st.info("Upload your holdings CSV to compute market value.")

elif nav == "Asset Allocation":
    st.markdown(f"<div class='titlebig'>Asset Allocation</div>", unsafe_allow_html=True)
    age = st.number_input("Age", 20, 80, 35)
    risk = st.selectbox("Risk", ["Low", "Moderate", "High"], index=1)
    horizon = st.slider("Horizon (years)", 1, 30, 10)
    if st.button("Suggest Allocation"):
        alloc, rationale = suggest_allocation(age, risk, horizon)
        df_alloc = pd.DataFrame(list(alloc.items()), columns=["Asset Class", "Allocation %"])
        st.dataframe(df_alloc, use_container_width=True)
        st.plotly_chart(px.pie(df_alloc, names="Asset Class", values="Allocation %"), use_container_width=True)
        st.write(rationale)

elif nav == "Goals & SIP":
    st.markdown(f"<div class='titlebig'>SIP Projection</div>", unsafe_allow_html=True)
    sip = st.number_input("Monthly SIP (₹)", 1000, 1000000, 5000)
    years = st.slider("Years", 1, 40, 10)
    ret = st.slider("Expected Return (%)", 4.0, 20.0, 10.0)
    inf = st.slider("Inflation (%)", 0.0, 10.0, 5.0)
    fv, real = sip_projection(sip, years, ret / 100, inf / 100)
    st.metric("Projected Corpus", f"₹{fv:,.0f}", f"Real Value: ₹{real:,.0f}")

# --- Footer ---
st.markdown("---")
st.markdown("<div class='footer'>💠 Built with AI — Designed for Ishani’s Fintech vision.</div>", unsafe_allow_html=True)
