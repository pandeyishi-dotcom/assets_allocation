# your_investment_guide_v5_1.py
# Your Investment Guide — Portfolio Intelligence Suite (Unicode-safe PDF)
# Requirements: streamlit, yfinance, pandas, numpy, plotly, fpdf

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import random
from io import BytesIO

try:
    from fpdf import FPDF
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# ---------------- CONFIG & THEME ----------------
st.set_page_config(page_title="Your Investment Guide", layout="wide", page_icon="💠")

THEME = {
    "bg": "#0b1220",
    "panel": "#121a2b",
    "accent": "#5ae0c8",
    "accent2": "#9cc3ff",
    "muted": "#a9b3c7",
    "text": "#e6eef5"
}

st.markdown(f"""
<style>
  .stApp {{ background: linear-gradient(180deg, {THEME['bg']} 0%, #0c1a2a 100%); color:{THEME['text']}; }}
  div[data-testid="stSidebar"] {{ background:{THEME['panel']}; color:{THEME['text']}; }}
  .titlebig{{ font-size:28px; color:{THEME['accent']}; font-weight:800; margin-bottom:6px; }}
  .muted{{ color:{THEME['muted']}; }}
  .card{{ background:{THEME['panel']}; padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.06); }}
  .chip{{ display:inline-block; padding:2px 10px; border-radius:999px; background:#0f2740; color:{THEME['accent2']}; font-size:12px; margin-right:6px; }}
  .greeting {{ font-size:18px; color:{THEME['accent']}; font-weight:700; }}
  .quote {{ color:{THEME['muted']}; font-style:italic; }}
</style>
""", unsafe_allow_html=True)

# ---------------- UTILITIES ----------------
QUOTES = [
    "Invest early. Time is the compounding engine.",
    "Diversify to control risk, not eliminate it.",
    "Plan like a pessimist, invest like an optimist.",
    "Volatility is the price of returns.",
    "Small, repeatable edges beat heroic bets."
]

def greeting():
    h = datetime.now().hour
    return "Good morning" if h < 12 else ("Good afternoon" if h < 17 else "Good evening")

def fmt_inr(x):
    try:
        return "₹{:,.0f}".format(float(x))
    except:
        return str(x)

@st.cache_data(ttl=300)
def fetch_symbol(symbol, period="5d", interval="15m"):
    try:
        return yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception:
        return pd.DataFrame()

def fetch_nifty_smart():
    df = fetch_symbol("^NSEI", period="5d", interval="15m")
    if not df.empty:
        last_day = pd.to_datetime(df.index[-1]).date()
        return df, "Live", "Intraday 15m (5d)", last_day
    df = fetch_symbol("^NSEI", period="1mo", interval="1d")
    if not df.empty:
        last_day = pd.to_datetime(df.index[-1]).date()
        return df, "Fallback", "Daily Close (1mo)", last_day
    now = pd.Timestamp.now()
    sample = pd.DataFrame({
        "Open": [20050, 20100, 19980, 20090, 20200, 20300],
        "High": [20200, 20250, 20080, 20400, 20350, 20450],
        "Low": [19900, 20010, 19850, 20000, 20150, 20210],
        "Close":[20100, 19980, 20050, 20300, 20250, 20350],
        "Volume":[1000,1100,1050,1200,900,950]
    }, index=pd.date_range(end=now, periods=6, freq="D"))
    return sample, "Offline", "Sample dataset", pd.to_datetime(sample.index[-1]).date()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<div class='titlebig'>Your Investment Guide</div>", unsafe_allow_html=True)
    colL, colR = st.columns([1,1])
    with colL:
        name_in = st.text_input("Investor name", value=st.session_state.get("user_name", "Ishani"))
    with colR:
        if st.button("🔁 New Quote", use_container_width=True):
            st.session_state["quote"] = random.choice(QUOTES)

    if name_in:
        st.session_state["user_name"] = name_in.strip().title()
    if "quote" not in st.session_state:
        st.session_state["quote"] = random.choice(QUOTES)

    st.markdown(f"<div class='greeting'>{greeting()}, {st.session_state['user_name']} 👋</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='quote'>“{st.session_state['quote']}”</div>", unsafe_allow_html=True)
    st.markdown("---")

    nav = st.radio("Navigate", [
        "🏠 Overview",
        "💹 Live Market",
        "📂 Portfolio",
        "📊 Sector Analytics",
        "🧾 PDF Report"
    ], index=0)
    st.markdown("---")
    st.caption("Pro Theme • Smart Fallbacks • Unicode-safe PDF")

# ---------------- PAGES ----------------

# 🏠 Overview
if nav == "🏠 Overview":
    st.markdown("<div class='titlebig'>Dashboard Overview</div>", unsafe_allow_html=True)
    df_n, mode, info, ltd = fetch_nifty_smart()
    last_refresh = datetime.now().strftime("%d %b %Y, %I:%M %p")
    c1, c2, c3 = st.columns(3)
    latest = float(df_n["Close"].iloc[-1])
    prev = float(df_n["Close"].iloc[-2]) if len(df_n)>=2 else latest
    delta = latest - prev
    pct = (delta/prev*100) if prev else 0
    c1.metric("NIFTY 50", f"{latest:,.2f}", f"{pct:+.2f}%")
    c2.markdown(f"<span class='chip'>Mode: {mode}</span>", unsafe_allow_html=True)
    c3.markdown(f"<span class='chip'>Last refreshed: {last_refresh}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>Last trading day: {ltd.strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
    st.line_chart(df_n["Close"], use_container_width=True)

# 💹 Live Market
elif nav == "💹 Live Market":
    st.markdown("<div class='titlebig'>Live Indian Market</div>", unsafe_allow_html=True)
    df_n, mode, info, ltd = fetch_nifty_smart()
    latest = float(df_n["Close"].iloc[-1])
    prev = float(df_n["Close"].iloc[-2]) if len(df_n)>=2 else latest
    d = latest - prev
    pct = (d/prev*100) if prev else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Last Price", f"{latest:,.2f}")
    c2.metric("Δ (pts)", f"{d:+.2f}")
    c3.metric("Δ (%)", f"{pct:+.2f}%")
    st.markdown(f"<span class='chip'>Mode: {mode}</span> <span class='chip'>Source: {info}</span>", unsafe_allow_html=True)
    st.line_chart(df_n["Close"], use_container_width=True)
    st.dataframe(df_n.tail(10).round(2), use_container_width=True)

# 📂 Portfolio
elif nav == "📂 Portfolio":
    st.markdown("<div class='titlebig'>Portfolio Tracker</div>", unsafe_allow_html=True)
    st.write("Upload CSV: `Symbol,Quantity,BuyPrice` (BuyPrice optional). Example: `TCS.NS,5,3500`")
    up = st.file_uploader("Upload holdings CSV", type=["csv"])
    if up:
        raw = pd.read_csv(up, header=None)
        if raw.shape[1] == 2: raw.columns = ["Symbol","Quantity"]
        else:
            raw = raw.iloc[:,:3]; raw.columns = ["Symbol","Quantity","BuyPrice"]
        raw["Symbol"] = raw["Symbol"].astype(str).str.strip().str.upper()
        raw["Quantity"] = raw["Quantity"].astype(float)
        if "BuyPrice" in raw.columns: raw["BuyPrice"] = raw["BuyPrice"].astype(float)
        prices = {}
        for s in raw["Symbol"].unique():
            d = fetch_symbol(s, period="5d", interval="1d")
            prices[s] = float(d["Close"].iloc[-1]) if not d.empty else np.nan
        raw["LTP"] = raw["Symbol"].map(prices)
        raw["Value"] = raw["Quantity"] * raw["LTP"]
        if "BuyPrice" in raw.columns:
            raw["Cost"] = raw["Quantity"] * raw["BuyPrice"]
            raw["P&L"] = raw["Value"] - raw["Cost"]
            raw["P&L %"] = np.where(raw["Cost"]!=0, raw["P&L"]/raw["Cost"]*100, np.nan)

        st.dataframe(raw.round(2), use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Market Value", fmt_inr(raw["Value"].sum()))
        if "Cost" in raw.columns:
            pnl = raw["P&L"].sum()
            c2.metric("Total P&L", fmt_inr(pnl), f"{(pnl/raw['Cost'].sum()*100):+.2f}%")
        c3.metric("Tickers", raw["Symbol"].nunique())
        st.session_state["portfolio_df"] = raw
    else:
        st.info("Upload your portfolio CSV to analyze holdings.")

# 📊 Sector Analytics
elif nav == "📊 Sector Analytics":
    st.markdown("<div class='titlebig'>Sector Analytics</div>", unsafe_allow_html=True)
    pf = st.session_state.get("portfolio_df")
    if pf is None:
        st.warning("Upload your portfolio in the Portfolio tab first.")
    else:
        st.dataframe(pf[["Symbol","Value"]].sort_values("Value",ascending=False), use_container_width=True)
        st.success("Sector analysis placeholder — extend with mappings if needed.")

# 🧾 PDF Report (Unicode-safe)
elif nav == "🧾 PDF Report":
    st.markdown("<div class='titlebig'>PDF Investor Report</div>", unsafe_allow_html=True)
    if not HAS_PDF:
        st.error("Install `fpdf`: pip install fpdf")
    else:
        pf = st.session_state.get("portfolio_df")
        df_n, mode, info, ltd = fetch_nifty_smart()
        latest = float(df_n["Close"].iloc[-1])

        def clean_text(txt):
            """Replace non-Latin chars safely for FPDF."""
            return str(txt).encode("latin-1", "replace").decode("latin-1")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, clean_text(f"Your Investment Guide — Report for {st.session_state.get('user_name','Investor')}"), ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, clean_text(f"Greeting: {greeting()}"), ln=True)
        pdf.cell(0, 8, clean_text(f"Quote: {st.session_state.get('quote','')}"), ln=True)
        pdf.cell(0, 8, clean_text(f"NIFTY: {latest:,.2f}  | Mode: {mode}  | Source: {info}"), ln=True)
        pdf.ln(4)

        if pf is not None:
            pdf.cell(0, 8, "Portfolio Allocation:", ln=True)
            bysym = pf.groupby("Symbol")["Value"].sum().reset_index().sort_values("Value", ascending=False)
            for _,r in bysym.iterrows():
                pdf.cell(0, 7, clean_text(f" - {r['Symbol']}: Rs {r['Value']:,.0f}"), ln=True)
        else:
            pdf.cell(0, 8, "Portfolio: Not uploaded", ln=True)

        buf = BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
        buf.write(pdf_bytes)
        buf.seek(0)
        st.download_button("📄 Download PDF Report", buf, file_name="Your_Investment_Report.pdf", mime="application/pdf")
        st.info("Report includes greeting, quote, NIFTY snapshot, and portfolio allocation (if uploaded).")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Your Investment Guide — Educational only. Verify before investing.")
