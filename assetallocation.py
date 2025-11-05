# your_investment_guide.py
# Your Investment Guide — Portfolio Intelligence Suite (V5)
# Combines: V2 modules + Pro upgrades (fallback live data, sector analytics, alpha/beta, MC, PDF)
# Requirements: streamlit, yfinance, pandas, numpy, plotly, fpdf (for PDF)
# Optional: pip install streamlit yfinance pandas numpy plotly fpdf

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import random
from io import BytesIO
try:
    from fpdf import FPDF
    HAS_PDF = True
except Exception:
    HAS_PDF = False

# ------------- CONFIG & THEME -------------
st.set_page_config(page_title="Your Investment Guide", layout="wide", page_icon="💠", initial_sidebar_state="expanded")

THEME = {
    "bg": "#0b1220",         # deep bloomberg-ish blue/black
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
  .hdr {{ color:{THEME['accent2']}; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

# ------------- QUOTES & HELPERS -------------
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
    """Live -> daily close -> sample. Returns (df, mode, info_str, last_trading_day)."""
    # Try intraday
    df = fetch_symbol("^NSEI", period="5d", interval="15m")
    if not df.empty:
        last_day = pd.to_datetime(df.index[-1]).date()
        info = "intraday 15m (5d)"
        return df, "Live", info, last_day
    # Fallback daily
    df = fetch_symbol("^NSEI", period="1mo", interval="1d")
    if not df.empty:
        last_day = pd.to_datetime(df.index[-1]).date()
        info = "daily close (1mo)"
        return df, "Fallback", info, last_day
    # Sample
    now = pd.Timestamp.now()
    sample = pd.DataFrame({
        "Open": [20050, 20100, 19980, 20090, 20200, 20300],
        "High": [20200, 20250, 20080, 20400, 20350, 20450],
        "Low":  [19900, 20010, 19850, 20000, 20150, 20210],
        "Close":[20100, 19980, 20050, 20300, 20250, 20350],
        "Volume":[1000,1100,1050,1200,900,950]
    }, index=pd.date_range(end=now, periods=6, freq="D"))
    last_day = pd.to_datetime(sample.index[-1]).date()
    return sample, "Offline", "sample dataset", last_day

# Quick sector map (extend as needed)
SECTOR_MAP = {
    "TCS.NS":"IT","INFY.NS":"IT","WIPRO.NS":"IT",
    "HDFCBANK.NS":"Banking","ICICIBANK.NS":"Banking","KOTAKBANK.NS":"Banking",
    "RELIANCE.NS":"Energy","ONGC.NS":"Energy","BPCL.NS":"Energy",
    "SUNPHARMA.NS":"Pharma","DRREDDY.NS":"Pharma","CIPLA.NS":"Pharma",
    "HINDUNILVR.NS":"FMCG","NESTLEIND.NS":"FMCG","BRITANNIA.NS":"FMCG"
}

# ------------- SIDEBAR (Professional + Interactive) -------------
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
        "📈 Market Pulse",
        "📂 Portfolio",
        "📊 Sector Analytics",
        "🧩 Asset Allocation",
        "🤖 Allocation Advisor",
        "🎯 Goals & SIP",
        "🔬 Monte Carlo",
        "🧾 PDF Report",
        "📡 Watchlist"
    ], index=0)

    st.markdown("---")
    st.caption("Pro theme • Smart fallbacks • Investor-grade analytics")

# ------------- PAGES -------------

# 🏠 Overview
if nav == "🏠 Overview":
    st.markdown("<div class='titlebig'>Dashboard Overview</div>", unsafe_allow_html=True)
    df_n, mode, info, ltd = fetch_nifty_smart()
    last_refresh = datetime.now().strftime("%d %b %Y, %I:%M %p")
    c1, c2, c3, c4 = st.columns(4)
    latest = float(df_n["Close"].iloc[-1])
    prev = float(df_n["Close"].iloc[-2]) if len(df_n)>=2 else latest
    delta = latest - prev
    pct = (delta/prev*100) if prev else 0
    c1.metric("NIFTY 50", f"{latest:,.2f}", f"{pct:+.2f}%")
    c2.markdown(f"<span class='chip'>Mode: {mode}</span>", unsafe_allow_html=True)
    c3.markdown(f"<span class='chip'>Source: {info}</span>", unsafe_allow_html=True)
    c4.markdown(f"<span class='chip'>Last refreshed: {last_refresh}</span>", unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>Last trading day: {ltd.strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
    st.write("")
    st.line_chart(df_n["Close"], use_container_width=True)

# 💹 Live Market (with refresh + fallbacks)
elif nav == "💹 Live Market":
    st.markdown("<div class='titlebig'>Live Indian Market</div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
    df_n, mode, info, ltd = fetch_nifty_smart()
    latest = float(df_n["Close"].iloc[-1])
    prev = float(df_n["Close"].iloc[-2]) if len(df_n)>=2 else latest
    d = latest - prev
    pct = (d/prev*100) if prev else 0
    a,b,c = st.columns(3)
    a.metric("Last Price", f"{latest:,.2f}")
    b.metric("Δ (pts)", f"{d:+.2f}")
    c.metric("Δ (%)", f"{pct:+.2f}%")
    st.markdown(f"<span class='chip'>Mode: {mode}</span> <span class='chip'>Source: {info}</span> <span class='chip'>Last day: {ltd.strftime('%d %b %Y')}</span>", unsafe_allow_html=True)
    st.line_chart(df_n["Close"], use_container_width=True)
    st.dataframe(df_n.tail(10).round(2), use_container_width=True)

# 📈 Market Pulse
elif nav == "📈 Market Pulse":
    st.markdown("<div class='titlebig'>Market Pulse</div>", unsafe_allow_html=True)
    tickers = {"NIFTY":"^NSEI","SENSEX":"^BSESN","USD/INR":"INR=X","GOLD":"GC=F","BTC":"BTC-USD"}
    cols = st.columns(len(tickers))
    for i,(lbl,sym) in enumerate(tickers.items()):
        d = fetch_symbol(sym, period="5d", interval="1d")
        if d.empty: 
            cols[i].metric(lbl,"N/A")
        else:
            last = float(d["Close"].iloc[-1]); prev = float(d["Close"].iloc[-2]) if len(d)>=2 else last
            pct = (last-prev)/prev*100 if prev else 0
            disp = f"{last:,.2f}" if lbl in ["USD/INR","GOLD","BTC"] else f"₹{last:,.2f}"
            cols[i].metric(lbl, disp, f"{pct:+.2f}%")
    st.caption("Pulse cached ~5 minutes.")

# 📂 Portfolio
elif nav == "📂 Portfolio":
    st.markdown("<div class='titlebig'>Portfolio Tracker</div>", unsafe_allow_html=True)
    st.write("Upload CSV: `Symbol,Quantity,BuyPrice` (BuyPrice optional). Example: `TCS.NS,5,3500`")
    up = st.file_uploader("Upload holdings CSV", type=["csv"])
    if up:
        try:
            raw = pd.read_csv(up, header=None)
            if raw.shape[1] == 2: raw.columns = ["Symbol","Quantity"]
            else:
                raw = raw.iloc[:,:3]; raw.columns = ["Symbol","Quantity","BuyPrice"]
            raw["Symbol"] = raw["Symbol"].astype(str).str.strip().str.upper()
            raw["Quantity"] = raw["Quantity"].astype(float)
            if "BuyPrice" in raw.columns: raw["BuyPrice"] = raw["BuyPrice"].astype(float)
            # attach LTP
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
                pnl = raw["P&L"].sum(); c2.metric("Total P&L", fmt_inr(pnl), f"{(pnl/raw['Cost'].sum()*100):+.2f}%")
            c3.metric("Tickers", raw["Symbol"].nunique())

            # pie
            alloc = raw.groupby("Symbol")["Value"].sum().reset_index()
            if not alloc.empty:
                fig = px.pie(alloc, names="Symbol", values="Value", title="Allocation", color_discrete_sequence=px.colors.sequential.Tealgrn)
                st.plotly_chart(fig, use_container_width=True)

            st.session_state["portfolio_df"] = raw
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Upload your CSV to see valuation and allocation.")

# 📊 Sector Analytics (uses uploaded portfolio)
elif nav == "📊 Sector Analytics":
    st.markdown("<div class='titlebig'>Sector Analytics</div>", unsafe_allow_html=True)
    pf = st.session_state.get("portfolio_df")
    if pf is None:
        st.warning("Upload your portfolio in the Portfolio tab first.")
    else:
        tmp = pf.assign(Sector=pf["Symbol"].map(SECTOR_MAP).fillna("Other"))
        sector_alloc = tmp.groupby("Sector")["Value"].sum().reset_index().sort_values("Value", ascending=False)
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Sector allocation")
            fig = px.pie(sector_alloc, names="Sector", values="Value", hole=0.3, color_discrete_sequence=px.colors.sequential.Tealgrn)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Avg sector % change (last day)")
            rows=[]
            for sec, g in tmp.groupby("Sector"):
                ch=[]
                for s in g["Symbol"].unique():
                    d = fetch_symbol(s, period="5d", interval="1d")
                    if not d.empty and len(d)>=2:
                        last, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
                        if prev: ch.append((last-prev)/prev*100)
                rows.append({"Sector":sec, "Change%": np.mean(ch) if ch else 0.0})
            sec_perf = pd.DataFrame(rows).sort_values("Change%", ascending=False)
            bar = px.bar(sec_perf, x="Sector", y="Change%", text="Change%")
            st.plotly_chart(bar, use_container_width=True)
        if not sector_alloc.empty:
            top_sec = sector_alloc.iloc[0]
            st.success(f"Top allocation: {top_sec['Sector']} — {fmt_inr(top_sec['Value'])}")

# 🧩 Asset Allocation (manual sliders)
elif nav == "🧩 Asset Allocation":
    st.markdown("<div class='titlebig'>Asset Allocation Builder</div>", unsafe_allow_html=True)
    options = ["Equity","Debt","Gold","REITs","International","Cash"]
    w = {}
    cols = st.columns(3)
    for i,opt in enumerate(options):
        with cols[i%3]:
            w[opt] = st.slider(f"{opt} (%)", 0, 100, 15)
    total = sum(w.values())
    if total != 100:
        st.warning(f"Allocations must total 100%. Current: {total}%")
    else:
        df = pd.DataFrame(list(w.items()), columns=["Asset Class","Allocation (%)"])
        fig = px.pie(df, names="Asset Class", values="Allocation (%)", color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)
        st.success("Allocation saved (session).")

# 🤖 Allocation Advisor (rule-based)
elif nav == "🤖 Allocation Advisor":
    st.markdown("<div class='titlebig'>AI Allocation Advisor (Rule-based)</div>", unsafe_allow_html=True)
    age = st.number_input("Age", 18, 80, 35)
    risk = st.selectbox("Risk appetite", ["Low","Moderate","High"], index=1)
    horizon = st.slider("Horizon (years)", 1, 40, 10)
    # simple rule:
    base_equity = 70 if age<=30 else (40 if age>=45 else 60)
    equity = base_equity + (10 if risk=="High" else (-20 if risk=="Low" else 0))
    if horizon>=15: equity += 5
    if horizon<=5: equity -= 5
    equity = int(np.clip(equity, 10, 90))
    debt = 100 - equity
    alloc = pd.DataFrame({
        "Asset Class":["Large-cap Equity","Mid/Small-cap Equity","International Equity","Debt — Govt","Debt — Corporate","Gold","REITs","Cash/Liquid"],
        "Allocation %":[int(equity*0.45), int(equity*0.25), int(equity*0.10), int(debt*0.6), int(debt*0.3), 8, 5, 5]
    })
    # normalize
    diff = 100 - alloc["Allocation %"].sum()
    alloc.loc[alloc.index[0], "Allocation %"] += diff
    st.dataframe(alloc, use_container_width=True)
    bar = px.bar(alloc, x="Asset Class", y="Allocation %", text="Allocation %")
    st.plotly_chart(bar, use_container_width=True)
    st.info(f"Rationale: age {age}, risk {risk}, horizon {horizon}y → equity tilt {equity}%.")

# 🎯 Goals & SIP
elif nav == "🎯 Goals & SIP":
    st.markdown("<div class='titlebig'>Goals & SIP Simulator</div>", unsafe_allow_html=True)
    mode = st.selectbox("Mode", ["SIP (monthly)","Lump sum"], index=0)
    if mode == "SIP (monthly)":
        monthly = st.number_input("Monthly SIP (₹)", 100, 200000, 5000)
        years = st.slider("Years", 1, 40, 15)
        exp = st.slider("Expected annual return (%)", 3.0, 20.0, 10.0)
        r = exp/100/12
        months = years*12
        fv = monthly*((1+r)**months - 1)/r if r!=0 else monthly*months
        st.metric("Projected corpus", fmt_inr(fv), f"{exp}%")
        df = pd.DataFrame({"Month":range(1,months+1),
                           "Balance":[monthly*(((1+r)**m - 1)/r if r!=0 else m) for m in range(1, months+1)]})
        st.line_chart(df.set_index("Month"), use_container_width=True)
    else:
        principal = st.number_input("Lump sum (₹)", 10000, 100000000, 100000)
        years = st.slider("Years", 1, 40, 10)
        exp = st.slider("Expected annual return (%)", 3.0, 20.0, 9.0)
        fv = principal*((1+exp/100)**years)
        st.metric("Projected corpus", fmt_inr(fv), f"{exp}%")

# 🔬 Monte Carlo (goal probability)
elif nav == "🔬 Monte Carlo":
    st.markdown("<div class='titlebig'>Monte Carlo Goal Probability</div>", unsafe_allow_html=True)
    corpus_goal = st.number_input("Target corpus (₹)", 100000, 1000000000, 10000000, step=50000)
    invest_now = st.number_input("Current invested (₹)", 0, 100000000, 500000, step=10000)
    monthly_sip = st.number_input("Monthly SIP (₹)", 0, 500000, 10000, step=1000)
    years = st.slider("Horizon (years)", 1, 40, 15)
    exp_return = st.slider("Expected return (%)", 3.0, 20.0, 11.0)
    vol = st.slider("Volatility (%)", 5.0, 35.0, 18.0)
    sims = st.slider("Simulations", 200, 10000, 2000, step=200)
    # annual steps
    mu = exp_return/100.0
    sigma = vol/100.0
    w = 1.0  # single-asset proxy
    annual_sip = monthly_sip*12
    res = np.zeros(sims)
    for s in range(sims):
        val = invest_now
        for y in range(years):
            ret = np.random.normal(mu, sigma)
            val = val*(1+ret) + annual_sip
        res[s] = val
    prob = (res >= corpus_goal).mean()*100
    st.metric("Probability of success", f"{prob:.1f}%")
    st.plotly_chart(px.histogram(res, nbins=60, title="Distribution of outcomes"), use_container_width=True)

# 🧾 PDF Report (from session data)
elif nav == "🧾 PDF Report":
    st.markdown("<div class='titlebig'>PDF Investor Report</div>", unsafe_allow_html=True)
    if not HAS_PDF:
        st.error("Install `fpdf` to enable PDF export: pip install fpdf")
    else:
        pf = st.session_state.get("portfolio_df")
        df_n, mode, info, ltd = fetch_nifty_smart()
        latest = float(df_n["Close"].iloc[-1])
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Your Investment Guide — Report for {st.session_state.get('user_name','Investor')}", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"Greeting: {greeting()}", ln=True)
        pdf.cell(0, 8, f"Quote: {st.session_state.get('quote','')}", ln=True)
        pdf.cell(0, 8, f"NIFTY: {latest:,.2f}  | Mode: {mode}  | Source: {info}", ln=True)
        pdf.ln(4)
        if pf is not None:
            pdf.cell(0, 8, "Portfolio Allocation:", ln=True)
            bysym = pf.groupby("Symbol")["Value"].sum().reset_index().sort_values("Value", ascending=False)
            for _,r in bysym.iterrows():
                pdf.cell(0, 7, f" - {r['Symbol']}: {fmt_inr(r['Value'])}", ln=True)
        else:
            pdf.cell(0, 8, "Portfolio: Not uploaded", ln=True)
        # output
        buf = BytesIO()
        pdf.output(buf)
        st.download_button("Download PDF", buf.getvalue(), file_name="Your_Investment_Report.pdf", mime="application/pdf")
        st.info("Report includes greeting, quote, NIFTY snapshot, and portfolio allocation (if uploaded).")

# 📡 Watchlist
elif nav == "📡 Watchlist":
    st.markdown("<div class='titlebig'>Watchlist & Alerts</div>", unsafe_allow_html=True)
    default = "TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS"
    wl = st.text_input("Tickers (comma separated)", value=default)
    thresh = st.number_input("Alert threshold (%)", 0.5, 20.0, 2.0)
    if st.button("Fetch"):
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        rows=[]
        for t in tickers:
            d = fetch_symbol(t, period="5d", interval="1d")
            if d.empty:
                rows.append({"Symbol": t, "Price":"N/A", "Change%":"N/A", "Alert":""})
            else:
                last = float(d["Close"].iloc[-1]); prev = float(d["Close"].iloc[-2]) if len(d)>=2 else last
                pct = (last-prev)/prev*100 if prev else 0.0
                rows.append({"Symbol": t.replace(".NS",""), "Price": f"₹{last:,.2f}", "Change%": f"{pct:+.2f}%", "Alert":"⚠️" if abs(pct)>=thresh else ""})
        st.table(pd.DataFrame(rows))

# Footer
st.markdown("---")
st.caption("Your Investment Guide — Educational; not investment advice. Verify instruments, taxes, and suitability.")
