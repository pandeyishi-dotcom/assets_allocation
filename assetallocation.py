# app.py — Your Investment Guide (V6, Professional Edition)
# Requirements: streamlit, yfinance, pandas, numpy, plotly, fpdf, pillow (for logo fallback)
# pip install streamlit yfinance pandas numpy plotly fpdf pillow

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from fpdf import FPDF
from datetime import datetime, date, timedelta
import io
import random
from math import ceil

# -------------------------#
# Page config & Pro theme  #
# -------------------------#
st.set_page_config(
    page_title="Your Investment Guide",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#00FFC6"
BG1 = "#0b1320"
BG2 = "#0f1b2b"
TEXT = "#e6eef0"
MUTED = "#9fb4c8"
CARD = "rgba(255,255,255,0.04)"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: linear-gradient(180deg, {BG1} 0%, {BG2} 100%);
        color: {TEXT};
        font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
      }}
      div[data-testid="stSidebar"] {{
        background: #0a1220;
        border-right: 1px solid rgba(255,255,255,0.07);
      }}
      .titlebig {{ font-size:28px; font-weight:800; color:{ACCENT}; }}
      .muted {{ color:{MUTED}; }}
      .card {{
        background:{CARD};
        padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.06);
      }}
      .badge {{
        display:inline-block; padding:2px 8px; border-radius:999px;
        background:rgba(0,255,198,0.12); color:{ACCENT}; font-size:12px; border:1px solid rgba(0,255,198,0.25);
      }}
      .badge-red {{ background: rgba(220,38,38,0.15); color:#ffb3b3; border:1px solid rgba(220,38,38,0.35); }}
      .badge-amber {{ background: rgba(245,158,11,0.15); color:#ffd89a; border:1px solid rgba(245,158,11,0.35); }}
      .badge-green {{ background: rgba(16,185,129,0.15); color:#b7ffdf; border:1px solid rgba(16,185,129,0.35); }}
      .hi {{ font-weight:700; color:#b9ffe0; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------#
# Quotes & helpers         #
# -------------------------#
QUOTES = [
    "Invest early. Time is the compounding engine.",
    "Discipline compounds like capital.",
    "Diversify not to avoid risk, but to domesticate it.",
    "Plan like a pessimist; invest like an optimist.",
    "Markets are volatile; your process shouldn’t be."
]

def greet():
    h = datetime.now().hour
    return "Good Morning" if h < 12 else ("Good Afternoon" if h < 17 else "Good Evening")

def fmt_inr(x):
    try:
        return "₹{:,.0f}".format(float(x))
    except Exception:
        return "₹0"

def to_latin1(s: str) -> str:
    # Make any text safe for FPDF core fonts (latin-1)
    if s is None:
        return ""
    return s.encode("latin-1", "ignore").decode("latin-1")

# -------------------------#
# Sidebar (Pro)            #
# -------------------------#
with st.sidebar:
    st.markdown("### 💼 Your Investment Guide")
    user_name = st.text_input("Your name", value=st.session_state.get("user_name", "Ishani")).strip()
    if user_name:
        st.session_state["user_name"] = user_name
    st.caption(f"{greet()}, {st.session_state['user_name']} 👋")
    # Rotating quote (collapsible)
    with st.expander("Quote of the session", expanded=True):
        if "quote" not in st.session_state:
            st.session_state.quote = random.choice(QUOTES)
        if st.button("Shuffle quote", key="shuffle_quote"):
            st.session_state.quote = random.choice(QUOTES)
        st.markdown(f"_{st.session_state.quote}_")

    st.markdown("---")
    nav = st.radio(
        "Navigate",
        [
            "Overview",
            "Live Market",
            "Market Pulse",
            "Portfolio",
            "Sector Analytics",
            "Asset Allocation",
            "Allocation Advisor",
            "Goals & SIP",
            "Monte Carlo",
            "PDF Report",
            "Watchlist"
        ],
        index=0,
    )
    st.markdown("---")
    st.caption("Pro theme • Smart fallbacks • Investor-grade analytics")

# -------------------------#
# Data utils               #
# -------------------------#
@st.cache_data(ttl=300)
def fetch(symbol, period="5d", interval="15m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_nifty_smart():
    """Try intraday; fallback to 1d; else sample."""
    mode = "Live"
    reason = "intraday 15m"
    df = fetch("^NSEI", "5d", "15m")
    if df is not None and not df.empty:
        return df, mode, reason

    mode, reason = "Fallback", "daily close (1mo)"
    df = fetch("^NSEI", "1mo", "1d")
    if df is not None and not df.empty:
        return df, mode, reason

    mode, reason = "Offline", "sample data"
    sample = {
        "Datetime": pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D"),
        "Open": [20050, 20100, 19980, 20090, 20200, 20300, 20410, 20500, 20420, 20550],
        "High": [20200, 20250, 20080, 20400, 20350, 20450, 20530, 20610, 20500, 20620],
        "Low":  [19900, 20010, 19850, 20000, 20150, 20210, 20310, 20390, 20380, 20490],
        "Close":[20100, 19980, 20050, 20300, 20250, 20350, 20480, 20520, 20460, 20590],
        "Volume":[1000,1100,1050,1200,900,950,980,1005,990,1020]
    }
    df = pd.DataFrame(sample).set_index("Datetime")
    return df, mode, reason

def latest_and_prev(series):
    latest = float(series.iloc[-1])
    prev = float(series.iloc[-2]) if len(series) > 1 else latest
    change = latest - prev
    pct = (change / prev) * 100 if prev else 0.0
    return latest, change, pct

# -------------------------#
# Shared state for report  #
# -------------------------#
if "report_snap" not in st.session_state:
    st.session_state.report_snap = {}  # store small, printable data slices

# -------------------------#
# Overview                 #
# -------------------------#
if nav == "Overview":
    st.markdown('<div class="titlebig">Your Investment Guide</div>', unsafe_allow_html=True)
    st.markdown(f"<div class='muted'>Investor OS for analysis, planning, and reporting.</div>", unsafe_allow_html=True)
    st.write("")
    nifty_df, mode, reason = fetch_nifty_smart()
    series = nifty_df["Close"] if "Close" in nifty_df.columns else nifty_df.select_dtypes(include=[np.number]).iloc[:, 0]
    latest, chg, pct = latest_and_prev(series)
    last_index = pd.to_datetime(series.index[-1])
    last_trading_day = last_index.date()

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    c1.metric("NIFTY 50", f"{latest:,.2f}", f"{pct:+.2f}%")
    c2.metric("Δ (pts)", f"{chg:+.2f}")
    c3.markdown(f"**Last trading day**<br>{last_trading_day}", unsafe_allow_html=True)
    badge_cls = "badge-green" if mode == "Live" else ("badge-amber" if mode == "Fallback" else "badge-red")
    c4.markdown(f"<span class='badge {badge_cls}'>Mode: {mode}</span><br><span class='muted'>{reason}</span>", unsafe_allow_html=True)

    st.markdown("#### Recent movement")
    st.line_chart(series.rename("Price"), use_container_width=True)
    st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

    st.session_state.report_snap["overview"] = {
        "mode": mode, "reason": reason, "nifty": latest, "pct": pct, "date": str(last_trading_day)
    }

# -------------------------#
# Live Market              #
# -------------------------#
elif nav == "Live Market":
    st.markdown('<div class="titlebig">Live Market — NIFTY 50</div>', unsafe_allow_html=True)
    if st.button("🔄 Refresh data", type="secondary"):
        st.cache_data.clear()
    nifty_df, mode, reason = fetch_nifty_smart()
    st.markdown(f"Mode: **{mode}** · _{reason}_")
    series = nifty_df["Close"] if "Close" in nifty_df.columns else nifty_df.select_dtypes(include=[np.number]).iloc[:, 0]
    latest, chg, pct = latest_and_prev(series)

    c1, c2, c3 = st.columns(3)
    c1.metric("LTP", f"{latest:,.2f}")
    c2.metric("Δ (pts)", f"{chg:+.2f}")
    c3.metric("Δ (%)", f"{pct:+.2f}%")

    st.line_chart(series.rename("Price"), use_container_width=True)
    st.dataframe(nifty_df.tail(12).round(2), use_container_width=True)
    st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

# -------------------------#
# Market Pulse             #
# -------------------------#
elif nav == "Market Pulse":
    st.markdown('<div class="titlebig">Market Pulse</div>', unsafe_allow_html=True)
    tickers = {"NIFTY":"^NSEI", "SENSEX":"^BSESN", "USD/INR":"INR=X", "GOLD":"GC=F", "BTC":"BTC-USD"}
    cols = st.columns(len(tickers))
    for i, (label, sym) in enumerate(tickers.items()):
        d = fetch(sym, "5d", "1d")
        if d.empty:
            cols[i].metric(label, "N/A", "")
        else:
            last = float(d["Close"].iloc[-1])
            prev = float(d["Close"].iloc[-2]) if len(d) > 1 else last
            pct = (last - prev) / prev * 100 if prev else 0.0
            val = f"{last:,.2f}"
            if label not in ["BTC","USD/INR","GOLD"]:
                val = "₹" + val
            cols[i].metric(label, val, f"{pct:+.2f}%")
    st.caption("Pulse cached for 5 minutes.")

# -------------------------#
# Portfolio (upload)       #
# -------------------------#
elif nav == "Portfolio":
    st.markdown('<div class="titlebig">Portfolio Tracker</div>', unsafe_allow_html=True)
    st.markdown("Upload CSV with: `Symbol,Quantity,BuyPrice` (BuyPrice optional). Example: `TCS.NS,5,3500`")
    up = st.file_uploader("Upload holdings CSV", type=["csv"])
    if up:
        try:
            raw = pd.read_csv(up, header=None)
            if raw.shape[1] == 2:
                raw.columns = ["Symbol", "Quantity"]
            else:
                raw = raw.iloc[:, :3]
                raw.columns = ["Symbol", "Quantity", "BuyPrice"]
            raw["Symbol"] = raw["Symbol"].astype(str).str.strip().str.upper()
            raw["Quantity"] = raw["Quantity"].astype(float)
            if "BuyPrice" in raw.columns:
                raw["BuyPrice"] = raw["BuyPrice"].astype(float)

            # LTP map
            prices = {}
            for s in raw["Symbol"].unique():
                d = fetch(s, "1d", "1d")
                prices[s] = float(d["Close"].iloc[-1]) if not d.empty else np.nan
            raw["LTP"] = raw["Symbol"].map(prices)
            raw["Value"] = raw["LTP"] * raw["Quantity"]
            if "BuyPrice" in raw.columns:
                raw["Cost"] = raw["BuyPrice"] * raw["Quantity"]
                raw["P&L"] = raw["Value"] - raw["Cost"]
                raw["P&L %"] = np.where(raw["Cost"] != 0, raw["P&L"]/raw["Cost"]*100, np.nan)

            st.dataframe(raw.round(2), use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Market Value", fmt_inr(raw["Value"].sum()))
            if "Cost" in raw.columns:
                pnl = raw["Value"].sum() - raw["Cost"].sum()
                pct = pnl / raw["Cost"].sum() * 100 if raw["Cost"].sum() else 0
                c2.metric("Total P&L", fmt_inr(pnl), f"{pct:+.2f}%")
            c3.metric("Tickers", raw["Symbol"].nunique())

            # Allocation pie
            alloc = raw.groupby("Symbol")["Value"].sum().reset_index()
            fig = px.pie(alloc, names="Symbol", values="Value",
                         color_discrete_sequence=px.colors.sequential.Tealgrn,
                         title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)

            st.session_state.report_snap["portfolio_table"] = raw.copy()

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.info("Upload your holdings to see valuation and allocation.")

# -------------------------#
# Sector Analytics         #
# -------------------------#
elif nav == "Sector Analytics":
    st.markdown('<div class="titlebig">Sector Analytics</div>', unsafe_allow_html=True)
    st.caption("Auto-detect sectors for common NSE tickers. Demo classifier inside the app.")
    if "portfolio_table" not in st.session_state:
        st.warning("Upload your portfolio in the Portfolio tab first.")
    else:
        df = st.session_state.report_snap["portfolio_table"].copy()

        # Minimal lookup (extend this dict as you like or map via web api)
        sector_map = {
            "RELIANCE.NS":"Energy", "ONGC.NS":"Energy", "BPCL.NS":"Energy",
            "TCS.NS":"IT", "INFY.NS":"IT", "WIPRO.NS":"IT", "HCLTECH.NS":"IT",
            "HDFCBANK.NS":"Banking", "ICICIBANK.NS":"Banking", "KOTAKBANK.NS":"Banking",
            "SUNPHARMA.NS":"Pharma", "DRREDDY.NS":"Pharma", "CIPLA.NS":"Pharma",
            "HINDUNILVR.NS":"FMCG", "NESTLEIND.NS":"FMCG", "BRITANNIA.NS":"FMCG",
        }
        df["Sector"] = df["Symbol"].map(sector_map).fillna("Other")
        alloc = df.groupby("Sector")["Value"].sum().reset_index()
        alloc["Weight %"] = (alloc["Value"] / alloc["Value"].sum() * 100).round(2)
        st.dataframe(alloc.sort_values("Value", ascending=False), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_p = px.pie(alloc, names="Sector", values="Value", hole=0.35,
                           color_discrete_sequence=px.colors.sequential.Tealgrn,
                           title="Sector Allocation")
            st.plotly_chart(fig_p, use_container_width=True)
        with c2:
            # Average sector 1-day change
            perf_rows = []
            for s, grp in df.groupby("Sector"):
                diffs = []
                for t in grp["Symbol"].unique():
                    d = fetch(t, "5d", "1d")
                    if not d.empty and len(d) > 1:
                        last = d["Close"].iloc[-1]
                        prev = d["Close"].iloc[-2]
                        diffs.append((last - prev) / prev * 100 if prev else 0.0)
                perf_rows.append({"Sector": s, "Avg % (1D)": np.mean(diffs) if diffs else 0})
            perf = pd.DataFrame(perf_rows)
            fig_b = px.bar(perf, x="Sector", y="Avg % (1D)", title="Avg Sector Performance (1D)")
            st.plotly_chart(fig_b, use_container_width=True)

        st.session_state.report_snap["sector_alloc"] = alloc.copy()

# -------------------------#
# Asset Allocation (manual)#
# -------------------------#
elif nav == "Asset Allocation":
    st.markdown('<div class="titlebig">Asset Allocation</div>', unsafe_allow_html=True)
    options = ["Large Cap Equity","Mid/Small Cap Equity","International Equity","Debt Funds","Gold","REITs","Cash"]
    cols = st.columns(3)
    weights = {}
    for i, opt in enumerate(options):
        weights[opt] = cols[i%3].slider(f"{opt} (%)", 0, 100, 15)
    total = sum(weights.values())
    if total != 100:
        st.warning(f"Allocations must total 100%. Current: {total}%.")
    else:
        df_alloc = pd.DataFrame(list(weights.items()), columns=["Asset Class","Allocation (%)"])
        fig = px.pie(df_alloc, names="Asset Class", values="Allocation (%)",
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)
        st.success("Allocation saved for visualization.")
        st.session_state.report_snap["manual_alloc"] = df_alloc.copy()

# -------------------------#
# Allocation Advisor (AI)  #
# -------------------------#
elif nav == "Allocation Advisor":
    st.markdown('<div class="titlebig">AI Allocation Advisor</div>', unsafe_allow_html=True)
    age = st.slider("Age", 18, 75, 34)
    risk = st.selectbox("Risk", ["Low","Moderate","High"], index=1)
    horizon = st.slider("Horizon (years)", 1, 40, 12)

    # Simple rule engine
    base_equity = 60 if age <= 45 else (40 if age <= 60 else 30)
    if risk == "Low": equity = base_equity - 15
    elif risk == "Moderate": equity = base_equity
    else: equity = base_equity + 10
    equity = max(10, min(90, equity))
    debt = 100 - equity
    alloc_map = {
        "Large Cap Equity": int(round(equity*0.5)),
        "Mid/Small Cap Equity": int(round(equity*0.25)),
        "International Equity": int(round(equity*0.10)),
        "Debt Funds": int(round(debt*0.75)),
        "Gold": 7,
        "REITs": 4,
        "Cash": max(0, 100 - (int(round(equity*0.5))+int(round(equity*0.25))+int(round(equity*0.10))+int(round(debt*0.75))+7+4))
    }
    df_ai = pd.DataFrame({"Asset Class": list(alloc_map.keys()), "Allocation (%)": list(alloc_map.values())})
    st.dataframe(df_ai, use_container_width=True)
    st.plotly_chart(px.bar(df_ai, x="Asset Class", y="Allocation (%)", text="Allocation (%)"), use_container_width=True)
    st.caption("Rule-based and explainable. Tune the rule as needed.")
    st.session_state.report_snap["ai_alloc"] = df_ai.copy()

# -------------------------#
# Goals & SIP              #
# -------------------------#
elif nav == "Goals & SIP":
    st.markdown('<div class="titlebig">Goals & SIP</div>', unsafe_allow_html=True)
    mode = st.selectbox("Projection mode", ["SIP (monthly)", "Lump sum"], index=0)

    if mode == "SIP (monthly)":
        monthly = st.number_input("Monthly SIP (₹)", 100, 200000, 5000)
        years = st.slider("Years", 1, 40, 15)
        exp_ret = st.slider("Expected annual return (%)", 3.0, 20.0, 10.0)
        infl = st.slider("Inflation (%)", 0.0, 12.0, 4.5)
        months = years * 12
        r = exp_ret/100/12
        fv = monthly * (((1+r)**months - 1) / r) if r else monthly * months
        real = fv / ((1 + infl/100) ** years)
        st.metric("Projected corpus", fmt_inr(fv), f"Inflation-adjusted: {fmt_inr(real)}")
        data = pd.DataFrame({
            "Month": range(1, months+1),
            "Balance": [monthly*(((1+r)**m - 1)/r) if r else monthly*m for m in range(1, months+1)]
        })
        st.line_chart(data.set_index("Month")["Balance"], use_container_width=True)
        st.session_state.report_snap["sip_summary"] = {"monthly": monthly, "years": years, "fv": fv, "real": real}
    else:
        principal = st.number_input("Lump sum (₹)", 10000, 10_00_00_000, 1_00_000, step=10_000)
        years = st.slider("Years", 1, 40, 10, key="ls_years")
        exp_ret = st.slider("Expected annual return (%)", 3.0, 20.0, 9.0, key="ls_ret")
        infl = st.slider("Inflation (%)", 0.0, 12.0, 4.5, key="ls_inf")
        fv = principal * ((1 + exp_ret/100) ** years)
        real = fv / ((1 + infl/100) ** years)
        st.metric("Projected corpus", fmt_inr(fv), f"Inflation-adjusted: {fmt_inr(real)}")
        st.session_state.report_snap["lump_summary"] = {"principal": principal, "years": years, "fv": fv, "real": real}

# -------------------------#
# Monte Carlo              #
# -------------------------#
elif nav == "Monte Carlo":
    st.markdown('<div class="titlebig">Monte Carlo Simulator</div>', unsafe_allow_html=True)
    invest = st.number_input("Current invested (₹)", 0, 10_00_00_000, 5_00_000, step=10_000)
    monthly = st.number_input("Monthly SIP (₹)", 0, 2_00_000, 10_000, step=1000)
    years = st.slider("Horizon (years)", 1, 40, 15)
    sims = st.slider("Simulations", 200, 10000, 3000, step=200)
    exp_ret = st.slider("Expected portfolio return (%)", 3.0, 20.0, 10.0)
    vol = st.slider("Expected volatility (%)", 5.0, 40.0, 16.0)
    rf = st.slider("Risk-free (%)", 0.0, 8.0, 4.0)

    ann_r = exp_ret/100
    ann_s = vol/100
    annual_sip = monthly * 12

    @st.cache_data(ttl=600)
    def run_mc(invest, annual_sip, years, ann_r, ann_s, sims):
        out = np.zeros(sims)
        for s in range(sims):
            val = invest
            for y in range(years):
                ret = np.random.normal(ann_r, ann_s)
                val = val * (1 + ret) + annual_sip
            out[s] = val
        return out

    with st.spinner("Running simulations…"):
        mc = run_mc(invest, annual_sip, years, ann_r, ann_s, sims)

    median = np.median(mc)
    p10, p90 = np.percentile(mc, [10, 90])
    sharpe_like = (ann_r - rf/100) / (ann_s + 1e-9)

    c1, c2, c3 = st.columns(3)
    c1.metric("Median outcome", fmt_inr(median))
    c2.metric("P10 / P90", f"{fmt_inr(p10)} – {fmt_inr(p90)}")
    c3.metric("Sharpe-like", f"{sharpe_like:.2f}")

    st.plotly_chart(px.histogram(mc, nbins=60, title="Final corpus distribution"), use_container_width=True)

    st.session_state.report_snap["mc"] = {"median": median, "p10": p10, "p90": p90, "sims": sims}

# -------------------------#
# PDF Report (Unicode-safe)#
# -------------------------#
elif nav == "PDF Report":
    st.markdown('<div class="titlebig">Export Investor Report (PDF)</div>', unsafe_allow_html=True)
    snap = st.session_state.report_snap

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(0, 0, 0)
            self.cell(0, 10, to_latin1(f"Your Investment Guide — Report for {st.session_state.get('user_name','Investor')}"), ln=1)
            self.set_font("Helvetica", "", 10)
            self.cell(0, 7, to_latin1(datetime.now().strftime("Generated on %d %b %Y, %I:%M %p")), ln=1)
            self.ln(2)
            self.set_draw_color(200,200,200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

        def section_title(self, txt):
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(0, 0, 0)
            self.cell(0, 8, to_latin1(txt), ln=1)

        def normal(self, txt):
            self.set_font("Helvetica", "", 11)
            self.set_text_color(30, 30, 30)
            self.multi_cell(0, 6, to_latin1(txt))

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Overview
    o = snap.get("overview", {})
    pdf.section_title("Overview")
    pdf.normal(f"Mode: {o.get('mode','N/A')} ({o.get('reason','')}). "
               f"NIFTY: {o.get('nifty','N/A')} ({o.get('pct',0):.2f}%). "
               f"Last trading day: {o.get('date','N/A')}.")

    # Portfolio
    if "portfolio_table" in snap:
        pdf.section_title("Portfolio (Top Rows)")
        dfp = snap["portfolio_table"].copy().round(2)
        head = dfp.head(12)
        for _, r in head.iterrows():
            row = f"{r.get('Symbol')}  Qty:{r.get('Quantity')}  LTP:{r.get('LTP'):.2f}  Value:{r.get('Value'):.0f}"
            pdf.normal(row)
    else:
        pdf.section_title("Portfolio")
        pdf.normal("No portfolio uploaded.")

    # Sector Alloc
    if "sector_alloc" in snap:
        pdf.section_title("Sector Allocation (summary)")
        for _, r in snap["sector_alloc"].iterrows():
            pdf.normal(f"{r['Sector']}: {r['Weight %']}%")
    else:
        pdf.section_title("Sector Allocation")
        pdf.normal("Not available.")

    # AI Allocation
    if "ai_alloc" in snap:
        pdf.section_title("AI Allocation (rule-based)")
        for _, r in snap["ai_alloc"].iterrows():
            pdf.normal(f"{r['Asset Class']}: {r['Allocation (%)']}%")

    # Manual Alloc
    if "manual_alloc" in snap:
        pdf.section_title("Manual Allocation")
        for _, r in snap["manual_alloc"].iterrows():
            pdf.normal(f"{r['Asset Class']}: {r['Allocation (%)']}%")

    # SIP Summary
    if "sip_summary" in snap:
        s = snap["sip_summary"]
        pdf.section_title("SIP Summary")
        pdf.normal(f"Monthly: {fmt_inr(s['monthly'])}, Years: {s['years']}, "
                   f"Projected: {fmt_inr(s['fv'])}, Real (inflation-adj): {fmt_inr(s['real'])}")

    # Lump Sum
    if "lump_summary" in snap:
        s = snap["lump_summary"]
        pdf.section_title("Lump Sum Summary")
        pdf.normal(f"Principal: {fmt_inr(s['principal'])}, Years: {s['years']}, "
                   f"Projected: {fmt_inr(s['fv'])}, Real (inflation-adj): {fmt_inr(s['real'])}")

    # MC
    if "mc" in snap:
        m = snap["mc"]
        pdf.section_title("Monte Carlo")
        pdf.normal(f"Median: {fmt_inr(m['median'])}; P10: {fmt_inr(m['p10'])}; P90: {fmt_inr(m['p90'])}; Sims: {m['sims']}")

    out = io.BytesIO()
    pdf.output(out)  # latin-1 safe due to to_latin1
    st.download_button("📄 Download PDF Report", data=out.getvalue(), file_name="Your_Investment_Guide_Report.pdf", mime="application/pdf")
    st.success("PDF generated with Unicode-safe text handling.")

# -------------------------#
# Watchlist                #
# -------------------------#
elif nav == "Watchlist":
    st.markdown('<div class="titlebig">Watchlist & Alerts</div>', unsafe_allow_html=True)
    wl = st.text_input("Tickers (comma separated)", value="TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS")
    threshold = st.number_input("Alert threshold (%)", 0.5, 20.0, 2.0)
    if st.button("Fetch watchlist"):
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        rows = []
        for t in tickers:
            d = fetch(t, "5d", "1d")
            if d.empty:
                rows.append({"Symbol": t, "Price": "N/A", "Change%": "N/A", "Alert": ""})
            else:
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2]) if len(d) >= 2 else last
                pct = (last-prev)/prev*100 if prev else 0.0
                alert = "⚠️" if abs(pct) >= threshold else ""
                rows.append({"Symbol": t.replace(".NS",""), "Price": f"₹{last:,.2f}", "Change%": f"{pct:+.2f}%", "Alert": alert})
        st.table(pd.DataFrame(rows))
