# wealthos_v4.py
# AI WealthOS — Professional Portfolio Intelligence (V4)
# Features:
# 1) Live Market with smart fallbacks, mode badges, timestamps, refresh
# 2) Portfolio sector analytics (allocation, avg perf, highlight)
# 3) Alpha/Beta/Correlation vs NIFTY (portfolio vs benchmark)
# 4) Goals & SIP with Monte Carlo (10k paths) + success probability
# 5) Export PDF Investor Report (summary, allocation, sectors, alpha/beta, MC, SIP)
# 6) Modern sidebar: logo, profile, rotating quote, compact nav, live refresh

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, date
from fpdf import FPDF
import io
import random
import math

# -------------------- CONFIG & THEME --------------------
st.set_page_config(
    page_title="AI WealthOS — Portfolio Intelligence",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Pro “terminal” dark style
st.markdown("""
<style>
  :root { --accent:#33C3F0; --accent2:#00FFC6; --muted:#9FB4C8; }
  .stApp { background: linear-gradient(180deg,#0A0E14 0%, #0E141B 100%); color:#E5EDF3; }
  h1,h2,h3,h4,h5 { color:#33C3F0; font-weight:700; letter-spacing:0.2px; }
  .metric-label, .css-10trblm { color:#DCE7F3 !important; }
  div[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#0C1118 0%, #0A0E14 100%);
    border-right:1px solid rgba(255,255,255,0.06);
  }
  .box { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.06);
         border-radius:12px; padding:14px; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
           border:1px solid rgba(255,255,255,0.12); margin-left:6px; }
  .badge-live { color:#0f0; border-color:#0f5; }
  .badge-fb { color:#ffb020; border-color:#cc8a12; }
  .badge-off { color:#ff6b6b; border-color:#cc4b4b; }
  .muted { color:#9FB4C8; }
  .small { font-size:12px; color:#8FA3B8; }
  .quote { font-style:italic; color:#cfeee6; }
  .navbtn button { width:100%; text-align:left; }
</style>
""", unsafe_allow_html=True)

# -------------------- UTILITIES --------------------
def greeting():
    hr = datetime.now().hour
    if hr < 12: return "Good morning"
    if hr < 17: return "Good afternoon"
    return "Good evening"

QUOTES = [
    "Price is what you pay. Value is what you get.",
    "Small edges, repeated, become big outcomes.",
    "Volatility is a feature, not a bug—if you’re prepared.",
    "Diversification is the only free lunch in finance.",
    "Liquidity is abundant—until you need it most."
]

@st.cache_data(ttl=300)
def fetch_prices(symbols, period="5d", interval="15m"):
    try:
        df = yf.download(symbols, period=period, interval=interval, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_history(ticker, period="5y", interval="1d"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        return df
    except Exception:
        return pd.DataFrame()

def safe_last_trading_day(hist_df):
    try:
        return pd.to_datetime(hist_df.index[-1]).date()
    except Exception:
        return None

def compute_portfolio_value(df_holdings):
    # df_holdings: Symbol, Quantity, BuyPrice (optional)
    df = df_holdings.copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    prices = {}
    for s in df["Symbol"].unique():
        h = fetch_history(s, period="1mo", interval="1d")
        prices[s] = float(h["Close"].iloc[-1]) if not h.empty else np.nan
    df["LTP"] = df["Symbol"].map(prices)
    df["Value"] = df["LTP"] * df["Quantity"].astype(float)
    if "BuyPrice" in df.columns:
        df["Cost"] = df["BuyPrice"].astype(float) * df["Quantity"].astype(float)
        df["P&L"] = df["Value"] - df["Cost"]
        df["P&L %"] = np.where(df["Cost"]!=0, df["P&L"]/df["Cost"]*100, np.nan)
    return df

def alpha_beta(port_ret, bench_ret):
    # daily returns aligned (Series)
    if len(port_ret) < 5:
        return np.nan, np.nan, np.nan
    port = port_ret.dropna()
    bench = bench_ret.dropna()
    idx = port.index.intersection(bench.index)
    if len(idx) < 5: return np.nan, np.nan, np.nan
    x = bench.loc[idx].values
    y = port.loc[idx].values
    # regression y = a + b x   (beta = cov/var)
    var_x = np.var(x)
    if var_x == 0: return np.nan, np.nan, np.nan
    beta = np.cov(x,y, ddof=1)[0,1] / var_x
    alpha = np.mean(y) - beta*np.mean(x)
    corr = np.corrcoef(x,y)[0,1]
    # annualize alpha assuming 252 trading days
    alpha_annual = (1 + alpha)**252 - 1 if alpha > -1 else np.nan
    return alpha_annual, beta, corr

def mc_simulation(init, monthly, years, mean_ann, vol_ann, sims=10000, seed=42):
    np.random.seed(seed)
    months = int(years*12)
    mean_m = (1+mean_ann)**(1/12)-1
    vol_m = vol_ann/np.sqrt(12)
    results = np.zeros(sims)
    for s in range(sims):
        bal = init
        for m in range(months):
            r = np.random.normal(mean_m, vol_m)
            bal = bal*(1+r) + monthly
        results[s] = bal
    return results

def make_report_pdf(name, alloc_df, sector_df, alpha_val, beta_val, corr_val,
                    mc_stats, sip_cfg):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"AI WealthOS — Investor Report", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Prepared for: {name}", ln=1)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", ln=1)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Allocation Summary", ln=1)
    pdf.set_font("Arial", "", 11)
    if alloc_df is not None and not alloc_df.empty:
        for _, r in alloc_df.iterrows():
            pdf.cell(0, 6, f"- {r['Symbol']}: Qty {r.get('Quantity','-')}, "
                           f"Value ₹{r.get('Value',0):,.0f}", ln=1)
    else:
        pdf.cell(0, 6, "No holdings uploaded.", ln=1)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Sector Breakdown (by Value)", ln=1)
    pdf.set_font("Arial", "", 11)
    if sector_df is not None and not sector_df.empty:
        for _, r in sector_df.iterrows():
            pdf.cell(0, 6, f"- {r['Sector']}: {r['Weight %']:.1f}%", ln=1)
    else:
        pdf.cell(0, 6, "Not available.", ln=1)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Performance vs NIFTY", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Alpha (annualized): {np.nan_to_num(alpha_val)*100:.2f}%", ln=1)
    pdf.cell(0, 6, f"Beta: {np.nan_to_num(beta_val):.2f}", ln=1)
    pdf.cell(0, 6, f"Correlation: {np.nan_to_num(corr_val):.2f}", ln=1)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "Monte Carlo (Final Corpus)", ln=1)
    pdf.set_font("Arial", "", 11)
    p10, p50, p90, prob = mc_stats
    pdf.cell(0, 6, f"P10: ₹{p10:,.0f}   Median: ₹{p50:,.0f}   P90: ₹{p90:,.0f}", ln=1)
    pdf.cell(0, 6, f"Probability of meeting target: {prob:.1f}%", ln=1)

    pdf.ln(2)
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "SIP Configuration", ln=1)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, f"Monthly SIP: ₹{sip_cfg.get('sip',0):,}", ln=1)
    pdf.cell(0, 6, f"Horizon: {sip_cfg.get('years',0)} years   "
                   f"Expected return: {sip_cfg.get('exp',0)}%   "
                   f"Target: ₹{sip_cfg.get('target',0):,}", ln=1)

    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return pdf_bytes

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6b/NSE_Logo.svg", width=120)
    st.markdown(f"### {greeting()}, Investor")
    if "username" not in st.session_state:
        st.session_state.username = "Guest"
    st.session_state.username = st.text_input("Name", value=st.session_state.username).strip() or "Guest"

    if "quote" not in st.session_state:
        st.session_state.quote = random.choice(QUOTES)
    with st.expander("Daily insight", expanded=False):
        if st.button("🔁 New quote"):
            st.session_state.quote = random.choice(QUOTES)
        st.markdown(f"<div class='quote'>“{st.session_state.quote}”</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Navigation")

    # Compact nav with icons
    menu = st.radio(
        "Go to",
        ["🏠 Overview", "💹 Live Market", "📂 Portfolio", "📊 Sector Analytics",
         "📈 Alpha & Beta", "🎯 Goals & Monte Carlo", "📄 Export Report"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    refresh_live = st.checkbox("Enable manual refresh on Live Market", value=True)
    if refresh_live and menu == "💹 Live Market":
        do_refresh = st.button("🔄 Refresh live data")

# -------------------- 1) OVERVIEW --------------------
if menu == "🏠 Overview":
    st.title("AI WealthOS — Portfolio Intelligence")
    st.markdown(f"**Welcome, {st.session_state.username}.** A professional cockpit for market insight, portfolio diagnostics, and goal planning.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Platform Latency", "Low", "Optimized")
    c2.metric("Theme", "Pro Dark")
    c3.metric("Status", "Ready")

    st.markdown("#### What you can do here")
    st.markdown("""
    - Track **live NIFTY** with smart fallbacks and mode badges.
    - Upload your **portfolio** and get **sector analytics**.
    - Measure **Alpha / Beta / Correlation** vs NIFTY.
    - Plan with **SIP + Monte Carlo** and export a **PDF investor report**.
    """)

# -------------------- 2) LIVE MARKET --------------------
elif menu == "💹 Live Market":
    st.title("Live Market — NIFTY 50")
    mode = "Offline"
    df = pd.DataFrame()
    ts = datetime.now().strftime("%d %b %Y, %I:%M %p")

    # Intraday attempt
    intraday = fetch_prices("^NSEI", period="5d", interval="15m")
    if not intraday.empty and "Close" in intraday.columns:
        df = intraday["Close"].dropna()
        mode = "Live"
    else:
        # Daily fallback
        daily = fetch_prices("^NSEI", period="1mo", interval="1d")
        if not daily.empty and "Close" in daily.columns:
            df = daily["Close"].dropna()
            mode = "Fallback"
        else:
            # Sample
            idx = pd.date_range(end=pd.Timestamp.now(), periods=20, freq="D")
            df = pd.Series(np.cumsum(np.random.randn(20)) + 20000, index=idx, name="Close")
            mode = "Offline"

    last_trading = df.index[-1].date() if len(df) else None

    badge = {"Live":"badge-live", "Fallback":"badge-fb", "Offline":"badge-off"}[mode]
    st.markdown(f"**Mode:** <span class='badge {badge}'>{mode}</span> &nbsp; "
                f"<span class='small'>Last refreshed: {ts}</span>", unsafe_allow_html=True)
    if last_trading:
        st.markdown(f"<span class='small'>Last trading day: {last_trading.strftime('%d %b %Y')}</span>", unsafe_allow_html=True)

    if refresh_live and 'do_refresh' in locals() and do_refresh:
        st.cache_data.clear()
        st.rerun()

    if len(df) >= 2:
        latest = float(df.iloc[-1])
        prev = float(df.iloc[-2])
        delta = latest - prev
        pct = delta/prev*100 if prev != 0 else 0.0
        c1,c2,c3 = st.columns(3)
        c1.metric("NIFTY (LTP)", f"{latest:,.2f}")
        c2.metric("Δ (pts)", f"{delta:+.2f}")
        c3.metric("Δ (%)", f"{pct:+.2f}%")
        st.line_chart(df, use_container_width=True)
    else:
        st.info("Data insufficient to chart.")

# -------------------- 3) PORTFOLIO --------------------
elif menu == "📂 Portfolio":
    st.title("Portfolio — Upload & Summary")
    st.caption("CSV format: Symbol,Quantity,BuyPrice (BuyPrice optional). Example: TCS.NS,5,3500")
    file = st.file_uploader("Upload holdings CSV", type=["csv"])
    if file:
        raw = pd.read_csv(file, header=None)
        if raw.shape[1] == 2:
            raw.columns = ["Symbol","Quantity"]
        else:
            raw = raw.iloc[:, :3]
            raw.columns = ["Symbol","Quantity","BuyPrice"]
        try:
            df_hold = compute_portfolio_value(raw)
            st.subheader("Holdings")
            st.dataframe(df_hold.round(2), use_container_width=True)
            total_value = df_hold["Value"].sum()
            total_cost = df_hold["Cost"].sum() if "Cost" in df_hold.columns else np.nan
            pnl = total_value - total_cost if not np.isnan(total_cost) else np.nan
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Market Value", f"₹{total_value:,.0f}")
            c2.metric("Total P&L", "N/A" if np.isnan(pnl) else f"₹{pnl:,.0f}")
            c3.metric("Tickers", df_hold["Symbol"].nunique())
            st.session_state.portfolio_df = df_hold
        except Exception as e:
            st.error(f"Parsing error: {e}")
    else:
        st.info("Upload CSV to compute values.")

# -------------------- 4) SECTOR ANALYTICS --------------------
elif menu == "📊 Sector Analytics":
    st.title("Sector Analytics — Allocation & Pulse")
    df_hold = st.session_state.get("portfolio_df")
    if df_hold is None or df_hold.empty:
        st.warning("Upload your portfolio in the **Portfolio** tab first.")
    else:
        # Minimal ticker→sector mapping (extend as needed)
        SECTOR_MAP = {
            "RELIANCE.NS":"Energy","ONGC.NS":"Energy","BPCL.NS":"Energy",
            "TCS.NS":"IT","INFY.NS":"IT","WIPRO.NS":"IT","HCLTECH.NS":"IT","TECHM.NS":"IT",
            "HDFCBANK.NS":"Banking","ICICIBANK.NS":"Banking","KOTAKBANK.NS":"Banking","SBIN.NS":"Banking","AXISBANK.NS":"Banking",
            "HINDUNILVR.NS":"FMCG","NESTLEIND.NS":"FMCG","BRITANNIA.NS":"FMCG",
            "SUNPHARMA.NS":"Pharma","DRREDDY.NS":"Pharma","CIPLA.NS":"Pharma",
        }
        df = df_hold.copy()
        df["Sector"] = df["Symbol"].map(SECTOR_MAP).fillna("Other")

        alloc = df.groupby("Sector")["Value"].sum().reset_index()
        total = alloc["Value"].sum()
        alloc["Weight %"] = alloc["Value"]/total*100 if total>0 else 0

        c1,c2 = st.columns([1.1,1])
        with c1:
            st.subheader("Allocation by sector")
            fig = px.pie(alloc, names="Sector", values="Value", hole=0.35,
                         color_discrete_sequence=px.colors.sequential.Tealgrn)
            st.plotly_chart(fig, use_container_width=True)

        # Sector performance (avg of last daily % change per sector)
        perfs = []
        for sec in alloc["Sector"]:
            tickers = df[df["Sector"]==sec]["Symbol"].unique().tolist()
            pct_list = []
            for t in tickers:
                h = fetch_history(t, period="5d", interval="1d")
                try:
                    last = float(h["Close"].iloc[-1]); prev = float(h["Close"].iloc[-2])
                    pct_list.append((last-prev)/prev*100 if prev!=0 else 0.0)
                except Exception:
                    pass
            perfs.append(np.nanmean(pct_list) if pct_list else np.nan)
        perf_df = pd.DataFrame({"Sector":alloc["Sector"], "Avg % Change":perfs}).dropna()

        with c2:
            st.subheader("Sector daily pulse")
            if not perf_df.empty:
                top_row = perf_df.sort_values("Avg % Change", ascending=False).iloc[0]
                st.metric("Top sector today", f"{top_row['Sector']}", f"{top_row['Avg % Change']:+.2f}%")
                bar = px.bar(perf_df, x="Sector", y="Avg % Change", text="Avg % Change")
                st.plotly_chart(bar, use_container_width=True)
            else:
                st.info("Insufficient recent data for sector pulse.")

        st.subheader("Sector table")
        st.dataframe(alloc[["Sector","Weight %"]].sort_values("Weight %", ascending=False),
                     use_container_width=True)
        st.session_state.sector_alloc = alloc[["Sector","Weight %"]].sort_values("Weight %", ascending=False)

# -------------------- 5) ALPHA & BETA --------------------
elif menu == "📈 Alpha & Beta":
    st.title("Performance vs NIFTY — Alpha, Beta & Correlation")
    df_hold = st.session_state.get("portfolio_df")
    if df_hold is None or df_hold.empty:
        st.warning("Upload your portfolio first in **Portfolio**.")
    else:
        # Build daily portfolio value series from last 1y
        tickers = df_hold["Symbol"].unique().tolist()
        weights_val = (df_hold.groupby("Symbol")["Value"].sum() / df_hold["Value"].sum()).to_dict()

        # fetch histories (1y daily)
        series = {}
        for t in tickers:
            h = fetch_history(t, period="1y", interval="1d")
            if not h.empty:
                series[t] = h["Close"].rename(t)

        if not series:
            st.info("No price history available to compute analytics.")
        else:
            pxdf = pd.concat(series.values(), axis=1).dropna(how="all")
            pxdf = pxdf.fillna(method="ffill")
            # portfolio value (weighted)
            w = pd.Series(weights_val)
            w = w.reindex(pxdf.columns).fillna(0)
            port_px = (pxdf * w).sum(axis=1)
            port_ret = port_px.pct_change().dropna()

            # benchmark (NIFTY)
            nifty = fetch_history("^NSEI", period="1y", interval="1d")
            bench_ret = nifty["Close"].pct_change().dropna() if not nifty.empty else pd.Series(dtype=float)

            a, b, corr = alpha_beta(port_ret, bench_ret)

            c1,c2,c3 = st.columns(3)
            c1.metric("Alpha (annualized)", f"{np.nan_to_num(a)*100:.2f}%")
            c2.metric("Beta", f"{np.nan_to_num(b):.2f}")
            c3.metric("Correlation", f"{np.nan_to_num(corr):.2f}")

            # Chart: indexed performance
            try:
                df_chart = pd.DataFrame({
                    "Portfolio": (1+port_ret).cumprod()*100,
                    "NIFTY": (1+bench_ret.reindex(port_ret.index, method="ffill")).cumprod()*100
                }).dropna()
                line = px.line(df_chart, labels={"value":"Indexed (100=Start)","index":"Date"})
                st.plotly_chart(line, use_container_width=True)
            except Exception:
                st.info("Could not render comparison chart.")

            st.session_state.alpha_val = a
            st.session_state.beta_val = b
            st.session_state.corr_val = corr

# -------------------- 6) GOALS & MONTE CARLO --------------------
elif menu == "🎯 Goals & Monte Carlo":
    st.title("Goals, SIP & Monte Carlo")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        lump = st.number_input("Current Lump Sum (₹)", 0, 10_00_00_000, 2_00_000, step=10_000)
    with c2:
        sip = st.number_input("Monthly SIP (₹)", 0, 10_00_000, 10_000, step=1_000)
    with c3:
        years = st.slider("Horizon (years)", 1, 40, 15)
    with c4:
        exp_ret = st.slider("Expected Return (p.a. %)", 3.0, 20.0, 10.0)

    # Approx volatility assumption by risk class (simple)
    vol = st.select_slider("Volatility band (p.a.)", options=[0.08,0.12,0.16,0.20], value=0.16)

    target = st.number_input("Target Corpus (₹)", 0, 100_00_00_000, 1_00_00_000, step=50_000)

    with st.spinner("Running Monte Carlo (10k paths)…"):
        sims = mc_simulation(lump, sip, years, exp_ret/100, vol, sims=10_000)
    p10 = float(np.percentile(sims, 10))
    p50 = float(np.percentile(sims, 50))
    p90 = float(np.percentile(sims, 90))
    prob = float((sims >= target).mean()*100)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("P10", f"₹{p10:,.0f}")
    c2.metric("Median", f"₹{p50:,.0f}")
    c3.metric("P90", f"₹{p90:,.0f}")
    c4.metric("Success Prob.", f"{prob:.1f}%")

    hist = px.histogram(sims, nbins=60, title="Final Corpus Distribution")
    st.plotly_chart(hist, use_container_width=True)

    st.session_state.mc_stats = (p10, p50, p90, prob)
    st.session_state.sip_cfg = {"sip": sip, "years": years, "exp": exp_ret, "target": target}

# -------------------- 7) EXPORT REPORT --------------------
elif menu == "📄 Export Report":
    st.title("Export — PDF Investor Report")
    alloc_df = st.session_state.get("portfolio_df")
    sector_df = st.session_state.get("sector_alloc")
    alpha_val = st.session_state.get("alpha_val", np.nan)
    beta_val = st.session_state.get("beta_val", np.nan)
    corr_val = st.session_state.get("corr_val", np.nan)
    mc_stats = st.session_state.get("mc_stats", (np.nan, np.nan, np.nan, np.nan))
    sip_cfg = st.session_state.get("sip_cfg", {"sip":0,"years":0,"exp":0,"target":0})

    pdf_bytes = make_report_pdf(
        name=st.session_state.username,
        alloc_df=alloc_df if isinstance(alloc_df, pd.DataFrame) else None,
        sector_df=sector_df if isinstance(sector_df, pd.DataFrame) else None,
        alpha_val=alpha_val, beta_val=beta_val, corr_val=corr_val,
        mc_stats=mc_stats, sip_cfg=sip_cfg
    )

    st.download_button(
        "📄 Download Investor Report (PDF)",
        data=pdf_bytes,
        file_name="AI_WealthOS_Investor_Report.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown("---")
st.caption("AI WealthOS — Professional tools for insights. This is not investment advice.")
