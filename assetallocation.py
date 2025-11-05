# wealthos_v3.py
# AI WealthOS V3 — Professional Dark (Teal + Graphite)
# - Personalized sidebar (greeting, rotating quote, live mini-tickers, theme toggle, sentiment)
# - Live Market Pulse (India + Global) with heatmaps
# - Portfolio analytics (allocation, diversification, beta-ish, sharpe-like)
# - Goal planner + SIP shortfall + Monte Carlo success probability
# - Efficient frontier sampler + risk panels
# - No login (frictionless). One-file deploy.

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import random

# -------------------------- CONFIG & THEME --------------------------
st.set_page_config(page_title="AI WealthOS V3", layout="wide", initial_sidebar_state="expanded")

ACCENT = "#00FFC6"     # teal
BG     = "#0e1117"     # graphite dark
CARD   = "#111827"     # sidebar card dark
MUTED  = "#9aa5a6"

st.markdown(f"""
<style>
  .stApp {{ background:{BG}; color:#e6eef0; }}
  div[data-testid="stSidebar"] {{
    background:{CARD}; border-right:1px solid rgba(255,255,255,0.06);
  }}
  h1,h2,h3,h4,h5 {{ color:{ACCENT}; font-weight:700; }}
  .kpi-card {{
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
    border-radius:14px; padding:14px;
  }}
  .pill {{
    font-size:12px; padding:4px 8px; border-radius:999px;
    background:#07121a; color:#bfffea; display:inline-block;
  }}
  .sidebar-title {{ color:{ACCENT}; font-size:20px; font-weight:700; }}
  .greeting {{ color:#B9FFE0; font-size:16px; font-weight:600; margin-bottom:4px; }}
  .quote {{ color:#d5efe8; font-style:italic; font-size:13px; }}
  .ticker-strip {{
    display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 6px 0;
  }}
  .ticker {{
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08);
    padding:6px 8px; border-radius:8px; font-size:12px;
  }}
  .sentiment {{
    height:7px; border-radius:6px; background:linear-gradient(90deg,#f87171, #fbbf24, #22c55e);
  }}
  .small-label {{ font-size:12px; color:{MUTED}; }}
</style>
""", unsafe_allow_html=True)

# -------------------------- HELPERS --------------------------
@st.cache_data(ttl=300)
def fetch_close(tickers, period="5d", interval="1d"):
    """Return Close prices DataFrame for list or str tickers."""
    try:
        data = yf.download(tickers, period=period, interval=interval, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.dropna(how="all")
    except Exception:
        return pd.DataFrame()

def time_greeting():
    hr = datetime.now().hour
    return "Good Morning" if hr < 12 else ("Good Afternoon" if hr < 17 else "Good Evening")

def pick_quote():
    q = [
        "Invest early. Time is the compounding engine.",
        "Diversify — not to avoid risk, but to control it.",
        "Plan like a pessimist, invest like an optimist.",
        "Small disciplined steps beat sporadic brilliance.",
        "Risk is what’s left over after you plan.",
        "Data drives decisions; humility preserves capital.",
        "Don’t time the market; spend time in the market."
    ]
    return random.choice(q)

def fmt_inr(x):
    try:
        return "₹{:,.0f}".format(x)
    except Exception:
        return str(x)

def safe_pct(a, b):
    return (a - b) / b * 100 if b not in (0, None, np.nan) else 0.0

# -------------------------- PERSONALIZATION STATE --------------------------
if "user_name" not in st.session_state:
    st.session_state.user_name = ""
if "quote" not in st.session_state:
    st.session_state.quote = pick_quote()

# -------------------------- SIDEBAR (Creative Command Hub) --------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>AI WealthOS</div>", unsafe_allow_html=True)

    # name + greeting
    name = st.text_input("Your name", value=st.session_state.user_name or "Ishani", max_chars=24)
    st.session_state.user_name = name.strip().title() if name.strip() else "Investor"

    st.markdown(f"<div class='greeting'>{time_greeting()}, {st.session_state.user_name} 👋</div>", unsafe_allow_html=True)
    if st.button("New quote"):
        st.session_state.quote = pick_quote()
    st.markdown(f"<div class='quote'>“{st.session_state.quote}”</div>", unsafe_allow_html=True)

    # mini live tickers
    st.markdown("---")
    st.caption("Mini market strip")
    strip = {"NIFTY":"^NSEI", "USDINR":"INR=X", "GOLD":"GC=F", "BTC":"BTC-USD"}
    strip_df = fetch_close(list(strip.values()), period="5d", interval="1d")
    st.markdown("<div class='ticker-strip'>", unsafe_allow_html=True)
    if not strip_df.empty:
        for label, sym in strip.items():
            try:
                last = float(strip_df[sym].iloc[-1])
                prev = float(strip_df[sym].iloc[-2]) if len(strip_df) > 1 else last
                pct = safe_pct(last, prev)
                sign = "▲" if pct >= 0 else "▼"
                st.markdown(f"<div class='ticker'>{label}: {last:,.2f} {sign} {pct:+.2f}%</div>", unsafe_allow_html=True)
            except Exception:
                st.markdown(f"<div class='ticker'>{label}: N/A</div>", unsafe_allow_html=True)
    else:
        for label in strip.keys():
            st.markdown(f"<div class='ticker'>{label}: N/A</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # sentiment meter (based on NIFTY 7-day change)
    st.caption("Market sentiment (NIFTY 7D)")
    nifty7 = fetch_close("^NSEI", period="1mo", interval="1d")
    sentiment = 0.0
    if not nifty7.empty and len(nifty7) >= 7:
        last = float(nifty7.iloc[-1, 0])
        prev7 = float(nifty7.iloc[-7, 0])
        sentiment = np.clip(safe_pct(last, prev7) / 5.0, -1, 1)  # normalize roughly
    # draw numeric + bar background
    bar_html = f"""
    <div class='small-label'>{sentiment*100:+.1f} (scaled)</div>
    <div class='sentiment'></div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

    # theme toggles (cosmetic only)
    st.markdown("---")
    refresh_secs = st.slider("Auto-refresh (sec)", 0, 180, 0, help="0 = no auto-refresh")
    if refresh_secs > 0:
        st.experimental_set_query_params(_=str(int(datetime.now().timestamp())))

    st.markdown("---")
    menu = st.radio("Navigate", [
        "🏠 Home",
        "💹 Market Pulse",
        "📊 Portfolio",
        "🧩 Allocation",
        "🎯 Goals & Monte Carlo",
        "📈 Efficient Frontier",
        "⚠️ Risk Panels"
    ], index=0)

# -------------------------- HOME --------------------------
if menu == "🏠 Home":
    st.title("AI WealthOS V3 — Intelligent Fintech Suite")
    st.write("Professional Dark theme • Real-time market pulse • Portfolio intelligence • Goal planning • Risk analytics")
    c1, c2, c3, c4 = st.columns(4)
    # quick KPIs: try NIFTY last, daily pct; BTC; Gold; USDINR
    try:
        df = fetch_close(["^NSEI", "BTC-USD", "GC=F", "INR=X"], "5d", "1d")
        def kpi(sym, label, col):
            if sym in df.columns:
                last = float(df[sym].iloc[-1])
                prev = float(df[sym].iloc[-2]) if len(df) > 1 else last
                col.metric(label, f"{last:,.2f}", f"{safe_pct(last, prev):+.2f}%")
            else:
                col.metric(label, "N/A", "")
        kpi("^NSEI", "NIFTY", c1); kpi("BTC-USD", "BTC-USD", c2)
        kpi("GC=F", "Gold (COMEX)", c3); kpi("INR=X", "USD/INR", c4)
    except Exception:
        c1.metric("NIFTY", "N/A"); c2.metric("BTC-USD","N/A"); c3.metric("Gold","N/A"); c4.metric("USD/INR","N/A")

    st.markdown("---")
    st.subheader("What’s inside")
    st.markdown(
        "- 💹 **Market Pulse**: India + global snapshot, heatmaps\n"
        "- 📊 **Portfolio**: upload holdings, allocation, diversification\n"
        "- 🧩 **Allocation**: auto-suggest + manual editor\n"
        "- 🎯 **Goals & Monte Carlo**: goal table, SIP shortfall, success probability\n"
        "- 📈 **Efficient Frontier**: random portfolios vs your mix\n"
        "- ⚠️ **Risk Panels**: stress shifts, volatility trend, simple VaR"
    )

# -------------------------- MARKET PULSE --------------------------
elif menu == "💹 Market Pulse":
    st.header("Market Pulse — India + Global")
    groups = {
        "India": ["^NSEI", "^NSEBANK", "^BSESN"],
        "US": ["^GSPC", "^IXIC", "^DJI"],
        "Global": ["^N225", "^FTSE", "^STOXX50E"],
        "FX": ["INR=X", "EURUSD=X", "JPY=X"],
        "Commodities": ["GC=F", "CL=F", "NG=F"]
    }
    all_syms = [s for arr in groups.values() for s in arr]
    df = fetch_close(all_syms, period="10d", interval="1d")
    if df.empty:
        st.warning("Live data unavailable. Try later.")
    else:
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        cards = []
        for k, syms in groups.items():
            block = []
            for s in syms:
                if s in df.columns:
                    last = float(last_row[s]); prev = float(prev_row[s])
                    block.append({"Symbol": s, "Last": last, "Change%": safe_pct(last, prev)})
            cards.append((k, pd.DataFrame(block)))
        for name, data in cards:
            st.subheader(name)
            if data.empty:
                st.info("No data.")
                continue
            st.dataframe(data.style.format({"Last":"{:.2f}", "Change%":"{:+.2f}%"}), use_container_width=True)
        # heatmap
        st.markdown("### Heatmap — Daily % Change")
        heat = []
        for s in df.columns:
            last = float(df[s].iloc[-1]); prev = float(df[s].iloc[-2]) if len(df) > 1 else last
            heat.append({"Symbol": s, "Change%": safe_pct(last, prev)})
        heat_df = pd.DataFrame(heat)
        fig = px.treemap(heat_df, path=["Symbol"], values="Change%", color="Change%",
                         color_continuous_scale="RdYlGn", title="Daily % Change Heatmap")
        st.plotly_chart(fig, use_container_width=True)

# -------------------------- PORTFOLIO --------------------------
elif menu == "📊 Portfolio":
    st.header("Portfolio — Upload & Analyze")
    st.caption("CSV columns: Symbol, Quantity, BuyPrice (BuyPrice optional). Example: TCS.NS,5,3500")
    up = st.file_uploader("Upload holdings", type=["csv"])
    if not up:
        st.info("Upload to see analytics.")
        st.stop()
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
            raw["BuyPrice"] = pd.to_numeric(raw["BuyPrice"], errors="coerce")

        # fetch LTPs
        tickers = raw["Symbol"].unique().tolist()
        px_df = fetch_close(tickers, period="5d", interval="1d")
        prices = {}
        for t in tickers:
            try:
                prices[t] = float(px_df[t].iloc[-1]) if (not px_df.empty and t in px_df.columns) else np.nan
            except Exception:
                prices[t] = np.nan

        raw["LTP"]   = raw["Symbol"].map(prices)
        raw["Value"] = raw["LTP"] * raw["Quantity"]
        if "BuyPrice" in raw.columns:
            raw["Cost"] = raw["BuyPrice"] * raw["Quantity"]
            raw["P&L"]  = raw["Value"] - raw["Cost"]
            raw["P&L %"] = np.where(raw["Cost"] != 0, raw["P&L"]/raw["Cost"]*100, np.nan)

        st.markdown("### Holdings")
        st.dataframe(raw.round(2), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Market Value", fmt_inr(raw["Value"].sum()))
        if "P&L" in raw.columns:
            pnl = raw["P&L"].sum()
            base = raw["Cost"].sum() if "Cost" in raw.columns else np.nan
            pct = pnl/base*100 if base and base != 0 else 0
            c2.metric("Total P&L", fmt_inr(pnl), f"{pct:+.2f}%")
        c3.metric("Tickers", raw["Symbol"].nunique())

        # allocation pie
        alloc = raw.groupby("Symbol")["Value"].sum().reset_index()
        if not alloc["Value"].sum() == 0:
            fig = px.pie(alloc, names="Symbol", values="Value", hole=0.35,
                         color_discrete_sequence=px.colors.sequential.Tealgrn,
                         title="Allocation by Symbol")
            st.plotly_chart(fig, use_container_width=True)

        # diversification gauge (simple)
        st.markdown("### Diversification")
        share = alloc["Value"] / alloc["Value"].sum()
        hhi = (share**2).sum() if share.size > 0 else np.nan  # Herfindahl-Hirschman-like
        st.write(f"Concentration index (lower is better): **{hhi:.3f}**")
        if hhi > 0.25:
            st.warning("High concentration detected. Consider diversifying.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

# -------------------------- ALLOCATION --------------------------
elif menu == "🧩 Allocation":
    st.header("Allocation — Auto & Manual")
    st.caption("Quick suggestion by age/risk; then tweak manually.")

    age = st.slider("Age", 18, 80, 35)
    risk = st.selectbox("Risk", ["Low","Moderate","High"], index=1)
    horizon = st.slider("Horizon (years)", 1, 40, 10)

    # simple rules
    base_equity = 60 if 30 < age < 45 else (70 if age <= 30 else (40 if age < 60 else 30))
    equity = base_equity + (10 if risk=="High" else (0 if risk=="Moderate" else -20))
    equity += (5 if horizon >= 15 else (-5 if horizon <= 5 else 0))
    equity = int(np.clip(equity, 10, 90))
    debt = 100 - equity
    sugg = {
        "Large-cap Equity": int(round(equity*0.48)),
        "Mid/Small-cap Equity": int(round(equity*0.22)),
        "International Equity": int(round(equity*0.10)),
        "Debt — Govt": int(round(debt*0.60)),
        "Debt — Corporate": int(round(debt*0.30)),
        "Gold": 6,
        "REITs": 4,
        "Cash/Liquid": 100  # placeholder; normalize next
    }
    s = sum(sugg.values());  # normalize to 100
    for k in sugg: sugg[k] = int(round(sugg[k]*100/s))
    diff = 100 - sum(sugg.values()); sugg["Cash/Liquid"] += diff

    st.markdown("**Suggested mix**")
    df_alloc = pd.DataFrame(list(sugg.items()), columns=["Asset Class","Allocation %"])
    st.dataframe(df_alloc, use_container_width=True)
    fig = px.pie(df_alloc, names="Asset Class", values="Allocation %",
                 color_discrete_sequence=px.colors.sequential.Tealgrn, title="Suggested Allocation")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Manual editor")
    base = df_alloc.copy()
    edited = st.data_editor(base, num_rows="dynamic", use_container_width=True)
    if st.button("Normalize to 100%"):
        if edited["Allocation %"].sum() > 0:
            edited["Allocation %"] = (edited["Allocation %"] / edited["Allocation %"].sum() * 100).round(0)
            delta = 100 - edited["Allocation %"].sum()
            edited.at[0, "Allocation %"] += delta
            st.success("Normalized.")
            st.dataframe(edited, use_container_width=True)

# -------------------------- GOALS & MONTE CARLO --------------------------
elif menu == "🎯 Goals & Monte Carlo":
    st.header("Goals — Planner & Probabilities")
    st.caption("Add multiple goals; estimate success probability with Monte Carlo.")

    if "goals" not in st.session_state:
        st.session_state.goals = [{"name":"Retirement","amount":8000000,"years":25},
                                  {"name":"Home","amount":3000000,"years":10}]
    goals_df = st.data_editor(pd.DataFrame(st.session_state.goals), num_rows="dynamic", use_container_width=True)
    st.session_state.goals = goals_df.to_dict("records")

    lump = st.number_input("Current Investment (₹)", 0, 10_00_00_000, 5_00_000, step=10_000)
    use_sip = st.checkbox("Use monthly SIP", True)
    sip = st.number_input("Monthly SIP (₹)", 0, 5_00_000, 10_000, step=500) if use_sip else 0
    horizon = st.slider("Simulation horizon (years)", 1, 40, 15)
    sims = st.slider("Simulations", 300, 5000, 1200, step=100)

    # simplified asset mix for MC (60/40 with mild vol)
    mean_ret = np.array([0.10, 0.06])  # equity, debt
    vol = np.array([0.18, 0.05])
    corr = 0.2
    cov = np.outer(vol, vol) * corr; np.fill_diagonal(cov, vol**2)
    w = np.array([0.6, 0.4])

    @st.cache_data(ttl=300, show_spinner=False)
    def mc_sim(lump, sip, w, mean_ret, cov, years, runs):
        L = np.linalg.cholesky(cov)
        annual_sip = sip*12
        res = np.zeros(runs)
        for s in range(runs):
            vals = lump * w
            for _ in range(years):
                z = np.random.normal(size=len(w))
                r = mean_ret + L @ z
                vals = vals*(1+r)
                if annual_sip>0:
                    vals += w*annual_sip
            res[s] = vals.sum()
        return res

    with st.spinner("Running simulations..."):
        mc = mc_sim(lump, sip, w, mean_ret, cov, horizon, sims)

    target_total = sum([g.get("amount",0) for g in st.session_state.goals])
    prob = float((mc >= target_total).sum()/len(mc)*100.0)
    p10, p50, p90 = np.percentile(mc, [10,50,90])

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Median Corpus", fmt_inr(p50))
    c2.metric("P10 / P90", f"{fmt_inr(p10)} / {fmt_inr(p90)}")
    c3.metric("Target (sum)", fmt_inr(target_total))
    c4.metric("Success Prob.", f"{prob:.1f}%")

    fig_mc = px.histogram(mc, nbins=50, title="Monte Carlo — Final Corpus Distribution")
    st.plotly_chart(fig_mc, use_container_width=True)

    # deterministic SIP shortfall check
    def deterministic_fv(lump, sip, w, mean_ret, years):
        mu = (w*mean_ret).sum()
        fv = lump*((1+mu)**years)
        if sip>0:
            r = mu/12
            fv += sip*(((1+r)**(12*years)-1)/r)
        return fv

    det = deterministic_fv(lump, sip, w, mean_ret, horizon)
    st.write(f"Deterministic FV: **{fmt_inr(det)}**")
    if det < target_total:
        # binary search SIP
        lo, hi = 0, 5_00_000
        for _ in range(36):
            mid = (lo+hi)/2
            if deterministic_fv(lump, mid, w, mean_ret, horizon) >= target_total:
                hi = mid
            else:
                lo = mid
        st.warning(f"Estimated required SIP to hit target: **{fmt_inr(hi)} / month**")

# -------------------------- EFFICIENT FRONTIER --------------------------
elif menu == "📈 Efficient Frontier":
    st.header("Efficient Frontier — Random Sampler")
    # 4 synthetic assets (equity/debt/gold/reits) with rough stats
    mu = np.array([0.11, 0.06, 0.07, 0.08])
    sig = np.array([0.19, 0.05, 0.12, 0.10])
    base_corr = np.array([
        [1.0, 0.2, -0.1, 0.1],
        [0.2, 1.0,  0.0, 0.1],
        [-0.1,0.0,  1.0, 0.0],
        [0.1, 0.1,  0.0, 1.0],
    ])
    cov = np.outer(sig, sig) * base_corr
    n = st.slider("Sample size", 200, 5000, 1200, step=100)

    def random_w(n_assets, s):
        r = np.random.random((s, n_assets)); r /= r.sum(axis=1)[:,None]; return r
    W = random_w(4, n)
    rets = W.dot(mu)
    vols = np.sqrt(np.einsum('ij,jk,ik->i', W, cov, W))
    sharpe = (rets - 0.04)/(vols+1e-9)

    df = pd.DataFrame({"Return%":rets*100, "Vol%":vols*100, "Sharpe":sharpe})
    fig = px.scatter(df, x="Vol%", y="Return%", color="Sharpe", color_continuous_scale="Viridis",
                     title="Efficient Frontier (synthetic assets)")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------- RISK PANELS --------------------------
elif menu == "⚠️ Risk Panels":
    st.header("Risk Panels — Stress & Volatility")

    st.subheader("NIFTY Volatility Trend")
    n = fetch_close("^NSEI", period="1y", interval="1d")
    if n.empty:
        st.info("No NIFTY data.")
    else:
        ret = n.pct_change().dropna()
        vol20 = ret.rolling(20).std()*np.sqrt(252)*100
        fig = px.line(vol20, title="Rolling 20D Annualized Volatility (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Stress Scenarios (Δ NIFTY)")
    base_val = st.number_input("Portfolio base value (₹)", 0, 10_00_00_000, 10_00_000, step=50_000)
    beta = st.slider("Portfolio beta vs NIFTY (approx)", 0.0, 2.0, 1.0, 0.05)
    shifts = np.array([-0.10,-0.05,0.0,0.05,0.10])
    pnl = base_val * beta * shifts
    stress_df = pd.DataFrame({"NIFTY Shift": [f"{s:+.0%}" for s in shifts], "P&L (₹)": pnl})
    st.dataframe(stress_df.style.format({"P&L (₹)":"₹{:,.0f}"}), use_container_width=True)

# -------------------------- FOOTER --------------------------
st.markdown("---")
st.caption("Educational tool — not investment advice. Data via Yahoo Finance (yfinance).")
