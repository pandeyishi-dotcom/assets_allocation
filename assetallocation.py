# asset_allocation_app.py
"""
Sleek Fintech Asset Allocation Dashboard — India Edition
Features:
- Dark fintech UI (cyan + saffron accents)
- Live market (yfinance) for selected tickers
- Editable allocation and CSV upload
- Volatility-driven adaptive rebalancer
- Monte Carlo goal probabilites + mentor-style AI sidebar
- Interactive Plotly charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
import io

st.set_page_config(page_title="AI Mentor Portfolio — India", layout="wide", page_icon="💼")

# --------------------
# Styles / Theme
# --------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #071025;
  --card: rgba(255,255,255,0.02);
  --glass: rgba(255,255,255,0.03);
  --cyan: #00E5FF;
  --teal: #26FFE6;
  --saffron: #FFD166;
  --muted: #9FB4C8;
  --white: #EAF6F8;
}
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(180deg,#05111a,#071425); color:var(--white); font-family: 'Inter', sans-serif; }
h1,h2,h3 { color:var(--cyan); font-family: 'Rajdhani', sans-serif; }
.card { background: var(--card); border-radius:12px; padding:12px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 30px rgba(2,6,10,0.6); }
.small { color:var(--muted); font-size:13px; }
.metric { background:#071425; border-radius:10px; padding:12px; text-align:center; }
.goal-card { background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02)); border-radius:12px; padding:12px; border:1px solid rgba(38,255,230,0.05); margin-bottom:12px; }
a, a:link { color: var(--cyan); }
</style>
""", unsafe_allow_html=True)

# --------------------
# Constants & defaults
# --------------------
ASSET_UNIVERSE = [
    "Large Cap Equity", "Mid/Small Cap Equity", "Index ETFs",
    "Debt Funds", "Government Bonds", "Corporate Bonds",
    "Gold ETF / SGB", "REITs / InvITs", "Cash / Liquid", "Fixed Deposits"
]

DEFAULT_RETURNS = {
    "Large Cap Equity": 0.10, "Mid/Small Cap Equity": 0.13, "Index ETFs": 0.095,
    "Debt Funds": 0.06, "Government Bonds": 0.05, "Corporate Bonds": 0.06,
    "Gold ETF / SGB": 0.07, "REITs / InvITs": 0.08, "Cash / Liquid": 0.035, "Fixed Deposits": 0.05
}
DEFAULT_VOL = {k: (0.18 if "Equity" in k else 0.08) for k in DEFAULT_RETURNS.keys()}
RISK_FREE = 0.06

DEFAULT_TICKERS = {
    "NIFTY": "^NSEI", "SENSEX": "^BSESN", "Gold": "GOLDBEES.NS"
}

# --------------------
# Utilities
# --------------------
@st.cache_data(ttl=60*20)
def fetch_series(ticker, years=3):
    if ticker is None:
        return None
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{years}y")
        if "Close" in hist:
            return hist["Close"].dropna()
    except Exception:
        return None
    return None

@st.cache_data(ttl=60*10)
def fetch_price(ticker):
    if not ticker:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.history(period="5d")
        if len(info) == 0:
            return None
        return float(info["Close"].iloc[-1])
    except Exception:
        return None

def cagr_and_vol(series):
    if series is None or len(series) < 10:
        return None, None
    days = (series.index[-1] - series.index[0]).days
    years = max(days/365.25, 1e-9)
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
    vol = series.pct_change().dropna().std() * np.sqrt(252)
    return float(cagr), float(vol)

def monte_carlo_final_values(lump, sip_month, weights, means, cov, years, sims=200, seed=42):
    years = int(years)
    sims = int(sims)
    weights = np.array(weights, dtype=float)
    means = np.array(means, dtype=float)
    cov = np.array(cov, dtype=float)
    n = len(weights)
    cov = cov + np.eye(n) * 1e-10
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    finals = np.zeros(sims)
    annual_sip = float(sip_month) * 12.0
    base = weights * lump
    for s in range(sims):
        vals = base.copy()
        for y in range(years):
            z = rng.normal(size=n)
            ret = means + L @ z
            vals = vals * (1 + ret)
            if annual_sip > 0:
                vals = vals + annual_sip * weights
        finals[s] = vals.sum()
    return finals

def normalize_alloc(d):
    s = sum(d.values())
    if s == 0: 
        n = len(d)
        return {k: round(100/n,2) for k in d}
    return {k: round(v/s*100,2) for k,v in d.items()}

def robo_suggest(age, income, risk_label, include_optional=None):
    include_optional = include_optional or []
    base = {
        "Large Cap Equity":30, "Mid/Small Cap Equity":10, "Index ETFs":10,
        "Debt Funds":25, "Gold ETF / SGB":8, "REITs / InvITs":5, "Cash / Liquid":7
    }
    map_r = {"Very Low": -15, "Low": -7, "Moderate":0, "High":7, "Very High":15}
    tilt = map_r.get(risk_label,0)
    base["Large Cap Equity"] = max(0, base["Large Cap Equity"] + int(tilt*0.5))
    base["Mid/Small Cap Equity"] = max(0, base["Mid/Small Cap Equity"] + int(tilt*0.3))
    base["Debt Funds"] = max(0, base["Debt Funds"] - int(tilt*0.6))
    if age < 35:
        base["Mid/Small Cap Equity"] += 5
        base["Debt Funds"] = max(0, base["Debt Funds"] - 5)
    if income > 150000:
        base["Index ETFs"] += 3
    for a in include_optional:
        if a in ASSET_UNIVERSE and a not in base:
            base[a] = 2
    return normalize_alloc(base)

def adaptive_rebalance(allocation_pct, market_vol_pct, risk_label):
    alloc = allocation_pct.copy()
    if market_vol_pct is None:
        return alloc, 0.0, "No live volatility available, skipped adaptive move."
    sens_map = {"Very Low":1.4, "Low":1.1, "Moderate":1.0, "High":0.9, "Very High":0.8}
    sens = sens_map.get(risk_label,1.0)
    neutral = 16.0
    delta = (market_vol_pct - neutral) / neutral * 0.5 * sens
    delta = max(min(delta, 0.35), -0.35)
    equity_keys = [k for k in alloc.keys() if "Equity" in k or "Index ETFs" in k]
    equity_total = sum(alloc.get(k,0) for k in equity_keys)
    shift = equity_total * delta
    if shift > 0:
        # reduce equities, increase debt/gold/cash
        for k in equity_keys:
            if equity_total>0:
                alloc[k] = max(0, alloc.get(k,0) - (alloc.get(k,0)/equity_total)*shift)
        recipients = ["Debt Funds", "Government Bonds", "Gold ETF / SGB", "Cash / Liquid"]
        recipients = [r for r in recipients if r in alloc]
        rec_total = sum(alloc.get(r,0) for r in recipients)
        if rec_total<=0:
            per = shift/len(recipients) if recipients else 0
            for r in recipients:
                alloc[r] = alloc.get(r,0) + per
        else:
            for r in recipients:
                alloc[r] = alloc.get(r,0) + (alloc.get(r,0)/rec_total)*shift
    elif shift < 0:
        # low vol: shift small amount back to equities
        inc = -shift
        recipients = ["Debt Funds", "Government Bonds", "Gold ETF / SGB", "Cash / Liquid"]
        recipients = [r for r in recipients if r in alloc]
        rec_total = sum(alloc.get(r,0) for r in recipients)
        if rec_total<=0:
            per = inc/len(equity_keys) if equity_keys else 0
            for k in equity_keys:
                alloc[k] = alloc.get(k,0) + per
        else:
            for r in recipients:
                alloc[r] = max(0, alloc.get(r,0) - (alloc.get(r,0)/rec_total)*inc)
            for k in equity_keys:
                alloc[k] = alloc.get(k,0) + (alloc.get(k,0)/equity_total if equity_total>0 else 1/len(equity_keys))*inc
    alloc = {k: float(round(v,2)) for k,v in normalize_alloc(alloc).items()}
    explanation = f"Adaptive rebalance: market vol {market_vol_pct:.2f}%, delta factor {delta:.3f}."
    return alloc, round(abs(shift),2), explanation

# --------------------
# Sidebar: AI Mentor + controls
# --------------------
with st.sidebar:
    st.markdown("<h3 style='color:var(--cyan)'>AI Mentor</h3>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Mentor-style guidance — concise, helpful notes appear here as you change inputs.</div>", unsafe_allow_html=True)
    # user profile inputs
    age = st.number_input("Age", min_value=18, max_value=80, value=34)
    monthly_income = st.number_input("Monthly income (₹)", min_value=0, value=70000, step=5000)
    risk_label = st.selectbox("Risk appetite", ["Very Low","Low","Moderate","High","Very High"], index=2)
    st.markdown("---")
    lump = st.number_input("Lump-sum invested (₹)", min_value=0, value=500000, step=10000)
    sip = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500)
    horizon = st.slider("Default horizon (yrs)", 1, 40, 10)
    st.markdown("---")
    st.write("Auto-refresh and data")
    auto_refresh = st.checkbox("Auto-refresh live data (5 min)", value=True)
    mc_sims_ui = st.slider("Monte Carlo sims", min_value=200, max_value=5000, value=800, step=100)
    st.markdown("---")
    # place for mentor text, will be updated in code below
    mentor_placeholder = st.empty()
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    # footer small
    st.markdown("<div class='small' style='margin-top:20px'>Built-in mentor — educational only. Not investment advice.</div>", unsafe_allow_html=True)

# --------------------
# Top header
# --------------------
hdr1, hdr2 = st.columns([0.8, 3.2])
with hdr1:
    st.markdown("<div style='width:86px;height:86px;border-radius:12px;background:linear-gradient(180deg,#002b34,#001218)'></div>", unsafe_allow_html=True)
with hdr2:
    st.markdown("<h1>AI Mentor Portfolio — India</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Sleek fintech dashboard • Live Indian market • Mentor guidance</div>", unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# --------------------
# Tabs layout
# --------------------
tab_overview, tab_market, tab_portfolio, tab_insights = st.tabs(["Dashboard","Live Market","Portfolio Editor","Goals & Simulator"])

# --------------------
# Tab: Live Market
# --------------------
with tab_market:
    st.markdown("## Live Market Snapshot")
    cols = st.columns(3)
    for i, (label, tk) in enumerate(DEFAULT_TICKERS.items()):
        price = fetch_price(tk)
        with cols[i]:
            if price:
                st.markdown(f"<div class='metric'><div style='font-size:14px;color:var(--muted)'>{label}</div><div style='font-weight:700;font-size:20px;color:var(--cyan)'>₹{price:,.2f}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric'><div style='font-size:14px;color:var(--muted)'>{label}</div><div style='font-weight:700;font-size:20px;color:var(--saffron)'>—</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("### Watchlist (editable)")
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","GOLDBEES.NS"]
    wl_df = pd.DataFrame({"Ticker": st.session_state.watchlist})
    wl_editor = st.data_editor(wl_df, num_rows="dynamic")
    new_watch = [str(x).strip() for x in wl_editor["Ticker"].tolist() if str(x).strip()!=""]
    st.session_state.watchlist = new_watch

    # show sparkline cards
    watch = st.session_state.watchlist[:8]
    if watch:
        cols = st.columns(min(len(watch),6))
        for i, tk in enumerate(watch):
            col = cols[i%len(cols)]
            s = fetch_series(tk, years=1)
            try:
                price = s.iloc[-1]
                pct = (s.iloc[-1] - s.iloc[-2]) / s.iloc[-2] * 100 if len(s)>=2 else 0
            except Exception:
                price = None
                pct = None
            if price is None:
                col.markdown(f"<div class='card' style='text-align:center'><div class='small'>{tk}</div><div style='color:var(--muted);margin-top:8px'>No data</div></div>", unsafe_allow_html=True)
            else:
                color = "var(--teal)" if pct>=0 else "#FF6B6B"
                col.markdown(f"<div class='card' style='text-align:center'><div class='small'>{tk}</div><div style='font-weight:700;color:{color};font-size:16px'>{price:,.2f}</div><div class='small' style='color:{color}'>{pct:+.2f}%</div></div>", unsafe_allow_html=True)
                # small sparkline
                sp = pd.DataFrame({"x": s.index, "y": s.values})
                fig_sp = px.line(sp, x="x", y="y", height=90)
                fig_sp.update_traces(line=dict(color=color.replace("var(--teal)","#26FFE6") if "var(" in color else color, width=2))
                fig_sp.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
                col.plotly_chart(fig_sp, use_container_width=True)

# --------------------
# Tab: Portfolio Editor
# --------------------
with tab_portfolio:
    st.markdown("## Portfolio Editor")
    st.markdown("Upload a CSV (two columns: Asset Class, Allocation %) or edit suggested allocation.")
    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv","txt","xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".xlsx"):
                df_port = pd.read_excel(uploaded)
            else:
                df_port = pd.read_csv(uploaded)
            # standardize columns
            if "Asset Class" not in df_port.columns and "asset" in [c.lower() for c in df_port.columns]:
                df_port.columns = ["Asset Class","Allocation (%)"]
        except Exception:
            st.error("Failed to read uploaded file. Make sure it has Asset Class and Allocation (%) columns.")
            df_port = None
    else:
        df_port = None

    include_optional = st.multiselect("Include optional asset classes", options=ASSET_UNIVERSE, default=[])
    suggested = robo_suggest(age=age, income=monthly_income, risk_label=risk_label, include_optional=include_optional)
    if df_port is None:
        suggestion_df = pd.DataFrame({"Asset Class": list(suggested.keys()), "Allocation (%)": list(suggested.values())})
        edited_alloc = st.data_editor(suggestion_df, num_rows="dynamic", use_container_width=True)
    else:
        edited_alloc = st.data_editor(df_port, num_rows="dynamic", use_container_width=True)

    # read user allocation
    try:
        alloc_map = {row["Asset Class"]: float(row["Allocation (%)"]) for _, row in edited_alloc.iterrows()}
    except Exception:
        alloc_map = suggested.copy()
    alloc_map = {k:v for k,v in alloc_map.items() if v>0}
    alloc_map = normalize_alloc(alloc_map)

    # show donut allocation
    alloc_df = pd.DataFrame({"Asset Class": list(alloc_map.keys()), "Allocation (%)": list(alloc_map.values())})
    st.markdown("### Allocation (post-normalization)")
    fig = px.pie(alloc_df, names="Asset Class", values="Allocation (%)", hole=0.4)
    fig.update_layout(template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=-0.15), margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # apply adaptive rebalance using NIFTY volatility if available
    nifty_series = fetch_series(DEFAULT_TICKERS.get("NIFTY"), years=3)
    nifty_cagr, nifty_vol = (None, None)
    if nifty_series is not None:
        _, nifty_vol = cagr_and_vol := cagr_and_vol if False else cagr_and_vol  # harmless placeholder
        # compute vol properly:
        try:
            _, nifty_vol = cagr_and_vol(nifty_series)
            nifty_vol = nifty_vol*100 if nifty_vol else None
        except Exception:
            nifty_vol = None
    # fallback if fetch failed
    new_alloc = alloc_map.copy()
    if st.checkbox("Enable adaptive rebalance (tilt allocations based on market volatility)", value=True):
        new_alloc, shift_amt, explain = adaptive_rebalance(new_alloc, nifty_vol, risk_label)
        st.markdown(f"<div class='small'>Adaptive note: {explain}</div>", unsafe_allow_html=True)

    # show radar / diversification
    st.markdown("### Diversification Radar")
    radar_df = alloc_df.copy()
    radar_df = radar_df.sort_values("Allocation (%)", ascending=False)
    try:
        fig_r = px.line_polar(radar_df, r="Allocation (%)", theta="Asset Class", line_close=True)
        fig_r.update_traces(fill='toself', line_color="#00E5FF")
        fig_r.update_layout(template="plotly_dark", margin=dict(t=10,b=10))
        st.plotly_chart(fig_r, use_container_width=True)
    except Exception:
        pass

    # show small snapshot numbers (expected return / vol / sharpe-like)
    means = np.array([DEFAULT_RETURNS.get(a,0.06) for a in alloc_df["Asset Class"]], dtype=float)
    vols = np.array([DEFAULT_VOL.get(a,0.15) for a in alloc_df["Asset Class"]], dtype=float)
    cov = np.outer(vols, vols)*0.25
    np.fill_diagonal(cov, vols**2)
    weights_vec = np.array(alloc_df["Allocation (%)"]/100.0, dtype=float)
    p_return = float(np.dot(weights_vec, means)) if len(weights_vec)>0 else 0.0
    p_vol = float(np.sqrt(np.dot(weights_vec.T, np.dot(cov, weights_vec)))) if len(weights_vec)>0 else 0.0
    p_sharpe = (p_return - RISK_FREE)/p_vol if p_vol>0 else 0.0

    c1,c2,c3 = st.columns(3)
    c1.markdown(f"<div class='metric'><div class='small'>Expected Return</div><div style='font-weight:700;color:var(--cyan);font-size:18px'>{p_return*100:.2f}%</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><div class='small'>Est. Volatility (σ)</div><div style='font-weight:700;color:var(--saffron);font-size:18px'>{p_vol*100:.2f}%</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><div class='small'>Sharpe-like</div><div style='font-weight:700;color:#cfeef0;font-size:18px'>{p_sharpe:.2f}</div></div>", unsafe_allow_html=True)

    # export allocation
    try:
        st.download_button("Download allocation (CSV)", alloc_df.to_csv(index=False).encode('utf-8'), "allocation.csv", mime="text/csv")
    except Exception:
        pass

# --------------------
# Tab: Dashboard overview + Goals
# --------------------
with tab_overview:
    st.markdown("## Dashboard Overview")
    left, mid, right = st.columns([1.4, 2.2, 1])
    with left:
        st.markdown("### Snapshot")
        st.markdown(f"<div class='card'><div class='small'>Profile: Age {age} · Risk: {risk_label} · Lump ₹{lump:,} · SIP ₹{sip:,}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>")
        st.markdown("### Allocation (current)")
        alloc_disp_df = pd.DataFrame({"Asset Class": list(new_alloc.keys()), "Allocation (%)": list(new_alloc.values())})
        st.dataframe(alloc_disp_df.style.format({"Allocation (%)":"{:.2f}"}), use_container_width=True)

    with mid:
        st.markdown("### Allocation Visual")
        try:
            fig2 = px.pie(alloc_disp_df, names="Asset Class", values="Allocation (%)", hole=0.45)
            fig2.update_layout(template="plotly_dark", margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            pass

        st.markdown("### Risk vs Return (point-estimates)")
        rr_df = pd.DataFrame({
            "Asset Class": list(alloc_disp_df["Asset Class"]),
            "Return": [DEFAULT_RETURNS.get(a,0.06)*100 for a in alloc_disp_df["Asset Class"]],
            "Vol": [DEFAULT_VOL.get(a,0.15)*100 for a in alloc_disp_df["Asset Class"]],
            "Weight": list(alloc_disp_df["Allocation (%)"])
        })
        fig_rr = go.Figure()
        fig_rr.add_trace(go.Scatter(x=rr_df["Vol"], y=rr_df["Return"], mode="markers+text",
                                    text=rr_df["Asset Class"], marker=dict(size=rr_df["Weight"]/2, color=rr_df["Weight"], colorscale="Viridis")))
        fig_rr.update_layout(xaxis_title="Volatility (%)", yaxis_title="Exp Return (%)", template="plotly_dark", height=360)
        st.plotly_chart(fig_rr, use_container_width=True)

    with right:
        st.markdown("### Quick Actions")
        if st.button("Re-run Monte Carlo for goals"):
            st.success("Simulations will run below in Goals tab.")
        st.markdown("<div style='height:8px'></div>")
        st.markdown("### Export")
        st.markdown("Download snapshots for reports:")
        try:
            st.download_button("Download allocation CSV", alloc_disp_df.to_csv(index=False).encode('utf-8'), "alloc_snapshot.csv", mime="text/csv")
        except Exception:
            pass

# --------------------
# Tab: Goals & Simulation
# --------------------
with tab_insights:
    st.markdown("## Goals & Simulator")
    if "goals" not in st.session_state:
        st.session_state.goals = [{"name":"Retirement","amount":5000000,"years":20},{"name":"Home","amount":3000000,"years":8}]
    goals_df = pd.DataFrame(st.session_state.goals)
    edited_goals = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)
    # sanitize
    cleaned = []
    for r in edited_goals.to_dict(orient="records"):
        n = r.get("name") or r.get("Name") or "Goal"
        try: amt = float(r.get("amount") or r.get("Amount") or 0)
        except: amt = 0.0
        try: yrs = int(float(r.get("years") or r.get("Years") or 0))
        except: yrs = 0
        cleaned.append({"name":n,"amount":amt,"years":yrs})
    st.session_state.goals = cleaned

    # weights vect and cov
    weights_list = [new_alloc.get(k,0)/100.0 for k in alloc_disp_df["Asset Class"]]
    means_list = [DEFAULT_RETURNS.get(k,0.06) for k in alloc_disp_df["Asset Class"]]
    vols_list = [DEFAULT_VOL.get(k,0.15) for k in alloc_disp_df["Asset Class"]]
    cov_mat = np.outer(vols_list, vols_list)*0.25
    np.fill_diagonal(cov_mat, np.array(vols_list)**2)

    # compute and render cards
    cards = []
    for g in st.session_state.goals:
        det = 0.0
        yrs = max(0, int(g["years"]))
        for i,a in enumerate(alloc_disp_df["Asset Class"]):
            w = weights_list[i] if i<len(weights_list) else 0
            r = means_list[i] if i<len(means_list) else 0.06
            # fv lumpsum
            det += lump*w*((1+r)**yrs)
            # fv sip
            r_m = (1+r)**(1/12)-1
            n = yrs*12
            if abs(r_m) < 1e-12:
                sip_fv = sip*w*n
            else:
                sip_fv = sip*w*(((1+r_m)**n -1)/r_m)*(1+r_m)
            det += sip_fv
        sims = max(300, int(mc_sims_ui/2))
        mc_vals = monte_carlo_final_values(lump, sip, weights_list, means_list, cov_mat, yrs, sims=sims)
        prob = float((mc_vals >= g["amount"]).sum() / len(mc_vals) * 100.0)
        median = float(np.median(mc_vals))
        cards.append({"name":g["name"], "target":int(g["amount"]), "years":yrs, "deterministic":int(det), "prob":round(prob,1), "median":int(median)})

    # render cards 2 per row
    for i in range(0, len(cards), 2):
        row = cards[i:i+2]
        cols = st.columns(len(row))
        for j, item in enumerate(row):
            with cols[j]:
                st.markdown(f"""
                    <div class='goal-card'>
                      <div style='display:flex;justify-content:space-between;align-items:center'>
                        <div style='font-weight:700;color:var(--cyan)'>{item['name']}</div>
                        <div style='text-align:right'><div style='font-weight:700;color:var(--teal)'>{item['prob']}%</div><div class='small'>P(achieve)</div></div>
                      </div>
                      <div style='height:8px'></div>
                      <div class='small'>Target: <b>₹{item['target']:,}</b> · Horizon: {item['years']} yrs</div>
                      <div style='height:8px'></div>
                      <div style='background:rgba(255,255,255,0.02);padding:8px;border-radius:8px'>
                        <div class='small'>Deterministic projection</div><div style='font-weight:700;color:#cfeef0'>₹{item['deterministic']:,}</div>
                        <div style='height:6px'></div>
                        <div class='small'>Monte Carlo median</div><div style='font-weight:700;color:#cfeef0'>₹{item['median']:,}</div>
                      </div>
                      <div style='height:8px'></div>
                      <div class='small' style='color:var(--muted)'>Note: probability calculated from {sims} simulated paths using current allocation.</div>
                    </div>
                """, unsafe_allow_html=True)

# --------------------
# AI Mentor sidebar content (mentor style)
# --------------------
# Build messages based on portfolio metrics and goals
mentor_lines = []
mentor_lines.append(f"Hello — I analysed your profile (age {age}, risk {risk_label}).")
mentor_lines.append(f"Current allocation has estimated return {p_return*100:.2f}% and volatility {p_vol*100:.2f}%.")
if p_sharpe > 1.2:
    mentor_lines.append("Good job — your portfolio shows strong risk-adjusted returns.")
elif p_sharpe > 0.6:
    mentor_lines.append("Portfolio is reasonable. Consider minor diversification if goals are ambitious.")
else:
    mentor_lines.append("I recommend reducing equity exposure and increasing debt or gold to improve stability.")

# Goal advice
for c in cards:
    if c["prob"] < 60:
        mentor_lines.append(f"For goal '{c['name']}', probability {c['prob']}% — consider increasing SIP or horizon.")
    elif c["prob"] < 80:
        mentor_lines.append(f"'{c['name']}' shows moderate chance ({c['prob']}%). Review allocation or SIP to improve confidence.")
    else:
        mentor_lines.append(f"'{c['name']}' looks healthy ({c['prob']}% chance). Keep monitoring.")

# Short concise mentor text
mentor_text = " ".join(mentor_lines[:6])  # keep short

mentor_placeholder.markdown(f"<div class='card'><div style='font-weight:600;color:var(--teal)'>Mentor summary</div><div class='small' style='margin-top:6px'>{mentor_text}</div></div>", unsafe_allow_html=True)

# --------------------
# Footer small
# --------------------
st.markdown("<hr style='border:1px solid rgba(255,255,255,0.04)'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:var(--muted)'>Educational tool. Not financial advice.</div>", unsafe_allow_html=True)
