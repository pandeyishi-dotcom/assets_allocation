# asset_allocation_app.py
"""
Futuristic AI Asset Allocation Dashboard — India Edition
Changes: upgraded Goals UI to futuristic neon goal cards with Monte Carlo probability rings,
         median outcomes and suggested SIPs. Defensive, cached computations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import requests
import time
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(page_title="AI Futuristic Asset Allocator (India)", layout="wide", initial_sidebar_state="expanded")

# ---------------- Auto reload (client-side) ----------------
AUTO_RELOAD_MS = 300000  # 5 minutes
st.components.v1.html(f"""
<script>
setTimeout(function() {{
    if (document.visibilityState === 'visible') {{
        window.location.reload();
    }}
}}, {AUTO_RELOAD_MS});
</script>
""", height=0)

# ---------------- Theme CSS (futuristic neon) ----------------
st.markdown("""
<style>
:root {
  --bg:#05060a;
  --panel: rgba(255,255,255,0.02);
  --glass: rgba(255,255,255,0.03);
  --neon-cyan:#00e5ff;
  --neon-teal:#26ffe6;
  --saffron:#ffd166;
  --muted:#9FB4C8;
  --card-bg: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
}
body { background: linear-gradient(180deg,#05060a,#071425); color:#EAF6F8; }
.card { background: var(--card-bg); border-radius:12px; padding:14px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 30px rgba(0,0,0,0.6);}
.header-left { display:flex; gap:12px; align-items:center; }
.title { color:var(--neon-cyan); font-weight:700; font-size:22px; margin:0; }
.subtitle { color:var(--muted); margin:0; font-size:13px; }
.neon-divider { height:3px; background: linear-gradient(90deg, rgba(0,229,255,0), rgba(38,255,230,0.9), rgba(255,209,102,0.4)); border-radius:4px; margin:10px 0 18px 0; }
.small { color:var(--muted); font-size:13px; }
.ticker-slab { background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02)); padding:8px; border-radius:8px; }
.marquee { overflow:hidden; white-space:nowrap; }
.marquee-inner { display:inline-block; animation:scroll 24s linear infinite; }
@keyframes scroll { 0% { transform: translateX(100%);} 100% { transform: translateX(-100%);} }

/* Goal cards */
.goal-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 18px; margin-top:10px; }
.goal-card {
    background: linear-gradient(180deg, rgba(3,7,12,0.55), rgba(6,10,15,0.35));
    border: 1px solid rgba(38,255,230,0.06);
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 8px 30px rgba(2,10,20,0.6);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.goal-card:hover { transform: translateY(-6px); box-shadow: 0 18px 40px rgba(38,255,230,0.08); }
.goal-title { color: var(--neon-teal); font-weight:700; font-size:18px; margin-bottom:6px; }
.goal-sub { color: var(--muted); font-size:13px; margin-bottom:8px; }
.goal-meta { display:flex; gap:12px; align-items:center; }

/* small badge */
.badge { background: rgba(255,255,255,0.02); border-radius:8px; padding:6px 8px; font-size:12px; color:var(--muted); border:1px solid rgba(255,255,255,0.02); }

/* responsive tweaks */
@media (max-width:800px) {
  .goal-meta { flex-direction:column; align-items:flex-start; }
}
</style>
""", unsafe_allow_html=True)

# ---------------- Optional Lottie (if available) ----------------
try:
    from streamlit_lottie import st_lottie
    LOTTIE_OK = True
except Exception:
    LOTTIE_OK = False

def load_lottie(url, timeout=6):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

LOTTIE_URL = "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_json = load_lottie(LOTTIE_URL)

# ---------------- Sidebar: inputs ----------------
with st.sidebar:
    st.markdown("<div style='display:flex;align-items:center;gap:10px'><div style='font-weight:700;color:#26ffe6'>AI Futuristic Allocator</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>India edition — custom profiles</div>", unsafe_allow_html=True)
    st.markdown("---")
    age = st.slider("Age", 18, 75, 34)
    monthly_income = st.number_input("Monthly income (₹)", min_value=0, value=70000, step=1000)
    risk_label = st.selectbox("Risk appetite (custom)", ["Very Low", "Low", "Moderate", "High", "Very High"])
    st.markdown("---")
    invest_now = st.number_input("Lump-sum invested (₹)", min_value=0, value=500000, step=10000)
    sip_now = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500)
    horizon_default = st.slider("Default planning horizon (yrs)", 1, 40, 10)
    st.markdown("---")
    st.header("Market / Simulation")
    mc_sims = st.slider("Monte Carlo sims", 200, 4000, 1200, step=100)
    frontier_samples = st.slider("Frontier samples", 50, 2000, 400, step=50)
    lookback_years = st.selectbox("Lookback (yrs) for live blending", [1,3,5], index=2)
    st.markdown("---")
    st.caption("Auto-refresh every 5 mins. If live data missing, app uses conservative defaults.")

# ---------------- Watchlist (persistent) ----------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "GOLDBEES.NS", "ICICIBANK.NS"]

# ---------------- Goal storage ----------------
if "goals" not in st.session_state:
    st.session_state.goals = [{"name":"Retirement","amount":5000000,"years":20},{"name":"Home","amount":3000000,"years":8}]

# ---------------- Helpers ----------------
@st.cache_data(ttl=60*20)
def fetch_close_series(ticker, years=5, period="5y"):
    """Return pandas Series of Close prices or None"""
    if not ticker or str(ticker).strip()=="":
        return None
    try:
        hist = yf.download(ticker, period=period, interval="1d", progress=False)
        if hist is None or hist.empty:
            return None
        series = hist["Close"].dropna()
        if series.empty:
            return None
        return series
    except Exception:
        return None

def compute_cagr_and_vol(series):
    if series is None or len(series) < 10:
        return None, None
    years_total = (series.index[-1] - series.index[0]).days / 365.25
    if years_total <= 0:
        return None, None
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1.0 / years_total) - 1.0
    vol = series.pct_change().dropna().std() * np.sqrt(252)
    return float(cagr), float(vol)

@st.cache_data(ttl=60*10)
def monte_carlo_sim_cached(invest, monthly_sip, weights, means, cov, years, sims, seed=42):
    """Cached Monte Carlo run (annual correlated returns)"""
    years = int(float(years))
    sims = int(sims)
    weights = np.array(weights, dtype=float)
    means = np.array(means, dtype=float)
    cov = np.array(cov, dtype=float)
    n = len(weights)
    cov = cov + np.eye(n) * 1e-12
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    finals = np.zeros(sims)
    annual_sip = float(monthly_sip) * 12.0
    base = weights * invest
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

def fv_sip_monthly(monthly, r, yrs):
    yrs = int(float(yrs))
    if yrs <= 0:
        return 0.0
    r_m = (1 + r) ** (1/12) - 1
    n = yrs * 12
    if abs(r_m) < 1e-12:
        return monthly * n
    return monthly * (((1 + r_m) ** n - 1) / r_m) * (1 + r_m)

def deterministic_portfolio_fv(invest, monthly, weights, means, years):
    total = 0.0
    years = int(float(years))
    for i, w in enumerate(weights):
        r = float(means[i])
        total += (invest * w) * ((1 + r) ** years) + fv_sip_monthly(monthly * w, r, years)
    return total

# ---------------- Robo allocation (simple) ----------------
def robo_allocation_custom(age, income, risk_label, include_optional=None):
    include_optional = include_optional or []
    base = {
        "Large Cap Equity":30, "Mid/Small Cap Equity":10, "International Equity":5,
        "Index ETFs":10, "Debt Funds":25, "Gold ETF / SGB":8, "REITs / InvITs":5, "Cash / Liquid":7
    }
    risk_map = {"Very Low": -15, "Low": -7, "Moderate": 0, "High": 7, "Very High": 15}
    tilt = risk_map.get(risk_label, 0)
    base["Large Cap Equity"] = max(0, base["Large Cap Equity"] + int(tilt*0.5))
    base["Mid/Small Cap Equity"] = max(0, base["Mid/Small Cap Equity"] + int(tilt*0.3))
    base["Debt Funds"] = max(0, base["Debt Funds"] - int(tilt*0.6))
    if age < 35:
        base["Mid/Small Cap Equity"] += 5
        base["Debt Funds"] = max(0, base["Debt Funds"] - 5)
    elif age > 60:
        base["Debt Funds"] += 7
        base["Large Cap Equity"] = max(0, base["Large Cap Equity"] - 7)
    if income > 150000:
        base["International Equity"] += 3
    for a in include_optional:
        if a not in base and a in ["Gold ETF / SGB", "REITs / InvITs", "International Equity", "Fixed Deposits", "Crypto (speculative)"]:
            base[a] = 2
    total = sum(base.values())
    if total == 0:
        return {k: round(100 / len(base),2) for k in base}
    return {k: round(v / total * 100, 2) for k, v in base.items()}

# ---------------- Header ----------------
h1, h2 = st.columns([0.8, 3.2])
with h1:
    if LOTTIE_OK and lottie_json is not None:
        try:
            st_lottie(lottie_json, height=100, key="lottie")
        except Exception:
            st.markdown("<div style='width:100px;height:100px;border-radius:12px;background:linear-gradient(180deg,#00121a,#002b2b)'></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='width:100px;height:100px;border-radius:12px;background:linear-gradient(180deg,#00121a,#002b2b)'></div>", unsafe_allow_html=True)
with h2:
    st.markdown("<div class='title'>AI-Powered Futuristic Asset Allocation — India</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Neon interface · Live watchlist · Goal cards with probabilistic progress</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🏦 Portfolio Allocation", "📡 Live Market", "🧮 Goals & Analytics"])

# -------------- Tab 1 (simplified allocation) --------------
with tab1:
    st.markdown("## Portfolio Allocation — custom inputs")
    left, right = st.columns([1, 1.6])
    with left:
        st.markdown("### Inputs")
        optional_assets = st.multiselect("Optional asset classes", options=["Gold ETF / SGB","REITs / InvITs","International Equity","Fixed Deposits","Crypto (speculative)"], default=[])
        horizon = st.number_input("Investment horizon (yrs)", min_value=1, max_value=50, value=horizon_default)
        suggestion = robo_allocation_custom(age, monthly_income, risk_label, optional_assets)
        suggestion_df = pd.DataFrame({"Asset Class": list(suggestion.keys()), "Allocation (%)": list(suggestion.values())})
        st.markdown("### Suggested allocation (editable)")
        edited_alloc = st.data_editor(suggestion_df, num_rows="dynamic", use_container_width=True)
    with right:
        st.markdown("### Quick metrics")
        try:
            if "Allocation (%)" in edited_alloc:
                total_alloc = float(edited_alloc["Allocation (%)"].sum())
                if abs(total_alloc - 100) > 0.75:
                    st.warning("Total allocation is not ~100% — consider adjusting or rebalancing.")
        except Exception:
            pass
        st.markdown("<div class='card'><b>Logic</b><div class='small'>Allocation blends age & risk. Younger investors skew equity; older investors skew debt. You can edit percentages directly.</div></div>", unsafe_allow_html=True)
    st.markdown("### Allocation Visual")
    try:
        fig = px.pie(edited_alloc, names="Asset Class", values="Allocation (%)", hole=0.36)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Allocation visual unavailable (check allocation table).")

# -------------- Tab 2 (Live Market) --------------
with tab2:
    st.markdown("## Live Market Tracker — neon watchlist")
    st.markdown("<div class='small'>Auto-refresh every 5 minutes. Add/remove tickers in the watchlist table below.</div>", unsafe_allow_html=True)

    # Watchlist editing
    wl_col1, wl_col2 = st.columns([3,1])
    with wl_col1:
        wl_df = pd.DataFrame({"Ticker": st.session_state.watchlist})
        wl_df = st.data_editor(wl_df, num_rows="dynamic", use_container_width=True)
        new_watch = [str(x).strip() for x in wl_df["Ticker"].tolist() if str(x).strip() != ""]
        if new_watch != st.session_state.watchlist:
            st.session_state.watchlist = new_watch
    with wl_col2:
        st.markdown("<div class='small muted'>Tip: use Yahoo tickers (eg RELIANCE.NS)</div>", unsafe_allow_html=True)

    # build marquee & cards
    WATCHLIST = st.session_state.watchlist.copy()
    if len(WATCHLIST) == 0:
        st.info("Watchlist empty — add tickers to begin.")
    else:
        strip_items = []
        detail_rows = []
        for t in WATCHLIST:
            ser = fetch_close_series(t, years=lookback_years, period=f"{lookback_years}y")
            if ser is None or ser.empty:
                strip_items.append((t, None, None))
                detail_rows.append({"Ticker": t, "Last Close": None, "Pct Change": None, "Status": "No data", "Series": None})
                continue
            last = float(ser.iloc[-1])
            prev = float(ser.iloc[-2]) if len(ser) >= 2 else last
            pct = (last - prev) / prev * 100 if prev != 0 else 0.0
            strip_items.append((t, last, pct))
            detail_rows.append({"Ticker": t, "Last Close": last, "Pct Change": pct, "Status": "OK", "Series": ser.tail(7)})

        marquee_parts = []
        for t, price, pct in strip_items:
            if price is None:
                marquee_parts.append(f"<span style='padding:0 18px;color:#6C7A89'>{t}: —</span>")
            else:
                color = "#26FFE6" if pct >= 0 else "#FF6B6B"
                marquee_parts.append(f"<span style='padding:0 18px;color:{color}; font-weight:600'>{t}: {price:,.2f} ({pct:+.2f}%)</span>")
        marquee_html = "<div class='ticker-slab marquee'><div class='marquee-inner'>" + " &#160; &#160; ".join(marquee_parts) + "</div></div>"
        st.markdown(marquee_html, unsafe_allow_html=True)

        # sparkline cards
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        cols = st.columns(min(len(strip_items), 6))
        for i, (t, price, pct) in enumerate(strip_items):
            col = cols[i % len(cols)]
            with col:
                if price is None:
                    col.markdown(f"<div class='card' style='text-align:center'><div class='small'>{t}</div><div style='font-size:16px;color:#6C7A89'>No data</div></div>", unsafe_allow_html=True)
                    continue
                color = "#26FFE6" if pct >= 0 else "#FF6B6B"
                col.markdown(f"<div class='card' style='text-align:center'><div class='small'>{t}</div><div style='font-weight:700;color:{color};font-size:18px'>{price:,.2f}</div><div class='small' style='color:{color}'>{pct:+.2f}%</div></div>", unsafe_allow_html=True)
                # sparkline
                series = None
                for r in detail_rows:
                    if r["Ticker"] == t:
                        series = r.get("Series")
                        break
                if series is not None and len(series) >= 3:
                    spdf = pd.DataFrame({"x": series.index, "y": series.values})
                    figsp = px.line(spdf, x="x", y="y", height=90)
                    figsp.update_traces(line=dict(color=color, width=2), hoverinfo="skip")
                    figsp.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
                    st.plotly_chart(figsp, use_container_width=True)
                else:
                    st.markdown("<div style='height:90px'></div>", unsafe_allow_html=True)

        st.markdown("---")
        detail_df = pd.DataFrame([{"Ticker": r["Ticker"], "Last Close": (f"{r['Last Close']:,.2f}" if r["Last Close"] else "—"), "Pct Change": (f"{r['Pct Change']:+.2f}%" if r["Pct Change"] is not None else "—"), "Status": r["Status"]} for r in detail_rows])
        st.table(detail_df)
        missing = [r["Ticker"] for r in detail_rows if r["Status"] != "OK"]
        if missing:
            st.warning("Some tickers returned no data. Yahoo may block certain index tickers; try stock/ETF tickers instead.")

# -------------- Tab 3: Goals & Analytics (futuristic goal cards) --------------
with tab3:
    st.markdown("## Goals & Analytics — neon goal cards with AI probabilities")
    left_col, right_col = st.columns([1.8, 1])

    with left_col:
        st.markdown("### Manage goals")
        goals_df = pd.DataFrame(st.session_state.goals)
        edited_goals = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)
        # sanitize & store
        rows = edited_goals.to_dict(orient="records")
        cleaned = []
        corrections = []
        for i, r in enumerate(rows):
            if not isinstance(r, dict):
                try:
                    r = dict(r)
                except Exception:
                    r = {}
            name = r.get("name") or r.get("Goal") or f"Goal {i+1}"
            # amount
            try:
                amt = float(r.get("amount", 0) or 0)
            except Exception:
                cleaned_amt = ''.join(ch for ch in str(r.get("amount","")) if (ch.isdigit() or ch=='.'))
                amt = float(cleaned_amt) if cleaned_amt else 0.0
                corrections.append((i+1, r.get("amount"), amt))
            # years
            try:
                val = r.get("years") or r.get("Years") or 0
                if isinstance(val, (int, float)):
                    yrs = int(val)
                elif isinstance(val, str):
                    cleaned_yrs = ''.join(ch for ch in val if (ch.isdigit() or ch=='.'))
                    yrs = int(float(cleaned_yrs)) if cleaned_yrs else 0
                    if cleaned_yrs != val:
                        corrections.append((i+1, val, yrs))
                else:
                    yrs = 0
            except Exception:
                yrs = 0
                corrections.append((i+1, r.get("years"), 0))
            cleaned.append({"name": name, "amount": amt, "years": yrs})
        st.session_state.goals = cleaned

        if corrections:
            with st.expander("Auto-corrections applied"):
                for idx, old, new in corrections:
                    st.markdown(f"- Row **{idx}**: `{old}` → `{new}`")
            st.success("Goal inputs cleaned.")

        # Run simulations and show neon cards
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        run = st.button("Simulate goal probabilities (Monte Carlo)")
        # fetch allocation weights from Tab1 edited_alloc if available
        try:
            weights_df = edited_alloc.copy()
            if "Allocation (%)" not in weights_df.columns:
                raise Exception
        except Exception:
            # fallback to robo suggestion
            weights_df = pd.DataFrame({"Asset Class": list(robo_allocation_custom(age, monthly_income, risk_label, optional_assets).keys()), "Allocation (%)": list(robo_allocation_custom(age, monthly_income, risk_label, optional_assets).values())})

        # prepare means & cov for weights_df (use defaults per asset type)
        default_returns = {
            "Large Cap Equity":0.10,"Mid/Small Cap Equity":0.13,"International Equity":0.08,"Index ETFs":0.095,
            "Debt Funds":0.06,"Gold ETF / SGB":0.07,"REITs / InvITs":0.08,"Cash / Liquid":0.035
        }
        means = np.array([default_returns.get(a, 0.06) for a in weights_df["Asset Class"]], dtype=float)
        vols = np.array([0.18 if "Equity" in a else 0.08 for a in weights_df["Asset Class"]], dtype=float)
        cov = np.outer(vols, vols) * 0.25
        np.fill_diagonal(cov, vols**2)
        w = np.array(weights_df["Allocation (%)"] / 100.0, dtype=float)

        # goal cards container
        st.markdown("<div class='goal-grid'>", unsafe_allow_html=True)
        # we will either run MC on button click or show placeholders
        for idx, g in enumerate(st.session_state.goals):
            gname = g.get("name", f"Goal {idx+1}")
            gamt = float(g.get("amount", 0) or 0)
            gyrs = int(g.get("years", 0) or 0)
            # compute deterministic projection & suggested SIP for shortfall (simple heuristic)
            det_val = deterministic_portfolio_fv(invest_now, sip_now, w, means, gyrs)
            # default suggested SIP: if det_val < gamt, estimate SIP needed with average expected portfolio return
            avg_r = max(0.03, float(np.dot(w, means)))  # fallback min 3%
            # estimate required additional monthly SIP to meet goal (approx using FV SIP formula inversion)
            # We'll use simple numeric search to avoid algebraic inversion to be safe.
            def estimate_sip_needed(target, current_invest, monthly_existing_sip, weights, means, years):
                # starting guess
                lo, hi = 0.0, 10_00_000.0
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    fv = deterministic_portfolio_fv(current_invest, monthly_existing_sip + mid, weights, means, years)
                    if fv >= target:
                        hi = mid
                    else:
                        lo = mid
                return hi

            suggested_sip = 0.0
            if det_val < gamt:
                suggested_sip = estimate_sip_needed(gamt, invest_now, sip_now, w, means, gyrs)
                suggested_sip = float(np.round(suggested_sip, -2))
            else:
                suggested_sip = 0.0

            # run MC only if user clicked run — otherwise show placeholder probability from deterministic logic
            prob = None
            median_end = None
            if run and gyrs > 0:
                sims = max(300, int(min(mc_sims, 3000)))
                mc = monte_carlo_sim_cached(invest_now, sip_now, w, means, cov, gyrs, sims)
                prob = float((mc >= gamt).sum() / len(mc) * 100.0)
                median_end = float(np.median(mc))
            else:
                # coarse proxy: if det_val >= target -> prob > 70; else low
                if det_val >= gamt:
                    prob = 78.0
                else:
                    # scale of shortfall (0..1)
                    shortfall_ratio = (gamt - det_val) / max(1.0, gamt)
                    prob = max(6.0, max(10.0, 70.0 * max(0.0, 1 - shortfall_ratio)))
                median_end = det_val

            # Build donut chart using Plotly to show probability
            achieved = prob
            remaining = max(0.0, 100.0 - achieved)
            fig = go.Figure(data=[go.Pie(values=[achieved, remaining],
                                         hole=0.7,
                                         marker=dict(colors=[ "#26FFE6", "rgba(255,255,255,0.04)"]),
                                         sort=False,
                                         textinfo="none")])
            fig.update_layout(showlegend=False,
                              margin=dict(l=0, r=0, t=0, b=0),
                              annotations=[dict(text=f"<b>{achieved:.0f}%</b><br><span style='font-size:11px;color:#9FB4C8'>P(achieve)</span>", x=0.5, y=0.5, showarrow=False)])
            fig.update_traces(hoverinfo='label+percent')

            # Render card HTML + plotly
            st.markdown(f"""
                <div class="goal-card">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <div class="goal-title">{gname}</div>
                            <div class="goal-sub">Target: <span style="color:#E6F0F2">₹{int(gamt):,}</span> · Horizon: <span style="color:#E6F0F2">{gyrs} yrs</span></div>
                        </div>
                        <div style="width:120px;height:120px">
                        """, unsafe_allow_html=True)
            # plotly donut
            st.plotly_chart(fig, use_container_width=False, width=120, height=120)
            st.markdown(f"""
                        </div>
                    </div>
                    <div style="height:8px"></div>
                    <div style="display:flex;gap:18px;align-items:center" class="goal-meta">
                        <div>
                            <div style="color:#9FB4C8;font-size:13px">Median (sim)</div>
                            <div style="font-weight:700;color:#E6F0F2">₹{int(median_end):,}</div>
                        </div>
                        <div>
                            <div style="color:#9FB4C8;font-size:13px">Suggested extra SIP</div>
                            <div style="font-weight:700;color:#ffd166">₹{int(suggested_sip):,}/mo</div>
                        </div>
                        <div>
                            <div style="color:#9FB4C8;font-size:13px">Deterministic FV</div>
                            <div style="font-weight:700;color:#64ffda">₹{int(det_val):,}</div>
                        </div>
                    </div>
                    <div style="height:10px"></div>
                    <div style="color:#8892b0;font-size:13px">Notes: This is a probabilistic projection using a simple annual-return Monte Carlo with assumed correlations. Use it as directional guidance.</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("### Portfolio snapshot (suggested)")
        preview = robo_allocation_custom(age, monthly_income, risk_label, optional_assets)
        df_preview = pd.DataFrame({"Asset Class": list(preview.keys()), "Allocation (%)": list(preview.values())})
        w_pre = np.array(df_preview["Allocation (%)"]/100.0, dtype=float)
        means_pre = np.array([0.09 if "Equity" in a else 0.05 for a in df_preview["Asset Class"]], dtype=float)
        vols_pre = np.array([0.18 if "Equity" in a else 0.08 for a in df_preview["Asset Class"]], dtype=float)
        cov_pre = np.outer(vols_pre, vols_pre)*0.25
        np.fill_diagonal(cov_pre, vols_pre**2)
        p_ret = float(np.dot(w_pre, means_pre))
        p_vol = float(np.sqrt_
