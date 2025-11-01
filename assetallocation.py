# asset_allocation_app.py
"""
AI-Futuristic Asset Allocation Dashboard — India Edition
Features:
- Futuristic dark UI + Lottie header
- Editable watchlist + live market (yfinance)
- Custom allocation suggestion + editable table
- Volatility-driven adaptive rebalancer (Indian assets only)
- Portfolio metrics (return, volatility, Sharpe)
- Professional AI commentary explaining reallocations
- Goal planner with futuristic glowing cards + Monte Carlo probabilities
- Downloadable CSV exports
- Auto-refresh (client-side JS)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from datetime import datetime

# ---------------- Page Config ----------------
st.set_page_config(page_title="AI Asset Allocator — India", layout="wide", initial_sidebar_state="expanded")
AUTO_RELOAD_MS = 300000  # Auto-refresh every 5 minutes (client-side)

# ---------------- Auto reload script ----------------
st.components.v1.html(f"""
<script>
setTimeout(function() {{
    if (document.visibilityState === 'visible') {{
        window.location.reload();
    }}
}}, {AUTO_RELOAD_MS});
</script>
""", height=0)

# ---------------- CSS / Theme ----------------
st.markdown(
    """
    <style>
    :root {
      --bg:#060812;
      --panel: rgba(255,255,255,0.02);
      --glass: rgba(255,255,255,0.03);
      --neon:#00e5ff;
      --teal:#26ffe6;
      --saffron:#ffd166;
      --muted:#9FB4C8;
    }
    body { background: linear-gradient(180deg,#05060a,#071425); color: #EAF6F8; }
    .title { font-size:22px; color:var(--neon); font-weight:700; margin:0; }
    .subtitle { color:var(--muted); margin-top:0; margin-bottom:8px; font-size:13px; }
    .card { background: var(--glass); border-radius:12px; padding:12px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 10px 26px rgba(0,0,0,0.45); }
    .goal-card { background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02)); border-radius:14px; padding:14px; margin-bottom:12px; border: 1px solid rgba(38,255,230,0.06); }
    .muted { color:var(--muted); font-size:13px; }
    .metric { background:#071425; border-radius:10px; padding:10px; text-align:center; }
    .prob-badge { font-weight:700; font-size:13px; color:#00121a; padding:6px 8px; border-radius:8px; background:linear-gradient(90deg,#26ffe6,#00e5ff); display:inline-block; }
    .neon-divider { height:3px; background: linear-gradient(90deg, rgba(0,229,255,0), rgba(38,255,230,0.9), rgba(255,209,102,0.2)); border-radius:4px; margin:10px 0 18px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Lottie helper ----------------
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

# ---------------- Sidebar inputs ----------------
with st.sidebar:
    st.markdown("<div class='title'>AI Asset Allocator — India</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Custom portfolios • Live market • Professional insights</div>", unsafe_allow_html=True)
    st.markdown("---")
    age = st.slider("Age", 18, 75, 34)
    monthly_income = st.number_input("Monthly income (₹)", min_value=0, value=70000, step=1000)
    risk_label = st.selectbox("Risk appetite (custom)", ["Very Low", "Low", "Moderate", "High", "Very High"])
    st.markdown("---")
    invest_now = st.number_input("Lump-sum invested (₹)", min_value=0, value=500000, step=10000)
    sip_now = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500)
    horizon_default = st.slider("Default planning horizon (yrs)", 1, 40, 10)
    st.markdown("---")
    st.header("Sim / Data")
    mc_sims = st.slider("Monte Carlo sims", 200, 4000, 1200, step=100)
    lookback_years = st.selectbox("Lookback (yrs) for live blending", [1,3,5], index=2)
    st.markdown("---")
    st.caption("Auto-refresh enabled. If live data is missing, app uses conservative defaults.")

# ---------------- Indian asset universe & defaults ----------------
ASSET_UNIVERSE = [
    "Large Cap Equity", "Mid/Small Cap Equity", "Index ETFs", "International Equity",
    "Debt Funds", "Government Bonds", "Corporate Bonds",
    "Gold ETF / SGB", "REITs / InvITs", "Cash / Liquid", "Fixed Deposits"
]

DEFAULT_RETURNS = {
    "Large Cap Equity": 0.10, "Mid/Small Cap Equity": 0.13, "Index ETFs": 0.095,
    "International Equity": 0.07, "Debt Funds": 0.06, "Government Bonds": 0.05, "Corporate Bonds": 0.06,
    "Gold ETF / SGB": 0.07, "REITs / InvITs": 0.08, "Cash / Liquid": 0.035, "Fixed Deposits": 0.05
}
DEFAULT_VOL = {k: (0.18 if "Equity" in k else 0.08) for k in DEFAULT_RETURNS.keys()}

DEFAULT_TICKERS = {
    "Large Cap Equity": "^NSEI",  # attempt index; may be blocked
    "Gold ETF / SGB": "GOLDBEES.NS",
    "REITs / InvITs": "NIFTYREIT.NS"  # placeholder; may not exist on yfinance
}

RISK_FREE = 0.06  # 6% risk-free proxy for India

# ---------------- Caching helpers ----------------
@st.cache_data(ttl=60*20)
def fetch_close_series(ticker, years=5):
    if not ticker:
        return None
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{years}y", interval="1d")
        close = hist["Close"].dropna()
        if len(close) < 8:
            return None
        return close
    except Exception:
        return None

def compute_cagr_and_vol(series):
    if series is None or len(series) < 8:
        return None, None
    years_total = (series.index[-1] - series.index[0]).days / 365.25
    if years_total <= 0:
        return None, None
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1.0 / years_total) - 1.0
    vol = series.pct_change().dropna().std() * np.sqrt(252)
    return float(cagr), float(vol)

@st.cache_data(ttl=60*10)
def mc_sim(invest, monthly_sip, weights, means, cov, years, sims, seed=42):
    years = int(float(years))
    sims = int(sims)
    weights = np.array(weights, dtype=float)
    means = np.array(means, dtype=float)
    cov = np.array(cov, dtype=float)
    n = len(weights)
    cov = cov + np.eye(n) * 1e-10
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

# ---------------- Robo suggestion (baseline) ----------------
def robo_allocation_custom(age, income, risk_label, include_optional=None):
    include_optional = include_optional or []
    base = {
        "Large Cap Equity": 30, "Mid/Small Cap Equity": 10, "Index ETFs": 10, "International Equity": 5,
        "Debt Funds": 25, "Gold ETF / SGB": 8, "REITs / InvITs": 5, "Cash / Liquid": 7
    }
    risk_map = {"Very Low": -15, "Low": -7, "Moderate": 0, "High": 7, "Very High": 15}
    tilt = risk_map.get(risk_label, 0)
    base["Large Cap Equity"] = max(0, base["Large Cap Equity"] + int(tilt * 0.5))
    base["Mid/Small Cap Equity"] = max(0, base["Mid/Small Cap Equity"] + int(tilt * 0.3))
    base["Debt Funds"] = max(0, base["Debt Funds"] - int(tilt * 0.6))
    if age < 35:
        base["Mid/Small Cap Equity"] += 5
        base["Debt Funds"] = max(0, base["Debt Funds"] - 5)
    elif age > 60:
        base["Debt Funds"] += 7
        base["Large Cap Equity"] = max(0, base["Large Cap Equity"] - 7)
    if income > 150000:
        base["International Equity"] += 3
    for a in include_optional:
        if a in ASSET_UNIVERSE and a not in base:
            base[a] = 2
    total = sum(base.values())
    return {k: round(v / total * 100, 2) for k, v in base.items()}

# ---------------- Adaptive rebalance logic ----------------
def adaptive_rebalance(allocation_pct, market_vol_pct, risk_label):
    """
    Adjust allocation by reducing equities when volatility rises,
    shifting into Debt/Gold/Cash proportionally.
    market_vol_pct: annualized volatility percent (e.g., 18.5)
    risk_label: user label to scale sensitivity
    """
    alloc = allocation_pct.copy()
    # sensitivity scale by risk appetite
    sensitivity_map = {"Very Low": 1.4, "Low": 1.1, "Moderate": 1.0, "High": 0.9, "Very High": 0.8}
    sens = sensitivity_map.get(risk_label, 1.0)
    # define thresholds (aggressive when vol high)
    if market_vol_pct is None:
        return alloc, 0.0, "No live volatility data; no adaptive change applied."
    vol = float(market_vol_pct)
    # baseline neutral vol ~ 16% (for reference) — adjust around it
    neutral = 16.0
    # scale change proportional to (vol - neutral)
    delta = (vol - neutral) / neutral * 0.5 * sens  # ratio mapped to percentage shift magnitude
    # cap delta between -0.35 and +0.35 (i.e., move at most 35% of equity weight)
    delta = max(min(delta, 0.35), -0.35)
    # compute total equity share (sum of equity buckets)
    equity_keys = [k for k in alloc.keys() if "Equity" in k or "Index ETFs" in k]
    equity_total = sum(alloc.get(k, 0) for k in equity_keys)
    shift = equity_total * delta  # absolute percent points to shift out of equity (if delta>0 => reduce equities)
    # if shift > 0 reduce equities equally and add proportionally to debt/gold/cash
    if shift > 0:
        # proportionally reduce equity keys
        for k in equity_keys:
            pct = alloc.get(k, 0)
            if equity_total > 0:
                alloc[k] = round(max(0, pct - (pct / equity_total) * shift), 2)
        # recipients
        recipients = ["Debt Funds", "Government Bonds", "Gold ETF / SGB", "Cash / Liquid"]
        # compute sum of recipients current amounts
        recipients_total = sum(alloc.get(r, 0) for r in recipients if r in alloc)
        # distribute shift in proportion to current recipient weights (or equally if zero)
        if recipients_total <= 0:
            per = round(shift / len(recipients), 2)
            for r in recipients:
                if r in alloc:
                    alloc[r] = round(alloc.get(r, 0) + per, 2)
        else:
            for r in recipients:
                if r in alloc:
                    alloc[r] = round(alloc.get(r, 0) + (alloc.get(r, 0) / recipients_total) * shift, 2)
    elif shift < 0:
        # volatility lower than neutral -> increase equities slightly (reverse)
        increase = -shift  # positive amount
        recipients = ["Debt Funds", "Government Bonds", "Gold ETF / SGB", "Cash / Liquid"]
        recipients_total = sum(alloc.get(r, 0) for r in recipients if r in alloc)
        if recipients_total <= 0:
            per = round(increase / len(recipients), 2)
            for r in recipients:
                if r in alloc:
                    alloc[r] = round(max(0, alloc.get(r, 0) - per), 2)
            # add to equities equally
            per_eq = round(increase / len(equity_keys), 2) if len(equity_keys) else 0
            for k in equity_keys:
                alloc[k] = round(alloc.get(k, 0) + per_eq, 2)
        else:
            for r in recipients:
                if r in alloc:
                    alloc[r] = round(max(0, alloc.get(r, 0) - (alloc.get(r, 0) / recipients_total) * increase), 2)
            for k in equity_keys:
                if k in alloc:
                    alloc[k] = round(alloc.get(k, 0) + (alloc.get(k, 0) / equity_total if equity_total>0 else 1/len(equity_keys)) * increase, 2)
    # normalize to 100
    s = sum(alloc.values())
    if s == 0:
        # fallback even split
        n = len(alloc)
        return {k: round(100 / n, 2) for k in alloc}, 0.0, "Adaptive fallback applied (zero-sum)."
    alloc = {k: round(v / s * 100, 2) for k, v in alloc.items()}
    # return allocation, absolute shift percent, and explanation
    explanation = f"Adaptive rebalance applied with market vol {vol:.2f}%. Delta factor {delta:.3f}."
    return alloc, round(abs(shift), 2), explanation

# ---------------- Session defaults ----------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "GOLDBEES.NS"]
if "goals" not in st.session_state:
    st.session_state.goals = [{"name":"Retirement","amount":5000000,"years":20},{"name":"Home","amount":3000000,"years":8}]

# ---------------- Header ----------------
hdr_col1, hdr_col2 = st.columns([0.8, 3.2])
with hdr_col1:
    if LOTTIE_OK and lottie_json:
        try:
            st_lottie(lottie_json, height=110, key="lottie")
        except Exception:
            st.markdown("<div style='width:110px;height:110px;background:linear-gradient(180deg,#00121a,#002b2b);border-radius:12px'></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='width:110px;height:110px;background:linear-gradient(180deg,#00121a,#002b2b);border-radius:12px'></div>", unsafe_allow_html=True)
with hdr_col2:
    st.markdown("<div class='title'>AI Asset Allocator — India (Professional AI Commentary)</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Volatility-driven adaptive allocation • Live market • Goal planning</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Portfolio Allocation", "Live Market Tracker", "Goals & Analytics"])

# ---------------- Tab 1: Portfolio Allocation ----------------
with tab1:
    st.markdown("## Portfolio Allocation — suggestion & adaptive rebalancer")
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### Inputs & Suggestion")
        optional_assets = st.multiselect("Add optional asset classes", options=[a for a in ASSET_UNIVERSE], default=[])
        suggestion = robo_allocation_custom(age, monthly_income, risk_label, include_optional=optional_assets)
        suggestion_df = pd.DataFrame({"Asset Class": list(suggestion.keys()), "Allocation (%)": list(suggestion.values())})
        st.markdown("Suggested allocation (editable)")
        edited_alloc = st.data_editor(suggestion_df, num_rows="dynamic", use_container_width=True)

        # Rebalance controls
        st.markdown("### Adaptive rebalancer settings")
        use_adaptive = st.checkbox("Enable volatility-driven adaptive rebalance", value=True)
        st.markdown("<div class='muted'>The AI will tilt allocations away from equities when market volatility rises.</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### Portfolio Snapshot")
        try:
            alloc_df = edited_alloc.copy()
            if "Allocation (%)" not in alloc_df.columns:
                alloc_df["Allocation (%)"] = 0
            alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"] / 100.0 * invest_now
            st.dataframe(alloc_df.style.format({"Allocation (%)":"{:.2f}", "Allocation (₹)":"₹{:,.0f}"}), use_container_width=True)
        except Exception:
            st.info("Edit allocation to view snapshot.")

    # get live market volatility (try Nifty)
    nifty_series = fetch_close_series(DEFAULT_TICKERS.get("Large Cap Equity", "^NSEI"), years=lookback_years)
    if nifty_series is not None:
        _, nifty_vol = compute_cagr_and_vol(nifty_series)
        nifty_vol_pct = nifty_vol * 100 if nifty_vol is not None else None
    else:
        # fallback: blend vols of sample watchlist
        nifty_vol_pct = None

    # apply adaptive rebalance
    try:
        user_alloc = {r["Asset Class"]: float(r["Allocation (%)"]) for _, r in alloc_df.iterrows()}
    except Exception:
        user_alloc = suggestion.copy()
    adaptive_explanation = ""
    shift_amt = 0.0
    if use_adaptive:
        new_alloc, shift_amt, adaptive_explanation = adaptive_rebalance(user_alloc, nifty_vol_pct, risk_label)
    else:
        new_alloc = user_alloc.copy()

    # show allocation chart
    alloc_display_df = pd.DataFrame({"Asset Class": list(new_alloc.keys()), "Allocation (%)": list(new_alloc.values())})
    st.markdown("### Allocation Visual (post-adaptive)")
    try:
        fig_alloc = px.pie(alloc_display_df, names="Asset Class", values="Allocation (%)", hole=0.38)
        fig_alloc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF"), legend=dict(orientation="h"))
        st.plotly_chart(fig_alloc, use_container_width=True)
    except Exception:
        st.info("Allocation visual unavailable.")

    # Portfolio metrics (expected return, vol, sharpe-like)
    # Build means and cov from defaults or blended live if available
    means = np.array([DEFAULT_RETURNS.get(a, 0.06) for a in alloc_display_df["Asset Class"]], dtype=float)
    vols = np.array([DEFAULT_VOL.get(a, 0.15) for a in alloc_display_df["Asset Class"]], dtype=float)
    cov = np.outer(vols, vols) * 0.25
    np.fill_diagonal(cov, vols ** 2)
    weights = np.array(alloc_display_df["Allocation (%)"] / 100.0, dtype=float)
    p_return = float(np.dot(weights, means)) if len(weights) else 0.0
    p_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights)))) if len(weights) else 0.0
    sharpe = (p_return - RISK_FREE) / p_vol if p_vol != 0 else 0.0

    # professional AI commentary
    commentary = []
    # explain adaptive change if applied
    if use_adaptive:
        if nifty_vol_pct is None:
            commentary.append("Live market volatility data was not available; adaptive rebalance used default conservative settings.")
        else:
            if shift_amt > 0.05:
                commentary.append(f"Market volatility ({nifty_vol_pct:.2f}%) is elevated. The model reduced equity exposure by approximately {shift_amt:.2f} percentage points and increased allocations to debt/gold to preserve capital.")
            elif shift_amt > 0:
                commentary.append(f"Market volatility ({nifty_vol_pct:.2f}%) is modestly elevated. The model applied a slight reduction in equity exposure ({shift_amt:.2f} pp).")
            else:
                commentary.append(f"Market volatility ({nifty_vol_pct:.2f}%) is within normal range; no significant equity reduction was required.")
    else:
        commentary.append("Adaptive rebalance is turned off; allocations reflect user input or baseline suggestion.")

    # performance summary card (professional tone)
    st.markdown(f"""
        <div class='card'>
          <div style='display:flex;justify-content:space-between;align-items:center'>
            <div>
              <div style='font-weight:700;color:var(--neon);font-size:16px'>Portfolio Performance</div>
              <div class='muted'>Expected annual return · Estimated volatility · Sharpe-like</div>
            </div>
            <div style='text-align:right'>
              <div style='font-size:18px;color:var(--teal);font-weight:700'>{p_return*100:.2f}%</div>
              <div class='muted'>Return</div>
            </div>
            <div style='text-align:right'>
              <div style='font-size:18px;color:var(--saffron);font-weight:700'>{p_vol*100:.2f}%</div>
              <div class='muted'>Volatility (σ)</div>
            </div>
            <div style='text-align:right'>
              <div style='font-size:18px;color:#cfeef0;font-weight:700'>{sharpe:.2f}</div>
              <div class='muted'>Sharpe-like</div>
            </div>
          </div>
        </div>
    """, unsafe_allow_html=True)

    # show professional commentary
    st.markdown("<br><div class='card'><b>Professional AI Commentary</b><div class='muted' style='margin-top:8px'>" + " ".join(commentary) + "</div></div>", unsafe_allow_html=True)

    # allow download of allocation
    try:
        csv_alloc = alloc_display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download allocation CSV", csv_alloc, file_name="allocation_adaptive.csv", mime="text/csv")
    except Exception:
        pass

# ---------------- Tab 2: Live Market Tracker ----------------
with tab2:
    st.markdown("## Live Market Tracker — watchlist & sparkline cards")
    st.markdown("<div class='muted'>Edit your watchlist; live prices are best-effort via Yahoo Finance.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns([3,1])
    with c1:
        wl_df = pd.DataFrame({"Ticker": st.session_state.watchlist})
        wl_df = st.data_editor(wl_df, num_rows="dynamic", use_container_width=True)
        new_watch = [str(x).strip() for x in wl_df["Ticker"].tolist() if str(x).strip() != ""]
        if new_watch != st.session_state.watchlist:
            st.session_state.watchlist = new_watch
    with c2:
        if st.button("Reset watchlist"):
            st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "GOLDBEES.NS"]

    WATCHLIST = st.session_state.watchlist.copy()[:8]
    strip_items = []
    details = []
    for t in WATCHLIST:
        try:
            s = fetch_close_series(t, years=1)
        except Exception:
            s = None
        if s is None or len(s) < 2:
            strip_items.append((t, None, None))
            details.append({"Ticker": t, "Last Close": None, "Pct Change": None, "Status": "No data"})
            continue
        last = float(s.iloc[-1])
        prev = float(s.iloc[-2]) if len(s) >= 2 else last
        pct = (last - prev) / prev * 100 if prev != 0 else 0.0
        strip_items.append((t, last, pct))
        details.append({"Ticker": t, "Last Close": last, "Pct Change": pct, "Status": "OK", "Series": s.tail(8)})

    # marquee strip
    parts = []
    for t, price, pct in strip_items:
        if price is None:
            parts.append(f"<span style='padding:0 18px;color:#6C7A89'>{t}: —</span>")
        else:
            color = "#26FFE6" if pct >= 0 else "#FF6B6B"
            parts.append(f"<span style='padding:0 18px;color:{color}; font-weight:600'>{t}: {price:,.2f} ({pct:+.2f}%)</span>")
    marquee_html = "<div style='padding:8px;border-radius:8px;background:linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02));'><div style='overflow:hidden;white-space:nowrap;'><div style='display:inline-block;animation:scroll 20s linear infinite;'>" + " &#160; &#160; ".join(parts) + "</div></div></div>"
    st.markdown(marquee_html, unsafe_allow_html=True)

    # sparkline cards
    if strip_items:
        cols = st.columns(min(len(strip_items), 6))
        for i, (t, price, pct) in enumerate(strip_items):
            col = cols[i % len(cols)]
            with col:
                if price is None:
                    col.markdown(f"<div class='card' style='text-align:center'><div class='muted'>{t}</div><div style='font-size:18px;color:#6C7A89'>No data</div></div>", unsafe_allow_html=True)
                else:
                    color = "#26FFE6" if pct >= 0 else "#FF6B6B"
                    col.markdown(f"<div class='card' style='text-align:center'><div class='muted'>{t}</div><div style='font-weight:700;color:{color};font-size:18px'>{price:,.2f}</div><div class='muted' style='color:{color}'>{pct:+.2f}%</div></div>", unsafe_allow_html=True)
                    # sparkline
                    series = None
                    for d in details:
                        if d["Ticker"] == t and d.get("Series") is not None:
                            series = d["Series"]
                            break
                    if series is not None and len(series) >= 3:
                        sp = pd.DataFrame({"x": series.index, "y": series.values})
                        fig_sp = px.line(sp, x="x", y="y", height=90)
                        fig_sp.update_traces(line=dict(color=color, width=2), hoverinfo="skip")
                        fig_sp.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
                        st.plotly_chart(fig_sp, use_container_width=True)
                    else:
                        st.markdown("<div style='height:90px'></div>", unsafe_allow_html=True)

    # detailed table
    detail_df = pd.DataFrame([{"Ticker": r["Ticker"], "Last Close": (f"{r['Last Close']:,}" if r["Last Close"] else "—"), "Pct Change": (f"{r['Pct Change']:+.2f}%" if r["Pct Change"] is not None else "—"), "Status": r["Status"]} for r in details])
    st.table(detail_df)

# ---------------- Tab 3: Goals & Analytics ----------------
with tab3:
    st.markdown("## Goals & Analytics — futuristic cards with Monte Carlo confidence")
    st.markdown("<div class='muted'>Edit goals below. Cards show deterministic progress and probability of success (professional commentary provided).</div>", unsafe_allow_html=True)

    # goals editor
    goals_df = pd.DataFrame(st.session_state.goals)
    edited_goals = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)
    # clean inputs
    rows = edited_goals.to_dict(orient="records")
    cleaned = []
    for r in rows:
        name = r.get("name") or r.get("Goal") or "Goal"
        try:
            amt = float(r.get("amount", 0) or 0)
        except Exception:
            cleaned_amt = ''.join(ch for ch in str(r.get("amount","")) if (ch.isdigit() or ch=='.'))
            amt = float(cleaned_amt) if cleaned_amt else 0.0
        try:
            yrs = int(float(r.get("years", 0) or 0))
        except Exception:
            yrs = 0
        cleaned.append({"name": name, "amount": amt, "years": yrs})
    st.session_state.goals = cleaned

    # weights & simulation setup (from alloc_display_df)
    weights_df = alloc_display_df.copy()
    weights_vec = np.array(weights_df["Allocation (%)"]/100.0, dtype=float)
    means_vec = np.array([DEFAULT_RETURNS.get(a, 0.06) for a in weights_df["Asset Class"]], dtype=float)
    vols_vec = np.array([DEFAULT_VOL.get(a, 0.15) for a in weights_df["Asset Class"]], dtype=float)
    cov_mat = np.outer(vols_vec, vols_vec) * 0.25
    np.fill_diagonal(cov_mat, vols_vec ** 2)

    # build cards
    cards = []
    for g in st.session_state.goals:
        name = g["name"]
        target = float(g["amount"])
        yrs = int(g["years"])
        # deterministic future value using mean returns
        # lumpsum + SIP split by weights
        def fv_lumpsum(pv, r, n):
            return pv * ((1 + r) ** n)
        def fv_sip_monthly(monthly, r, n):
            if n <= 0:
                return 0.0
            r_m = (1 + r) ** (1/12) - 1
            N = n * 12
            if abs(r_m) < 1e-12:
                return monthly * N
            return monthly * (((1 + r_m) ** N - 1) / r_m) * (1 + r_m)
        det_total = 0.0
        for i, a in enumerate(weights_df["Asset Class"]):
            w = weights_vec[i]
            r = means_vec[i]
            det_total += fv_lumpsum(invest_now * w, r, yrs) + fv_sip_monthly(sip_now * w, r, yrs)
        # monte carlo
        sims = max(300, int(mc_sims / 3))
        mc_res = mc_sim(invest_now, sip_now, weights_vec, means_vec, cov_mat, yrs, sims)
        prob = float((mc_res >= target).sum() / len(mc_res) * 100.0)
        median_end = float(np.median(mc_res))
        cards.append({
            "name": name, "target": int(target), "years": yrs, "deterministic": int(det_total), "prob": round(prob,1), "median": int(median_end)
        })

    # render cards two per row
    for i in range(0, len(cards), 2):
        row = cards[i:i+2]
        cols = st.columns(len(row))
        for j, c in enumerate(row):
            with cols[j]:
                st.markdown(f"""
                    <div class="goal-card">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div>
                          <div style="font-weight:700;color:var(--neon);font-size:15px">{c['name']}</div>
                          <div class="muted">Target: <b>₹{c['target']:,}</b> · Horizon: {c['years']} yrs</div>
                        </div>
                        <div style="text-align:right">
                          <div class="prob-badge">{c['prob']}%</div>
                          <div class="muted" style="font-size:12px;margin-top:4px">P(achieve)</div>
                        </div>
                      </div>
                      <div style="height:10px"></div>
                      <div style="background:rgba(255,255,255,0.02);border-radius:8px;padding:8px">
                        <div style="font-size:12px;color:var(--muted)">Deterministic projection</div>
                        <div style="font-weight:700;color:#cfeef0">₹{c['deterministic']:,}</div>
                        <div style="height:8px"></div>
                        <div style="font-size:12px;color:var(--muted)">Median (Monte Carlo)</div>
                        <div style="font-weight:700;color:#cfeef0">₹{c['median']:,}</div>
                      </div>
                      <div style="height:8px"></div>
                      <div class="muted">Professional note: This probability is calculated from {sims} simulated return paths using your current allocation. Consider adjusting SIP or allocation to improve probability if it is below your target confidence level.</div>
                    </div>
                """, unsafe_allow_html=True)

    # export goal summary
    try:
        export_df = pd.DataFrame(cards)
        st.download_button("Download goals summary CSV", export_df.to_csv(index=False).encode("utf-8"), "goals_summary.csv", mime="text/csv")
    except Exception:
        pass

# ---------------- Footer ----------------
st.markdown("<hr style='border:1px solid rgba(255,255,255,0.04)'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#9FB4C8'>Built for Indian investors — educational tool. Not financial advice.</div>", unsafe_allow_html=True)
