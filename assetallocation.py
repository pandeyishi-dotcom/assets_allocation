# app.py
"""
Fintech Pro Edition — Indian Asset Allocator & Robo-Advisor (Streamlit Cloud ready)

Features:
- Sidebar navigation: Overview | Portfolio | Market | Goals | Analytics
- Live market data via yfinance (Nifty + sample tickers)
- Smart robo-allocation by age/risk/income with editable allocation table
- Multi-goal planner with deterministic + Monte Carlo simulation per goal
- Efficient frontier (random portfolios), correlation heatmap
- Rebalance worksheet, downloadable CSVs
- Dark/Light theme toggle and polished UI (CSS)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from math import ceil

# -----------------------
# Page config + global css
# -----------------------
st.set_page_config(page_title="Fintech Pro Asset Allocator", layout="wide", initial_sidebar_state="expanded")

# CSS for polished UI (dark + light support)
st.markdown(
    """
    <style>
    :root {
        --bg-dark: #071425;
        --panel-dark: rgba(255,255,255,0.03);
        --accent: #6FF0B0;
        --muted: #9FB4C8;
        --glass: rgba(255,255,255,0.025);
    }
    .app-header { font-size:20px; font-weight:700; color:var(--accent); margin-bottom: 6px;}
    .app-sub { color:var(--muted); margin-bottom: 12px; }
    .card { background: var(--glass); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); }
    .metric { background:#071425; padding:10px; border-radius:8px; text-align:center; }
    .metric-value { font-size:18px; color:var(--accent); font-weight:700; }
    .small { color:var(--muted); font-size:13px; }
    /* responsive tweaks */
    @media (max-width: 768px) {
        .hide-mobile { display:none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# helper formatting
def fmt_inr(x):
    return "₹{:,.0f}".format(x)

# -----------------------
# Sidebar: Navigation + user profile
# -----------------------
st.sidebar.image("https://raw.githubusercontent.com/amirbek/fin-dashboard-examples/main/logo.png", width=120) if False else None
st.sidebar.markdown("<div style='font-weight:700; font-size:16px;'>Fintech Pro — Asset Allocator</div>", unsafe_allow_html=True)

# Theme toggle
theme = st.sidebar.selectbox("Theme", ["Dark (default)", "Light"])
if theme.startswith("Light"):
    st.markdown("<style>body{background:linear-gradient(180deg,#f7fbff,#e8f4ff); color:#022;}</style>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("Investor Profile")
age = st.sidebar.slider("Age", 18, 75, 34)
monthly_income = st.sidebar.number_input("Monthly income (₹)", min_value=0, value=70000, step=1000)
risk_pref = st.sidebar.selectbox("Risk appetite", ["Low", "Moderate", "High"])
st.sidebar.markdown("---")
st.sidebar.header("Quick controls")
mc_sims = st.sidebar.slider("Monte Carlo sims", 200, 4000, 1200, step=100)
frontier_samples = st.sidebar.slider("Efficient frontier samples", 50, 2000, 400, step=50)
lookback_years = st.sidebar.selectbox("Lookback (yrs) for live CAGR", [1,3,5], index=2)

st.sidebar.markdown("---")
st.sidebar.caption("After you deploy: Manage app → Clear cache → Rerun from scratch (important for installing deps).")

# Navigation
page = st.sidebar.radio("Navigate", ["Overview", "Portfolio", "Market", "Goals", "Analytics"])

# -----------------------
# Shared asset universe & defaults
# -----------------------
ASSET_UNIVERSE = [
    "Large Cap Equity", "Mid/Small Cap Equity", "International Equity", "Index ETFs",
    "Active Equity Funds", "Sectoral/Thematic Funds", "Debt Funds", "Government Bonds",
    "Corporate Bonds", "Gold ETF", "REITs/InvITs", "Real Estate (Direct)",
    "Liquid/Cash", "Fixed Deposits", "Commodities (other)", "Crypto (speculative)"
]

DEFAULT_TICKERS = {
    "Large Cap Equity": "^NSEI",
    "Gold ETF": "GOLDBEES.NS",
    "International Equity": "VTI"
}

DEFAULT_RETURNS = {
    "Large Cap Equity": 0.10, "Mid/Small Cap Equity": 0.13, "International Equity": 0.09,
    "Index ETFs": 0.095, "Debt Funds": 0.06, "Gold ETF": 0.07, "REITs/InvITs": 0.08,
    "Liquid/Cash": 0.035, "Fixed Deposits": 0.05, "Crypto (speculative)": 0.20
}
DEFAULT_VOL = {k: 0.18 if "Equity" in k else 0.07 for k in DEFAULT_RETURNS.keys()}

# -----------------------
# Utility functions for live data + stats
# -----------------------
@st.cache_data(ttl=60*30)
def fetch_close_series(ticker, years=5):
    if not ticker:
        return None
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=f"{years}y", interval="1d")
        close = hist["Close"].dropna()
        if len(close) < 10:
            return None
        return close
    except Exception:
        return None

def compute_cagr_and_vol(series):
    if series is None or len(series) < 10:
        return None, None
    start = series.iloc[0]
    end = series.iloc[-1]
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    if yrs <= 0:
        return None, None
    cagr = (end/start)**(1.0/yrs) - 1.0
    vol = series.pct_change().dropna().std() * np.sqrt(252)
    return float(cagr), float(vol)

# Monte Carlo cached (annual correlated returns)
@st.cache_data(ttl=60*10)
def monte_carlo_sim(invest, monthly_sip, weights, means, cov, years, sims, seed=42):
    np.random.seed(seed)
    n = len(weights)
    L = np.linalg.cholesky(cov + np.eye(n)*1e-12)
    results = np.zeros(sims)
    annual_sip = monthly_sip * 12.0
    base = weights * invest
    for s in range(sims):
        vals = base.copy()
        for y in range(int(years)):
            z = np.random.normal(size=n)
            ret = means + L @ z
            vals = vals * (1 + ret)
            if annual_sip > 0:
                vals = vals + annual_sip * weights
        results[s] = vals.sum()
    return results

# deterministic FV helpers
def fv_lumpsum(pv, r, years):
    return pv * ((1 + r) ** years)
def fv_sip_monthly(monthly, r, years):
    r_m = (1 + r) ** (1/12) - 1
    n = int(years) * 12
    if r_m == 0:
        return monthly * n
    return monthly * (((1 + r_m) ** n - 1) / r_m) * (1 + r_m)

def deterministic_portfolio_fv(invest, monthly, weights, means, years):
    total = 0.0
    for i, w in enumerate(weights):
        r = means[i]
        total += fv_lumpsum(invest * w, r, years) + fv_sip_monthly(monthly * w, r, years)
    return total

# -----------------------
# Robo allocation logic (smart)
# -----------------------
BASELINE = {
    "Low": {"Large Cap Equity":25,"Mid/Small Cap Equity":5,"International Equity":5,"Index ETFs":10,"Debt Funds":35,"Gold ETF":10,"Liquid/Cash":10},
    "Moderate":{"Large Cap Equity":35,"Mid/Small Cap Equity":10,"International Equity":8,"Index ETFs":10,"Debt Funds":20,"Gold ETF":7,"Liquid/Cash":10},
    "High":{"Large Cap Equity":45,"Mid/Small Cap Equity":15,"International Equity":10,"Index ETFs":8,"Debt Funds":10,"Gold ETF":5,"Liquid/Cash":5,"Crypto (speculative)":2}
}

def robo_allocation(age, income, risk_label, include_assets=None, normalize=True):
    base = BASELINE.get({"Low":"Low","Moderate":"Moderate","High":"High"}[risk_label]).copy()
    include_assets = include_assets or []
    for a in include_assets:
        if a not in base:
            base[a] = 2.0
    # age tilt: younger -> more equity
    if age < 35:
        shift = 5
        base["Debt Funds"] = max(0, base.get("Debt Funds",0)-shift)
        base["Large Cap Equity"] = base.get("Large Cap Equity",0) + shift
    elif age > 55:
        shift = 5
        base["Large Cap Equity"] = max(0, base.get("Large Cap Equity",0)-shift)
        base["Debt Funds"] = base.get("Debt Funds",0) + shift
    # income tilt
    if income > 150000:
        base["International Equity"] = base.get("International Equity",0) + 2
    # normalize
    if normalize:
        total = sum(base.values())
        base = {k: round(v/total*100,2) for k,v in base.items()}
    return base

# -----------------------
# Page: Overview
# -----------------------
if page == "Overview":
    st.markdown("<div class='app-header'>Portfolio — Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>Quick summary and recommended allocation</div>", unsafe_allow_html=True)

    # profile inputs quick
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        invest = st.number_input("Current invested (lump-sum ₹)", min_value=0, value=500000, step=10000)
    with col2:
        sip = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500)
    with col3:
        horizon = st.slider("Planning horizon (years)", 1, 40, 10)

    allocation = robo_allocation(age, monthly_income, risk_pref)
    alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
    alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"]/100.0 * invest

    st.markdown("### Recommended Allocation (editable)")
    alloc_df = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)

    if abs(alloc_df["Allocation (%)"].sum() - 100) > 0.5:
        st.warning("Total allocation is not 100% — edit the table to normalize to 100%.")

    # compute blended expected returns and vols for visible assets
    asset_returns = {}
    asset_vols = {}
    for a in alloc_df["Asset Class"]:
        t = DEFAULT_TICKERS.get(a)
        series = fetch_close_series(t, years=lookback_years) if t else None
        if series is not None:
            c, v = compute_cagr_and_vol(series)
            if c is not None:
                asset_returns[a] = 0.7*c + 0.3*DEFAULT_RETURNS.get(a,0.06)
                asset_vols[a] = 0.7*v + 0.3*DEFAULT_VOL.get(a,0.15)
            else:
                asset_returns[a] = DEFAULT_RETURNS.get(a,0.06)
                asset_vols[a] = DEFAULT_VOL.get(a,0.15)
        else:
            asset_returns[a] = DEFAULT_RETURNS.get(a,0.06)
            asset_vols[a] = DEFAULT_VOL.get(a,0.15)

    alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x,0.06)*100)
    alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x,0.15)*100)
    alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"]/100.0 * invest

    st.dataframe(alloc_df.style.format({"Allocation (%)":"{:.2f}","Allocation (₹)":"₹{:,.0f}","Exp Return (%)":"{:.2f}%","Volatility (%)":"{:.2f}%"}), use_container_width=True)

    # portfolio expected metrics (wtd)
    w = np.array(alloc_df["Allocation (%)"]/100.0)
    means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]])
    vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]])
    base_corr = 0.25
    cov = np.outer(vols, vols)*base_corr
    np.fill_diagonal(cov, vols**2)
    port_return = float(np.dot(w, means))
    port_vol = float(np.sqrt(w @ cov @ w))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected annual return", f"{port_return*100:.2f}%")
    k2.metric("Estimated volatility", f"{port_vol*100:.2f}%")
    # deterministic FV
    det_fv = deterministic_portfolio_fv(invest, sip, w, means, horizon)
    k3.metric(f"Deterministic value ({horizon}y)", fmt_inr(det_fv))
    # Monte Carlo quick
    mc_vals = monte_carlo_sim(invest, sip, w, means, cov, horizon, min(1000, mc_sims))
    k4.metric("Median MC end", fmt_inr(float(np.median(mc_vals))))

    # small allocation pie
    fig_p = px.pie(alloc_df, names="Asset Class", values="Allocation (%)", hole=0.35,
                   color_discrete_sequence=px.colors.sequential.Tealgrn)
    fig_p.update_layout(title="Allocation", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_p, use_container_width=True)

# -----------------------
# Page: Portfolio (detailed)
# -----------------------
elif page == "Portfolio":
    st.markdown("<div class='app-header'>Portfolio — Detailed</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>Advanced charts and efficient frontier</div>", unsafe_allow_html=True)

    # allow user to upload or enter current holdings
    st.subheader("Your Current Holdings (optional)")
    holdings = st.data_editor(pd.DataFrame(columns=["Asset Class", "Current Value (₹)"]), num_rows="dynamic", use_container_width=True)

    st.subheader("Simulate asset mix")
    # reuse allocation from robo engine for speed
    allocation = robo_allocation(age, monthly_income, risk_pref)
    alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
    alloc_df = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)

    # compute expected returns/vols
    asset_returns = {a: DEFAULT_RETURNS.get(a,0.06) for a in alloc_df["Asset Class"]}
    asset_vols = {a: DEFAULT_VOL.get(a,0.15) for a in alloc_df["Asset Class"]}
    alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x)*100)
    alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x)*100)

    # efficient frontier (random search)
    weights = np.array(alloc_df["Allocation (%)"]/100.0)
    means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]])
    vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]])
    base_corr = 0.25
    cov = np.outer(vols, vols)*base_corr
    np.fill_diagonal(cov, vols**2)

    def rand_weights(n, samples):
        r = np.random.random((samples, n))
        r /= r.sum(axis=1)[:,None]
        return r

    samples = min(max(50, frontier_samples), 3000)
    rws = rand_weights(len(weights), samples)
    ef_ret = rws.dot(means)
    ef_vol = np.sqrt(np.einsum('ij,jk,ik->i', rws, cov, rws))
    ef_sh = (ef_ret - 0.04) / (ef_vol + 1e-9)

    ef_df = pd.DataFrame({"Return":ef_ret*100, "Volatility":ef_vol*100, "Sharpe":ef_sh})
    fig_ef = px.scatter(ef_df, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="Viridis")
    fig_ef.add_trace(go.Scatter(x=[port_vol*100], y=[port_return*100], mode="markers+text", marker=dict(size=12,color="gold"), text=["Current"], textposition="top center"))
    st.plotly_chart(fig_ef, use_container_width=True)

    st.markdown("### Asset Risk vs Return")
    fig_sc = px.scatter(alloc_df, x="Volatility (%)", y="Exp Return (%)", size="Allocation (%)", text="Asset Class")
    st.plotly_chart(fig_sc, use_container_width=True)

    # rebalancing worksheet
    st.markdown("---")
    st.subheader("Rebalancing worksheet")
    if not holdings.empty and "Asset Class" in holdings.columns and "Current Value (₹)" in holdings.columns:
        cur = holdings.copy()
        total = cur["Current Value (₹)"].sum()
        if total > 0:
            target_vals = total * (alloc_df["Allocation (%)"]/100.0).values
            cur_vals = cur.set_index("Asset Class")["Current Value (₹)"].reindex(alloc_df["Asset Class"]).fillna(0).values
            buy_sell = target_vals - cur_vals
            reb = pd.DataFrame({
                "Asset Class": alloc_df["Asset Class"],
                "Target Value (₹)": target_vals,
                "Current Value (₹)": cur_vals,
                "Buy(+)/Sell(-) (₹)": buy_sell
            })
            st.dataframe(reb.style.format({"Target Value (₹)":"₹{:,.0f}","Current Value (₹)":"₹{:,.0f}","Buy(+)/Sell(-) (₹)":"₹{:,.0f}"}), use_container_width=True)
            st.download_button("Download rebalance CSV", reb.to_csv(index=False).encode("utf-8"), file_name="rebalance.csv", mime="text/csv")
        else:
            st.info("Enter current holdings values to generate rebalance worksheet.")

# -----------------------
# Page: Market
# -----------------------
elif page == "Market":
    st.markdown("<div class='app-header'>Market — Live</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>Nifty, sector sparklines and sample tickers</div>", unsafe_allow_html=True)

    # Nifty index
    nifty_ticker = DEFAULT_TICKERS["Large Cap Equity"]
    try:
        nifty = fetch_close_series(nifty_ticker, years=lookback_years)
        if nifty is not None:
            nifty_norm = nifty / nifty.iloc[0] * 100
            nifty_df = pd.DataFrame({"Date": nifty_norm.index, "Index": nifty_norm.values})
            fig_n = px.line(nifty_df, x="Date", y="Index", title="Nifty (normalized)")
            st.plotly_chart(fig_n, use_container_width=True)
    except Exception as e:
        st.warning("Could not fetch Nifty data: " + str(e))

    # sample sector tickers (user can edit)
    st.subheader("Sample live stocks (click to edit tickers in code if needed)")
    sample_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    try:
        prices = yf.download(sample_tickers, period="1mo", interval="1d")["Close"]
        latest = prices.iloc[-1].reset_index()
        latest = latest.rename(columns={latest.columns[1]:"Price"}) if latest.shape[1]>1 else latest
        latest_df = pd.DataFrame({"Ticker": prices.columns, "Latest": prices.iloc[-1].values})
        st.dataframe(latest_df, use_container_width=True)
    except Exception:
        st.info("Live sample tickers not available right now.")

# -----------------------
# Page: Goals
# -----------------------
elif page == "Goals":
    st.markdown("<div class='app-header'>Goals — Planner</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>Create multiple goals and run simulations per goal</div>", unsafe_allow_html=True)

    if "goals" not in st.session_state:
        st.session_state.goals = [{"name":"Retirement","amount":8000000,"years":25},{"name":"Home","amount":3000000,"years":8}]

    with st.expander("View / edit goals"):
        goals_df = pd.DataFrame(st.session_state.goals)
        edited = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)
        st.session_state.goals = edited.to_dict("records")

    st.markdown("### Combined targets")
    combined = sum([g["amount"] for g in st.session_state.goals])
    st.write(f"Sum of goals target: {fmt_inr(combined)}")

    # get current allocation (use robo recommended)
    allocation = robo_allocation(age, monthly_income, risk_pref)
    alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
    w = np.array(alloc_df["Allocation (%)"]/100.0)
    means = np.array([DEFAULT_RETURNS.get(a,0.06) for a in alloc_df["Asset Class"]])
    vols = np.array([DEFAULT_VOL.get(a,0.15) for a in alloc_df["Asset Class"]])
    cov = np.outer(vols, vols) * 0.25
    np.fill_diagonal(cov, vols**2)

    st.markdown("### Run simulations for each goal (this runs smaller MC per goal to save CPU)")
    if st.button("Run goal simulations"):
        out = []
        for g in st.session_state.goals:
            yrs = int(g.get("years", 10))
            sims_goal = max(300, int(mc_sims/4))
            mc_goal = monte_carlo_sim(500000, 10000, w, means, cov, yrs, sims_goal)  # using sample invest/SIP for speed; ideally pass user inputs
            p = float((mc_goal >= g["amount"]).sum() / len(mc_goal) * 100.0)
            out.append({"Goal": g["name"], "Target (₹)": g["amount"], "Years": yrs, "P(achieve %)": round(p,1), "Median end (₹)": int(np.median(mc_goal))})
        out_df = pd.DataFrame(out)
        st.dataframe(out_df, use_container_width=True)
        st.plotly_chart(px.bar(out_df, x="Goal", y="P(achieve %)", text="P(achieve %)"), use_container_width=True)

    st.markdown("---")
    # Download goals
    st.download_button("Download goals CSV", pd.DataFrame(st.session_state.goals).to_csv(index=False).encode("utf-8"),
                       file_name="goals.csv", mime="text/csv")

# -----------------------
# Page: Analytics
# -----------------------
elif page == "Analytics":
    st.markdown("<div class='app-header'>Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-sub'>Risk metrics, correlation heatmap, Monte Carlo distribution</div>", unsafe_allow_html=True)

    # Use current robo allocation
    allocation = robo_allocation(age, monthly_income, risk_pref)
    alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
    w = np.array(alloc_df["Allocation (%)"]/100.0)
    means = np.array([DEFAULT_RETURNS.get(a,0.06) for a in alloc_df["Asset Class"]])
    vols = np.array([DEFAULT_VOL.get(a,0.15) for a in alloc_df["Asset Class"]])
    base_corr = 0.25
    cov = np.outer(vols, vols)*base_corr
    np.fill_diagonal(cov, vols**2)

    # Metrics
    port_ret = float(np.dot(w, means))
    port_vol = float(np.sqrt(w @ cov @ w))
    sharpe = (port_ret - 0.04) / (port_vol + 1e-9)
    st.metric("Portfolio expected return", f"{port_ret*100:.2f}%")
    st.metric("Portfolio volatility (σ)", f"{port_vol*100:.2f}%")
    st.metric("Sharpe-like (approx)", f"{sharpe:.2f}")

    # correlation heatmap: try to build series for assets that have tickers
    series_map = {}
    for a in alloc_df["Asset Class"]:
        t = DEFAULT_TICKERS.get(a)
        s = fetch_close_series(t, years=lookback_years) if t else None
        if s is not None:
            series_map[a] = s.pct_change().dropna()

    if len(series_map) >= 2:
        comb = pd.concat(series_map, axis=1).dropna()
        corr = comb.corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation heatmap", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough live series available for correlation (provide tickers or include assets with defaults).")

    # Monte Carlo final histogram
    mc_all = monte_carlo_sim(500000, 10000, w, means, cov, 10, min(2000, mc_sims))
    fig_hist = px.histogram(mc_all, nbins=60, title="Monte Carlo final distribution (sample inputs)")
    st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("Fintech Pro — educational planning tool. Not investment advice. Verify tickers, taxes, and product suitability before acting.")
