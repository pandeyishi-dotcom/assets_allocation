# app.py
"""
Fintech Pro Edition — Robust Asset Allocator & Robo-Advisor
- Multi-page Streamlit dashboard (Overview, Portfolio, Market, Goals, Analytics)
- Live data via yfinance (optional, with fallbacks)
- Smart robo-allocation tuned by age/income/risk
- Multi-goal planner, deterministic projection & Monte Carlo simulation
- Efficient frontier (random portfolios), correlation heatmap
- Rebalancing worksheet + CSV downloads
- Defensive code to avoid NameError/TypeError and friendly UX for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from math import ceil
from datetime import datetime

# ------------------- Page config -------------------
st.set_page_config(page_title="Fintech Pro Asset Allocator", layout="wide", initial_sidebar_state="expanded")

# ------------------- CSS / theme -------------------
st.markdown(
    """
    <style>
    :root {
      --bg:#071425;
      --glass: rgba(255,255,255,0.02);
      --muted:#9FB4C8;
      --accent:#6FF0B0;
    }
    .stApp { background: linear-gradient(180deg,#071425,#032b3a); color: #E8F0F2;}
    .card { background: var(--glass); padding:12px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); }
    .app-title { color:var(--accent); font-weight:700; font-size:20px; margin-bottom:4px; }
    .muted { color:var(--muted); font-size:13px; }
    .metric { background:#071425; padding:8px; border-radius:8px; text-align:center; }
    .metric-value { color:var(--accent); font-weight:700; font-size:16px; }
    </style>
    """, unsafe_allow_html=True
)

def fmt_inr(x):
    return "₹{:,.0f}".format(x)

# ------------------- Sidebar inputs -------------------
st.sidebar.markdown("<div class='app-title'>Fintech Pro — Asset Allocator</div>", unsafe_allow_html=True)
age = st.sidebar.slider("Age", 18, 75, 34)
monthly_income = st.sidebar.number_input("Monthly income (₹)", min_value=0, value=70000, step=1000)
risk_pref = st.sidebar.selectbox("Risk appetite", ["Low", "Moderate", "High"])
st.sidebar.markdown("---")
st.sidebar.subheader("Computation limits (Streamlit Cloud)")
mc_sims = st.sidebar.slider("Monte Carlo sims", 200, 4000, 1200, step=100)
frontier_samples = st.sidebar.slider("Efficient frontier samples", 50, 2000, 400, step=50)
lookback_years = st.sidebar.selectbox("Lookback (yrs) for live CAGR", [1,3,5], index=2)
st.sidebar.markdown("---")
st.sidebar.caption("After updating code, Manage app → Clear cache → Rerun from scratch (Streamlit Cloud).")

page = st.sidebar.radio("Navigate", ["Overview", "Portfolio", "Market", "Goals", "Analytics"])

# ------------------- Asset universe & defaults -------------------
ASSET_UNIVERSE = [
    "Large Cap Equity","Mid/Small Cap Equity","International Equity","Index ETFs",
    "Active Equity Funds","Sectoral/Thematic Funds","Debt Funds","Government Bonds",
    "Corporate Bonds","Gold ETF","REITs/InvITs","Real Estate (Direct)","Liquid/Cash",
    "Fixed Deposits","Commodities (other)","Crypto (speculative)","Insurance-linked (ULIPs/Annuities)"
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

# ------------------- Utility: fetch & compute -------------------
@st.cache_data(ttl=60*30)
def fetch_close_series(ticker, years=5):
    if not ticker or ticker.strip()=="":
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
    total_years = (series.index[-1] - series.index[0]).days / 365.25
    if total_years <= 0:
        return None, None
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1.0/total_years) - 1.0
    vol = series.pct_change().dropna().std() * np.sqrt(252)
    return float(cagr), float(vol)

# Monte Carlo (annual correlated returns) - defensive
@st.cache_data(ttl=60*10)
def monte_carlo_sim(invest, monthly_sip, weights, means, cov, years, sims, seed=42):
    # ensure numeric types
    years = int(float(years))
    sims = int(sims)
    weights = np.array(weights, dtype=float)
    means = np.array(means, dtype=float)
    cov = np.array(cov, dtype=float)
    n = len(weights)
    # ensure valid covariance matrix
    cov = cov + np.eye(n) * 1e-12
    L = np.linalg.cholesky(cov)
    rng = np.random.default_rng(seed)
    final = np.zeros(sims)
    annual_sip = float(monthly_sip) * 12.0
    base_alloc = weights * invest
    for s in range(sims):
        vals = base_alloc.copy()
        for y in range(int(years)):
            z = rng.normal(size=n)
            ret = means + L @ z
            vals = vals * (1 + ret)
            if annual_sip > 0:
                vals = vals + annual_sip * weights
        final[s] = vals.sum()
    return final

def fv_lumpsum(pv, r, yrs):
    yrs = int(float(yrs))
    return pv * ((1 + r) ** yrs)

def fv_sip_monthly(monthly, r, yrs):
    yrs = int(float(yrs))
    r_m = (1 + r) ** (1/12) - 1
    n = yrs * 12
    if r_m == 0:
        return monthly * n
    return monthly * (((1 + r_m) ** n - 1) / r_m) * (1 + r_m)

def deterministic_portfolio_fv(invest, monthly, weights, means, years):
    total = 0.0
    years = int(float(years))
    for i, w in enumerate(weights):
        r = float(means[i])
        pv = invest * w
        total += fv_lumpsum(pv, r, years) + fv_sip_monthly(monthly * w, r, years)
    return total

# ------------------- Robo allocation -------------------
BASELINE = {
    "Low": {"Large Cap Equity":25,"Mid/Small Cap Equity":5,"International Equity":5,"Index ETFs":10,"Debt Funds":35,"Gold ETF":10,"Liquid/Cash":10},
    "Moderate":{"Large Cap Equity":35,"Mid/Small Cap Equity":10,"International Equity":8,"Index ETFs":10,"Debt Funds":20,"Gold ETF":7,"Liquid/Cash":10},
    "High":{"Large Cap Equity":45,"Mid/Small Cap Equity":15,"International Equity":10,"Index ETFs":8,"Debt Funds":10,"Gold ETF":5,"Liquid/Cash":5,"Crypto (speculative)":2}
}

def robo_allocation(age, income, risk_label, include_assets=None, normalize=True):
    base = BASELINE.get({"Low":"Low","Moderate":"Moderate","High":"High"}[risk_label]).copy()
    include_assets = include_assets or []
    for a in include_assets:
        if a not in base and a in ASSET_UNIVERSE:
            base[a] = 2.0
    # age tilt
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
    if normalize:
        total = sum(base.values())
        if total == 0:
            return {k: round(100/len(base), 2) for k in base}
        return {k: round(v/total * 100, 2) for k, v in base.items()}
    return base

# ------------------- Page: Overview -------------------
if page == "Overview":
    st.markdown("<div class='app-title'>Portfolio — Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Quick summary and recommended allocation</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        invest = st.number_input("Current invested (lump-sum ₹)", min_value=0, value=500000, step=10000)
    with col2:
        sip = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500)
    with col3:
        horizon = st.slider("Planning horizon (years)", 1, 40, 10)

    allocation = robo_allocation(age, monthly_income, risk_pref)
    alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
    # editable allocation
    st.markdown("### Recommended Allocation (editable)")
    alloc_df = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)
    # normalization warning
    if abs(alloc_df["Allocation (%)"].sum() - 100) > 0.5:
        st.warning("Total allocation is not ~100%. Edit table to normalize.")

    # compute returns/vols (blend live where possible)
    asset_returns = {}
    asset_vols = {}
    for a in alloc_df["Asset Class"]:
        t = DEFAULT_TICKERS.get(a)
        series = fetch_close_series(t, years=lookback_years) if t else None
        if series is not None:
            c, v = compute_cagr_and_vol(series)
            if c is not None:
                asset_returns[a] = 0.7 * c + 0.3 * DEFAULT_RETURNS.get(a, 0.06)
                asset_vols[a] = 0.7 * v + 0.3 * DEFAULT_VOL.get(a, 0.15)
            else:
                asset_returns[a] = DEFAULT_RETURNS.get(a, 0.06)
                asset_vols[a] = DEFAULT_VOL.get(a, 0.15)
        else:
            asset_returns[a] = DEFAULT_RETURNS.get(a, 0.06)
            asset_vols[a] = DEFAULT_VOL.get(a, 0.15)

    alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x, 0.06) * 100)
    alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x, 0.15) * 100)
    alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"] / 100.0 * invest

    st.dataframe(alloc_df.style.format({"Allocation (%)":"{:.2f}","Allocation (₹)":"₹{:,.0f}","Exp Return (%)":"{:.2f}%","Volatility (%)":"{:.2f}%"}), use_container_width=True)

    # portfolio metrics (ensure these variables always exist)
    w = np.array(alloc_df["Allocation (%)"] / 100.0, dtype=float)
    means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]], dtype=float)
    vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]], dtype=float)
    base_corr = 0.25
    cov = np.outer(vols, vols) * base_corr
    np.fill_diagonal(cov, vols ** 2)
    port_return = float(np.dot(w, means))
    port_vol = float(np.sqrt(w @ cov @ w))

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected annual return", f"{port_return*100:.2f}%")
    k2.metric("Estimated volatility (σ)", f"{port_vol*100:.2f}%")
    det_value = deterministic_portfolio_fv(invest, sip, w, means, horizon)
    k3.metric(f"Deterministic value ({horizon}y)", fmt_inr(det_value))
    mc_sample = monte_carlo_sim(invest, sip, w, means, cov, horizon, min(1200, mc_sims))
    k4.metric("Median MC end", fmt_inr(float(np.median(mc_sample))))

    # pie chart
    pie = px.pie(alloc_df, names="Asset Class", values="Allocation (%)", hole=0.36,
                 color_discrete_sequence=px.colors.sequential.Tealgrn)
    pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(pie, use_container_width=True)

# ------------------- Page: Portfolio -------------------
elif page == "Portfolio":
    st.markdown("<div class='app-title'>Portfolio — Detailed</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Efficient frontier, risk-return and rebalance worksheet</div>", unsafe_allow_html=True)
    # user's holdings editor (optional)
    st.subheader("Current holdings (optional)")
    holdings = st.data_editor(pd.DataFrame(columns=["Asset Class", "Current Value (₹)"]), num_rows="dynamic", use_container_width=True)

    # show allocation (robo baseline editable)
    allocation = robo_allocation(age, monthly_income, risk_pref)
    alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
    st.markdown("### Edit simulated allocation")
    alloc_df = st.data_editor(alloc_df, num_rows="dynamic", use_container_width=True)

    # compute basic returns/vols for charting
    asset_returns = {a: DEFAULT_RETURNS.get(a, 0.06) for a in alloc_df["Asset Class"]}
    asset_vols = {a: DEFAULT_VOL.get(a, 0.15) for a in alloc_df["Asset Class"]}
    alloc_df["Exp Return (%)"] = alloc_df["Asset Class"].map(lambda x: asset_returns.get(x) * 100)
    alloc_df["Volatility (%)"] = alloc_df["Asset Class"].map(lambda x: asset_vols.get(x) * 100)

    # prepare efficient frontier samples
    weights = np.array(alloc_df["Allocation (%)"] / 100.0, dtype=float)
    means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]], dtype=float)
    vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]], dtype=float)
    base_corr = 0.25
    cov = np.outer(vols, vols) * base_corr
    np.fill_diagonal(cov, vols ** 2)

    def random_weights(n, samples):
        r = np.random.random((samples, n))
        r = r / r.sum(axis=1)[:, None]
        return r

    samples = min(max(50, frontier_samples), 3000)
    rws = random_weights(len(weights), samples)
    ef_ret = rws.dot(means)
    ef_vol = np.sqrt(np.einsum('ij,jk,ik->i', rws, cov, rws))
    ef_sh = (ef_ret - 0.04) / (ef_vol + 1e-9)

    # compute port_return/port_vol defensively for marker
    port_return = float(np.dot(weights, means))
    port_vol = float(np.sqrt(weights @ cov @ weights))

    ef_df = pd.DataFrame({"Return": ef_ret * 100, "Volatility": ef_vol * 100, "Sharpe": ef_sh})
    fig_ef = px.scatter(ef_df, x="Volatility", y="Return", color="Sharpe", color_continuous_scale="Viridis", title="Efficient frontier (random samples)")
    fig_ef.add_trace(go.Scatter(x=[port_vol * 100], y=[port_return * 100], mode="markers+text",
                                marker=dict(size=12, color="gold"), text=["Current"], textposition="top center"))
    fig_ef.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_ef, use_container_width=True)

    st.markdown("### Asset Risk vs Return")
    fig_sc = px.scatter(alloc_df, x="Volatility (%)", y="Exp Return (%)", size="Allocation (%)", text="Asset Class")
    fig_sc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.subheader("Rebalancing worksheet")
    if not holdings.empty and "Asset Class" in holdings.columns and "Current Value (₹)" in holdings.columns:
        cur = holdings.copy()
        total = cur["Current Value (₹)"].sum()
        if total > 0:
            target_vals = total * (alloc_df["Allocation (%)"] / 100.0).values
            cur_vals = cur.set_index("Asset Class")["Current Value (₹)"].reindex(alloc_df["Asset Class"]).fillna(0).values
            buy_sell = target_vals - cur_vals
            reb_df = pd.DataFrame({
                "Asset Class": alloc_df["Asset Class"],
                "Target Value (₹)": target_vals,
                "Current Value (₹)": cur_vals,
                "Buy(+)/Sell(-) (₹)": buy_sell
            })
            st.dataframe(reb_df.style.format({"Target Value (₹)":"₹{:,.0f}","Current Value (₹)":"₹{:,.0f}","Buy(+)/Sell(-) (₹)":"₹{:,.0f}"}), use_container_width=True)
            st.download_button("Download rebalance CSV", reb_df.to_csv(index=False).encode("utf-8"), file_name="rebalance.csv", mime="text/csv")
        else:
            st.info("Add holdings values to generate rebalance worksheet.")

# ------------------- Page: Market -------------------
elif page == "Market":
    st.markdown("<div class='app-title'>Market — Live</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Nifty and sample tickers (live via Yahoo, optional)</div>", unsafe_allow_html=True)

    nifty_ticker = DEFAULT_TICKERS.get("Large Cap Equity")
    if nifty_ticker:
        series = fetch_close_series(nifty_ticker, years=lookback_years)
        if series is not None:
            norm = series / series.iloc[0] * 100
            df_norm = pd.DataFrame({"Date": norm.index, "Index": norm.values})
            fig = px.line(df_norm, x="Date", y="Index", title="Nifty (normalized)")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nifty series not available (ticker may be blocked by Yahoo).")

    st.markdown("### Sample live stocks")
    sample_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
    try:
        prices = yf.download(sample_tickers, period="1mo", interval="1d")["Close"]
        if not prices.empty:
            latest = prices.iloc[-1]
            table = pd.DataFrame({"Ticker": latest.index, "Latest": latest.values})
            st.dataframe(table, use_container_width=True)
        else:
            st.info("No recent data for sample tickers.")
    except Exception:
        st.info("Live sample tickers not available right now (Yahoo fetch failed).")

# ------------------- Page: Goals -------------------
elif page == "Goals":
    st.markdown("<div class='app-title'>Goals — Planner</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Create multiple goals, run deterministic and Monte Carlo projections</div>", unsafe_allow_html=True)

    if "goals" not in st.session_state:
        st.session_state.goals = [{"name":"Retirement","amount":8000000,"years":25},{"name":"Home","amount":3000000,"years":8}]

    with st.expander("View / edit goals"):
        goals_df = pd.DataFrame(st.session_state.goals)
        edited_goals = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)
        # coerce numeric types & store back
        for r in edited_goals.to_dict("records"):
            try:     val = r.get("years") or r.get("Years") or 0     r["years"] = int(float(val)) if str(val).replace('.', '', 1).isdigit() else 0 except Exception:     r["years"] = 0
        # normalize keys to expected names
        st.session_state.goals = []
        for row in edited_goals.to_dict("records"):
            name = row.get("name") or row.get("Goal") or "Goal"
            amt = float(row.get("amount") or row.get("Target Amount (₹)") or 0)
            yrs = int(float(row.get("years") or row.get("Years") or 0))
            st.session_state.goals.append({"name": name, "amount": amt, "years": yrs})

    st.markdown("### Combined target")
    combined_target = sum(g["amount"] for g in st.session_state.goals)
    st.write(f"Combined target: {fmt_inr(combined_target)}")

    # use robo allocation for weights
    allocation = robo_allocation(age, monthly_income, risk_pref)
    alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
    w = np.array(alloc_df["Allocation (%)"] / 100.0, dtype=float)
    means = np.array([DEFAULT_RETURNS.get(a, 0.06) for a in alloc_df["Asset Class"]], dtype=float)
    vols = np.array([DEFAULT_VOL.get(a, 0.15) for a in alloc_df["Asset Class"]], dtype=float)
    cov = np.outer(vols, vols) * 0.25
    np.fill_diagonal(cov, vols ** 2)

    st.markdown("### Run goal simulations (smaller MC per goal to save CPU)")
    if st.button("Run goal simulations"):
        out = []
        for g in st.session_state.goals:
            years_g = int(float(g.get("years", g.get("Years", 10))))
            sims_goal = max(300, int(mc_sims / 4))
            mc_goal = monte_carlo_sim(500000, 10000, w, means, cov, years_g, sims_goal)  # sample invest & SIP
            p = float((mc_goal >= g["amount"]).sum() / len(mc_goal) * 100.0)
            out.append({"Goal": g["name"], "Target (₹)": g["amount"], "Years": years_g, "P(achieve %)": round(p, 1), "Median end (₹)": int(np.median(mc_goal))})
        st.dataframe(pd.DataFrame(out), use_container_width=True)
        st.plotly_chart(px.bar(pd.DataFrame(out), x="Goal", y="P(achieve %)", text="P(achieve %)"), use_container_width=True)

    st.download_button("Download goals CSV", pd.DataFrame(st.session_state.goals).to_csv(index=False).encode("utf-8"), file_name="goals.csv", mime="text/csv")

# ------------------- Page: Analytics -------------------
elif page == "Analytics":
    st.markdown("<div class='app-title'>Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Risk metrics, correlation heatmap and Monte Carlo</div>", unsafe_allow_html=True)

    # compute portfolio from robo allocation
    allocation = robo_allocation(age, monthly_income, risk_pref)
    alloc_df = pd.DataFrame({"Asset Class": list(allocation.keys()), "Allocation (%)": list(allocation.values())})
    w = np.array(alloc_df["Allocation (%)"]/100.0, dtype=float)
    means = np.array([DEFAULT_RETURNS.get(a,0.06) for a in alloc_df["Asset Class"]], dtype=float)
    vols = np.array([DEFAULT_VOL.get(a,0.15) for a in alloc_df["Asset Class"]], dtype=float)
    cov = np.outer(vols, vols) * 0.25
    np.fill_diagonal(cov, vols**2)

    port_ret = float(np.dot(w, means))
    port_vol = float(np.sqrt(w @ cov @ w))
    sharpe = (port_ret - 0.04) / (port_vol + 1e-9)

    st.metric("Portfolio expected return", f"{port_ret*100:.2f}%")
    st.metric("Portfolio volatility (σ)", f"{port_vol*100:.2f}%")
    st.metric("Sharpe-like (approx)", f"{sharpe:.2f}")

    # correlation heatmap using available tickers
    series_map = {}
    for a in alloc_df["Asset Class"]:
        t = DEFAULT_TICKERS.get(a)
        s = fetch_close_series(t, years=lookback_years) if t else None
        if s is not None:
            series_map[a] = s.pct_change().dropna()

    if len(series_map) >= 2:
        comb = pd.concat(series_map, axis=1).dropna()
        comb.columns = list(series_map.keys())
        corr = comb.corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation (returns)")
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Not enough live series for correlation heatmap (add tickers).")

    st.markdown("### Monte Carlo sample distribution (10y, sample invest)")
    mc_all = monte_carlo_sim(500000, 10000, w, means, cov, 10, min(2000, mc_sims))
    fig_mc = px.histogram(mc_all, nbins=60, title="Monte Carlo final distribution (sample inputs)")
    st.plotly_chart(fig_mc, use_container_width=True)

# ------------------- Footer -------------------
st.markdown("---")
st.caption("Fintech Pro — educational planning tool. Not financial advice. Verify tickers, taxes and product suitability before acting.")
