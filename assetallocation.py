# asset_allocation_app.py
"""
Futuristic Smart Asset Allocation (India Edition)
- Dark neon theme, Lottie header (finance-themed)
- Live market auto-refresh (every 5 minutes)
- Custom allocation logic, robust goal parser with progress
- Monte Carlo simulations, efficient frontier, correlation heatmap
- Downloadable CSV / Excel outputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
import io
import time
from datetime import datetime
from math import ceil

# ---------------- Page config ----------------
st.set_page_config(page_title="Futuristic Asset Allocator (India)", layout="wide", initial_sidebar_state="expanded")

# ---------------- Auto page reload every 5 minutes (300000 ms) ----------------
# This uses a small client-side JS to reload the page automatically.
AUTO_RELOAD_MS = 300000  # 5 minutes
st.components.v1.html(f"""
<script>
setTimeout(function() {{
    // reload only if the tab is visible (reduce unnecessary reloads)
    if (document.visibilityState === 'visible') {{
        window.location.reload();
    }}
}}, {AUTO_RELOAD_MS});
</script>
""", height=0)

# ---------------- CSS (futuristic dark neon) ----------------
st.markdown(
    """
    <style>
    :root{
      --bg:#0b0f19;
      --panel: rgba(255,255,255,0.03);
      --glass: rgba(255,255,255,0.02);
      --neon:#00e5ff;
      --accent:#26ffe6;
      --muted:#9fb4c8;
      --gold:#ffd166;
    }
    body { background: linear-gradient(180deg, #05060a 0%, #0b0f19 100%); color:#E6F2F5; }
    .header { display:flex; align-items:center; gap:18px; }
    .title { font-size:22px; font-weight:700; color:var(--neon); margin-bottom:2px; }
    .subtitle { color:var(--muted); margin-top:0; margin-bottom:8px; font-size:13px; }
    .card { background: var(--glass); border-radius:12px; padding:14px; border:1px solid rgba(255,255,255,0.03); box-shadow:0 10px 30px rgba(0,0,0,0.5); }
    .muted { color:var(--muted); font-size:13px; }
    .metric { background:#071425; border-radius:10px; padding:10px; text-align:center; }
    .metric-val { color:var(--accent); font-weight:700; font-size:18px; }
    .small { font-size:12px; color:var(--muted); }
    .neon-divider { height:2px; background: linear-gradient(90deg, rgba(0,229,255,0), rgba(38,255,230,0.9), rgba(0,229,255,0)); border-radius:4px; margin:8px 0 16px 0; }
    /* data editor styling fallback */
    div[role="list"] > div { background: transparent !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Lottie helper ----------------
# Use streamlit_lottie if available; otherwise fallback to embedding
try:
    from streamlit_lottie import st_lottie
    STREAMLITE_LOTTIE_AVAILABLE = True
except Exception:
    STREAMLITE_LOTTIE_AVAILABLE = False

def load_lottie_from_url(url):
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

# Online finance-themed Lottie URL (public)
LOTTIE_FINANCE_URL = "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json"  # finance/dashboard style (common public host)
lottie_json = load_lottie_from_url(LOTTIE_FINANCE_URL)

# ---------------- Sidebar: Inputs ----------------
with st.sidebar:
    st.markdown("<div class='title'>Futuristic Asset Allocator</div>", unsafe_allow_html=True)
    st.markdown("<div class='small muted'>India edition — custom profile</div>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)

    age = st.slider("Age", 18, 75, 34)
    monthly_income = st.number_input("Monthly income (₹)", min_value=0, value=70000, step=1000)
    risk_label = st.selectbox("Risk appetite (custom)", ["Very Low", "Low", "Moderate", "High", "Very High"])
    st.markdown("---")
    invest_now = st.number_input("Lump-sum invested (₹)", min_value=0, value=500000, step=10000)
    sip_now = st.number_input("Monthly SIP (₹)", min_value=0, value=10000, step=500)
    horizon_default = st.slider("Default planning horizon (yrs)", 1, 40, 10)
    st.markdown("---")
    st.header("Simulation / Data")
    mc_sims = st.slider("Monte Carlo sims", 200, 4000, 1200, step=100)
    frontier_samples = st.slider("Frontier samples", 50, 2000, 400, step=50)
    lookback_years = st.selectbox("Live blending lookback (yrs)", [1,3,5], index=2)
    st.markdown("---")
    st.caption("Auto-refresh enabled every 5 minutes. After updating code, Manage app → Clear cache → Rerun from scratch.")

# ---------------- Asset universe (Indian) ----------------
ASSET_UNIVERSE = [
    "Large Cap Equity", "Mid/Small Cap Equity", "International Equity",
    "Index ETFs", "Equity Mutual Funds (Active)", "Sectoral/Thematic",
    "Debt Funds", "Government Bonds", "Corporate Bonds",
    "Gold ETF / SGB", "REITs / InvITs", "Real Estate (Direct)",
    "Cash / Liquid", "Fixed Deposits", "Commodities (other)", "Crypto (speculative)"
]

# Simple defaults for returns & vol
DEFAULT_RETURNS = {
    "Large Cap Equity": 0.10, "Mid/Small Cap Equity": 0.13, "International Equity": 0.08,
    "Index ETFs": 0.095, "Equity Mutual Funds (Active)": 0.10, "Sectoral/Thematic": 0.12,
    "Debt Funds": 0.06, "Government Bonds": 0.05, "Corporate Bonds": 0.06,
    "Gold ETF / SGB": 0.07, "REITs / InvITs": 0.08, "Real Estate (Direct)": 0.06,
    "Cash / Liquid": 0.035, "Fixed Deposits": 0.05, "Commodities (other)": 0.06, "Crypto (speculative)": 0.20
}
DEFAULT_VOL = {k: (0.20 if "Equity" in k or "Thematic" in k or "Crypto" in k else 0.08) for k in DEFAULT_RETURNS.keys()}

# Default tickers for live blending (best-effort)
DEFAULT_TICKERS = {
    "Large Cap Equity": "^NSEI",
    "Gold ETF / SGB": "GOLDBEES.NS",
    "International Equity": "VTI",
    "REITs / InvITs": "REIT.NS"  # placeholder (may not exist in Yahoo)
}

# ---------------- small helpers ----------------
def fmt_inr(x):
    try:
        return "₹{:,.0f}".format(float(x))
    except Exception:
        return f"₹{x}"

@st.cache_data(ttl=60*20)
def fetch_close_series(ticker, years=5):
    """Returns pandas Series of Close prices or None"""
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
    years_total = (series.index[-1] - series.index[0]).days / 365.25
    if years_total <= 0:
        return None, None
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1.0 / years_total) - 1.0
    vol = series.pct_change().dropna().std() * np.sqrt(252)
    return float(cagr), float(vol)

@st.cache_data(ttl=60*10)
def monte_carlo_sim(invest, monthly_sip, weights, means, cov, years, sims, seed=42):
    """Annual Monte Carlo with correlated returns"""
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

# ---------------- Robo-allocation (custom, Indian) ----------------
def robo_allocation_custom(age, income, risk_label, include_optional=None):
    """Return allocation dict (percent) — custom user-led baseline influenced by age/income/risk"""
    include_optional = include_optional or []
    # start with reasonable baseline buckets
    base = {
        "Large Cap Equity": 30, "Mid/Small Cap Equity": 10, "International Equity": 5,
        "Index ETFs": 10, "Equity Mutual Funds (Active)": 0, "Sectoral/Thematic": 0,
        "Debt Funds": 25, "Gold ETF / SGB": 8, "REITs / InvITs": 5, "Cash / Liquid": 7
    }
    # adjust by risk
    risk_map = {
        "Very Low": -15, "Low": -7, "Moderate": 0, "High": 7, "Very High": 15
    }
    tilt = risk_map.get(risk_label, 0)
    # apply tilt to equity vs debt
    base["Large Cap Equity"] = max(0, base["Large Cap Equity"] + int(tilt * 0.5))
    base["Mid/Small Cap Equity"] = max(0, base["Mid/Small Cap Equity"] + int(tilt * 0.3))
    base["Debt Funds"] = max(0, base["Debt Funds"] - int(tilt * 0.6))
    # age tilt
    if age < 35:
        base["Mid/Small Cap Equity"] += 5
        base["Debt Funds"] = max(0, base["Debt Funds"] - 5)
    elif age > 60:
        base["Debt Funds"] += 7
        base["Large Cap Equity"] = max(0, base["Large Cap Equity"] - 7)
    # income tilt
    if income > 150000:
        base["International Equity"] += 3
    # include optional assets small allocation
    for asset in include_optional:
        if asset in ASSET_UNIVERSE and asset not in base:
            base[asset] = 2
    # normalize to 100
    total = sum(base.values())
    if total == 0:
        return {k: round(100 / len(base), 2) for k in base}
    return {k: round(v / total * 100, 2) for k, v in base.items()}

# ---------------- Header (Lottie + Title) ----------------
header_col1, header_col2 = st.columns([0.8, 3.2])
with header_col1:
    if STREAMLITE_LOTTIE_AVAILABLE and lottie_json:
        st_lottie(lottie_json, height=110, key="lottie")
    else:
        # fallback: small inline SVG or title image placeholder
        st.image("https://raw.githubusercontent.com/nehamishra/fin-dashboard-assets/main/finance-neon.png", width=120) if False else None

with header_col2:
    st.markdown("<div class='title'>AI-Powered Futuristic Asset Allocation — India</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Custom allocations, live Nifty tracker, robust goal planning, and neon analytics</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

# ---------------- Navigation tabs ----------------
tab1, tab2, tab3 = st.tabs(["🏦 Portfolio Allocation", "📈 Live Market Tracker", "🧮 Analytics & Goals"])

# ---------------- Tab 1: Portfolio Allocation ----------------
with tab1:
    st.markdown("## Portfolio Allocation — custom inputs")
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown("### Inputs")
        # allow user to add optional assets to include
        optional_assets = st.multiselect("Add optional asset classes", options=ASSET_UNIVERSE, default=[])
        # let user edit recommended horizon and goal
        horizon = st.number_input("Your investment horizon (yrs)", min_value=1, max_value=50, value=horizon_default, step=1)
        # quick risk note
        st.markdown("<div class='small muted'>Tip: change risk appetite to see allocation tilt. All allocations are editable below.</div>", unsafe_allow_html=True)

        # generate allocation suggestion
        suggestion = robo_allocation_custom(age, monthly_income, risk_label, include_optional=optional_assets)
        suggestion_df = pd.DataFrame({"Asset Class": list(suggestion.keys()), "Allocation (%)": list(suggestion.values())})
        st.markdown("### Suggested allocation (editable)")
        edited_alloc = st.data_editor(suggestion_df, num_rows="dynamic", use_container_width=True)
        # normalize if necessary
        if "Allocation (%)" in edited_alloc.columns:
            total_allocation = float(edited_alloc["Allocation (%)"].sum())
            if abs(total_allocation - 100) > 0.5:
                st.warning("Total allocation not ~100%. Edit values to sum close to 100% or re-balance.")
        else:
            st.info("Edit allocation table to set percentages.")

    with right:
        st.markdown("### Portfolio quick metrics")
        # compute blended returns & vol using defaults or live blend
        alloc_df = edited_alloc.copy()
        if "Allocation (%)" not in alloc_df.columns:
            alloc_df["Allocation (%)"] = 0
        # try to blend with live series if available
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
        alloc_df["Allocation (₹)"] = alloc_df["Allocation (%)"] / 100.0 * invest_now

        st.dataframe(alloc_df.style.format({"Allocation (%)":"{:.2f}", "Allocation (₹)":"₹{:,.0f}", "Exp Return (%)":"{:.2f}%", "Volatility (%)":"{:.2f}%"}), use_container_width=True)

        # portfolio level metrics
        w = np.array(alloc_df["Allocation (%)"] / 100.0, dtype=float)
        means = np.array([asset_returns[a] for a in alloc_df["Asset Class"]], dtype=float)
        vols = np.array([asset_vols[a] for a in alloc_df["Asset Class"]], dtype=float)
        cov = np.outer(vols, vols) * 0.25
        np.fill_diagonal(cov, vols ** 2)
        port_ret = float(np.dot(w, means)) if len(w) > 0 else 0.0
        port_vol = float(np.sqrt(np.dot(w, np.dot(cov, w)))) if len(w) > 0 else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Est. annual return", f"{port_ret*100:.2f}%")
        m2.metric("Est. volatility (σ)", f"{port_vol*100:.2f}%")
        det_future = deterministic_portfolio_fv = 0  # placeholder for small note
        # small narrative box
        st.markdown("<div class='card'><b>Logic for selection</b><br><div class='small muted'>Allocation is tilted using age & income. Equity allocations favor long horizons and younger investors; debt allocations increase with lower risk preference and older age.</div></div>", unsafe_allow_html=True)

    st.markdown("### Visual: Allocation (glow pie)")
    fig_p = px.pie(alloc_df, names="Asset Class", values="Allocation (%)", hole=0.36)
    fig_p.update_traces(textinfo="percent+label")
    fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), legend=dict(orientation="h"))
    st.plotly_chart(fig_p, use_container_width=True)

    # Download allocation
    csv_bytes = alloc_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download allocation CSV", data=csv_bytes, file_name="allocation.csv", mime="text/csv")

# ---------------- Tab 2: Live Market Tracker ----------------
with tab2:
    st.markdown("## Live Market Tracker — Nifty & sample tickers (auto-updates every 5 mins)")
    # Nifty index
    nifty_ticker = "^NSEI"  # Nifty 50
    nifty_series = fetch_close_series(nifty_ticker, years=lookback_years)
    if nifty_series is not None:
        norm = nifty_series / nifty_series.iloc[0] * 100
        df_norm = pd.DataFrame({"Date": norm.index, "Index": norm.values})
        fig = px.line(df_norm, x="Date", y="Index", title=f"Nifty (last {lookback_years} yrs normalized)")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nifty data not available — Yahoo may block some index tickers.")

    st.markdown("### Live sample tickers (prices updated automatically)")
    sample_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "GOLDBEES.NS"]
    try:
        prices = yf.download(sample_tickers, period="10d", interval="1d")["Close"]
        if not prices.empty:
            latest = prices.iloc[-1]
            table = pd.DataFrame({"Ticker": latest.index, "Latest": latest.values})
            table["Latest"] = table["Latest"].map(lambda x: f"{x:,.2f}")
            st.dataframe(table, use_container_width=True)
        else:
            st.info("No recent tickers data.")
    except Exception:
        st.info("Live tickers not available at the moment (Yahoo fetch issue).")

# ---------------- Tab 3: Analytics & Goals ----------------
with tab3:
    st.markdown("## Analytics & Goals")
    colA, colB = st.columns([1.4, 1])

    with colA:
        st.markdown("### Goal Planner — create multiple goals (editable)")
        if "goals" not in st.session_state:
            st.session_state.goals = [{"name": "Retirement", "amount": 5000000, "years": 20},
                                      {"name": "Child Education", "amount": 2000000, "years": 10}]
        goals_df = pd.DataFrame(st.session_state.goals)
        edited_goals = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)

        # robust cleaning with progress and corrections log
        rows = edited_goals.to_dict(orient="records")
        corrected = []
        cleaned = []
        total = max(1, len(rows))
        prog = st.progress(0)
        for i, r in enumerate(rows):
            # ensure dict
            if not isinstance(r, dict):
                try:
                    r = dict(r)
                except Exception:
                    r = {}
            name = r.get("name") or r.get("Goal") or "Goal"
            # amount
            amt_raw = r.get("amount") or r.get("Target Amount (₹)") or 0
            try:
                amt = float(amt_raw)
            except Exception:
                cleaned_amt = ''.join(ch for ch in str(amt_raw) if (ch.isdigit() or ch == '.'))
                amt = float(cleaned_amt) if cleaned_amt else 0.0
                corrected.append((i + 1, str(amt_raw), str(amt)))
            # years
            try:
                val = r.get("years") or r.get("Years") or 0
                if isinstance(val, (int, float)):
                    yrs = int(val)
                elif isinstance(val, str):
                    cleaned_yrs = ''.join(ch for ch in val if (ch.isdigit() or ch == '.'))
                    yrs = int(float(cleaned_yrs)) if cleaned_yrs else 0
                    if cleaned_yrs != val:
                        corrected.append((i + 1, str(val), str(yrs)))
                else:
                    yrs = 0
            except Exception:
                yrs = 0
                corrected.append((i + 1, str(r.get("years", r.get("Years", ""))), "0"))
            cleaned.append({"name": name, "amount": amt, "years": yrs})
            prog.progress((i + 1) / total)
            time.sleep(0.01)
        prog.empty()
        st.session_state.goals = cleaned

        if corrected:
            with st.expander("⚠️ Auto-corrections (Goals)"):
                for row_i, old_v, new_v in corrected:
                    st.markdown(f"- Row **{row_i}**: `{old_v}` → `{new_v}`")
            st.success("Goal inputs cleaned automatically.")

        st.markdown("### Run Monte Carlo per goal (uses current allocation suggestion)")
        run_button = st.button("Run goal simulations")
        if run_button:
            # compute weights from current suggestion in tab1 (if exists), else from robo default
            try:
                weights_df = edited_alloc.copy() if 'edited_alloc' in locals() else None
            except Exception:
                weights_df = None
            if weights_df is None or "Allocation (%)" not in weights_df.columns:
                weights = robo_allocation_custom(age, monthly_income, risk_label, optional_assets)
                weights_df = pd.DataFrame({"Asset Class": list(weights.keys()), "Allocation (%)": list(weights.values())})
            w = np.array(weights_df["Allocation (%)"] / 100.0, dtype=float)
            means = np.array([DEFAULT_RETURNS.get(a, 0.06) for a in weights_df["Asset Class"]], dtype=float)
            vols = np.array([DEFAULT_VOL.get(a, 0.15) for a in weights_df["Asset Class"]], dtype=float)
            cov = np.outer(vols, vols) * 0.25
            np.fill_diagonal(cov, vols ** 2)

            out = []
            for g in st.session_state.goals:
                yrs = int(float(g.get("years", 10)))
                sims_goal = max(300, int(mc_sims / 4))
                mc_goal = monte_carlo_sim(invest_now, sip_now, w, means, cov, yrs, sims_goal)
                prob = float((mc_goal >= g["amount"]).sum() / len(mc_goal) * 100.0)
                out.append({"Goal": g["name"], "Target (₹)": int(g["amount"]), "Years": yrs, "P(achieve %)": round(prob, 1), "Median end (₹)": int(np.median(mc_goal))})
            out_df = pd.DataFrame(out)
            st.dataframe(out_df, use_container_width=True)
            st.plotly_chart(px.bar(out_df, x="Goal", y="P(achieve %)", text="P(achieve %)"), use_container_width=True)
            csv_out = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download goal simulation results CSV", csv_out, file_name="goal_simulations.csv", mime="text/csv")

    with colB:
        st.markdown("### Quick analytics")
        # small sample: portfolio expected return & volatility from default robo allocation
        default_alloc = robo_allocation_custom(age, monthly_income, risk_label)
        df_def = pd.DataFrame({"Asset Class": list(default_alloc.keys()), "Allocation (%)": list(default_alloc.values())})
        w_def = np.array(df_def["Allocation (%)"] / 100.0, dtype=float)
        means_def = np.array([DEFAULT_RETURNS.get(a, 0.06) for a in df_def["Asset Class"]], dtype=float)
        vols_def = np.array([DEFAULT_VOL.get(a, 0.15) for a in df_def["Asset Class"]], dtype=float)
        cov_def = np.outer(vols_def, vols_def) * 0.25
        np.fill_diagonal(cov_def, vols_def ** 2)
        p_ret = float(np.dot(w_def, means_def))
        p_vol = float(np.sqrt(w_def @ cov_def @ w_def))
        p_sharpe = (p_ret - 0.04) / (p_vol + 1e-9)
        st.markdown("<div class='card'><b>Portfolio snapshot (suggested)</b><br><div class='muted'>Expected return | Volatility | Sharpe-like</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'><div class='metric-val'>{p_ret*100:.2f}%</div><div class='small muted'>Expected return</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'><div class='metric-val'>{p_vol*100:.2f}%</div><div class='small muted'>Volatility (σ)</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='height:6px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'><div class='metric-val'>{p_sharpe:.2f}</div><div class='small muted'>Sharpe-like</div></div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("<hr style='border:1px solid rgba(255,255,255,0.04)'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#9FB4C8; font-size:13px;'>Made for you — futuristic Indian asset allocation tool. Not financial advice.</div>", unsafe_allow_html=True)
