# asset_allocation_app.py
"""
Futuristic AI Asset Allocation Dashboard — India Edition
- Dark neon theme, Lottie header (finance-themed)
- Auto-refresh market data every 5 minutes (client-side JS)
- Editable persistent watchlist (st.session_state)
- Live ticker marquee and neon sparkline cards
- Portfolio Allocation, Goals, Analytics tabs (simplified)
- Defensive: falls back if yfinance/index tickers blocked
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
}
body { background: linear-gradient(180deg,#05060a,#071425); color:#EAF6F8; }
.card { background: var(--glass); border-radius:12px; padding:14px; border:1px solid rgba(255,255,255,0.03); box-shadow: 0 8px 30px rgba(0,0,0,0.6);}
.title { color:var(--neon-cyan); font-weight:700; font-size:22px; margin:0; }
.subtitle { color:var(--muted); margin:0; font-size:13px; }
.neon-divider { height:3px; background: linear-gradient(90deg, rgba(0,229,255,0), rgba(38,255,230,0.9), rgba(255,209,102,0.4)); border-radius:4px; margin:10px 0 18px 0; }
.small { color:var(--muted); font-size:13px; }
.ticker-slab { background: linear-gradient(90deg, rgba(255,255,255,0.01), rgba(255,255,255,0.02)); padding:8px; border-radius:8px; }
.marquee { overflow:hidden; white-space:nowrap; }
.marquee-inner { display:inline-block; animation:scroll 24s linear infinite; }
@keyframes scroll { 0% { transform: translateX(100%);} 100% { transform: translateX(-100%);} }
</style>
""", unsafe_allow_html=True)

# ---------------- Lottie helper (optional) ----------------
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

# ---------------- Watchlist (persistent in session_state) ----------------
if "watchlist" not in st.session_state:
    # sensible default set (Nifty may be blocked; keep equities)
    st.session_state.watchlist = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "GOLDBEES.NS", "ICICIBANK.NS"]

# small helper functions
@st.cache_data(ttl=60*10)
def fetch_close(ticker, period="1y", interval="1d"):
    """Return Series of close prices or None."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        if "Close" not in df.columns:
            return None
        return df["Close"].dropna()
    except Exception:
        return None

def fmt_inr(x):
    try:
        return "₹{:,.0f}".format(float(x))
    except Exception:
        return str(x)

# ---------------- Header ----------------
h1, h2 = st.columns([0.7, 3.3])
with h1:
    if LOTTIE_OK and lottie_json:
        try:
            st_lottie(lottie_json, height=110, key="lottie")
        except Exception:
            st.image("https://raw.githubusercontent.com/nehamishra/fin-dashboard-assets/main/finance-neon.png", width=110)
    else:
        # minimal icon fallback (no external asset assumed)
        st.markdown("<div style='width:110px;height:110px;background:linear-gradient(180deg,#00121a,#002b2b);border-radius:12px'></div>", unsafe_allow_html=True)
with h2:
    st.markdown("<div class='title'>AI Futuristic Asset Allocator — India</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Live watchlist, custom allocation suggestions, goal planning, efficient frontier</div>", unsafe_allow_html=True)

st.markdown("<div class='neon-divider'></div>", unsafe_allow_html=True)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🏦 Portfolio Allocation", "📡 Live Market (AI) Tracker", "🧮 Analytics & Goals"])

# ---------------- Tab 1: Portfolio Allocation (simplified) ----------------
with tab1:
    st.markdown("## Portfolio Allocation — custom & editable")
    c1, c2 = st.columns([1, 1.6])

    with c1:
        st.markdown("### Inputs")
        optional_assets = st.multiselect("Optional asset classes to include", options=[
            "Gold ETF / SGB", "REITs / InvITs", "International Equity", "Fixed Deposits", "Crypto (speculative)"
        ], default=[])
        horizon = st.number_input("Investment horizon (yrs)", min_value=1, max_value=50, value=horizon_default)

        # build suggestion (simple, transparent logic)
        def robo_custom(age, income, risk_label, optional_assets):
            base = {
                "Large Cap Equity":30, "Mid/Small Cap Equity":10, "International Equity":5,
                "Index ETFs":10, "Debt Funds":25, "Gold ETF / SGB":8, "REITs / InvITs":5, "Cash / Liquid":7
            }
            risk_map = {"Very Low": -15, "Low": -7, "Moderate": 0, "High": 7, "Very High": 15}
            tilt = risk_map.get(risk_label,0)
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
            for a in optional_assets:
                if a not in base:
                    base[a] = 2
            total = sum(base.values())
            return {k: round(v/total*100,2) for k,v in base.items()}

        suggestion = robo_custom(age, monthly_income, risk_label, optional_assets)
        suggestion_df = pd.DataFrame({"Asset Class": list(suggestion.keys()), "Allocation (%)": list(suggestion.values())})
        st.markdown("### Suggested allocation (editable)")
        edited_alloc = st.data_editor(suggestion_df, num_rows="dynamic", use_container_width=True)

    with c2:
        st.markdown("### Quick metrics & logic")
        if "Allocation (%)" in edited_alloc.columns:
            total_alloc = edited_alloc["Allocation (%)"].sum()
            if abs(total_alloc - 100) > 0.5:
                st.warning("Allocation does not sum to 100% — adjust the table or rebalance.")
        # show short logic box
        st.markdown("<div class='card'><b>Logic for selection</b><div class='small'>Allocation is a function of age, income, and risk preferences. Equity bias rises with lower age and higher risk appetite; debt and cash act as shock absorbers for income and retirement horizons.</div></div>", unsafe_allow_html=True)

    st.markdown("### Allocation visual")
    try:
        fig_p = px.pie(edited_alloc, names="Asset Class", values="Allocation (%)", hole=0.36)
        fig_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_p, use_container_width=True)
    except Exception:
        st.info("Allocation chart unavailable (check the allocation table).")

    # allow download
    try:
        csv = edited_alloc.to_csv(index=False).encode("utf-8")
        st.download_button("Download allocation CSV", csv, file_name="allocation.csv", mime="text/csv")
    except Exception:
        pass

# ---------------- Tab 2: Live Market (AI) Tracker — new AI-futuristic market UI ----------------
with tab2:
    st.markdown("## 📡 Live Market Tracker — AI-futuristic (auto-refresh every 5 minutes)")
    st.markdown("<div class='small'>Neon ticker, sparkline cards, and detailed table. Add / remove tickers in the watchlist below.</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Watchlist editor
    st.markdown("### Watchlist (editable & persistent)")
    wl_col1, wl_col2 = st.columns([3,1])
    with wl_col1:
        # show editable watchlist (data editor)
        wl_df = pd.DataFrame({"Ticker": st.session_state.watchlist})
        wl_df = st.data_editor(wl_df, num_rows="dynamic", use_container_width=True)
        # coerce to session state
        new_watch = [str(x).strip() for x in wl_df["Ticker"].tolist() if str(x).strip() != ""]
        if new_watch != st.session_state.watchlist:
            st.session_state.watchlist = new_watch
    with wl_col2:
        st.markdown("<div style='display:flex;flex-direction:column;gap:6px'>"
                    "<button id='add_default' style='padding:6px;border-radius:8px;background:#00121a;color:#00e5ff;border:0'>Add Nifty</button>"
                    "</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Build marquee strip
    WATCHLIST = st.session_state.watchlist.copy()
    # Ensure Nifty presence optionally (local). Keep '^NSEI' optional: Yahoo index may be blocked.
    if "^NSEI" not in WATCHLIST and len(WATCHLIST) < 10:
        WATCHLIST.insert(0, "RELIANCE.NS")  # ensure at least one Indian name

    # Fetch latest and short series for each ticker (defensive)
    strip_items = []
    detailed_rows = []
    for t in WATCHLIST:
        # try last 7 days for sparkline & last two closes for % change
        series = fetch_close(t, period="14d", interval="1d")
        if series is None or series.empty:
            strip_items.append((t, None, None))
            detailed_rows.append({"Ticker": t, "Last Close": None, "Pct Change": None, "Status": "No data"})
            continue
        last = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if len(series) >= 2 else last
        pct = (last - prev) / prev * 100 if prev != 0 else 0.0
        strip_items.append((t, last, pct))
        detailed_rows.append({"Ticker": t, "Last Close": last, "Pct Change": pct, "Status": "OK", "Series": series.tail(7)})

    # marquee html
    marquee_parts = []
    for t, price, pct in strip_items:
        if price is None:
            marquee_parts.append(f"<span style='padding:0 18px;color:#6C7A89'>{t}: —</span>")
        else:
            color = "#26FFE6" if pct >= 0 else "#FF6B6B"
            marquee_parts.append(f"<span style='padding:0 18px;color:{color}; font-weight:600'>{t}: {price:,.2f} ({pct:+.2f}%)</span>")
    marquee_html = "<div class='ticker-slab marquee'><div class='marquee-inner'>" + " &#160; &#160; ".join(marquee_parts) + "</div></div>"
    st.markdown(marquee_html, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Show sparkline cards in a responsive grid
    if strip_items:
        cols = st.columns(min(len(strip_items), 6))
        for i, (t, price, pct) in enumerate(strip_items):
            col = cols[i % len(cols)]
            with col:
                if price is None:
                    col.markdown(f"<div class='card' style='text-align:center'><div class='small'>{t}</div><div style='font-size:18px;color:#6C7A89'>No data</div></div>", unsafe_allow_html=True)
                    continue
                color = "#26FFE6" if pct >= 0 else "#FF6B6B"
                col.markdown(f"<div class='card' style='text-align:center'><div class='small'>{t}</div><div style='font-weight:700;color:{color};font-size:18px'>{price:,.2f}</div><div class='small' style='color:{color}'>{pct:+.2f}%</div></div>", unsafe_allow_html=True)
                # sparkline
                # find the corresponding series
                series = None
                for row in detailed_rows:
                    if row["Ticker"] == t and row.get("Series") is not None:
                        series = row["Series"]
                        break
                if series is not None and len(series) >= 3:
                    sp_df = pd.DataFrame({"x": series.index, "y": series.values})
                    fig_sp = px.line(sp_df, x="x", y="y", height=90)
                    fig_sp.update_traces(line=dict(color=color, width=2), hoverinfo="skip")
                    fig_sp.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
                    st.plotly_chart(fig_sp, use_container_width=True)
                else:
                    st.markdown("<div style='height:90px'></div>", unsafe_allow_html=True)

    st.markdown("---")
    # Detailed table
    detail_df = pd.DataFrame([{"Ticker": r["Ticker"], "Last Close": (fmt_inr(r["Last Close"]) if r["Last Close"] else "—"), "Pct Change": (f"{r['Pct Change']:+.2f}%" if r["Pct Change"] is not None else "—"), "Status": r["Status"]} for r in detailed_rows])
    st.table(detail_df)

    # note about unavailable tickers
    missing = [r["Ticker"] for r in detailed_rows if r["Status"] != "OK"]
    if missing:
        st.warning("Some tickers returned no data. Yahoo may block certain index tickers like ^NSEI; add individual stocks/ETFs instead.")

# ---------------- Tab 3: Analytics & Goals ----------------
with tab3:
    st.markdown("## Analytics & Goals — projections & Monte Carlo")
    colL, colR = st.columns([1.6, 1])

    with colL:
        st.markdown("### Goals — create and simulate")
        if "goals" not in st.session_state:
            st.session_state.goals = [{"name":"Retirement","amount":5000000,"years":20},{"name":"Home","amount":3000000,"years":8}]
        goals_df = pd.DataFrame(st.session_state.goals)
        edited_goals = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)

        # robust cleaning of goals
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
                if isinstance(val, (int,float)):
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
            cleaned.append({"name":name,"amount":amt,"years":yrs})
        st.session_state.goals = cleaned

        if corrections:
            with st.expander("Auto-corrections applied"):
                for idx, old, new in corrections:
                    st.markdown(f"- Row **{idx}**: `{old}` → `{new}`")
            st.success("Goal inputs cleaned.")

        if st.button("Run Monte Carlo for goals"):
            # get allocation weights from earlier edited_alloc if present
            try:
                weights_df = edited_alloc.copy()
                if "Allocation (%)" not in weights_df:
                    raise Exception
            except Exception:
                # fallback to robo suggestion
                weights_df = pd.DataFrame({"Asset Class": list(robo_custom(age, monthly_income, risk_label, optional_assets).keys()), "Allocation (%)": list(robo_custom(age, monthly_income, risk_label, optional_assets).values())})

            w = np.array(weights_df["Allocation (%)"]/100.0, dtype=float)
            # use default returns & vol mapping for asset classes
            default_returns = { # conservative defaults (annual)
                "Large Cap Equity":0.10,"Mid/Small Cap Equity":0.13,"International Equity":0.08,"Index ETFs":0.095,
                "Debt Funds":0.06,"Gold ETF / SGB":0.07,"REITs / InvITs":0.08,"Cash / Liquid":0.035
            }
            means = np.array([default_returns.get(a,0.06) for a in weights_df["Asset Class"]], dtype=float)
            vols = np.array([0.18 if "Equity" in a else 0.08 for a in weights_df["Asset Class"]], dtype=float)
            cov = np.outer(vols, vols) * 0.25
            np.fill_diagonal(cov, vols**2)

            results = []
            for g in st.session_state.goals:
                yrs = int(float(g.get("years",10)))
                sims = max(300, int(mc_sims/3))
                mc = monte_carlo_sim = None
                # run MC (simple annual correlated returns)
                # reuse cache function pattern locally to avoid redefinition
                def run_mc(invest, sip, w, means, cov, yrs, sims):
                    yrs = int(yrs)
                    sims = int(sims)
                    w = np.array(w, dtype=float)
                    means = np.array(means, dtype=float)
                    cov = np.array(cov, dtype=float)
                    n = len(w)
                    cov = cov + np.eye(n)*1e-10
                    L = np.linalg.cholesky(cov)
                    rng = np.random.default_rng(42)
                    finals = np.zeros(sims)
                    annual_sip = float(sip)*12.0
                    base = w * invest
                    for s in range(sims):
                        vals = base.copy()
                        for y in range(yrs):
                            z = rng.normal(size=n)
                            ret = means + L @ z
                            vals = vals * (1 + ret)
                            if annual_sip>0:
                                vals = vals + annual_sip * w
                        finals[s] = vals.sum()
                    return finals
                mc = run_mc(invest_now, sip_now, w, means, cov, yrs, sims)
                prob = float((mc >= g["amount"]).sum() / len(mc) * 100.0)
                results.append({"Goal":g["name"], "Target (₹)": int(g["amount"]), "Years":yrs, "P(achieve %)": round(prob,1), "Median end (₹)": int(np.median(mc))})
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True)
            st.plotly_chart(px.bar(res_df, x="Goal", y="P(achieve %)", text="P(achieve %)"), use_container_width=True)
            st.download_button("Download goal results CSV", res_df.to_csv(index=False).encode("utf-8"), "goal_results.csv", mime="text/csv")

    with colR:
        st.markdown("### Snapshot — suggested allocation metrics")
        # create a preview of suggested robo allocation
        preview = robo_custom(age, monthly_income, risk_label, optional_assets)
        df_preview = pd.DataFrame({"Asset Class": list(preview.keys()), "Allocation (%)": list(preview.values())})
        w_pre = np.array(df_preview["Allocation (%)"]/100.0, dtype=float)
        means_pre = np.array([0.09 if "Equity" in a else 0.05 for a in df_preview["Asset Class"]], dtype=float)
        vols_pre = np.array([0.18 if "Equity" in a else 0.08 for a in df_preview["Asset Class"]], dtype=float)
        cov_pre = np.outer(vols_pre, vols_pre)*0.25
        np.fill_diagonal(cov_pre, vols_pre**2)
        p_ret = float(np.dot(w_pre, means_pre))
        p_vol = float(np.sqrt(w_pre @ cov_pre @ w_pre))
        st.markdown(f"<div class='card'><b>Exp. return</b><div style='font-size:18px;color:#26ffe6'>{p_ret*100:.2f}%</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Volatility (σ)</b><div style='font-size:18px;color:#ffd166'>{p_vol*100:.2f}%</div></div>", unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("<hr style='border:1px solid rgba(255,255,255,0.04)'>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;color:#9FB4C8'>Built for Indian investors — educational tool. Not financial advice.</div>", unsafe_allow_html=True)
