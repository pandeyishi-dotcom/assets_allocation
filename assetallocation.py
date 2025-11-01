# assetallocation_fixed.py
# Full fintech cockpit (fixed plotting + safe data editor)
# Requirements: streamlit, yfinance, pandas, numpy, plotly, pytz

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date
from io import StringIO

# -------------------------
# Page config & theme
# -------------------------
st.set_page_config(page_title="AI Portfolio Cockpit — Fixed", layout="wide", page_icon="💠")
ACCENT = "#00FFC6"
BG = "#0e1117"
CARD = "#0f1720"
MUTED = "#9aa5a6"

st.markdown(
    f"""
    <style>
      body {{ background: {BG}; color: #e6eef0; }}
      div[data-testid="stSidebar"]{{background:#071427;}}
      .titlebig{{ font-size:28px; color:{ACCENT}; font-weight:700; margin-bottom:6px;}}
      .muted{{ color:{MUTED}; }}
      .card{{ background:{CARD}; padding:12px; border-radius:10px; }}
      .smallpill{{ font-size:12px; color:#cfeee6; background:#07121a; padding:6px 10px;border-radius:6px;}}
      .footer{{ color: {MUTED}; font-size:12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utility functions
# -------------------------
@st.cache_data(ttl=300)
def fetch_symbol(symbol, period="5d", interval="15m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_safe_nifty():
    """Try multiple strategies for ^NSEI and return (df, mode, reason)"""
    reasons = []
    # intraday 15m
    try:
        df = fetch_symbol("^NSEI", period="5d", interval="15m")
        if not df.empty:
            return df, "live", "intraday 15m (5d)"
    except Exception as e:
        reasons.append(f"intraday error {e}")
    # daily 1mo
    try:
        df = fetch_symbol("^NSEI", period="1mo", interval="1d")
        if not df.empty:
            return df, "ltp", "daily 1mo fallback"
    except Exception as e:
        reasons.append(f"daily error {e}")
    # ticker.history
    try:
        t = yf.Ticker("^NSEI")
        df = t.history(period="1mo", interval="1d")
        if not df.empty:
            return df, "ltp", "Ticker.history 1mo"
    except Exception as e:
        reasons.append(f"ticker.history error {e}")
    return None, "error", "; ".join(reasons)

def compute_portfolio_value(df_holdings):
    """Add LTP, Value, Cost, P&L columns to holdings DataFrame"""
    df = df_holdings.copy()
    symbols = df["Symbol"].unique().tolist()
    prices = {}
    for s in symbols:
        try:
            d = fetch_symbol(s, period="5d", interval="1d")
            if not d.empty:
                prices[s] = float(d["Close"].iloc[-1])
            else:
                prices[s] = np.nan
        except Exception:
            prices[s] = np.nan
    df["LTP"] = df["Symbol"].map(prices)
    df["Value"] = df["LTP"] * df["Quantity"]
    if "BuyPrice" in df.columns:
        df["Cost"] = df["BuyPrice"] * df["Quantity"]
        df["P&L"] = df["Value"] - df["Cost"]
        # avoid division by zero
        df["P&L %"] = np.where(df["Cost"] != 0, df["P&L"] / df["Cost"] * 100, np.nan)
    return df

def suggest_allocation(age, risk_level, horizon_years):
    """Simple rule-based allocation advice"""
    base_equity = 60
    if age >= 60:
        base_equity = 30
    elif age >= 45:
        base_equity = 40
    elif age <= 30:
        base_equity = 70

    if risk_level == "Low":
        equity = base_equity - 20
    elif risk_level == "Moderate":
        equity = base_equity
    else:
        equity = base_equity + 10

    if horizon_years >= 15:
        equity += 5
    elif horizon_years <= 5:
        equity -= 5

    equity = max(10, min(90, int(round(equity))))
    debt = 100 - equity
    gold = int(round(equity * 0.08))
    reits = 5
    cash = 5
    allocation = {
        "Large-cap Equity": int(round(equity * 0.45)),
        "Mid/Small-cap Equity": int(round(equity * 0.25)),
        "International Equity": int(round(equity * 0.10)),
        "Debt — Govt": int(round(debt * 0.6)),
        "Debt — Corporate": int(round(debt * 0.3)),
        "Cash/Liquid": int(round(cash)),
        "Gold": gold,
        "Real Estate / REITs": reits,
        "Commodities (ex-gold)": 0,
        "Alternatives (PE/VC)": 0,
        "Crypto (small)": 0,
        "Sector Bets": 0,
        "ESG / Thematic": 0,
        "Short-term Bonds": 0,
        "Others": 0
    }
    tot = sum(allocation.values())
    if tot == 0:
        for k in allocation:
            allocation[k] = int(round(100/len(allocation)))
    else:
        for k in allocation:
            allocation[k] = int(round(allocation[k] * 100 / tot))
    diff = 100 - sum(allocation.values())
    allocation["Cash/Liquid"] += diff
    rationale = f"Rule-of-thumb allocation based on age {age}, risk {risk_level}, horizon {horizon_years}y."
    return allocation, rationale

def sip_projection(monthly_sip, years, annual_return, inflation=0.05):
    months = years * 12
    r = annual_return / 12
    if r == 0:
        fv = monthly_sip * months
    else:
        fv = monthly_sip * (((1 + r) ** months - 1) / r)
    real = fv / ((1 + inflation) ** years)
    return fv, real

# -------------------------
# Sidebar — top-level nav & settings
# -------------------------
with st.sidebar:
    st.markdown(f"<div style='text-align:center'><img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/NSE_Logo.svg' width='120'></div>", unsafe_allow_html=True)
    st.title("AI Portfolio Cockpit")
    nav = st.radio("Navigate", ["Home", "Live Market", "Market Pulse", "Portfolio", "Asset Allocation", "Allocation Advisor", "Goals & SIP", "Sector Heatmap", "Watchlist"], index=1)
    st.markdown("---")
    st.caption("Built for Indian markets • Auto-fallbacks included")

# -------------------------
# Home
# -------------------------
if nav == "Home":
    st.markdown(f"<div class='titlebig'>AI Portfolio Cockpit</div>", unsafe_allow_html=True)
    st.write("A futuristic, professional dashboard for Indian investors — live market, portfolio, AI allocation tips, and scenario simulations.")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY (open)", "Use Live Market tab")
    c2.metric("Portfolio", "Upload CSV in Portfolio tab")
    c3.metric("Allocation", "Use Asset Allocation tab")
    st.markdown("---")
    st.info("Tip: Use 'Allocation Advisor' for a quick allocation suggestion based on Age, Risk and Horizon.")

# -------------------------
# Market Pulse
# -------------------------
elif nav == "Market Pulse":
    st.markdown(f"<div class='titlebig'>Market Pulse — Snapshot</div>", unsafe_allow_html=True)
    symbols = {
        "NIFTY":"^NSEI", "SENSEX":"^BSESN", "NIFTY BANK":"^NSEBANK",
        "USD/INR":"INR=X", "GOLD (COMEX)":"GC=F", "BTC":"BTC-USD"
    }
    cols = st.columns(len(symbols))
    for i, (label, sym) in enumerate(symbols.items()):
        d = fetch_symbol(sym, period="5d", interval="1d")
        if d.empty:
            cols[i].metric(label, "N/A")
        else:
            last = float(d["Close"].iloc[-1])
            prev = float(d["Close"].iloc[-2]) if len(d) >= 2 else last
            pct = (last - prev) / prev * 100 if prev != 0 else 0.0
            display = f"₹{last:,.2f}" if label not in ["BTC","USD/INR","GOLD (COMEX)"] else f"{last:,.2f}"
            cols[i].metric(label, display, f"{pct:+.2f}%")
    st.caption("Pulse data cached to reduce fetches.")

# -------------------------
# Live Market (NIFTY) with LTP fallback
# -------------------------
elif nav == "Live Market":
    st.markdown(f"<div class='titlebig'>Live Indian Market Tracker (NIFTY)</div>", unsafe_allow_html=True)
    nifty_df, mode, reason = fetch_safe_nifty()
    if nifty_df is None or nifty_df.empty:
        st.warning("⚠️ Could not fetch live NIFTY data. Displaying sample dataset.")
        sample = {
            "Datetime": pd.date_range(end=pd.Timestamp.now(), periods=6, freq="D"),
            "Open": [20050, 20100, 19980, 20090, 20200, 20300],
            "High": [20200, 20250, 20080, 20400, 20350, 20450],
            "Low":  [19900, 20010, 19850, 20000, 20150, 20210],
            "Close":[20100, 19980, 20050, 20300, 20250, 20350],
            "Volume":[1000,1100,1050,1200,900,950]
        }
        nifty_df = pd.DataFrame(sample).set_index("Datetime")
        mode = "sample"
        reason = reason or "no data"
    st.caption(f"Mode: {mode} — {reason}")

    # compute latest/previous safely
    # if 'Close' exists, use it; else pick first numeric column
    if "Close" in nifty_df.columns:
        series_to_plot = nifty_df["Close"]
    else:
        numeric_cols = nifty_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            series_to_plot = nifty_df[numeric_cols[0]]
        else:
            # as ultimate fallback, convert first column to numeric
            series_to_plot = pd.to_numeric(nifty_df.iloc[:, 0], errors="coerce")

    latest = float(series_to_plot.iloc[-1])
    previous = float(series_to_plot.iloc[-2]) if len(series_to_plot) > 1 else latest
    change = latest - previous
    pct = (change / previous) * 100 if previous != 0 else 0.0
    last_date = pd.to_datetime(series_to_plot.index[-1]).date()
    today = date.today()

    colA, colB, colC = st.columns(3)
    if last_date < today:
        colA.subheader("📅 Market Closed — showing LTP")
        colA.metric("Last Traded Price (LTP)", f"₹{latest:,.2f}")
    else:
        colA.subheader("✅ Market Live")
        colA.metric("NIFTY 50", f"₹{latest:,.2f}", f"{pct:+.2f}%")
    colB.metric("Recent Δ (pts)", f"{change:+.2f}")
    colC.metric("Δ (%)", f"{pct:+.2f}%")

    st.markdown("### Price movement (recent)")
    # use series_to_plot for plotting to avoid MultiIndex issues
    try:
        st.line_chart(series_to_plot.rename("Price"), use_container_width=True)
    except Exception:
        # final fallback: convert DataFrame to single series
        try:
            st.line_chart(pd.to_numeric(nifty_df.iloc[:, 0], errors="coerce"), use_container_width=True)
        except Exception:
            st.write("Chart unavailable.")

    st.markdown("### Recent data")
    # show dataframe safely
    try:
        st.dataframe(nifty_df.tail(10).round(2), use_container_width=True)
    except Exception:
        st.write(nifty_df.tail(10))

# -------------------------
# Portfolio upload & analysis
# -------------------------
elif nav == "Portfolio":
    st.markdown(f"<div class='titlebig'>Portfolio Tracker</div>", unsafe_allow_html=True)
    st.markdown("Upload CSV: Symbol,Quantity,BuyPrice (BuyPrice optional). Example: TCS.NS,5,3500")
    uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded, header=None)
            # Support headerless or headered
            if raw.shape[1] == 2:
                raw.columns = ["Symbol", "Quantity"]
            elif raw.shape[1] >= 3:
                raw = raw.iloc[:, :3]
                raw.columns = ["Symbol", "Quantity", "BuyPrice"]
            raw["Symbol"] = raw["Symbol"].astype(str).str.strip().str.upper()
            raw["Quantity"] = raw["Quantity"].astype(float)
            if "BuyPrice" in raw.columns:
                raw["BuyPrice"] = raw["BuyPrice"].astype(float)
            df_hold = compute_portfolio_value(raw)
            st.markdown("### Holdings")
            st.dataframe(df_hold.round(2), use_container_width=True)

            total_value = df_hold["Value"].sum()
            total_cost = df_hold["Cost"].sum() if "Cost" in df_hold.columns else np.nan
            pnl = total_value - total_cost if not np.isnan(total_cost) else np.nan
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Market Value", f"₹{total_value:,.2f}")
            c2.metric("Total P&L", f"₹{pnl:,.2f}" if not np.isnan(pnl) else "N/A")
            c3.metric("Tickers", df_hold["Symbol"].nunique())
            alloc = df_hold.groupby("Symbol")["Value"].sum().reset_index()
            fig = px.pie(alloc, names="Symbol", values="Value", title="Portfolio Allocation", color_discrete_sequence=px.colors.sequential.Tealgrn)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    else:
        st.info("Upload a portfolio CSV to compute current value and allocation.")

# -------------------------
# Asset Allocation (simplified UI)
# -------------------------
elif nav == "Asset Allocation":
    st.markdown(f"<div class='titlebig'>Asset Allocation Builder</div>", unsafe_allow_html=True)
    st.markdown("Minimal allocation UI (auto-suggest + manual edit).")
    asset_classes = [
        "Large-cap Equity", "Mid/Small-cap Equity", "International Equity",
        "Debt — Govt", "Debt — Corporate", "Short-term Bonds",
        "Gold", "Real Estate / REITs", "Commodities (ex-gold)",
        "Alternatives (PE/VC)", "Crypto (small)", "Sector Bets",
        "ESG / Thematic", "Cash/Liquid", "Others"
    ]
    cols = st.columns(3)
    selected = []
    for i, ac in enumerate(asset_classes):
        if cols[i % 3].checkbox(ac, value=(ac in ["Large-cap Equity", "Debt — Govt", "Cash/Liquid"])):
            selected.append(ac)

    st.markdown("### Auto-suggest allocation")
    with st.form("allocation_form"):
        age = st.number_input("Age", 20, 80, 35)
        risk = st.selectbox("Risk appetite", ["Low", "Moderate", "High"], index=1)
        horizon = st.slider("Horizon (years)", 1, 30, 10)
        submitted = st.form_submit_button("Suggest allocation")
    if submitted:
        alloc_map, rationale = suggest_allocation(age, risk, horizon)
        alloc_display = {k: v for k, v in alloc_map.items() if k in selected} or alloc_map
        df_alloc = pd.DataFrame(list(alloc_display.items()), columns=["Asset Class", "Allocation %"])
        total = df_alloc["Allocation %"].sum()
        if total != 100 and total > 0:
            df_alloc["Allocation %"] = (df_alloc["Allocation %"] / total * 100).round(0)
            diff = 100 - df_alloc["Allocation %"].sum()
            df_alloc.loc[df_alloc.index[0], "Allocation %"] += diff
        st.dataframe(df_alloc, use_container_width=True)
        fig = px.pie(df_alloc, names="Asset Class", values="Allocation %", title="Suggested Allocation", color_discrete_sequence=px.colors.sequential.Tealgrn)
        st.plotly_chart(fig, use_container_width=True)
        st.write(rationale)

    st.markdown("### Manual edit (if editor is available)")
    default_df = pd.DataFrame({"Asset Class": selected or asset_classes, "Allocation %": [round(100/len(selected or asset_classes), 0)]*(len(selected or asset_classes))})

    # use experimental_data_editor if available, else fallback to static table + input
    try:
        edited = st.experimental_data_editor(default_df)
        # 'Normalize to 100' button to fix sums
        if st.button("Normalize to 100"):
            s = edited["Allocation %"].sum()
            if s == 0:
                st.error("Sum is 0 — adjust values.")
            else:
                edited["Allocation %"] = (edited["Allocation %"] / s * 100).round(0)
                diff = 100 - edited["Allocation %"].sum()
                edited.at[0, "Allocation %"] += diff
                st.success("Normalized to 100%")
                st.dataframe(edited, use_container_width=True)
    except Exception:
        st.warning("Interactive data editor not available in this Streamlit runtime. Showing static table.")
        st.dataframe(default_df, use_container_width=True)

# -------------------------
# Allocation Advisor
# -------------------------
elif nav == "Allocation Advisor":
    st.markdown(f"<div class='titlebig'>AI Allocation Advisor</div>", unsafe_allow_html=True)
    st.write("Quick explainable suggestions.")
    age2 = st.number_input("Your age", 20, 80, 35, key="age2")
    risk2 = st.selectbox("Risk appetite", ["Low","Moderate","High"], index=1, key="risk2")
    horizon2 = st.slider("Goal horizon (years)", 1, 30, 10, key="hor2")
    if st.button("Get Advice"):
        alloc_map2, rationale2 = suggest_allocation(age2, risk2, horizon2)
        df_alloc2 = pd.DataFrame(list(alloc_map2.items()), columns=["Class","Pct"])
        st.dataframe(df_alloc2, use_container_width=True)
        fig = px.bar(df_alloc2, x="Class", y="Pct", text="Pct")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Mentor note:**")
        st.info("This is a concise conservative-to-aggressive allocation. For tailoring, upload portfolio and consult an adviser.")

# -------------------------
# Goals & SIP
# -------------------------
elif nav == "Goals & SIP":
    st.markdown(f"<div class='titlebig'>Goals & SIP Simulator</div>", unsafe_allow_html=True)
    mode = st.selectbox("Projection mode", ["SIP (monthly)", "Lump sum"], index=0)
    if mode == "SIP (monthly)":
        monthly = st.number_input("Monthly SIP (₹)", 100, 200000, 5000)
        years = st.slider("Years", 1, 40, 10)
        exp_ret = st.slider("Expected annual return (%)", 3.0, 20.0, 10.0)
        inflation = st.slider("Inflation (%)", 0.0, 12.0, 4.5)
        fv, real = sip_projection(monthly, years, exp_ret/100, inflation/100)
        st.metric("Projected corpus (nominal)", f"₹{fv:,.0f}", f"Inflation-adjusted: ₹{real:,.0f}")
        months = years * 12
        vals = []
        bal = 0.0
        monthly_r = exp_ret/100/12
        for m in range(1, months+1):
            bal = bal*(1+monthly_r) + monthly
            vals.append(bal)
        df_curve = pd.DataFrame({"Month": range(1, months+1), "Balance": vals})
        st.line_chart(df_curve.set_index("Month")["Balance"], use_container_width=True)
    else:
        principal = st.number_input("Principal (₹)", 10000, 100000000, 100000)
        years = st.slider("Years", 1, 40, 10, key="ls_years")
        exp_ret = st.slider("Expected annual return (%)", 3.0, 20.0, 9.0, key="ls_ret")
        inflation = st.slider("Inflation (%)", 0.0, 12.0, 4.5, key="ls_inf")
        fv = principal * ((1 + exp_ret/100) ** years)
        fv_real = fv / ((1 + inflation/100) ** years)
        st.metric("Projected corpus", f"₹{fv:,.0f}", f"Inflation-adjusted: ₹{fv_real:,.0f}")

# -------------------------
# Sector Heatmap
# -------------------------
elif nav == "Sector Heatmap":
    st.markdown(f"<div class='titlebig'>Sector Heatmap (sample)</div>", unsafe_allow_html=True)
    sectors = {
        "IT":["TCS.NS","INFY.NS","WIPRO.NS"],
        "Banking":["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS"],
        "Energy":["RELIANCE.NS","ONGC.NS","BPCL.NS"],
        "Pharma":["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"],
        "FMCG":["HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS"]
    }
    perf = []
    for s, tickers in sectors.items():
        vals = []
        for t in tickers:
            d = fetch_symbol(t, period="5d", interval="1d")
            if not d.empty:
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2]) if len(d) >= 2 else last
                vals.append((last-prev)/prev*100 if prev != 0 else 0.0)
        avg = float(np.nanmean(vals)) if len(vals) else 0.0
        perf.append({"Sector": s, "Change%": avg})
    df_sector = pd.DataFrame(perf)
    fig = px.treemap(df_sector, path=["Sector"], values="Change%", color="Change%", color_continuous_scale="RdYlGn", title="Sector avg % change")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Watchlist & Alerts
# -------------------------
elif nav == "Watchlist":
    st.markdown(f"<div class='titlebig'>Watchlist & Alerts</div>", unsafe_allow_html=True)
    default = "TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS"
    wl = st.text_input("Tickers (comma separated)", value=default)
    threshold = st.number_input("Alert threshold (%)", 0.5, 20.0, 2.0)
    if st.button("Fetch watchlist"):
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        rows = []
        for t in tickers:
            d = fetch_symbol(t, period="5d", interval="1d")
            if d.empty:
                rows.append({"Symbol": t, "Price": "N/A", "Change%": "N/A", "Alert": ""})
            else:
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2]) if len(d) >= 2 else last
                pct = (last-prev)/prev*100 if prev != 0 else 0.0
                alert = "⚠️" if abs(pct) >= threshold else ""
                rows.append({"Symbol": t.replace(".NS",""), "Price": f"₹{last:,.2f}", "Change%": f"{pct:+.2f}%", "Alert": alert})
        st.table(pd.DataFrame(rows))

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("<div class='footer'>Built with ❤️ — rule-based AI adviser. Always verify before investing.</div>", unsafe_allow_html=True)
