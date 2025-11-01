"""
assetallocation_v2.py
AI-Driven Fintech Dashboard — v2 Master

Notes:
- Requires: streamlit, yfinance, pandas, numpy, plotly, pytz
- Paste into Streamlit Cloud or local environment and run.
- This code uses yfinance; sometimes intraday index data is blocked — graceful fallback included.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import pytz
from io import StringIO

# ----------------------------
# Page config & small helpers
# ----------------------------
st.set_page_config(
    page_title="AI Portfolio Cockpit — v2",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded",
)

KOLORS = {"accent": "#00FFC6", "bg": "#0e1117", "card": "#0f1720", "muted": "#9aa5a6"}

# theme toggle
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def set_theme(t):
    st.session_state.theme = t

# small style
st.markdown(
    f"""
    <style>
      body {{ background: {KOLORS['bg']}; color: #e6eef0; }}
      div[data-testid="stSidebar"]{{background:#0b1220}}
      .titlebig{{ font-size:34px; color:{KOLORS['accent']}; font-weight:800}}
      .muted{{ color:#9aa5a6; }}
      .card{{ background:{KOLORS['card']}; padding:12px; border-radius:10px; }}
      .smallpill{{ font-size:12px; color:#cfeee6; background:#07121a; padding:6px 10px;border-radius:6px;}}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Utility functions
# ----------------------------
@st.cache_data(ttl=300)
def fetch_symbol(symbol, period="5d", interval="15m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

def fetch_safe_nifty():
    """Robust multi-strategy fetch for ^NSEI. Returns (df, mode, reason)"""
    reason = []
    try:
        df = fetch_symbol("^NSEI", period="5d", interval="15m")
        if not df.empty:
            return df, "live", "intraday 15m (5d)"
    except Exception as e:
        reason.append(f"intraday error {e}")
    try:
        df = fetch_symbol("^NSEI", period="1mo", interval="1d")
        if not df.empty:
            return df, "ltp", "daily 1mo (fallback)"
    except Exception as e:
        reason.append(f"daily error {e}")
    try:
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(period="1mo", interval="1d")
        if not df.empty:
            return df, "ltp", "Ticker.history 1mo"
    except Exception as e:
        reason.append(f"Ticker.history error {e}")
    return None, "error", "; ".join(reason)

def compute_portfolio_value(holdings_df):
    """holdings_df columns: Symbol, Quantity, BuyPrice (optional)"""
    symbols = holdings_df["Symbol"].unique().tolist()
    prices = {}
    for s in symbols:
        try:
            d = yf.download(s, period="1d", interval="1d", progress=False)
            if not d.empty:
                prices[s] = float(d["Close"].iloc[-1])
            else:
                prices[s] = np.nan
        except Exception:
            prices[s] = np.nan
    holdings_df["LTP"] = holdings_df["Symbol"].map(prices)
    holdings_df["Value"] = holdings_df["LTP"] * holdings_df["Quantity"]
    if "BuyPrice" in holdings_df.columns:
        holdings_df["Cost"] = holdings_df["BuyPrice"] * holdings_df["Quantity"]
        holdings_df["P&L"] = holdings_df["Value"] - holdings_df["Cost"]
        holdings_df["P&L %"] = holdings_df["P&L"] / holdings_df["Cost"] * 100
    return holdings_df

def suggest_allocation(age, risk_level, horizon_years):
    """Simple rule-based allocation advice"""
    # risk_level: 'Low','Moderate','High'
    base_equity = 60
    if age >= 60:
        base_equity = 30
    elif age >= 45:
        base_equity = 40
    elif age <= 30:
        base_equity = 70

    if risk_level == "Low":
        equity = int(base_equity - 20)
    elif risk_level == "Moderate":
        equity = int(base_equity)
    else:
        equity = int(base_equity + 10)

    # horizon adjustment
    if horizon_years >= 15:
        equity += 5
    elif horizon_years <= 5:
        equity -= 5

    equity = max(10, min(90, equity))
    debt = 100 - equity
    gold = int(round(equity * 0.08))  # small hedge
    reits = 5
    others = 100 - (equity + debt - gold - reits)  # simple balancing (not exact)
    # make nicer buckets
    allocation = {
        "Equity": equity,
        "Debt": debt - gold - reits,
        "Gold": gold,
        "REITs": reits,
        "Cash": 5
    }
    # normalize sum to 100
    s = sum(allocation.values())
    for k in allocation:
        allocation[k] = int(round(allocation[k] * 100 / s))
    # ensure sum 100
    diff = 100 - sum(allocation.values())
    allocation["Cash"] += diff
    rationale = f"Age {age}, Risk {risk_level}, Horizon {horizon_years}y -> equity-focused if younger"
    return allocation, rationale

def sip_projection(monthly_sip, years, expected_return, inflation=0.05):
    months = years * 12
    r = expected_return / 12
    if r == 0:
        fv = monthly_sip * months
    else:
        fv = monthly_sip * (((1 + r) ** months - 1) / r)
    real_fv = fv / ((1 + inflation) ** years)
    return fv, real_fv

# ----------------------------
# Sidebar controls + header
# ----------------------------
with st.sidebar:
    st.image("https://raw.githubusercontent.com/rezwan7/finance-icons/main/nifty_dark.png", width=120)
    st.title("AI Portfolio Cockpit")
    theme = st.selectbox("Theme", ["dark", "light"], index=0)
    set_theme(theme)
    st.markdown("---")
    st.markdown("Navigation")
    nav = st.radio("", ["Home", "Live Market", "Market Pulse", "Portfolio", "Allocation Advisor", "Goals & SIP", "Efficient Frontier", "Sector Heatmap", "Watchlist"], index=1)
    st.markdown("---")
    st.caption("Built for Indian markets • Auto-fallbacks included")

# ----------------------------
# Home
# ----------------------------
if nav == "Home":
    st.markdown('<div class="titlebig">AI Portfolio Cockpit — v2</div>', unsafe_allow_html=True)
    st.markdown("A modular, futuristic fintech app: live market, portfolio analysis, AI allocation advice, goals, and more.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY (approx)", "—", "live data in Live Market tab")
    c2.metric("Portfolio", "Upload CSV", "in Portfolio tab")
    c3.metric("AI Advice", "Age + Risk", "in Allocation Advisor")
    st.markdown("---")
    st.info("Tip: Use Live Market tab to check NIFTY status (live or LTP).")

# ----------------------------
# Market Pulse
# ----------------------------
elif nav == "Market Pulse":
    st.markdown('<div class="titlebig">Market Pulse</div>', unsafe_allow_html=True)
    # sample tickers to display at a glance
    pulse_symbols = {"NIFTY":"^NSEI", "SENSEX":"^BSESN", "USDINR":"INR=X", "GOLD":"GC=F", "BTC":"BTC-USD"}
    cols = st.columns(len(pulse_symbols))
    results = {}
    for i, (label, sym) in enumerate(pulse_symbols.items()):
        try:
            d = fetch_symbol(sym, period="5d", interval="1d")
            if not d.empty:
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2]) if len(d) >= 2 else last
                pct = (last - prev) / prev * 100 if prev != 0 else 0.0
                txt = f"₹{last:,.2f}" if label not in ["BTC","USDINR","GOLD"] else f"{last:,.2f}"
                delta = f"{pct:+.2f}%"
                cols[i].metric(label, txt, delta)
                results[label] = (last, pct)
            else:
                cols[i].metric(label, "N/A", "")
        except Exception:
            cols[i].metric(label, "N/A", "")

    st.markdown("---")
    st.caption("Pulse updates are cached for 5 minutes to reduce rate-limits.")

# ----------------------------
# Live Market
# ----------------------------
elif nav == "Live Market":
    st.markdown('<div class="titlebig">Live Indian Market Tracker (NIFTY 50)</div>', unsafe_allow_html=True)
    nifty_df, mode, reason = fetch_safe_nifty()
    if nifty_df is None or nifty_df.empty:
        st.warning("⚠️ Could not fetch live NIFTY data. Displaying sample data.")
        sample = {
            "Datetime": pd.date_range(end=pd.Timestamp.now(), periods=6, freq="D"),
            "Close": [20102, 19980, 20050, 20300, 20250, 20350],
            "Open": [20000, 20010, 19900, 20200, 20150, 20300]
        }
        nifty_df = pd.DataFrame(sample).set_index("Datetime")
        mode = "sample"
    st.caption(f"Mode: {mode} — reason: {reason}")

    latest_price = float(nifty_df["Close"].iloc[-1])
    last_date = pd.to_datetime(nifty_df.index[-1]).date()
    today = date.today()

    colA, colB = st.columns([3,1])
    if last_date < today:
        colA.subheader("📅 Market Closed — showing LTP")
        colA.metric("LTP", f"₹{latest_price:,.2f}")
    else:
        prev = float(nifty_df["Close"].iloc[-2]) if len(nifty_df) >= 2 else latest_price
        change = latest_price - prev
        pct = (change / prev) * 100 if prev != 0 else 0.0
        colA.subheader("✅ Market Live")
        colA.metric("NIFTY 50", f"₹{latest_price:,.2f}", f"{pct:+.2f}%")
    if len(nifty_df) >= 2:
        colB.metric("Recent Δ (pts)", f"{change:+.2f}")

    st.markdown("### Intraday / Recent movement")
    try:
        st.line_chart(nifty_df["Close"].rename("Close"), use_container_width=True)
    except Exception:
        st.line_chart(nifty_df.iloc[:,0], use_container_width=True)

    st.markdown("### Recent rows")
    st.dataframe(nifty_df.tail(10).round(2), use_container_width=True)

# ----------------------------
# Portfolio Upload & Analysis
# ----------------------------
elif nav == "Portfolio":
    st.markdown('<div class="titlebig">Portfolio Tracker</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with columns: Symbol,Quantity,BuyPrice (BuyPrice optional). Example:\n\n```\nTCS.NS,5,3500\nINFY.NS,10,1500\n```")
    uploaded = st.file_uploader("Upload holdings CSV", type=["csv"])
    if uploaded:
        try:
            df_hold = pd.read_csv(uploaded, header=None)
            if df_hold.shape[1] == 2:
                df_hold.columns = ["Symbol","Quantity"]
            else:
                df_hold.columns = ["Symbol","Quantity","BuyPrice"]
            df_hold["Quantity"] = df_hold["Quantity"].astype(float)
            df_hold = compute_portfolio_value(df_hold)
            st.markdown("### Holdings")
            st.dataframe(df_hold.round(2), use_container_width=True)

            total_value = df_hold["Value"].sum()
            total_cost = df_hold["Cost"].sum() if "Cost" in df_hold.columns else np.nan
            pnl = total_value - total_cost if not np.isnan(total_cost) else np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Market Value", f"₹{total_value:,.2f}")
            if not np.isnan(pnl):
                c2.metric("Total P&L", f"₹{pnl:,.2f}", f"{(pnl/total_cost*100):+.2f}%")
            c3.metric("Tickers", df_hold["Symbol"].nunique())

            # allocation pie
            alloc = df_hold.groupby("Symbol")["Value"].sum().reset_index()
            fig = px.pie(alloc, names="Symbol", values="Value", title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.info("Upload a CSV to see portfolio analytics.")

# ----------------------------
# Allocation Advisor
# ----------------------------
elif nav == "Allocation Advisor":
    st.markdown('<div class="titlebig">AI Allocation Advisor</div>', unsafe_allow_html=True)
    st.markdown("Quick recommendation based on age, risk, and horizon (simple explainable rules).")
    age = st.number_input("Age", 25, 80, 35)
    risk = st.selectbox("Risk appetite", ["Low","Moderate","High"], index=1)
    horizon = st.slider("Investment horizon (years)", 1, 40, 10)
    if st.button("Suggest allocation"):
        alloc, rationale = suggest_allocation(age, risk, horizon)
        df_alloc = pd.DataFrame(list(alloc.items()), columns=["Asset Class","Allocation %"])
        st.dataframe(df_alloc, use_container_width=True)
        fig = px.bar(df_alloc, x="Asset Class", y="Allocation %", text="Allocation %")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Rationale:**")
        st.write(rationale)
        st.success("This is a rule-of-thumb suggestion. Consult advisor for tailor-made plans.")

# ----------------------------
# Goals & SIP
# ----------------------------
elif nav == "Goals & SIP":
    st.markdown('<div class="titlebig">Goals & SIP Simulator</div>', unsafe_allow_html=True)
    st.markdown("Project future corpus with lump-sum or SIP (monthly) inputs. Shows inflation-adjusted outcome and scenarios.")

    goal_type = st.selectbox("Mode", ["Lump sum projection", "SIP projection"], index=1)
    if goal_type == "Lump sum projection":
        principal = st.number_input("Current Investment (₹)", 10000, 100000000, 100000)
        years = st.slider("Years", 1, 40, 10)
        exp_ret = st.slider("Expected Annual Return (%)", 3.0, 20.0, 9.0)
        inflation = st.slider("Expected Inflation (%)", 0.0, 12.0, 4.5)
        fv = principal * ((1 + exp_ret/100) ** years)
        fv_real = fv / ((1 + inflation/100) ** years)
        st.metric("Projected corpus", f"₹{fv:,.0f}", f"Inflation-adjusted: ₹{fv_real:,.0f}")
    else:
        monthly = st.number_input("Monthly SIP (₹)", 100, 200000, 5000)
        years = st.slider("Years", 1, 40, 10)
        exp_ret = st.slider("Expected Annual Return (%)", 3.0, 20.0, 10.0)
        inflation = st.slider("Expected Inflation (%)", 0.0, 12.0, 4.5)
        fv, real_fv = sip_projection(monthly, years, exp_ret/100, inflation/100)
        st.metric("Projected corpus (nominal)", f"₹{fv:,.0f}", f"Inflation-adjusted: ₹{real_fv:,.0f}")
        # show growth curve
        months = years * 12
        vals = []
        balance = 0.0
        monthly_r = exp_ret/100/12
        for m in range(1, months+1):
            balance = balance*(1+monthly_r) + monthly
            vals.append(balance)
        df_curve = pd.DataFrame({"Month": range(1, months+1), "Balance": vals})
        st.line_chart(df_curve.set_index("Month")["Balance"], use_container_width=True)

# ----------------------------
# Efficient Frontier
# ----------------------------
elif nav == "Efficient Frontier":
    st.markdown('<div class="titlebig">Efficient Frontier Simulator</div>', unsafe_allow_html=True)
    st.markdown("Random-portfolio simulation to visualize risk-return frontier.")
    n = st.slider("Number of portfolios", 200, 5000, 1000, step=100)
    np.random.seed(42)
    returns = np.random.normal(0.09, 0.06, n)
    risks = np.random.normal(0.16, 0.04, n)
    df_front = pd.DataFrame({"Return": returns, "Risk": risks})
    fig = px.scatter(df_front, x="Risk", y="Return", color="Return", color_continuous_scale="Tealgrn")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Sector Heatmap (sample)
# ----------------------------
elif nav == "Sector Heatmap":
    st.markdown('<div class="titlebig">Sector Heatmap</div>', unsafe_allow_html=True)
    st.markdown("Snapshot of sectors using representative tickers. Colors show % change over last day.")
    sectors = {
        "IT": ["TCS.NS","INFY.NS","WIPRO.NS"],
        "Banking": ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS"],
        "Energy": ["RELIANCE.NS","ONGC.NS","BPCL.NS"],
        "Pharma": ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"],
        "FMCG": ["HINDUNILVR.NS","NESTLEIND.NS","BRITANNIA.NS"]
    }
    sector_perf = []
    for s, tickers in sectors.items():
        vals = []
        for t in tickers:
            d = fetch_symbol(t, period="5d", interval="1d")
            if not d.empty:
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2]) if len(d) >= 2 else last
                vals.append((last-prev)/prev*100 if prev!=0 else 0.0)
        avg = np.nan if len(vals)==0 else np.nanmean(vals)
        sector_perf.append({"Sector": s, "Change%": avg if not np.isnan(avg) else 0.0})
    df_sector = pd.DataFrame(sector_perf)
    fig = px.treemap(df_sector, path=["Sector"], values="Change%", color="Change%", color_continuous_scale="RdYlGn", title="Sector performance (avg % change)")
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Watchlist & Alerts
# ----------------------------
elif nav == "Watchlist":
    st.markdown('<div class="titlebig">Watchlist & Alerts</div>', unsafe_allow_html=True)
    st.markdown("Add tickers to watch. App will show current price and small sparkline. Alerts shown if change > threshold.")
    wl_text = st.text_input("Add tickers (comma separated, e.g. TCS.NS, INFY.NS)", value="TCS.NS, INFY.NS, RELIANCE.NS")
    threshold = st.number_input("Alert threshold (% change vs prev close)", 0.5, 10.0, 2.0)
    if st.button("Fetch Watchlist"):
        tickers = [t.strip().upper() for t in wl_text.split(",") if t.strip()]
        rows = []
        for t in tickers:
            d = fetch_symbol(t, period="5d", interval="1d")
            if d.empty:
                rows.append({"Symbol": t, "Price": "N/A", "Change%": "N/A", "Alert": ""})
            else:
                last = float(d["Close"].iloc[-1])
                prev = float(d["Close"].iloc[-2]) if len(d)>=2 else last
                pct = (last-prev)/prev*100 if prev != 0 else 0.0
                alert = "⚠️" if abs(pct) >= threshold else ""
                rows.append({"Symbol": t.replace(".NS",""), "Price": f"₹{last:,.2f}", "Change%": f"{pct:+.2f}%", "Alert": alert})
        st.table(pd.DataFrame(rows))

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("<div class='muted'>Built with ❤️ — AI advisor is rule-based for now. Always verify before investing.</div>", unsafe_allow_html=True)
