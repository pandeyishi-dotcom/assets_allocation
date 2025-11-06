# app.py — Your Investment Guide (V6.1, Professional Edition)
# Requirements: streamlit, yfinance, pandas, numpy, plotly, fpdf, pillow
# pip install streamlit yfinance pandas numpy plotly fpdf pillow

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from fpdf import FPDF
from datetime import datetime, date
import io
import random

# -------------------------#
# CONFIGURATION & THEME
# -------------------------#
st.set_page_config(page_title="Your Investment Guide", page_icon="💼", layout="wide")

ACCENT = "#00FFC6"
BG1 = "#0b1320"
BG2 = "#0f1b2b"
TEXT = "#e6eef0"
MUTED = "#9fb4c8"

st.markdown(
    f"""
    <style>
      .stApp {{
        background: linear-gradient(180deg, {BG1} 0%, {BG2} 100%);
        color: {TEXT};
        font-family: 'Inter', sans-serif;
      }}
      div[data-testid="stSidebar"] {{
        background: #0a1220;
        border-right: 1px solid rgba(255,255,255,0.1);
      }}
      .titlebig {{ font-size:28px; font-weight:700; color:{ACCENT}; }}
      .muted {{ color:{MUTED}; }}
      .badge {{ padding:2px 8px; border-radius:10px; font-size:12px; border:1px solid {ACCENT}; color:{ACCENT}; }}
      .badge-red {{ color:#ff8080; border:1px solid #ff8080; }}
      .badge-amber {{ color:#ffcc66; border:1px solid #ffcc66; }}
      .badge-green {{ color:#66ffcc; border:1px solid #66ffcc; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------#
# UTILITIES
# -------------------------#

def greet():
    h = datetime.now().hour
    return "Good Morning" if h < 12 else ("Good Afternoon" if h < 17 else "Good Evening")


def fmt_inr(x):
    try:
        return f"₹{float(x):,.0f}"
    except Exception:
        return "₹0"


def to_latin1(s):
    return s.encode("latin-1", "ignore").decode("latin-1")


@st.cache_data(ttl=300)
def fetch(symbol, period="5d", interval="15m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def fetch_nifty_smart():
    try:
        df = fetch("^NSEI", "5d", "15m")
        if not df.empty:
            return df, "Live", "Intraday (15m)"
        df = fetch("^NSEI", "1mo", "1d")
        if not df.empty:
            return df, "Fallback", "Daily close (1mo)"
    except Exception:
        pass

    sample = {
        "Datetime": pd.date_range(end=pd.Timestamp.now(), periods=10, freq="D"),
        "Close": [20000, 20050, 20100, 20200, 20300, 20250, 20320, 20400, 20450, 20500],
    }
    return pd.DataFrame(sample).set_index("Datetime"), "Offline", "Sample data"


def safe_plot(series, title="Price Chart"):
    """Safe plotting utility to avoid rename errors."""
    try:
        if isinstance(series, pd.DataFrame):
            series = series.select_dtypes(include=[np.number]).iloc[:, 0]
        fig = px.line(x=series.index, y=series.values, labels={"x": "Date", "y": "Price"}, title=title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("⚠️ Could not plot chart — simplified fallback used.")
        try:
            st.line_chart(pd.to_numeric(series.squeeze(), errors="coerce"), use_container_width=True)
        except Exception:
            st.write("Chart unavailable.")


@st.cache_data
def get_supported_etfs():
    # Simple list of popular India ETFs / index-related tickers for Market Pulse and Sector analytics
    return [
        "^NSEI",
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "HINDUNILVR.NS",
        "ONGC.NS",
    ]

# -------------------------#
# SIDEBAR
# -------------------------#
QUOTES = [
    "Invest early. Time is the compounding engine.",
    "Discipline compounds like capital.",
    "Diversify not to avoid risk, but to domesticate it.",
    "Plan like a pessimist; invest like an optimist.",
    "Markets are volatile; your process shouldn’t be."
]

with st.sidebar:
    st.markdown("### 💼 Your Investment Guide")
    user_name = st.text_input("Your Name", value=st.session_state.get("user_name", "Ishani")).strip()
    if user_name:
        st.session_state["user_name"] = user_name
    st.caption(f"{greet()}, {st.session_state['user_name']} 👋")

    with st.expander("💬 Quote of the Day", expanded=True):
        if "quote" not in st.session_state:
            st.session_state.quote = random.choice(QUOTES)
        if st.button("🔄 Shuffle Quote", key="shuffle_quote"):
            st.session_state.quote = random.choice(QUOTES)
        st.markdown(f"_{st.session_state.quote}_")

    st.markdown("---")
    nav = st.radio(
        "Navigate",
        [
            "Overview",
            "Live Market",
            "Market Pulse",
            "Portfolio",
            "Sector Analytics",
            "Asset Allocation",
            "Allocation Advisor",
            "Goals & SIP",
            "Monte Carlo",
            "PDF Report",
            "Watchlist",
        ],
        index=0,
    )

# -------------------------#
# OVERVIEW
# -------------------------#
if nav == "Overview":
    st.markdown(f"<div class='titlebig'>Your Investment Guide</div>", unsafe_allow_html=True)
    st.write("Professional, personalized investor cockpit for Indian markets.")
    nifty_df, mode, reason = fetch_nifty_smart()
    if "Close" in nifty_df.columns:
        series = nifty_df["Close"]
    else:
        series = nifty_df.select_dtypes(include=[np.number]).iloc[:, 0]

    latest, prev = float(series.iloc[-1]), float(series.iloc[-2]) if len(series) > 1 else float(series.iloc[-1])
    chg = latest - prev
    pct = (chg / prev) * 100 if prev else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY 50", f"{latest:,.2f}", f"{pct:+.2f}%")
    c2.metric("Δ (pts)", f"{chg:+.2f}")
    badge_class = "badge-green" if mode == "Live" else ("badge-amber" if mode == "Fallback" else "badge-red")
    c3.markdown(f"<span class='{badge_class}'>Mode: {mode}</span><br><span class='muted'>{reason}</span>", unsafe_allow_html=True)

    st.markdown("#### Price Movement")
    safe_plot(series)
    st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

# -------------------------#
# LIVE MARKET
# -------------------------#
elif nav == "Live Market":
    st.markdown(f"<div class='titlebig'>💹 Live Indian Market</div>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
    nifty_df, mode, reason = fetch_nifty_smart()
    st.caption(f"Mode: {mode} — {reason}")

    series = nifty_df["Close"] if "Close" in nifty_df.columns else nifty_df.select_dtypes(include=[np.number]).iloc[:, 0]
    latest, prev = float(series.iloc[-1]), float(series.iloc[-2]) if len(series) > 1 else float(series.iloc[-1])
    chg = latest - prev
    pct = (chg / prev) * 100 if prev else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY 50", f"{latest:,.2f}")
    c2.metric("Δ (pts)", f"{chg:+.2f}")
    c3.metric("Δ (%)", f"{pct:+.2f}%")
    safe_plot(series)
    st.caption(f"Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

# -------------------------#
# MARKET PULSE
# -------------------------#
elif nav == "Market Pulse":
    st.markdown(f"<div class='titlebig'>📈 Market Pulse</div>", unsafe_allow_html=True)
    tickers = get_supported_etfs()
    selected = st.multiselect("Select tickers to view", tickers, default=tickers[:6])
    if not selected:
        st.info("Pick at least one ticker to view the pulse.")
    else:
        cols = st.columns(2)
        for i, ticker in enumerate(selected):
            with cols[i % 2]:
                df = fetch(ticker, period="5d", interval="15m")
                if df.empty:
                    st.write(ticker)
                    st.warning("No data (sample/offline).")
                else:
                    st.subheader(ticker)
                    series = df["Close"] if "Close" in df.columns else df.select_dtypes(include=[np.number]).iloc[:, 0]
                    safe_plot(series, title=f"{ticker} price")
                    latest = float(series.iloc[-1])
                    prev = float(series.iloc[-2]) if len(series) > 1 else latest
                    chg = latest - prev
                    st.metric("Latest", f"{latest:,.2f}", f"{chg:+.2f}")

# -------------------------#
# PORTFOLIO
# -------------------------#
elif nav == "Portfolio":
    st.markdown(f"<div class='titlebig'>💼 Portfolio</div>", unsafe_allow_html=True)
    st.write("Upload a CSV with columns: Ticker, Quantity, AvgPrice (optional). Ticker examples: RELIANCE.NS, INFY.NS")

    uploaded = st.file_uploader("Upload portfolio CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        required = ["Ticker", "Quantity"]
        if not all(c in df.columns for c in required):
            st.error(f"CSV must contain columns: {required}")
        else:
            df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(0)
            if "AvgPrice" not in df.columns:
                df["AvgPrice"] = np.nan
            df["AvgValue"] = df["Quantity"] * df["AvgPrice"].fillna(0)

            # fetch current prices in batch
            prices = {}
            tickers = df["Ticker"].unique().tolist()
            for t in tickers:
                try:
                    info = yf.Ticker(t)
                    hist = info.history(period="5d")
                    if not hist.empty:
                        prices[t] = hist["Close"].iloc[-1]
                    else:
                        prices[t] = np.nan
                except Exception:
                    prices[t] = np.nan

            df["MarketPrice"] = df["Ticker"].map(prices)
            df["MarketValue"] = df["Quantity"] * df["MarketPrice"].fillna(0)
            df["PnL"] = (df["MarketPrice"] - df["AvgPrice"]) * df["Quantity"]

            st.subheader("Holdings")
            st.dataframe(df)

            total_mv = df["MarketValue"].sum()
            st.markdown(f"**Total Market Value:** {fmt_inr(total_mv)}")

            if total_mv > 0:
                alloc = df.groupby("Ticker")["MarketValue"].sum().reset_index()
                alloc["Pct"] = alloc["MarketValue"] / alloc["MarketValue"].sum() * 100
                fig = px.pie(alloc, names="Ticker", values="MarketValue", title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Trade Simulator")
            t_ticker = st.selectbox("Select ticker to trade", df["Ticker"].tolist())
            t_action = st.selectbox("Action", ["Buy", "Sell"]) 
            t_qty = st.number_input("Quantity", min_value=1, value=1)
            if st.button("Simulate Trade"):
                price = prices.get(t_ticker, np.nan)
                if np.isnan(price):
                    st.warning("Market price unavailable for this ticker.")
                else:
                    if t_action == "Buy":
                        df.loc[df["Ticker"] == t_ticker, "Quantity"] += t_qty
                    else:
                        df.loc[df["Ticker"] == t_ticker, "Quantity"] = np.maximum(0, df.loc[df["Ticker"] == t_ticker, "Quantity"] - t_qty)
                    st.success("Trade applied (in-session). Refresh portfolio viewer to re-upload if you want to persist.)")

    else:
        st.info("Upload a portfolio CSV to get started. Or use the sample below.")
        if st.button("Load sample portfolio"):
            sample = pd.DataFrame({
                "Ticker": ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"],
                "Quantity": [10, 20, 15],
                "AvgPrice": [2500, 1200, 1500]
            })
            csv = sample.to_csv(index=False).encode("utf-8")
            st.download_button("Download sample CSV", data=csv, file_name="sample_portfolio.csv")

# -------------------------#
# SECTOR ANALYTICS
# -------------------------#
elif nav == "Sector Analytics":
    st.markdown(f"<div class='titlebig'>🔎 Sector Analytics</div>", unsafe_allow_html=True)
    st.write("Simple sector view using a user-provided mapping file or default sample sectors.")

    with st.expander("How it works"):
        st.write("Upload a CSV with columns: Ticker, Sector. The app will fetch current prices and show sector-level metrics.")

    uploaded = st.file_uploader("Upload ticker->sector mapping (optional)", type=["csv"], key="sector_map")
    if uploaded:
        mapping = pd.read_csv(uploaded)
        if not all(c in mapping.columns for c in ["Ticker", "Sector"]):
            st.error("Mapping must contain Ticker and Sector columns.")
        else:
            tickers = mapping["Ticker"].unique().tolist()
            prices = {}
            for t in tickers:
                try:
                    hist = yf.Ticker(t).history(period="1d")
                    prices[t] = hist["Close"].iloc[-1] if not hist.empty else np.nan
                except Exception:
                    prices[t] = np.nan
            mapping["Price"] = mapping["Ticker"].map(prices)
            sector = mapping.groupby("Sector").agg({"Price": ["count", "mean"]})
            sector.columns = ["Count", "AvgPrice"]
            st.dataframe(sector.reset_index())
            fig = px.bar(sector.reset_index(), x="Sector", y="AvgPrice", title="Average Price by Sector")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No mapping uploaded — showing sample sectors.")
        sample = pd.DataFrame({
            "Ticker": ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","ITC.NS","ONGC.NS"],
            "Sector": ["Energy","IT","Financials","IT","FMCG","Energy"]
        })
        st.dataframe(sample)
        st.write("Run the sample through the same flow by downloading and re-uploading it if you'd like to try the full pipeline.")

# -------------------------#
# ASSET ALLOCATION
# -------------------------#
elif nav == "Asset Allocation":
    st.markdown(f"<div class='titlebig'>📊 Asset Allocation</div>", unsafe_allow_html=True)
    st.write("Create a target allocation and visualize it. Supports Cash, Equity, Debt, Gold, Others.")

    with st.form("allocation_form"):
        eq = st.number_input("Equity (%)", min_value=0, max_value=100, value=60)
        debt = st.number_input("Debt (%)", min_value=0, max_value=100, value=25)
        gold = st.number_input("Gold (%)", min_value=0, max_value=100, value=5)
        cash = st.number_input("Cash (%)", min_value=0, max_value=100, value=5)
        others = st.number_input("Others (%)", min_value=0, max_value=100, value=5)
        submitted = st.form_submit_button("Set Allocation")
    if submitted:
        total = eq + debt + gold + cash + others
        if total != 100:
            st.error("Allocation must sum to 100% — please adjust values.")
        else:
            alloc = pd.DataFrame({"Asset": ["Equity","Debt","Gold","Cash","Others"], "Pct": [eq, debt, gold, cash, others]})
            st.dataframe(alloc)
            fig = px.pie(alloc, names="Asset", values="Pct", title="Target Asset Allocation")
            st.plotly_chart(fig, use_container_width=True)

# -------------------------#
# ALLOCATION ADVISOR
# -------------------------#
elif nav == "Allocation Advisor":
    st.markdown(f"<div class='titlebig'>🧭 Allocation Advisor</div>", unsafe_allow_html=True)
    age = st.slider("Your age", 18, 75, 30)
    risk = st.selectbox("Risk tolerance", ["Low","Moderate","High"], index=1)
    horizon = st.selectbox("Investment horizon", ["<1 year","1-3 years","3-5 years",">5 years"], index=3)

    def advisor(age, risk, horizon):
        # Simple rule-based advisor
        if risk == "High":
            eq = 80 if age < 50 else 70
        elif risk == "Moderate":
            eq = 60 if age < 50 else 50
        else:
            eq = 40 if age < 50 else 30

        if horizon == "<1 year":
            eq = max(0, eq - 30)
        elif horizon == "1-3 years":
            eq = max(0, eq - 10)

        debt = 100 - eq - 5
        gold = 5
        cash = 5
        return {"Equity": eq, "Debt": debt, "Gold": gold, "Cash": cash}

    rec = advisor(age, risk, horizon)
    st.markdown("### Recommended allocation")
    st.write(rec)
    fig = px.pie(values=list(rec.values()), names=list(rec.keys()), title="Recommended Allocation")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------#
# GOALS & SIP
# -------------------------#
elif nav == "Goals & SIP":
    st.markdown(f"<div class='titlebig'>🎯 Goals & SIP Calculator</div>", unsafe_allow_html=True)
    goal_amt = st.number_input("Target corpus (₹)", min_value=1000, value=1000000, step=1000)
    years = st.number_input("Years to achieve", min_value=1, max_value=50, value=10)
    est_return = st.slider("Expected annual return (%)", min_value=1.0, max_value=25.0, value=10.0)

    def sip_required(target, years, r):
        # r is annual return in %
        m = years * 12
        rm = r / 100 / 12
        if rm == 0:
            return target / m
        sip = (target * rm) / ((1 + rm) ** m - 1)
        return sip

    sip = sip_required(goal_amt, years, est_return)
    st.markdown(f"**Monthly SIP required:** {fmt_inr(sip)}")

    st.markdown("#### SIP Projection (nominal)")
    months = years * 12
    arr = []
    bal = 0
    rm = est_return / 100 / 12
    for i in range(1, months + 1):
        bal = bal * (1 + rm) + sip
        arr.append(bal)
    df_proj = pd.DataFrame({"Month": range(1, months + 1), "Balance": arr})
    fig = px.line(df_proj, x="Month", y="Balance", title="SIP Projection")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------#
# MONTE CARLO
# -------------------------#
elif nav == "Monte Carlo":
    st.markdown(f"<div class='titlebig'>🎲 Monte Carlo Simulator</div>", unsafe_allow_html=True)
    st.write("Estimate distribution of portfolio value after a horizon using geometric Brownian motion.")
    start_val = st.number_input("Current portfolio value (₹)", min_value=1000, value=1000000, step=1000)
    years = st.number_input("Years to simulate", min_value=1, max_value=50, value=10)
    mu = st.number_input("Expected annual return (%)", value=8.0)
    sigma = st.number_input("Annual volatility (%)", value=15.0)
    sims = st.number_input("Number of simulations", min_value=100, max_value=2000, value=400)

    if st.button("Run Simulation"):
        np.random.seed(42)
        dt = 1/252
        steps = int(years * 252)
        results = np.zeros((sims, steps + 1))
        results[:, 0] = start_val
        mu_d = mu / 100
        sigma_d = sigma / 100
        for t in range(1, steps + 1):
            z = np.random.standard_normal(sims)
            results[:, t] = results[:, t - 1] * np.exp((mu_d - 0.5 * sigma_d ** 2) * dt + sigma_d * np.sqrt(dt) * z)
        final_vals = results[:, -1]
        st.write(f"Median outcome: {fmt_inr(np.median(final_vals))}")
        st.write(f"5th percentile: {fmt_inr(np.percentile(final_vals, 5))}")
        st.write(f"95th percentile: {fmt_inr(np.percentile(final_vals, 95))}")
        fig = go.Figure()
        for i in range(min(50, sims)):
            fig.add_trace(go.Scatter(y=results[i, :], mode='lines', line=dict(width=0.5), opacity=0.6))
        fig.update_layout(title='Monte Carlo paths (sample)', xaxis_title='Trading day', yaxis_title='Portfolio value')
        st.plotly_chart(fig, use_container_width=True)

# -------------------------#
# PDF REPORT
# -------------------------#
elif nav == "PDF Report":
    st.markdown(f"<div class='titlebig'>📄 PDF Report</div>", unsafe_allow_html=True)
    st.write("Generate a simple PDF summary of your portfolio and key metrics.")

    uploaded = st.file_uploader("Upload portfolio CSV for PDF report", type=["csv"], key="pdf_portfolio")
    if uploaded:
        df = pd.read_csv(uploaded)
        if not all(c in df.columns for c in ["Ticker", "Quantity"]):
            st.error("CSV must contain Ticker and Quantity columns.")
        else:
            # basic metric calculations
            tickers = df["Ticker"].unique().tolist()
            prices = {}
            for t in tickers:
                try:
                    hist = yf.Ticker(t).history(period="5d")
                    prices[t] = hist["Close"].iloc[-1] if not hist.empty else np.nan
                except Exception:
                    prices[t] = np.nan
            df["Price"] = df["Ticker"].map(prices)
            df["MarketValue"] = df["Price"] * df["Quantity"]
            total = df["MarketValue"].sum()

            # create PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=14)
            pdf.cell(200, 10, txt="Portfolio Report", ln=True, align='C')
            pdf.set_font("Arial", size=10)
            pdf.ln(4)
            pdf.cell(200, 8, txt=f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}", ln=True)
            pdf.ln(4)
            pdf.cell(200, 8, txt=f"Total Market Value: ₹{total:,.2f}", ln=True)
            pdf.ln(6)
            pdf.set_font("Arial", size=9)
            for _, row in df.iterrows():
                pdf.cell(0, 6, txt=f"{row['Ticker']}: Qty {row['Quantity']} | Price {row['Price']:.2f} | MV ₹{row['MarketValue']:.2f}", ln=True)
            buf = io.BytesIO()
            pdf.output(buf)
            buf.seek(0)
            st.download_button("Download PDF report", data=buf, file_name="portfolio_report.pdf", mime='application/pdf')
    else:
        st.info("Upload a portfolio CSV to generate PDF report.")

# -------------------------#
# WATCHLIST
# -------------------------#
elif nav == "Watchlist":
    st.markdown(f"<div class='titlebig'>👀 Watchlist</div>", unsafe_allow_html=True)
    wl = st.text_area("Enter tickers (comma separated)", value=st.session_state.get("watchlist_input", "RELIANCE.NS, INFY.NS, TCS.NS"))
    if wl:
        st.session_state["watchlist_input"] = wl
    tickers = [t.strip() for t in wl.split(",") if t.strip()]
    if not tickers:
        st.info("Add tickers to your watchlist.")
    else:
        cols = st.columns(len(tickers))
        for i, t in enumerate(tickers):
            with cols[i]:
                df = fetch(t, period="5d", interval="15m")
                st.subheader(t)
                if df.empty:
                    st.warning("No data")
                else:
                    series = df["Close"] if "Close" in df.columns else df.select_dtypes(include=[np.number]).iloc[:, 0]
                    latest = float(series.iloc[-1])
                    prev = float(series.iloc[-2]) if len(series) > 1 else latest
                    chg = latest - prev
                    st.metric("Price", f"{latest:,.2f}", f"{chg:+.2f}")
                    safe_plot(series, title=f"{t} price")

# -------------------------#
# DEFAULT
# -------------------------#
else:
    st.write("Select a module from the sidebar.")

# End of app
