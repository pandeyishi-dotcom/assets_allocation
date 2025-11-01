# ------------------ LIVE MARKET (robust + fallback) ------------------ #
if menu == "💹 Live Market":
    import pytz
    from datetime import datetime, date

    st.header("💹 Live Indian Market Tracker (NIFTY 50)")

    # Helper: attempt multiple fetch strategies, return (df, reason)
    def fetch_nifty_safe():
        reason = []
        # 1) Try 15m intraday (5 days)
        try:
            df = yf.download("^NSEI", period="5d", interval="15m", progress=False)
            if not df.empty:
                reason.append("intraday 15m (5d)")
                return df, "live", "; ".join(reason)
        except Exception as e:
            reason.append(f"intraday failed: {e}")

        # 2) Try daily 1 month (fallback to LTP)
        try:
            df = yf.download("^NSEI", period="1mo", interval="1d", progress=False)
            if not df.empty:
                reason.append("daily 1mo (fallback)")
                return df, "ltp", "; ".join(reason)
        except Exception as e:
            reason.append(f"daily failed: {e}")

        # 3) Try Ticker.history (alternative call)
        try:
            t = yf.Ticker("^NSEI")
            df = t.history(period="1mo", interval="1d")
            if not df.empty:
                reason.append("Ticker.history 1mo")
                return df, "ltp", "; ".join(reason)
        except Exception as e:
            reason.append(f"Ticker.history failed: {e}")

        # 4) Final: empty -> return None
        return None, "error", "; ".join(reason)

    # Fetch once
    nifty_df, mode, fetch_reason = fetch_nifty_safe()

    # If fetching failed, use a small sample so UI stays alive
    if nifty_df is None or nifty_df.empty:
        st.warning("⚠️ Could not fetch live data from Yahoo Finance — using local sample data. (Reason logged below)")
        sample = {
            "Datetime": pd.date_range(end=pd.Timestamp.now(), periods=5, freq="D"),
            "Open": [20000, 20100, 19900, 20250, 20300],
            "High": [20150, 20250, 20050, 20400, 20450],
            "Low": [19950, 20050, 19850, 20100, 20200],
            "Close": [20100, 19950, 20200, 20300, 20350],
            "Volume": [1000, 1100, 1050, 1200, 900]
        }
        nifty_df = pd.DataFrame(sample).set_index("Datetime")
        data_mode = "Sample Data (no live)"
        fetch_reason = fetch_reason or "unknown"
    else:
        data_mode = "Live" if mode == "live" else "Market Closed / LTP"

    # Show reason (concise)
    st.caption(f"Data source: {data_mode} — fetch info: {fetch_reason}")

    # Prepare display (latest price, LTP logic)
    latest_price = round(float(nifty_df["Close"].iloc[-1]), 2)
    last_timestamp = nifty_df.index[-1]
    last_date = pd.to_datetime(last_timestamp).date()
    today = date.today()

    col1, col2 = st.columns([2, 1])

    # If last_date < today => market closed (show LTP)
    if last_date < today:
        col1.subheader("📅 Market Closed — showing LTP")
        col1.metric("Last Traded Price (LTP)", f"₹{latest_price:,.2f}")
    else:
        col1.subheader("✅ Market Live")
        # compute previous close robustly
        if len(nifty_df) >= 2:
            prev_close = float(nifty_df["Close"].iloc[-2])
            change = latest_price - prev_close
            pct = (change / prev_close) * 100 if prev_close != 0 else 0.0
            col1.metric("NIFTY 50", f"₹{latest_price:,.2f}", delta=f"{change:+.2f} ({pct:+.2f}%)")
        else:
            col1.metric("NIFTY 50", f"₹{latest_price:,.2f}")

    # Quick chart and table
    st.subheader("📈 Price Movement (recent)")
    # prefer using Close series
    try:
        st.line_chart(nifty_df["Close"].rename("Close"), use_container_width=True, height=350)
    except Exception:
        st.line_chart(nifty_df.iloc[:, 0], use_container_width=True, height=350)

    st.subheader("📊 Recent Data (latest rows)")
    try:
        st.dataframe(nifty_df.tail(10).round(2), use_container_width=True)
    except Exception:
        st.table(nifty_df.tail(5).round(2))

    # extra: top NIFTY samples (attempt fetch; tolerant)
    st.markdown("---")
    st.subheader("🔎 Sample Stocks — TCS | INFY | RELIANCE | HDFCBANK")
    sample_symbols = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS"]
    stocks = []
    for s in sample_symbols:
        try:
            d = yf.download(s, period="5d", interval="1d", progress=False)
            if not d.empty:
                last = round(d["Close"].iloc[-1], 2)
                prev = round(d["Close"].iloc[-2], 2) if len(d) >= 2 else last
                pct = round(((last - prev) / prev) * 100, 2) if prev != 0 else 0.0
                stocks.append({"Symbol": s.replace(".NS", ""), "Price (₹)": last, "Change (%)": pct})
        except Exception:
            stocks.append({"Symbol": s.replace(".NS", ""), "Price (₹)": "N/A", "Change (%)": "N/A"})
    st.table(pd.DataFrame(stocks))
