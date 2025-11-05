elif nav == "Live Market":
    st.markdown('<div class="titlebig">Live Indian Market Tracker (NIFTY)</div>', unsafe_allow_html=True)

    nifty_df, mode, reason, ts_str, last_trade_date = fetch_safe_nifty_v2()

    # Mode badge
    badge_color = {"live":"#16a34a", "fallback":"#d97706", "offline":"#ef4444"}[mode]
    st.markdown(
        f"""
        <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;">
            <span style="background:{badge_color};color:white;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700;text-transform:uppercase;">
                {mode}
            </span>
            <span style="color:#cfeee6;">{reason}</span>
            <span style="color:#9aa5a6;">• Last refreshed: <b>{ts_str}</b></span>
            <span style="color:#9aa5a6;">• Last trading day: <b>{last_trade_date:%d %b %Y}</b></span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Choose series to plot
    if "Close" in nifty_df.columns:
        series_to_plot = nifty_df["Close"]
    else:
        numeric_cols = nifty_df.select_dtypes(include=[np.number]).columns
        series_to_plot = nifty_df[numeric_cols[0]] if len(numeric_cols) else pd.to_numeric(nifty_df.iloc[:, 0], errors="coerce")

    latest = float(series_to_plot.iloc[-1])
    prev = float(series_to_plot.iloc[-2]) if len(series_to_plot) > 1 else latest
    change = latest - prev
    pct = (change / prev) * 100 if prev != 0 else 0.0

    # Live vs LTP message
    colA, colB, colC = st.columns(3)
    if mode == "live":
        colA.subheader("✅ Market Live")
        colA.metric("NIFTY 50", f"₹{latest:,.2f}", f"{pct:+.2f}%")
    else:
        colA.subheader("📅 Market Closed / Fallback — showing Last Close")
        colA.metric("Last Traded Price (LTP)", f"₹{latest:,.2f}")

    colB.metric("Δ (points)", f"{change:+.2f}")
    colC.metric("Δ (%)", f"{pct:+.2f}%")

    st.markdown("### Price movement")
    try:
        st.line_chart(series_to_plot.rename("Price"), use_container_width=True)
    except Exception:
        try:
            st.line_chart(pd.to_numeric(nifty_df.iloc[:, 0], errors="coerce"), use_container_width=True)
        except Exception:
            st.info("Chart unavailable.")

    st.markdown("### Recent rows")
    st.dataframe(nifty_df.tail(10).round(2), use_container_width=True)
