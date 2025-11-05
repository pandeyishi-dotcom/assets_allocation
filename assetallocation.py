# 🧾 PDF Report (from session data)
elif nav == "🧾 PDF Report":
    st.markdown("<div class='titlebig'>PDF Investor Report</div>", unsafe_allow_html=True)
    if not HAS_PDF:
        st.error("Install `fpdf` to enable PDF export: pip install fpdf")
    else:
        pf = st.session_state.get("portfolio_df")
        df_n, mode, info, ltd = fetch_nifty_smart()
        latest = float(df_n["Close"].iloc[-1])

        # Clean strings of Unicode that PDF can't handle
        def clean_text(txt):
            return str(txt).encode("latin-1", "replace").decode("latin-1")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, clean_text(f"Your Investment Guide — Report for {st.session_state.get('user_name','Investor')}"), ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, clean_text(f"Greeting: {greeting()}"), ln=True)
        pdf.cell(0, 8, clean_text(f"Quote: {st.session_state.get('quote','')}"), ln=True)
        pdf.cell(0, 8, clean_text(f"NIFTY: {latest:,.2f}  | Mode: {mode}  | Source: {info}"), ln=True)
        pdf.ln(4)

        if pf is not None:
            pdf.cell(0, 8, "Portfolio Allocation:", ln=True)
            bysym = pf.groupby("Symbol")["Value"].sum().reset_index().sort_values("Value", ascending=False)
            for _,r in bysym.iterrows():
                pdf.cell(0, 7, clean_text(f" - {r['Symbol']}: Rs {r['Value']:,.0f}"), ln=True)
        else:
            pdf.cell(0, 8, "Portfolio: Not uploaded", ln=True)

        # Output safely to buffer
        buf = BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin-1", "replace")
        buf.write(pdf_bytes)
        buf.seek(0)

        st.download_button("Download PDF", buf, file_name="Your_Investment_Report.pdf", mime="application/pdf")
        st.info("Report includes greeting, quote, NIFTY snapshot, and portfolio allocation (if uploaded).")
