# -------------------------
# Sidebar — Enhanced Design (Ishani Edition)
# -------------------------
with st.sidebar:
    st.markdown(f"""
    <style>
      div[data-testid="stSidebar"] {{
          background: linear-gradient(180deg, #071427 0%, #0c1b33 100%);
          color: #e6eef0;
          padding-top: 1rem;
          border-right: 1px solid rgba(255,255,255,0.05);
      }}
      .sidebar-title {{
          font-size: 24px;
          font-weight: 700;
          color: {ACCENT};
          text-align: center;
          margin-top: 10px;
      }}
      .sidebar-sub {{
          text-align:center;
          font-size:13px;
          color:{MUTED};
          margin-bottom:20px;
      }}
      .nav-radio label {{
          display: flex;
          align-items: center;
          gap: 8px;
          font-size:15px !important;
          color:#d8e2e3 !important;
      }}
      .nav-radio div[role='radiogroup'] > label:hover {{
          background-color: rgba(0,255,198,0.1);
          border-radius: 6px;
          transition: background 0.3s ease;
      }}
      .sidebar-footer {{
          margin-top:40px;
          text-align:center;
          font-size:12px;
          color:{MUTED};
      }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<div style='text-align:center;'><img src='https://upload.wikimedia.org/wikipedia/commons/6/6b/NSE_Logo.svg' width='90'></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-title'>AI Portfolio Cockpit</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Curated by Ishani</div>", unsafe_allow_html=True)

    # Icons for radio labels
    nav_labels = {
        "Home": "🏠 Home",
        "Live Market": "📈 Live Market",
        "Market Pulse": "🌐 Market Pulse",
        "Portfolio": "💼 Portfolio",
        "Asset Allocation": "📊 Asset Allocation",
        "Allocation Advisor": "🧠 Allocation Advisor",
        "Goals & SIP": "🎯 Goals & SIP",
        "Sector Heatmap": "🔥 Sector Heatmap",
        "Watchlist": "👁️ Watchlist"
    }

    nav = st.radio(
        "Navigate",
        list(nav_labels.keys()),
        format_func=lambda x: nav_labels[x],
        index=1,
        key="nav_radio",
        label_visibility="collapsed"
    )

    st.markdown("<div class='sidebar-footer'>Built for Indian markets • Auto-fallbacks included<br>⚙️ Intelligent. Minimal. Adaptive.</div>", unsafe_allow_html=True)
