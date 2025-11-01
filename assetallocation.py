# assetallocation_final.py
# Full fintech cockpit (with improved sidebar integration)
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
st.set_page_config(page_title="AI Portfolio Cockpit — Ishani", layout="wide", page_icon="💠")
ACCENT = "#00FFC6"
BG = "#0e1117"
CARD = "#0f1720"
MUTED = "#9aa5a6"

st.markdown(
    f"""
    <style>
      body {{ background: {BG}; color: #e6eef0; }}
      div[data-testid="stSidebar"]{{background:#071427;}}
      .titlebig{{ font-size:28px; color:{ACCENT}; font-weight:700; margin-bottom:6px; }}
      .muted{{ color:{MUTED
