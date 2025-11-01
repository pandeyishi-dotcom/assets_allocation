import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import random

# -------------------------
# CONFIGURATION
# -------------------------
st.set_page_config(page_title="Smart Asset Allocation Planner V2", layout="wide")

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def get_greeting(name):
    hour = datetime.now().hour
    if hour < 12:
        time_greet = "Good Morning"
    elif hour < 17:
        time_greet = "Good Afternoon"
    else:
        time_greet = "Good Evening"
    quotes = [
        "Invest in your future, one decision at a time.",
        "Wealth is not built in a day, but daily.",
        "Discipline is the best investment strategy.",
        "The best time to start was yesterday. The next best is today.",
        "Smart investing is 90% patience, 10% action."
    ]
    return f"{time_greet}, {name}! 🌞", random.choice(quotes)

# -------------------------
# USER INPUT
# -------------------------
with st.sidebar:
    st.title("⚙️ Personal Dashboard Setup")
    name = st.text_input("Enter your name", "")
    risk_profile = st.selectbox("Select your Risk Profile", ["Low", "Moderate", "High"])
    investment_goal = st.selectbox("Investment Goal", ["Wealth Creation", "Retirement", "Child Education", "Home Purchase"])
    investment_amount = st.number_input("Total Investment Amount (₹)", min_value=10000, step=5000)
    goal_years = st.slider("Years to Goal", 1, 30, 10)
    st.divider()

    if st.button("Save & Generate Plan"):
        st.session_state["name"] = name
        st.session_state["risk"] = risk_profile
        st.session_state["goal"] = investment_goal
        st.session_state["amount"] = investment_amount
        st.session_state["years"] = goal_years
        st.rerun()

# -------------------------
# MAIN PAGE
# -------------------------
if "name" not in st.session_state or st.session_state["name"] == "":
    st.warning("👈 Please enter your details in the sidebar to begin.")
else:
    name = st.session_state["name"]
    risk_profile = st.session_state["risk"]
    investment_goal = st.session_state["goal"]
    investment_amount = st.session_state["amount"]
    goal_years = st.session_state["years"]

    greet, quote = get_greeting(name)
    st.markdown(f"### {greet}")
    st.info(f"💬 *{quote}*")

    st.markdown(f"#### 🧭 Goal: {investment_goal}")
    st.markdown(f"**Investment Horizon:** {goal_years} years  |  **Total Invested:** ₹{investment_amount:,.0f}")
    st.divider()

    # -------------------------
    # ALLOCATION LOGIC
    # -------------------------
    if risk_profile == "Low":
        allocation = {"Debt Funds": 50, "Equity Large Cap": 25, "Gold": 15, "Cash": 10}
    elif risk_profile == "Moderate":
        allocation = {"Equity Large Cap": 40, "Mid Cap": 20, "Debt Funds": 25, "Gold": 10, "REITs": 5}
    else:  # High
        allocation = {"Equity Large Cap": 35, "Mid Cap": 25, "Small Cap": 15, "International": 10, "Debt Funds": 10, "Gold": 5}

    df = pd.DataFrame({
        "Asset Class": allocation.keys(),
        "Allocation (%)": allocation.values()
    })
    df["Amount (₹)"] = (df["Allocation (%)"] / 100) * investment_amount

    st.subheader("📊 Recommended Portfolio Allocation")
    st.dataframe(df, use_container_width=True)

    # -------------------------
    # EXPECTED RETURNS
    # -------------------------
    returns = {"Low": 0.07, "Moderate": 0.10, "High": 0.13}
    expected_return = returns[risk_profile]
    future_value = investment_amount * ((1 + expected_return) ** goal_years)
    st.metric("Expected Future Value", f"₹{future_value:,.0f}", f"{expected_return * 100:.1f}% annualized")

    # -------------------------
    # PROGRESS VISUALIZATION
    # -------------------------
    progress = min((goal_years / 30), 1.0)
    st.progress(progress, text=f"Goal progress: {goal_years}/30 years")

    # -------------------------
    # TIPS
    # -------------------------
    st.divider()
    st.subheader("💡 Personalized Insights")
    tips = {
        "Low": "Focus on stability and steady income. Consider tax-saving debt instruments.",
        "Moderate": "Balance growth and safety. Use SIPs in large and mid-cap funds.",
        "High": "Take calculated risks. Diversify globally and monitor volatility monthly."
    }
    st.success(tips[risk_profile])

# -------------------------
# FOOTER
# -------------------------
st.divider()
st.caption("🔒 Built for financial literacy and smarter asset planning — powered by Streamlit")
