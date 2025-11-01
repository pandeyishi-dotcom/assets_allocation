import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Smart Asset Allocator", layout="wide")

# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center; color:#00FFAA;'>💰 Smart Asset Allocation & Goal Planner</h1>",
    unsafe_allow_html=True
)
st.markdown("### A real-time interactive dashboard with diversification logic, Monte Carlo simulation, and goal tracking.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
risk_profile = st.sidebar.selectbox("Select Risk Profile", ["Conservative", "Moderate", "Aggressive"])
investment_amount = st.sidebar.number_input("Initial Investment (₹)", min_value=1000, step=1000)
monthly_sip = st.sidebar.number_input("Monthly SIP (₹)", min_value=0, step=500)
years = st.sidebar.number_input("Investment Horizon (Years)", min_value=1, step=1)

# ---------------- ALLOCATION LOGIC ----------------
if risk_profile == "Conservative":
    data = {"Asset Class": ["Equity", "Debt", "Gold"], "Allocation (%)": [30, 60, 10]}
elif risk_profile == "Moderate":
    data = {"Asset Class": ["Equity", "Debt", "Gold"], "Allocation (%)": [50, 40, 10]}
else:
    data = {"Asset Class": ["Equity", "Debt", "Gold"], "Allocation (%)": [70, 20, 10]}

df = pd.DataFrame(data)

# ---------------- PIE CHART ----------------
fig = px.pie(df, names="Asset Class", values="Allocation (%)",
             title="Portfolio Allocation",
             color_discrete_sequence=px.colors.sequential.Tealgrn)
st.plotly_chart(fig, use_container_width=True)

# ---------------- FETCH LIVE DATA ----------------
st.markdown("## 📈 Live Market Snapshot (NSE)")
tickers = ["^NSEI", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]
try:
    live_data = yf.download(tickers, period="5d")["Close"].iloc[-1]
    live_df = pd.DataFrame(live_data).reset_index()
    live_df.columns = ["Ticker", "Price (₹)"]
    st.dataframe(live_df, use_container_width=True)
except Exception as e:
    st.warning("⚠️ Could not fetch live data. Check internet connection or Yahoo API limits.")

# ---------------- MONTE CARLO SIMULATION ----------------
def monte_carlo_sim(initial_investment, monthly_investment, years, sims=1000):
    years = int(years)
    mean_returns = 0.08
    volatility = 0.12
    results = []

    for _ in range(sims):
        portfolio_value = initial_investment
        for y in range(years):
            annual_return = np.random.normal(mean_returns, volatility)
            portfolio_value = (portfolio_value + monthly_investment * 12) * (1 + annual_return)
        results.append(portfolio_value)
    return results

sim_results = monte_carlo_sim(investment_amount, monthly_sip, years)
df_sim = pd.DataFrame(sim_results, columns=["Portfolio Value"])
percentiles = np.percentile(sim_results, [10, 25, 50, 75, 90])

st.markdown("## 🎲 Monte Carlo Simulation Results")
st.write(f"Median Outcome: ₹{percentiles[2]:,.0f}")
st.write(f"10th–90th Percentile Range: ₹{percentiles[0]:,.0f} – ₹{percentiles[4]:,.0f}")

fig_sim = px.histogram(df_sim, x="Portfolio Value", nbins=50,
                       title="Distribution of Portfolio Outcomes (Monte Carlo)",
                       color_discrete_sequence=["#00FFAA"])
st.plotly_chart(fig_sim, use_container_width=True)

# ---------------- GOAL PLANNER ----------------
st.markdown("## 🎯 Multi-Goal Planner")

goals_df = pd.DataFrame({
    "Goal": ["Retirement", "Child Education", "Travel Fund"],
    "Target Amount (₹)": [5000000, 2000000, 500000],
    "Years": [20, 10, 5]
})

edited_goals = st.data_editor(goals_df, num_rows="dynamic", use_container_width=True)

if st.button("Simulate Goals"):
    goal_summary = []
    for _, row in edited_goals.iterrows():
        years_goal = int(row["Years"])
        required_amount = float(row["Target Amount (₹)"])
        sim_goal = monte_carlo_sim(investment_amount, monthly_sip, years_goal)
        median_value = np.median(sim_goal)
        success_rate = np.mean(np.array(sim_goal) >= required_amount) * 100
        goal_summary.append({
            "Goal": row["Goal"],
            "Target (₹)": required_amount,
            "Median Projection (₹)": median_value,
            "Success Probability (%)": round(success_rate, 2)
        })

    summary_df = pd.DataFrame(goal_summary)
    st.dataframe(summary_df, use_container_width=True)

    fig_goal = px.bar(summary_df, x="Goal", y="Success Probability (%)",
                      title="Goal Success Probability",
                      color="Success Probability (%)",
                      color_continuous_scale="Tealgrn")
    st.plotly_chart(fig_goal, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>© 2025 Smart Asset Allocator | Powered by Streamlit + Yahoo Finance + Plotly</p>",
    unsafe_allow_html=True
)
