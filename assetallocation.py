import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback

# Streamlit page setup
st.set_page_config(page_title="Smart Asset Allocation – by Ishani", layout="wide")

# === SIDEBAR DESIGN ===
with st.sidebar:
    st.markdown("### 🪙 Smart Asset Allocator")
    st.image("https://cdn-icons-png.flaticon.com/512/3845/3845841.png", width=100)
    st.markdown("**Developed by Ishani** 💫")
    st.markdown("---")

    # Sidebar inputs
    st.subheader("Investor Profile")
    risk_level = st.selectbox("Select Risk Level", ["Low", "Moderate", "High"])
    investment_amount = st.number_input("Investment Amount (₹)", min_value=10000, value=100000, step=10000)
    st.markdown("---")

# === MAIN TITLE ===
st.title("💹 Diversified Asset Allocation Dashboard")
st.caption("Build your ideal investment mix across multiple asset classes with dynamic visualization and analysis.")

# === ASSET DATA CREATION ===
default_data = {
    "Asset Class": [
        "Equity (Large Cap)",
        "Equity (Mid Cap)",
        "Equity (Small Cap)",
        "Debt Funds",
        "Gold",
        "Real Estate",
        "International Equity",
        "REITs",
        "Commodities",
        "Crypto",
    ],
    "Expected Return (%)": [12, 14, 16, 8, 9, 10, 11, 10, 12, 20],
    "Risk Level": ["Moderate", "High", "High", "Low", "Low", "Moderate", "High", "Moderate", "High", "High"],
}

default_df = pd.DataFrame(default_data)

# === ALLOCATION LOGIC ===
def allocate_assets(df, risk, amount):
    df = df.copy()

    if risk == "Low":
        weights = np.array([0.15, 0.05, 0.05, 0.4, 0.15, 0.05, 0.05, 0.05, 0.03, 0.02])
    elif risk == "Moderate":
        weights = np.array([0.25, 0.15, 0.1, 0.2, 0.1, 0.05, 0.05, 0.03, 0.05, 0.02])
    else:  # High
        weights = np.array([0.3, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05, 0.03, 0.05, 0.02])

    df["Allocation (%)"] = (weights * 100).round(2)
    df["Investment (₹)"] = (weights * amount).round(2)
    df["Expected Return (₹)"] = (df["Investment (₹)"] * df["Expected Return (%)"] / 100).round(2)

    return df

# === ALLOCATE & DISPLAY ===
try:
    allocated_df = allocate_assets(default_df, risk_level, investment_amount)
    st.subheader("📊 Suggested Portfolio Allocation")
    st.dataframe(allocated_df, use_container_width=True)

    total_return = allocated_df["Expected Return (₹)"].sum()
    st.metric("💰 Total Expected Annual Return", f"₹{total_return:,.0f}")

    # Chart visualization
    st.markdown("### 📈 Allocation Visualization")
    st.bar_chart(allocated_df.set_index("Asset Class")["Investment (₹)"])

except Exception as e:
    st.error("Something went wrong in allocation. Check your data inputs.")
    st.text(traceback.format_exc())

# === EDITABLE SECTION ===
st.markdown("### 🧮 Customize Portfolio (Dynamic Editor)")
try:
    edited = st.data_editor(default_df, num_rows="dynamic", use_container_width=True)
    st.write("Customized Portfolio Preview:", edited)
except Exception as e:
    st.warning("Editor encountered an issue. Using default values.")

# === SAVE & DOWNLOAD ===
st.markdown("### 💾 Export Your Plan")
csv = allocated_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Allocation as CSV",
    data=csv,
    file_name="ishani_asset_allocation.csv",
    mime="text/csv",
)

st.markdown("---")
st.caption("© 2025 Created with ❤️ by Ishani | For educational purposes only.")
