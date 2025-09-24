# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------
# APP CONFIG
# ----------------------------
st.set_page_config(
    page_title="Crypto Volatility Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# HEADER
# ----------------------------
st.title("📉 Crypto Volatility Forecasting Dashboard")
st.markdown("""
**Hybrid GARCH + LSTM Model** to forecast 7-day BTC volatility
Built for traders to anticipate risk & adjust position sizing
*Data: Yahoo Finance | Model: Trained in Google Colab*
""")

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('/content/crypto_volatility_forecast_results.csv', parse_dates=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Data file 'crypto_volatility_forecast_results.csv' not found. Please ensure it is in the same directory as the app.")
    st.stop()


# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("🎛️ Dashboard Controls")

# Date Range Filter
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
start_date, end_date = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Model Toggle
show_garch = st.sidebar.checkbox("Show GARCH Forecast", value=True)
show_lstm = st.sidebar.checkbox("Show LSTM+GARCH Forecast", value=True)

# Filter data
mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
df_filtered = df.loc[mask]

# ----------------------------
# KEY METRICS
# ----------------------------
st.subheader("📊 Model Performance at a Glance")

col1, col2, col3 = st.columns(3)

# Calculate RMSE for filtered period
actual = df_filtered['Actual_Volatility']
garch_pred = df_filtered['GARCH_Volatility']
lstm_pred = df_filtered['Predicted_Volatility']

# Check if filtered data is not empty before calculating metrics
if not df_filtered.empty:
    rmse_garch = ((actual - garch_pred) ** 2).mean() ** 0.5
    rmse_lstm = ((actual - lstm_pred) ** 2).mean() ** 0.5
    improvement = (rmse_garch - rmse_lstm) / rmse_garch * 100 if rmse_garch != 0 else 0

    col1.metric("GARCH RMSE", f"{rmse_garch:.4f}")
    col2.metric("LSTM+GARCH RMSE", f"{rmse_lstm:.4f}", f"{improvement:.1f}% 🎯")
    col3.metric("Improvement", f"{improvement:.1f}%", "vs GARCH")
else:
    col1.info("No data in selected date range.")
    col2.info("No data in selected date range.")
    col3.info("No data in selected date range.")


# ----------------------------
# MAIN CHART
# ----------------------------
st.subheader("📈 Volatility Forecast vs Actual")

fig = go.Figure()

# Actual
fig.add_trace(go.Scatter(
    x=df_filtered['Date'],
    y=df_filtered['Actual_Volatility'],
    mode='lines',
    name='Actual Volatility',
    line=dict(color='blue', width=3)
))

# GARCH
if show_garch:
    fig.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['GARCH_Volatility'],
        mode='lines',
        name='GARCH Forecast',
        line=dict(color='green', dash='dot', width=2)
    ))

# LSTM
if show_lstm:
    fig.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Predicted_Volatility'],
        mode='lines',
        name='LSTM+GARCH Forecast',
        line=dict(color='red', width=2)
    ))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="7-Day Volatility",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=500
)

st.plotly_chart(fig, width='stretch')

# ----------------------------
# ERROR DISTRIBUTION
# ----------------------------
st.subheader("📉 Forecast Error Distribution")

col1, col2 = st.columns(2)

if not df_filtered.empty:
    with col1:
        fig_garch_error = px.histogram(
            df_filtered,
            x=(df_filtered['Actual_Volatility'] - df_filtered['GARCH_Volatility']),
            nbins=30,
            title="GARCH Forecast Errors",
            color_discrete_sequence=['green']
        )
        st.plotly_chart(fig_garch_error, width='stretch')

    with col2:
        fig_lstm_error = px.histogram(
            df_filtered,
            x=(df_filtered['Actual_Volatility'] - df_filtered['Predicted_Volatility']),
            nbins=30,
            title="LSTM+GARCH Forecast Errors",
            color_discrete_sequence=['red']
        )
        st.plotly_chart(fig_lstm_error, width='stretch')
else:
    st.info("No data in selected date range to display error distribution.")


# ----------------------------
# DATA TABLE
# ----------------------------
st.subheader("📋 Raw Forecast Data (Filtered)")
st.dataframe(df_filtered.style.format({
    'Actual_Volatility': '{:.4f}',
    'Predicted_Volatility': '{:.4f}',
    'GARCH_Volatility': '{:.4f}'
}), width='stretch')

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.caption("💡 Built by [Your Name] | Model trained in Google Colab | GitHub: [your-repo-link]")
st.caption("Hybrid volatility model helps crypto traders manage risk during bull/bear swings.")
