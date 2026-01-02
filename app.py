import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Rossmann Sales Explorer", layout="wide")
sns.set_style("whitegrid")

@st.cache_data(show_spinner=False)
def load_data():
    base = Path(__file__).parent
    train = pd.read_csv(base / "train.csv")
    store = pd.read_csv(base / "store.csv")
    train["Date"] = pd.to_datetime(train["Date"])
    return train, store

train_df, store_df = load_data()

st.title("Rossmann Store Sales Explorer")
st.markdown("Explore historical sales and a quick seasonal-naive forecast for any store.")

# Sidebar controls
all_store_ids = sorted(train_df["Store"].unique())
selected_store = st.sidebar.selectbox("Select store", all_store_ids, index=0)
forecast_horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=42, value=42, step=7)
last_window = st.sidebar.slider("History window to display (days)", min_value=60, max_value=400, value=180, step=10)

# Filter for the selected store and open days
store_data = train_df[(train_df["Store"] == selected_store) & (train_df["Open"] == 1)].copy()
store_data = store_data.sort_values("Date")
store_data.set_index("Date", inplace=True)

if store_data.empty:
    st.warning("No open-day records for this store.")
    st.stop()

# Basic stats
st.subheader(f"Store {selected_store} overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", f"{len(store_data):,}")
col2.metric("Avg sales", f"{store_data['Sales'].mean():,.0f}")
col3.metric("Median sales", f"{store_data['Sales'].median():,.0f}")
col4.metric("Promo uplift vs no promo", f"{(store_data.groupby('Promo')['Sales'].mean().pct_change().iloc[-1]*100):.1f}%")

# Time series plot (recent window)
st.subheader("Recent sales and forecast")
recent = store_data.tail(last_window)
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(recent.index, recent["Sales"], label="History", color="#1f77b4")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title(f"Store {selected_store} sales (last {last_window} days)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig, clear_figure=True)

# Simple seasonal-naive forecast (repeat last 7-day pattern)
last_week = store_data["Sales"].tail(7).values
repeats = int(np.ceil(forecast_horizon / 7))
forecast_vals = np.tile(last_week, repeats)[:forecast_horizon]
forecast_index = pd.date_range(store_data.index.max() + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
forecast_series = pd.Series(forecast_vals, index=forecast_index, name="Forecast")

# Combine recent history + forecast for visualization
combo = pd.concat([store_data["Sales"].tail(last_window), forecast_series])
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(combo.index, combo.values, label="History + Forecast", color="#ff7f0e")
ax2.axvline(store_data.index.max(), color="green", linestyle="--", label="Forecast start")
ax2.set_title(f"Seasonal-naive forecast (h={forecast_horizon} days)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Sales")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2, clear_figure=True)

st.markdown("### Forecast table")
st.dataframe(forecast_series.reset_index().rename(columns={"index": "Date"}), use_container_width=True)

st.markdown(
    """
    **Method:** Seasonal naive (repeats the last 7-day pattern). This is a fast baseline;
    swap in your trained SARIMA/ARIMA/Holt-Winters models here for better accuracy.
    """
)
