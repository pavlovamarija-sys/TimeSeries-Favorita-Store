import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.title("Time Series Forecasting – Favorita Store Sales")

st.markdown("""
### Project Overview

This project analyzes and forecasts retail sales using time series methods and machine learning models.

The goal is to understand how **external drivers** influence sales and to build models that improve forecasting accuracy.

Key external drivers included:

- Weather conditions (solar radiation, precipitation)
- Holidays
- Macroeconomic indicators (CPI, minimum wage)
- Oil prices

An interactive dashboard allows exploration of these relationships.
""")

st.subheader("Example: Sales Over Time")

# --- robust path handling (works from any working directory) ---
ROOT = Path(__file__).resolve().parents[2]   # TimeSeries/
DATA_PATH = ROOT / "Data" / "cleaned_timeseries.csv"

if not DATA_PATH.exists():
    st.error(f"Could not find dataset at: {DATA_PATH}")
    st.info("Check that the file exists in the Data/ folder and the name matches exactly.")
    st.stop()

# load dataset
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["unit_sales"] = pd.to_numeric(df["unit_sales"], errors="coerce")
df = df.dropna(subset=["date", "unit_sales"]).sort_values("date")

# simple time series plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(df["date"], df["unit_sales"])
ax.set_xlabel("Date")
ax.set_ylabel("Unit Sales")
ax.set_title("Sales Time Series")

st.pyplot(fig)