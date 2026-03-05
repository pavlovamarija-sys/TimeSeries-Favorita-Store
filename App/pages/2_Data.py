import streamlit as st
import pandas as pd
from pathlib import Path

st.title("Data Overview")

# --- robust path handling (works regardless of Streamlit working directory) ---
ROOT = Path(__file__).resolve().parents[2]   # TimeSeries/
DATA_PATH = ROOT / "Data" / "cleaned_timeseries.csv"

if not DATA_PATH.exists():
    st.error(f"Could not find dataset at: {DATA_PATH}")
    st.info("Check that the file exists in the Data/ folder and the name matches exactly.")
    st.stop()

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Basic dataset stats
st.subheader("Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Start date", str(df["date"].min().date()) if df["date"].notna().any() else "N/A")
c4.metric("End date", str(df["date"].max().date()) if df["date"].notna().any() else "N/A")

st.markdown("---")

# Columns
st.subheader("Column Descriptions")

column_info = {
    "date": "Date of the observation",
    "unit_sales": "Number of units sold",
    "day_of_week": "Day of the week (0=Monday)",
    "day_of_month": "Day number within the month",
    "month": "Month number",
    "month_name": "Name of the month",
    "year": "Year of observation",

    "is_holiday": "Indicator if the day is a holiday",
    "is_national_holiday": "Indicator if it is a national holiday",
    "holiday_type": "Type of holiday",
    "locale": "Holiday location level",

    "before_1": "One day before a holiday",
    "before_2": "Two days before a holiday",
    "after_1": "One day after a holiday",

    "dcoilwtico": "Daily oil price",
    "oil_lag_7": "Oil price lagged by 7 days",

    "cpi": "Consumer Price Index",
    "min_wage": "Minimum wage indicator",

    "ALLSKY_SFC_SW_DWN": "Solar radiation (NASA POWER)",
    "PRECTOTCORR": "Precipitation (mm)",
    "T2M": "Average temperature (°C)",
    "T2M_MAX": "Maximum temperature (°C)",
    "T2M_MIN": "Minimum temperature (°C)",

    "is_sunny": "Indicator for sunny weather conditions",
    "is_rainy": "Indicator for rainy days",
    "solar_7d": "7-day rolling average of solar radiation"
}

desc_df = pd.DataFrame(list(column_info.items()), columns=["Column", "Description"])
st.dataframe(desc_df, use_container_width=True)

# Preview
st.subheader("Preview (first 15 rows)")
st.dataframe(df.head(15), use_container_width=True)

# Missing values overview
st.subheader("Missing Values (top 15)")
na_counts = df.isna().sum().sort_values(ascending=False)
st.dataframe(na_counts.head(15).to_frame("missing_values"), use_container_width=True)

st.markdown("---")

# Data sources (presentation-friendly text)
st.subheader("Data Sources (high-level)")
st.markdown("""
This project combines daily sales with external data sources:

- **Retail sales**: daily unit sales  
- **Weather (NASA POWER API)**: solar radiation, temperature, precipitation  
- **Holidays**: national/local holiday indicators  
- **Macroeconomic indicators**: CPI and minimum wage  
- **Oil price**: crude oil series (used as an external driver)
""")