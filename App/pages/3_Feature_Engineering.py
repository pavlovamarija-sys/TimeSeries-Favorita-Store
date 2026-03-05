import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.title("Feature Engineering")

# -----------------------------
# Robust dataset path (works regardless of working directory)
# -----------------------------
ROOT = Path(__file__).resolve().parents[2]  # TimeSeries/
DATA_PATH = ROOT / "Data" / "cleaned_timeseries.csv"

if not DATA_PATH.exists():
    st.error(f"Could not find dataset at: {DATA_PATH}")
    st.info("Check that the file exists in the Data/ folder and the name matches exactly.")
    st.stop()

# Load dataset
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

st.markdown("""
This page summarizes the engineered features used to improve forecasting performance.

**Update (current project status):**
- The final selected model is **Random Forest (tuned)**.
- **SARIMA** is included in the Streamlit forecast page **for comparison only** (loaded from a saved forecast, not trained inside Streamlit).

The engineered features capture **time structure**, **seasonality**, and **external drivers** (weather, holidays, macro, oil).
""")

# -----------------------------
# Feature groups from dataset columns
# -----------------------------
calendar_cols = [c for c in ["day_of_week", "day_of_month", "month", "month_name", "year"] if c in df.columns]
holiday_cols = [c for c in ["is_holiday", "is_national_holiday", "holiday_type", "locale", "before_1", "before_2", "after_1"] if c in df.columns]
macro_cols = [c for c in ["cpi", "min_wage", "dcoilwtico", "oil_lag_7"] if c in df.columns]
weather_cols = [c for c in ["ALLSKY_SFC_SW_DWN", "solar_7d", "PRECTOTCORR", "T2M", "T2M_MAX", "T2M_MIN", "is_sunny", "is_rainy"] if c in df.columns]

# -----------------------------
# Model-specific (engineered in app) features for Random Forest
# These may not exist as columns in the CSV, but are created in the Forecast page.
# -----------------------------
rf_engineered_in_app = ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7", "is_weekend"]

st.subheader("Feature Groups")

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Calendar Features (from dataset)")
    st.write(calendar_cols if calendar_cols else "None found")

    st.markdown("### Holiday Features (from dataset)")
    st.write(holiday_cols if holiday_cols else "None found")

with c2:
    st.markdown("### Macro & Oil Features (from dataset)")
    st.write(macro_cols if macro_cols else "None found")

    st.markdown("### Weather Features (from dataset)")
    st.write(weather_cols if weather_cols else "None found")

st.markdown("---")

st.subheader("Additional Engineered Features (used by Random Forest)")

st.markdown("""
These features are **created programmatically** (e.g., in the Forecast page) to help the Random Forest model learn short-term dynamics:

- **Lag features**: `lag_1`, `lag_7`, `lag_14`  
- **Rolling statistics**: `rolling_mean_7`, `rolling_std_7`  
- **Weekend flag**: `is_weekend`

These features are important because they capture recent sales momentum and weekly patterns that pure external drivers may not fully explain.
""")

st.markdown("---")

# -----------------------------
# Simple visualization: sales vs one selected feature
# -----------------------------
st.subheader("Quick Visualization")

numeric_candidates = [
    c for c in (weather_cols + macro_cols)
    if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
]
default_feature = "solar_7d" if "solar_7d" in numeric_candidates else (numeric_candidates[0] if numeric_candidates else None)

feature = (
    st.selectbox(
        "Select a numeric feature to visualize against unit_sales",
        options=numeric_candidates,
        index=numeric_candidates.index(default_feature) if default_feature in numeric_candidates else 0
    )
    if numeric_candidates else None
)

if feature:
    plot_df = df[["date", "unit_sales", feature]].dropna()

    fig, ax1 = plt.subplots(figsize=(8, 4))

    # Sales line
    ax1.plot(plot_df["date"], plot_df["unit_sales"], color="blue", label="unit_sales")
    ax1.set_xlabel("date")
    ax1.set_ylabel("unit_sales", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Feature line
    ax2 = ax1.twinx()
    ax2.plot(plot_df["date"], plot_df[feature], color="orange", label=feature)
    ax2.set_ylabel(feature, color="orange")
    ax2.tick_params(axis="y", labelcolor="orange")

    ax1.set_title(f"unit_sales vs {feature}")

    st.pyplot(fig)

    # -----------------------------
    # Automatic Interpretation
    # -----------------------------
    st.subheader("Automatic Interpretation")

    # 1) Pearson correlation (daily values)
    corr = plot_df["unit_sales"].corr(plot_df[feature])
    st.metric("Correlation with sales", f"{corr:.2f}")
    st.write(f"**Pearson correlation (daily values):** `{corr:.2f}`")

    # 2) Quartile comparison (presentation-friendly)
    try:
        bins = pd.qcut(plot_df[feature], q=4, duplicates="drop")
        bin_means = plot_df.groupby(bins)["unit_sales"].mean()

        low_mean = float(bin_means.iloc[0])
        high_mean = float(bin_means.iloc[-1])

        delta_pct = ((high_mean - low_mean) / low_mean * 100) if low_mean != 0 else 0.0

        st.write(
            f"**Average sales (low {feature} quartile):** `{low_mean:.1f}`  \n"
            f"**Average sales (high {feature} quartile):** `{high_mean:.1f}`"
        )

        if abs(delta_pct) < 5:
            st.info(
                f"When `{feature}` is **high vs low**, the average sales difference is small "
                f"(**{delta_pct:.1f}%**). This suggests **limited practical impact** on sales."
            )
        elif delta_pct > 0:
            st.success(
                f"When `{feature}` is **high**, average sales are **higher** than when it is low "
                f"by about **{delta_pct:.1f}%** (high vs low quartile)."
            )
        else:
            st.warning(
                f"When `{feature}` is **high**, average sales are **lower** than when it is low "
                f"by about **{abs(delta_pct):.1f}%** (high vs low quartile)."
            )

    except Exception as e:
        st.info("Could not compute quartile-based interpretation for this feature.")
        st.caption(f"Reason: {e}")

    st.caption(
        "Note: Correlation measures linear day-to-day co-movement. "
        "The quartile comparison shows average differences between low and high levels of the feature."
    )

    # -----------------------------
    # Feature Influence Ranking
    # -----------------------------
    st.subheader("Feature Influence Ranking")

    numeric_df = df.select_dtypes(include="number").copy()
    if "unit_sales" in numeric_df.columns:
        features = numeric_df.columns.drop("unit_sales")

        corr_values = numeric_df[features].corrwith(numeric_df["unit_sales"])
        corr_values = corr_values.abs().sort_values(ascending=False)

        top_features = corr_values.head(10)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        top_features.sort_values().plot(kind="barh", ax=ax2)

        ax2.set_title("Top Features Related to Sales")
        ax2.set_xlabel("Absolute Correlation with unit_sales")

        st.pyplot(fig2)
    else:
        st.warning("Column 'unit_sales' not found among numeric columns; cannot compute feature ranking.")
else:
    st.info("No numeric weather/macro features found to visualize.")